"""
evidence_analyzer.py
====================
Turns retrieved sources into classified, scored evidence chunks.

Design rules that make the output trustworthy:

1. Semantic similarity is a *pre-filter*, never a verdict. It can drop a
   chunk; it can never promote one.
2. An event match is judged on the claimed location, the claim's own specific
   details (keywords, numbers) and explicit correction language â€” not on
   "some capitalised word appears here".
3. Classification is gated: SUPPORT and CONTRADICTION require an exact date
   match (or an explicit correction of the claim), a strong event match, NLI
   agreement, and a primary source. A search snippet can only ever be context.
4. NLI receives input that is guaranteed to fit the model's token limit; the
   tokenizer budget is derived from the claim, not assumed.
"""

import re
from datetime import datetime

from claim_profile import (
    MONTH_NAMES,
    build_claim_profile,
    extract_dates,
    extract_years,
    extract_publication_date,
    keyword_forms,
    strip_publication_boilerplate,
)
from query_planner import WORDING_VARIANTS
from evidence_matcher import calculate_similarity, calculate_similarity_batch
from nli_checker import check_nli, NLI_MAX_TOKENS, NLI_SAFETY_MARGIN
from evidence_scorer import calculate_evidence_score, apply_class_ceiling
from source_reliability import get_source_profile
from tools import is_rolling_page


# ============================================================
# TOKENIZER (shared with nli_checker — loaded once per process)
# ============================================================

from nli_checker import check_nli, NLI_MAX_TOKENS, NLI_SAFETY_MARGIN, get_tokenizer

tokenizer = get_tokenizer()


def _count_tokens(text):
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=True, verbose=False))
    except Exception:
        return max(1, len(str(text)) // 4)


# ============================================================
# THRESHOLDS
# ============================================================

# Pre-filter only: below this a chunk is not even worth a date check.
MIN_SIMILARITY = 0.22

# A chunk about a different date needs to be clearly relevant to survive.
MIN_SIMILARITY_DIFFERENT_DATE = 0.45

# Upper bound on evidence chunks submitted to NLI per source.
MAX_NLI_CHUNKS_PER_SOURCE = 3

# A rolling page (liveblog, "live updates") is a sequence of timestamped
# entries, so it needs more passes before the claimed day's entry is found.
MAX_NLI_CHUNKS_ROLLING_PAGE = 6

# Upper bound on retained evidence chunks per source (post-classification).
MAX_CHUNKS_PER_SOURCE = 4


# ============================================================
# CHUNKING
# ============================================================

def _resolve_chunk_budget(claim):
    """
    Evidence-chunk budget that guarantees claim + chunk fit the NLI model.

    This is the structural fix for the old 512-token truncation: the chunker is
    given a budget derived from the claim length and the model's real limit, so
    NLI never sees a silently truncated premise.
    """
    claim_tokens = _count_tokens(claim)
    budget = NLI_MAX_TOKENS - NLI_SAFETY_MARGIN - claim_tokens
    return max(48, min(220, budget))


def split_into_chunks(text, max_tokens=220):
    """
    Paragraph- and sentence-aware chunking to produce small, cohesive,
    and focused evidence passages rather than large article blocks.
    """
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n|\n+", text)
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        token_count = _count_tokens(para)

        if token_count <= max_tokens:
            if len(para.split()) >= 6:
                chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = []
            current_tokens = 0

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                sent_tokens = _count_tokens(sent)

                if sent_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    words = sent.split()
                    sub_chunk = []
                    for word in words:
                        sub_chunk.append(word)
                        if _count_tokens(" ".join(sub_chunk)) >= max_tokens:
                            chunks.append(" ".join(sub_chunk))
                            sub_chunk = []
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                elif current_tokens + sent_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens

            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks


def _dedupe_chunks(chunks, threshold=0.80):
    """
    Drop near-identical chunks inside one source.

    Repeated boilerplate and duplicated paragraphs otherwise make a single
    article look like many pieces of evidence.
    """
    from source_reliability import jaccard, shingle_set

    kept = []
    signatures = []

    for chunk in chunks:
        signature = shingle_set(chunk, 5)
        if not signature:
            continue

        duplicate = False
        for existing in signatures:
            if jaccard(signature, existing) >= threshold:
                duplicate = True
                break

        if duplicate:
            continue

        signatures.append(signature)
        kept.append(chunk)

    return kept


# How many multi-paragraph windows, in addition to the single paragraphs, may
# be offered to the NLI model for one source.
#
# A publisher often states a claim across ADJACENT paragraphs: NASA writes
# "Apollo 11 astronauts ... sit atop another Saturn V", then "When the lunar
# module lands ... The Eagle has landed", then "Armstrong is ready to plant
# the first human foot on another world". Read on its own, each paragraph is
# NEUTRAL against "Apollo 11 landed humans on the Moon in July 1969", so the
# evidence layer can never support the claim even though the article states it
# plainly. Offering the surrounding paragraphs as a window lets the model read
# the passage the way a reader would.
MAX_WINDOW_CHUNKS_PER_SOURCE = 3


def _build_context_windows(chunks, anchors, max_windows, max_tokens):
    """
    Contiguous windows of adjacent paragraphs around the strongest paragraphs.

    A window contains only text that is already in the article, so nothing is
    invented. Windows are bounded by the NLI token budget, and the candidate
    anchors are visited round-robin so that one paragraph cannot consume the
    whole window budget.

    For each anchor we generate both forward windows (anchor is the first
    paragraph in the window) and backward windows (anchor is the last paragraph
    in the window).  This ensures that the critical evidence paragraph is
    reachable regardless of which adjacent paragraph the similarity pre-filter
    ranked highest.
    """
    if not chunks or not anchors or max_windows <= 0:
        return []

    # For each anchor paragraph, collect forward and backward windows of 2 and
    # 3 paragraphs.  Forward: anchor is the start (lo = anchor).  Backward:
    # anchor is the end (hi = anchor + 1, lo = hi - size).  Duplicate windows
    # are collapsed later via the `seen` set.
    per_anchor = []
    for anchor in anchors:
        options = []
        for size in (2, 3):
            # Forward window: [anchor, anchor+size)
            lo_f, hi_f = anchor, anchor + size
            if hi_f <= len(chunks):
                options.append((lo_f, hi_f))
            # Backward window: [anchor-size+1, anchor+1)
            lo_b = anchor - size + 1
            hi_b = anchor + 1
            if lo_b >= 0 and hi_b <= len(chunks) and (lo_b, hi_b) != (lo_f, hi_f):
                options.append((lo_b, hi_b))
        if options:
            per_anchor.append(options)

    windows = []
    seen = set()
    # Round-robin over the anchors so that one paragraph cannot consume the
    # whole window budget: every anchor gets its shortest window before any
    # anchor gets its longer ones.
    max_options = max((len(o) for o in per_anchor), default=0)
    for round_index in range(max_options):
        for options in per_anchor:
            if round_index >= len(options):
                continue
            lo, hi = options[round_index]
            window = "\n\n".join(chunks[lo:hi])
            if _count_tokens(window) > max_tokens:
                continue
            key = window[:160]
            if key in seen:
                continue
            seen.add(key)
            windows.append(window)
            if len(windows) >= max_windows:
                return windows
    return windows


# ============================================================
# CLAIM ATTRIBUTES
# ============================================================

def extract_claim_attributes(claim):
    """
    Claim attributes for evidence matching, read from the shared claim parser.
    """
    profile = build_claim_profile(claim)

    return {
        "clean_claim": profile["clean_claim"],
        "claim_dates": profile["dates"],
        "claim_years": profile["years"],
        "claim_date": profile["primary_date"],
        "locations": profile["locations"],
        "location_terms": profile["location_terms"],
        "keywords": profile["keywords"],
        "phrases": profile["phrases"],
        "has_date": profile["has_date"],
        "claim_type": profile.get("claim_type", "other_factual_claim"),
        "relative_date_ambiguous": profile.get("relative_date_ambiguous", False),
        "profile": profile,
    }


# ============================================================
# DATE COMPATIBILITY
# ============================================================

# ============================================================
# EVENT / DATE ASSOCIATION
# ============================================================

def _stem(word):
    """A crude stem, enough to match 'landing' with 'landed'."""
    word = re.sub(r"[^a-z0-9]", "", str(word or "").lower())
    for suffix in ("ings", "ing", "edly", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _claim_event_stems(claim_attrs):
    """The stems of the claim's event vocabulary."""
    stems = set()
    for keyword in (claim_attrs or {}).get("keywords") or []:
        stem = _stem(keyword)
        if stem:
            stems.add(stem)
    for phrase in (claim_attrs or {}).get("phrases") or []:
        for word in str(phrase).split():
            stem = _stem(word)
            if stem:
                stems.add(stem)
    return stems


def _split_sentences(text):
    """Sentences of a passage, in document order."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _event_date_binding(evidence, claim_attrs, claim_date):
    """
    Associate the dates in a passage with the event they describe.

    A passage can carry several dates that describe DIFFERENT events:
    "â€¦on their 20 July 1969 moon landing. They presented this gift â€¦ on
    20 July 1970." Finding the claim's date SOMEWHERE in the passage is not
    enough to say the passage supports the claim: the date has to be the date
    of the CLAIMED event.

    A date is bound to the event when it appears in a sentence that also
    mentions the claim's event vocabulary. A date in a sentence that does not
    mention the event describes something else and is ignored.

    Returns (clear_conflict, soft_conflict):
      clear - an event-bearing sentence gives the event the same month and day
              as the claim but a different year (the classic "one year later"
              discrepancy), or a different month or year for a claim that
              states no day;
      soft  - an event-bearing sentence gives the event a different date in
              another way.
    """
    if not claim_date:
        return False, False

    stems = _claim_event_stems(claim_attrs)
    if not stems:
        return False, False

    claim_month = claim_date.get("month")
    claim_year = claim_date.get("year")
    claim_day = claim_date.get("day")
    if claim_month is None or claim_year is None:
        return False, False

    clear = False
    soft = False

    for sentence in _split_sentences(evidence):
        sentence_dates = extract_dates(sentence)
        if not sentence_dates:
            continue

        words = [_stem(w) for w in re.findall(r"[A-Za-z]+", sentence.lower())]
        if not stems.intersection(words):
            # The sentence does not mention the claimed event, so any date in
            # it belongs to a different event.
            continue

        for date in sentence_dates:
            if date.get("month") is None or date.get("year") is None:
                continue
            if (date["month"], date["year"]) == (claim_month, claim_year):
                continue
            if claim_day is None or date.get("day") == claim_day:
                clear = True
            else:
                soft = True

    return clear, soft


def _check_date_compatibility_core(
    claim,
    evidence,
    claim_attrs=None,
    publication_date=None,
    source_dates=None,
    source_years=None,
):
    """
    Compare the *event* date of the evidence with the claim date.

    Statuses:
        match                the claimed day/month/year is explicitly written
        same_day_published   the page was published on the claimed date and
                             describes the claimed event (weaker than match,
                             because the date comes from page metadata)
        same_year            the claimed year is present, the exact date is not
        different            the evidence is explicitly about another date
        unknown              no usable date information

    The result also carries ``event_date_conflict`` and
    ``event_date_mismatch``: whether the passage states the CLAIMED event on a
    different date. A passage can contain the claim's date while dating the
    claimed event to another date, so the date has to be associated with the
    event it describes, not merely found somewhere in the passage.

    `source_dates` / `source_years` let a chunk inherit the date context of the
    article it came from, so a passing mention of 1916 inside a 2024 article is
    not mistaken for the event date.
    """
    claim_attrs = claim_attrs or extract_claim_attributes(claim)
    claim_date = claim_attrs.get("claim_date")
    claim_years = claim_attrs.get("claim_years") or set()

    evidence = strip_publication_boilerplate(evidence)

    chunk_dates = extract_dates(evidence)
    chunk_years = extract_years(evidence)

    all_dates = list(chunk_dates) + list(source_dates or [])
    all_years = set(chunk_years) | set(source_years or set())

    if not claim_date and not claim_years:
        return {
            "status": "unknown",
            "reason": "Claim does not contain a detectable date.",
            "claim_year_absent": False,
        }
    if not all_dates and not all_years:
        return {
            "status": "unknown",
            "reason": "Evidence does not contain a detectable date.",
            "claim_year_absent": False,
        }

    claim_year_absent = bool(all_years) and bool(claim_years) and not (
        claim_years & all_years
    )

    # --------------------------------------------------------
    # Month-precision claim ("... in July 1969")
    #
    # The claim names a month and a year but no day. Evidence that states the
    # same month and year satisfies the claim - it is simply more specific.
    # Evidence that states a DIFFERENT month of the same year contradicts it.
    # --------------------------------------------------------
    if claim_date and claim_date.get("day") is None:
        claim_month = claim_date["month"]
        claim_year = claim_date["year"]

        def _render(dates):
            return ", ".join(
                f"{d['day']} {d['month_name']}" if d.get("day") else d["month_name"]
                for d in dates[:3]
            )

        same_month = [
            d for d in all_dates
            if d["month"] == claim_month and d["year"] == claim_year
        ]
        if same_month:
            return {
                "status": "match",
                "reason": (
                    f"Evidence explicitly refers to {_render(same_month)} "
                    f"{claim_year}, which matches the claimed "
                    f"{claim_date['month_name']} {claim_year}."
                ),
                "claim_year_absent": False,
            }

        other_month = [
            d for d in all_dates
            if d["year"] == claim_year and d["month"] != claim_month
        ]
        if other_month:
            return {
                "status": "different",
                "reason": (
                    f"Evidence explicitly refers to {_render(other_month)} "
                    f"{claim_year} rather than the claimed "
                    f"{claim_date['month_name']} {claim_year}."
                ),
                "claim_year_absent": False,
            }

        if claim_years and all_years and (claim_years & all_years):
            return {
                "status": "same_year",
                "reason": (
                    f"Evidence mentions the claimed year "
                    f"({sorted(claim_years & all_years)[0]}) but not the claimed "
                    f"month."
                ),
                "claim_year_absent": False,
            }

        return {
            "status": "different",
            "reason": (
                f"Evidence refers to {sorted(all_years)} and never to the "
                f"claimed year {sorted(claim_years)}."
            ),
            "claim_year_absent": True,
        }

    if claim_date:
        claim_key = (claim_date["day"], claim_date["month"], claim_date["year"])
        claim_year = claim_date["year"]

        if claim_key in {(d["day"], d["month"], d["year"]) for d in chunk_dates}:
            return {
                "status": "match",
                "reason": (
                    f"Evidence explicitly refers to {claim_date['day']} "
                    f"{claim_date['month_name']} {claim_date['year']}."
                ),
                "claim_year_absent": False,
            }

        # A same-year, different day date is a genuine conflict. A date in a
        # different year is only a conflict when the source never mentions the
        # claimed year at all: a 2024 article that says "since 1916" is quoting
        # history, while an article that only ever says 2019 is about 2019.
        claim_year_present = bool(claim_years and all_years and (claim_years & all_years))

        # A date that does not name a day (a month-precision date such as
        # "July 1969") is the same month stated less precisely. It is NOT in
        # conflict with a date that names a day, so the comparison only
        # considers the fields that BOTH dates actually specify. Comparing
        # raw tuples would also mix None with int, which cannot be ordered.
        conflicting_keys = set()
        for key in {(d["day"], d["month"], d["year"]) for d in all_dates}:
            if key == claim_key:
                continue
            if key[2] == claim_year:
                if key[1] != claim_key[1]:
                    conflicting_keys.add(key)
                elif (
                    key[0] is not None
                    and claim_key[0] is not None
                    and key[0] != claim_key[0]
                ):
                    conflicting_keys.add(key)
            elif not claim_year_present:
                conflicting_keys.add(key)

        # Sort on a type-safe key: a missing day must never be ordered
        # against an integer day.
        conflicting = sorted(
            conflicting_keys,
            key=lambda k: (k[2] or 0, k[1] or 0, k[0] or 0),
        )

        if conflicting:
            conflicting_set = set(conflicting)
            rendered = ", ".join(
                (
                    f"{d['day']:02d} {d['month_name'][:3]} {d['year']}"
                    if d.get("day")
                    else f"{d['month_name'][:3]} {d['year']}"
                )
                for d in all_dates
                if (d["day"], d["month"], d["year"]) in conflicting_set
            )
            return {
                "status": "different",
                "reason": (
                    f"Evidence explicitly refers to a different date "
                    f"({rendered}) than the claim."
                ),
                "claim_year_absent": claim_year_absent,
            }

        # The page itself was published on the claimed date. That is weaker
        # than the text stating the date, but an article published on the
        # claimed day about the claimed event is legitimate same-day evidence.
        if publication_date and (
            publication_date.get("day"),
            publication_date.get("month"),
            publication_date.get("year"),
        ) == claim_key:
            return {
                "status": "same_day_published",
                "reason": (
                    f"Page was published on the claimed date "
                    f"({claim_date['day']} {claim_date['month_name']} "
                    f"{claim_date['year']}); the text does not restate it."
                ),
                "claim_year_absent": False,
            }

    if claim_years and all_years:
        shared = claim_years & all_years
        if shared:
            return {
                "status": "same_year",
                "reason": (
                    f"Evidence mentions the claimed year "
                    f"({sorted(shared)[0]}) but not the exact date."
                ),
                "claim_year_absent": False,
            }

        return {
            "status": "different",
            "reason": (
                f"Evidence refers to {sorted(all_years)} and never to the "
                f"claimed year {sorted(claim_years)}."
            ),
            "claim_year_absent": True,
        }

    return {
        "status": "unknown",
        "reason": "Date relationship could not be determined.",
        "claim_year_absent": claim_year_absent,
    }


def check_date_compatibility(
    claim,
    evidence,
    claim_attrs=None,
    publication_date=None,
    source_dates=None,
    source_years=None,
):
    """
    Compare the *event* date of the evidence with the claim date.

    Wraps the status computation and adds the event/date association: whether
    the passage states the CLAIMED event on a different date. A passage can
    contain the claim's date while dating the claimed event to another date
    ("â€¦on their 20 July 1969 moon landing. They presented this gift â€¦ on
    20 July 1970"), so a date has to be associated with the event it
    describes, not merely found somewhere in the passage.
    """
    result = _check_date_compatibility_core(
        claim, evidence, claim_attrs,
        publication_date, source_dates, source_years,
    )

    attrs = claim_attrs if claim_attrs is not None else extract_claim_attributes(claim)
    clear, soft = _event_date_binding(
        strip_publication_boilerplate(evidence),
        attrs,
        attrs.get("claim_date"),
    )
    result["event_date_conflict"] = clear
    result["event_date_mismatch"] = soft

    if clear:
        result["reason"] = (
            f"{result['reason']} The passage states the claimed event on a "
            f"different date."
        )
    elif soft:
        result["reason"] = (
            f"{result['reason']} The passage associates the claimed event with "
            f"a different date."
        )

    return result


# ============================================================
# EVENT MATCH
# ============================================================

_POLICY_MARKERS = [
    "safety regulations", "safety guidelines", "compliance rules",
    "safety norms", "building codes", "advisory guidelines",
    "safety manual", "regulations for", "guidelines issued",
]

_INCIDENT_MARKERS = [
    "broke out", "fire", "killed", "injured", "engulfed", "erupted",
    "occurred", "happened", "destroyed", "damaged", "blaze", "gutted",
    "accident", "incident", "collapse", "blast", "crash", "dead",
    "casualties", "flood", "waterlogged", "earthquake", "derailed",
    "landed", "launched", "defeated", "won", "beat", "elected", "resigned",
    "withdrew", "withdrawal", "announced", "signed", "passed",
]

# Aggregate / statistical reporting: quotes a date and a place but describes a
# pattern rather than the claimed incident.
_AGGREGATE_MARKERS = (
    "data showed", "data released", "data shows", "statistics showed",
    "figures showed", "in the first half", "in the second half",
    "year-on-year", "on average", "per cent of", "a total of", "cumulative",
    "data indicated", "record shows", "records show", "survey found",
    "compared with", "as against",
)

_DEBUNK_RE = re.compile(    r"\b(fact[\s\-]?check|debunk|falsely claimed|false claim|viral claim|"
    r"hoax|misleading|rumou?r|untrue|no such incident|clarified that no|"
    r"did not happen|did not occur|actually took place on|actually occurred on|"
    r"old video|old photo|old image|misinformation|"
    r"never happened|no evidence that|incorrect|false information)\b",
    re.IGNORECASE,
)

# A different place only counts as a conflict when the text actually places the
# event somewhere ("in Agra", "at Kolkata"). A stray capitalised word is not a
# location conflict â€” that mistake demoted real evidence before.
_GEO_CONTEXT_RE = re.compile(
    r"\b(?:in|at|near|from|around|outside|inside|across)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
)


def _contains_term(text_lower, term):
    if not term:
        return False
    if " " in term:
        return term.lower() in text_lower
    return re.search(rf"\b{re.escape(term.lower())}\b", text_lower) is not None


def _contains_keyword(text_lower, keyword):
    """Word-boundary match that also accepts simple English forms."""
    for form in keyword_forms(keyword):
        if re.search(rf"\b{re.escape(form)}\b", text_lower):
            return True
    return False


def _contains_phrase(text_lower, phrase):
    """Phrase match that tolerates one word differing in inflection."""
    words = phrase.lower().split()
    if not words:
        return False
    if phrase.lower() in text_lower:
        return True

    window = len(words)
    tokens = re.findall(r"[a-z0-9']+", text_lower)
    for start in range(0, max(0, len(tokens) - window + 1)):
        matched = 0
        for offset, word in enumerate(words):
            forms = {word} | set(keyword_forms(word))
            if tokens[start + offset] in forms:
                matched += 1
        if matched >= window - 1 and matched >= 2:
            return True
    return False


def _numeric_details(claim):
    """Numbers stated in the claim (casualties, floors, times) â€” hard details."""
    return set(re.findall(r"\b\d{1,4}\b", claim or ""))


def _detail_coverage(claim_attrs, chunk, chunk_lower):
    """
    How much of the claim's own specific content the chunk reproduces.

    Claim keywords are matched through their English forms, and a claim keyword
    also counts as covered when the chunk uses a known alternative wording
    ("flooded" in the claim, "waterlogging" in the article). A verbatim event
    phrase is worth more than loose keywords.
    """
    keywords = claim_attrs.get("keywords") or []
    phrases = claim_attrs.get("phrases") or []

    if not keywords:
        return 0.0, {"keyword_hits": [], "phrase_hits": []}

    keyword_hits = []
    synonym_hits = []

    for keyword in keywords:
        if _contains_keyword(chunk_lower, keyword):
            keyword_hits.append(keyword)
            continue
        for alternative in WORDING_VARIANTS.get(keyword, []):
            if _contains_phrase(chunk_lower, alternative) or _contains_keyword(
                chunk_lower, alternative
            ):
                synonym_hits.append(keyword)
                break

    phrase_hits = [ph for ph in phrases if _contains_phrase(chunk_lower, ph)]

    matched_keywords = len(set(keyword_hits) | set(synonym_hits))
    keyword_ratio = matched_keywords / len(keywords)

    # A verbatim event phrase is a much stronger signal than loose keywords.
    coverage = min(1.0, keyword_ratio + (0.30 * len(phrase_hits)))

    return round(coverage, 4), {
        "keyword_hits": sorted(set(keyword_hits)),
        "synonym_hits": sorted(set(synonym_hits)),
        "phrase_hits": phrase_hits,
    }


def evaluate_event_match(
    claim,
    chunk,
    claim_attrs,
    date_status,
    similarity,
    nli_entails=False,
    event_date_conflict=False,
):
    """
    Judge whether the chunk describes the SAME event as the claim.

    Returns (level, signals) where level is one of:
        strong    same place, same specific details, compatible date
        moderate  same place and topic, but some claim detail is missing
        weak      same topic only
        none      different place, different event, or pure policy text
    """
    chunk_lower = chunk.lower()
    signals = {}

    # --------------------------------------------------------
    # 1. Location alignment
    # --------------------------------------------------------
    locations = claim_attrs.get("locations") or []
    location_terms = claim_attrs.get("location_terms") or []

    matched_terms = [
        term for term in location_terms if _contains_term(chunk_lower, term)
    ]
    location_match = bool(matched_terms)

    # A claim that states no location cannot be penalised for the passage not
    # naming one. The classifier already applies this rule; the event matcher
    # must apply the same one, otherwise a location-less claim can never reach
    # event_match 'strong' and therefore can never be SUPPORT, however good the
    # evidence is.
    claim_has_location = bool(locations)
    location_ok = location_match or not claim_has_location

    location_conflict = False
    conflicting_place = ""

    if locations and not location_match:
        claim_lower = claim_attrs.get("clean_claim", "").lower()
        for match in _GEO_CONTEXT_RE.finditer(chunk):
            place = match.group(1)
            place_lower = place.lower()
            if len(place_lower) < 4:
                continue
            if place_lower in claim_lower:
                continue
            if _contains_term(claim_lower, place_lower):
                continue
            location_conflict = True
            conflicting_place = place
            break

    signals["location_match"] = location_match
    signals["location_ok"] = location_ok
    signals["claim_has_location"] = claim_has_location
    signals["location_terms_matched"] = matched_terms[:5]
    signals["location_conflict"] = location_conflict
    signals["conflicting_place"] = conflicting_place

    # --------------------------------------------------------
    # 2. Claim-specific detail coverage
    # --------------------------------------------------------
    coverage, detail = _detail_coverage(claim_attrs, chunk, chunk_lower)
    signals["detail_coverage"] = coverage
    signals["detail"] = detail

    claim_numbers = _numeric_details(claim_attrs.get("clean_claim", ""))
    number_hits = sorted(
        number for number in claim_numbers
        if re.search(rf"\b{re.escape(number)}\b", chunk_lower)
    )
    signals["claim_number_hits"] = number_hits
    signals["claim_numbers"] = sorted(claim_numbers)

    # --------------------------------------------------------
    # 3. Policy text and correction language
    # --------------------------------------------------------
    is_pure_policy = (
        any(marker in chunk_lower for marker in _POLICY_MARKERS)
        and not any(marker in chunk_lower for marker in _INCIDENT_MARKERS)
    )
    has_debunking = bool(_DEBUNK_RE.search(chunk))
    signals["pure_policy"] = is_pure_policy
    signals["debunking"] = has_debunking

    incident_match = any(marker in chunk_lower for marker in _INCIDENT_MARKERS)

    # Aggregate reporting ("data showed", "in the first half") quotes the claimed
    # date and place but reports a pattern, not the claimed incident.
    aggregate = any(marker in chunk_lower for marker in _AGGREGATE_MARKERS)
    signals["aggregate"] = aggregate

    # --------------------------------------------------------
    # 4. Verdict
    # --------------------------------------------------------
    if is_pure_policy:
        return "none", signals

    if location_conflict:
        return "none", signals

    if date_status == "different":
        if not has_debunking and not event_date_conflict:
            return "weak", signals
        # The passage debunks the claim or explicitly binds the claimed event to
        # a conflicting date. When the event and location are sufficiently
        # aligned, this is a strong event match: the event is identified, only
        # the date is wrong.
        if location_ok and coverage >= 0.30 and similarity >= 0.50:
            return "strong", signals
        if location_ok and coverage >= 0.30 and similarity >= 0.45:
            return "moderate", signals
        return "weak", signals

    if date_status in ("match", "same_day_published"):
        # The date gate has already been passed. What remains is whether this
        # passage is about the claimed event in the claimed place: the claimed
        # location, at least a third of the claim's own content words, and a
        # semantically close passage. An explicitly written date counts as a
        # claim-specific detail in its own right; a same-day publication date
        # does not, so it needs a further detail.
        base_ok = (
            location_ok
            and coverage >= 0.30
            and similarity >= 0.50
        )
        detail_ok = (
            date_status == "match"
            or bool(number_hits)
            or bool(detail["phrase_hits"])
            or coverage >= 0.65
            # When the date comes from page metadata rather than the text, the
            # NLI judgement (run against the same-day context) is what has to
            # confirm that this passage is about the claimed event.
            or nli_entails
        )

        # Reporting a pattern rather than the incident.
        if aggregate and not detail["phrase_hits"] and coverage < 0.65:
            return "weak", signals

        if base_ok and detail_ok:
            return "strong", signals
        if location_ok and coverage >= 0.30 and similarity >= 0.45:
            return "moderate", signals
        return "weak", signals

    if date_status == "same_year":
        if location_ok and coverage >= 0.60 and incident_match:
            return "moderate", signals
        return "weak", signals

    # Unknown date: the chunk must earn its place on location and detail.
    if location_ok and coverage >= 0.60 and incident_match and similarity >= 0.45:
        return "moderate", signals

    return "weak", signals


# ============================================================
# EVIDENCE CLASSIFICATION
# ============================================================

def classify_evidence(signals, evidence_type, source_profile, claim_type=None):
    """
    Classify one chunk into SUPPORT / CONTRADICTION / RELATED_CONTEXT /
    NON_EVIDENCE.

    Hard rules (none of them can be bypassed by similarity):
      * a search snippet can only ever be RELATED_CONTEXT or NON_EVIDENCE
      * a non-primary source (blog, social, aggregator) can never SUPPORT or
        CONTRADICT on its own
      * SUPPORT requires: NLI entailment + similarity + coverage + primary source
      * CONTRADICTION requires: NLI contradiction (or explicit date conflict)
        + similarity + coverage + primary source

    Event claims (historical_event, current_event) additionally require
    strong event match and date association. Scientific / geographical /
    statistical / other factual claims use proposition-based gates instead:
    they do not need a date, location, or incident marker.

    `claim_type` (from the claim profile) selects which gate set applies.
    """
    date_status = signals.get("date_status", "unknown")
    event_match = signals.get("event_match", "weak")
    similarity = signals.get("similarity", 0.0)
    nli_label = (signals.get("raw_nli_label") or "neutral").lower()
    nli_score = signals.get("raw_nli_score", 0.0)
    nli_status = signals.get("nli_status", "ok")

    location_match = signals.get("location_match", False)
    coverage = signals.get("detail_coverage", 0.0)
    has_debunking = signals.get("debunking", False)

    is_snippet = evidence_type == "search_snippet"
    is_primary = bool(source_profile.get("is_primary"))

    has_claim_location = bool(signals.get("claim_has_location"))
    location_ok = location_match or not has_claim_location

    # NLI could not be trusted: never classify on it.
    nli_unusable = nli_status in ("error", "unavailable")

    # A multi-paragraph window may SUPPORT a claim, but it may not REFUTE one.
    # The NLI model reads a diluted premise unreliably, and a spurious
    # contradiction is far more harmful than a missed one: refutation still
    # requires a single focused passage, so the evidence floor is unchanged.
    is_window = signals.get("passage_kind") == "window"

    # Claim type determines which evidence gate set applies.
    # When not provided, default to event-based gates (existing behaviour) to
    # preserve backward compatibility with callers that do not classify claims.
    claim_type = claim_type or signals.get("claim_type")
    if claim_type is None:
        claim_type = "historical_event"
    is_event_claim = claim_type in ("historical_event", "current_event")

    # --------------------------------------------------------
    # EVENT / DATE ASSOCIATION
    #
    # The passage states the CLAIMED event on a date other than the claimed
    # one. A passage that says so cannot entail the claim, whatever the NLI
    # model scored: the model is judging the topic, not the date, and it
    # routinely reports entailment for a passage that names a different year.
    # --------------------------------------------------------
    event_date_conflict = bool(signals.get("event_date_conflict"))
    event_date_mismatch = bool(signals.get("event_date_mismatch"))

    # --------------------------------------------------------
    # CONTRADICTION
    #
    # Event claims: strong event match + date association + NLI contradiction
    # (or explicit date conflict).
    # Non-event claims: direct contradiction of the proposition via NLI +
    # similarity + coverage, without requiring event match or dates.
    # --------------------------------------------------------
    contradiction_date_ok = (
        date_status == "match"
        or (date_status == "different" and has_debunking)
        or event_date_conflict
    )

    if claim_type in ("historical_event", "current_event"):
        # Event claim: keep the existing strict gate.
        if (
            not is_snippet
            and not is_window
            and is_primary
            and not nli_unusable
            and similarity >= 0.40
            and event_match == "strong"
            and location_ok
            and contradiction_date_ok
            and (
                (nli_label == "contradiction" and nli_score >= 0.75)
                or event_date_conflict
            )
        ):
            return "CONTRADICTION"
    else:
        # Non-event claim (scientific_fact, geographical_fact,
        # statistical_claim, other_factual_claim): proposition-based gate.
        # No date or incident marker required. A passage that directly
        # contradicts the proposition with NLI contradiction qualifies.
        if (
            not is_snippet
            and not is_window
            and is_primary
            and not nli_unusable
            and similarity >= 0.40
            and coverage >= 0.30
            and location_ok
            and not event_date_conflict
            and not event_date_mismatch
            and nli_label == "contradiction"
            and nli_score >= 0.75
        ):
            return "CONTRADICTION"

    # --------------------------------------------------------
    # SUPPORT
    #
    # Event claims: strong event match + exact claimed date + location +
    # claim-detail coverage + NLI entailment.
    # Non-event claims: NLI entailment + similarity + coverage + primary
    # source; no date or incident marker required.
    # --------------------------------------------------------
    if claim_type in ("historical_event", "current_event"):
        if (
            not is_snippet
            and is_primary
            and not nli_unusable
            and nli_label == "entailment"
            and nli_score >= 0.70
            and similarity >= 0.40
            and event_match == "strong"
            and location_ok
            and coverage >= 0.30
            and not event_date_conflict
            and not event_date_mismatch
            and (
                date_status in ("match", "same_day_published")
                or (
                    # High-confidence NLI entailment (>=0.90) with strong event
                    # match and substantial coverage means the model has already
                    # confirmed the passage entails the claim including its date.
                    # Explicit date restatement is then redundant; requiring it
                    # rejects genuinely supporting passages whose text refers to
                    # the event by relative date ("on Sunday") rather than
                    # restating the calendar date.
                    nli_score >= 0.90
                    and coverage >= 0.50
                    and date_status == "unknown"
                )
            )
        ):
            return "SUPPORT"
    else:
        # Non-event claim: proposition-based gate.
        if (
            not is_snippet
            and is_primary
            and not nli_unusable
            and nli_label == "entailment"
            and nli_score >= 0.70
            and similarity >= 0.40
            and coverage >= 0.30
            and location_ok
            and not event_date_conflict
            and not event_date_mismatch
        ):
            return "SUPPORT"

    # --------------------------------------------------------
    # CONTEXT
    #
    # Context still needs a substantive anchor (the claimed place, or the
    # claim's own specific details). Raw similarity alone is never enough.
    # --------------------------------------------------------
    has_anchor = location_match or coverage >= 0.30

    if event_match in ("moderate", "strong"):
        return "RELATED_CONTEXT"

    if date_status in ("same_year", "match") and similarity >= 0.35 and has_anchor:
        return "RELATED_CONTEXT"

    if date_status == "different" and similarity >= 0.45 and has_anchor:
        return "RELATED_CONTEXT"

    if is_snippet and has_anchor:
        return "RELATED_CONTEXT"

    return "NON_EVIDENCE"


# ============================================================
# EVIDENCE ANALYSIS
# ============================================================

def _score_for_class(evidence_class, signals, reliability, evidence_type):
    """
    Evidence score for a classified chunk.

    The composite score is computed from date / event / detail / location /
    NLI / quality, then bounded by the class ceiling and by the trust of the
    retrieval mode, so a snippet can never score like a scraped article.
    """
    score = calculate_evidence_score(
        similarity=signals.get("similarity", 0.0),
        nli_label=signals.get("evidence_class_label", "neutral"),
        nli_score=signals.get("raw_nli_score", 0.0),
        source_reliability=reliability,
        date_status=signals.get("date_status", "unknown"),
        event_match=signals.get("event_match", "weak"),
        detail_coverage=signals.get("detail_coverage", 0.0),
        location_match=signals.get("location_match", False),
        claim_has_location=signals.get("claim_has_location", True),
        debunking=signals.get("debunking", False),
    )

    return apply_class_ceiling(score, evidence_class, evidence_type)


def analyze_evidence(claim, sources, verbose=True):
    """
    Analyse every retrieved source and return classified evidence chunks.

    Each returned chunk records *why* it was classified as it was: date status
    and reason, location match, claim-detail coverage, event-match level, NLI
    label/score/status, source tier and independence group.
    """
    claim_attrs = extract_claim_attributes(claim)
    claim_has_location = bool(claim_attrs.get("locations"))
    claim_has_date = bool(claim_attrs.get("claim_date"))
    chunk_budget = _resolve_chunk_budget(claim)

    results = []

    for source_id, source in enumerate(sources, 1):
        url = source.get("url", "")
        content = source.get("content", "") or ""
        evidence_type = source.get("evidence_type", "scraped_article")
        domain = source.get("domain", "")
        title = source.get("title", "")

        source_profile = get_source_profile(url, evidence_type)
        reliability = source_profile["reliability"]

        print(
            "[PROFILE DEBUG]",
            {
                "url": url,
                "domain_field": domain,
                "evidence_type": evidence_type,
                "tier": source_profile.get("tier"),
                "is_primary": source_profile.get("is_primary"),
                "reliability": source_profile.get("reliability"),
            },
        )

        if not content or len(content.strip()) < 60:
            if verbose:
                print(f"\nSkipping unusable source {source_id}: {url}")
            continue

        chunks = split_into_chunks(content, max_tokens=chunk_budget)
        if not chunks and content.strip():
            # Never hand an unbounded blob to NLI: slice it to the budget.
            chunks = [content[: chunk_budget * 4]]
        chunks = _dedupe_chunks(chunks)

        if not chunks:
            continue

        # ---- source-level date context -------------------------------------
        # A chunk inherits the article's date context, so a passing mention of
        # an old year (1916, 2019) cannot be read as the event date, and a page
        # published on the claimed date keeps that date available to its chunks.
        publication_date = extract_publication_date(content)
        clean_content = strip_publication_boilerplate(content)
        source_dates = extract_dates(clean_content)
        source_years = extract_years(clean_content)

        # ---- similarity pre-filter (batched) ----
        # The document index is carried along so that a context window can be
        # built from the article's real neighbours later.
        similarities = calculate_similarity_batch(claim, chunks)
        scored_chunks = [
            (index, chunk, similarity)
            for index, (chunk, similarity) in enumerate(zip(chunks, similarities))
            if similarity >= MIN_SIMILARITY
        ]
        scored_chunks.sort(key=lambda item: item[2], reverse=True)

        if not scored_chunks:
            continue

        # ---- date-aware pre-filter ----
        prelim = []
        prelim_index = []
        for index, chunk, similarity in scored_chunks:
            date_check = check_date_compatibility(
                claim,
                chunk,
                claim_attrs,
                publication_date=publication_date,
                source_dates=source_dates,
                source_years=source_years,
            )

            if (
                date_check["status"] == "different"
                and similarity < MIN_SIMILARITY_DIFFERENT_DATE
            ):
                continue

            prelim.append((chunk, similarity, date_check))
            prelim_index.append(index)

        if not prelim:
            continue

        # ---- NLI only on the strongest candidates ----
        source_results = []

        rolling = is_rolling_page(url, title)
        chunk_limit = (
            MAX_NLI_CHUNKS_ROLLING_PAGE if rolling else MAX_NLI_CHUNKS_PER_SOURCE
        )

        if rolling and len(prelim) > chunk_limit:
            # A liveblog is many days of entries. The entry for the claimed day
            # is the evidence; whatever the top of the page happens to be is
            # not. Date status therefore outranks similarity when choosing.
            status_rank = {
                "match": 0,
                "same_day_published": 1,
                "same_year": 2,
                "unknown": 3,
                "different": 4,
            }
            order = sorted(
                range(len(prelim)),
                key=lambda i: (
                    status_rank.get(prelim[i][2]["status"], 5),
                    -prelim[i][1],
                ),
            )
            prelim = [prelim[i] for i in order]
            prelim_index = [prelim_index[i] for i in order]

        # ---- multi-paragraph context windows ----
        #
        # A publisher often states a claim across adjacent paragraphs. Read on
        # its own, each paragraph is NEUTRAL against the claim, so the evidence
        # layer can never support it. The surrounding paragraphs are offered to
        # the NLI model as a window so it sees the passage as a reader would.
        #
        # A window may SUPPORT a claim, but it may not REFUTE one: the NLI
        # model reads a diluted premise unreliably, and a spurious
        # contradiction is far more harmful than a missed one. Refutation
        # therefore continues to require a single focused passage.
        selected = prelim[:chunk_limit]
        anchors = prelim_index[:chunk_limit]
        windows = _build_context_windows(
            chunks, anchors, MAX_WINDOW_CHUNKS_PER_SOURCE, chunk_budget
        )
        window_similarities = (
            calculate_similarity_batch(claim, windows) if windows else []
        )

        candidates = [(chunk, sim, date_check, "paragraph")
                      for chunk, sim, date_check in selected]
        for window, window_similarity in zip(windows, window_similarities):
            window_date = check_date_compatibility(
                claim,
                window,
                claim_attrs,
                publication_date=publication_date,
                source_dates=source_dates,
                source_years=source_years,
            )
            candidates.append((window, window_similarity, window_date, "window"))

        for chunk, similarity, date_check, passage_kind in candidates:
            # For same-day reporting the NLI premise is given the page's own
            # publication date as context, so entailment is judged against what
            # the source actually says about that day instead of a bare passage
            # with no date in it. The deterministic date check remains the gate.
            nli_premise = chunk
            if date_check["status"] == "same_day_published" and publication_date:
                day = publication_date.get("day")
                month = publication_date.get("month")
                year = publication_date.get("year")
                if day and month and year:
                    month_name = MONTH_NAMES[month - 1]
                    nli_premise = (
                        f"(This article was published on {day} {month_name} "
                        f"{year}.) {chunk}"
                    )

            

            # NLI inference: every exception path must still assign `nli`
            # so downstream code never sees an unbound variable.
            try:
                nli = check_nli(claim, nli_premise)
            except Exception as nli_exc:
                nli = {
                    "label": "neutral",
                    "score": 0.0,
                    "status": "error",
                    "truncated": False,
                }

            nli_entails = (
                nli["label"] == "entailment" and nli["score"] >= 0.70
            )

            

            event_match, event_signals = evaluate_event_match(
                claim,
                chunk,
                claim_attrs,
                date_check["status"],
                similarity,
                nli_entails=nli_entails,
                event_date_conflict=bool(date_check.get("event_date_conflict")),
            )
            raw_nli_label = nli["label"]
            raw_nli_score = nli["score"]

            signals = {
                "similarity": similarity,
                "date_status": date_check["status"],
                "event_match": event_match,
                "raw_nli_label": raw_nli_label,
                "raw_nli_score": raw_nli_score,
                "nli_status": nli.get("status", "ok"),
                "nli_truncated": nli.get("truncated", False),
                "claim_has_location": claim_has_location,
                "claim_has_date": claim_has_date,
                "relative_date_ambiguous": claim_attrs.get("relative_date_ambiguous", False),
                "passage_kind": passage_kind,
                # Claim type determines which evidence gates apply.
                "claim_type": claim_attrs.get("claim_type", "other_factual_claim"),
                # Does the passage date the CLAIMED event to another date?
                "event_date_conflict": bool(date_check.get("event_date_conflict")),
                "event_date_mismatch": bool(date_check.get("event_date_mismatch")),
                **event_signals,
            }

            print(
                "[CLASSIFY DEBUG]",
                {
                    "url": url,
                    "evidence_type": evidence_type,
                    "claim_type": signals.get("claim_type"),
                    "primary": source_profile.get("is_primary"),
                    "nli_label": signals.get("raw_nli_label"),
                    "nli_score": signals.get("raw_nli_score"),
                    "nli_status": signals.get("nli_status"),
                    "similarity": signals.get("similarity"),
                    "coverage": signals.get("detail_coverage"),
                    "event_match": signals.get("event_match"),
                    "date_status": signals.get("date_status"),
                    "passage_kind": signals.get("passage_kind"),
                },
            )

            evidence_class = classify_evidence(
                signals, evidence_type, source_profile,
                claim_type=signals.get("claim_type")
            )

            # The scorer must see the label the *evidence layer* assigned, so a
            # neutral NLI result can never be scored as an entailment.
            signals["evidence_class_label"] = (
                "entailment" if evidence_class == "SUPPORT"
                else "contradiction" if evidence_class == "CONTRADICTION"
                else "neutral"
            )

            evidence_score = _score_for_class(
                evidence_class, signals, reliability, evidence_type
            )

            record = {
                "source_id": source_id,
                "url": url,
                "domain": domain,
                "title": title,
                "evidence_type": evidence_type,
                "chunk": chunk,
                # "paragraph" (a single passage) or "window" (adjacent
                # paragraphs of the same article, offered because a publisher
                # often splits a claim across them).
                "passage_kind": passage_kind,
                "similarity": round(similarity, 4),
                # The model's own NLI label, paired with the model's own NLI
                # score. Reporting the evidence CLASS here would pair a
                # "neutral" label with a 0.99 entailment score, which reads as
                # "the model was neutral with 99% confidence" when the model
                # in fact entailed the claim.
                "nli_label": raw_nli_label,
                "nli_score": raw_nli_score,
                "nli_status": nli.get("status", "ok"),
                "nli_truncated": nli.get("truncated", False),
                "raw_nli_label": raw_nli_label,
                "raw_nli_score": raw_nli_score,
                # The label the evidence layer assigned to this passage.
                "evidence_class_label": signals.get("evidence_class_label", "neutral"),
                "source_reliability": reliability,
                "source_tier": source_profile["tier"],
                "source_tier_reason": source_profile["tier_reason"],
                "independence_group": source_profile["independence_group"],
                "search_query": source.get("search_query", ""),
                "is_primary_source": source_profile["is_primary"],
                "is_rolling_page": rolling,
                "evidence_score": evidence_score,
                "evidence_class": evidence_class,
                "date_status": date_check["status"],
                "date_reason": date_check["reason"],
                "publication_date": (
                    {
                        "day": publication_date.get("day"),
                        "month": publication_date.get("month"),
                        "year": publication_date.get("year"),
                    }
                    if publication_date else None
                ),
                "event_match": event_match,
                "location_match": signals.get("location_match", False),
                "location_conflict": signals.get("location_conflict", False),
                "detail_coverage": signals.get("detail_coverage", 0.0),
                "debunking": signals.get("debunking", False),
                "event_signals": event_signals,
            }

            source_results.append(record)

            if verbose:
                print(f"\nSOURCE: {domain or url}")
                print(
                    "Evidence Type: "
                    + {
                        "scraped_article": "SCRAPED_ARTICLE",
                        "search_extract": "SEARCH_EXTRACT",
                    }.get(evidence_type, "SEARCH_SNIPPET_FALLBACK")
                )
                print(f"Source Tier: {source_profile['tier']} ({reliability})")
                print(f"Similarity: {record['similarity']}")
                print(f"NLI: {raw_nli_label} {raw_nli_score} ({record['nli_status']})")
                print(f"Date Status: {date_check['status']} â€” {date_check['reason']}")
                print(
                    f"Event Match: {event_match} | Location: "
                    f"{record['location_match']} | Coverage: {record['detail_coverage']}"
                )
                print(f"Evidence Class: {evidence_class}")
                print(f"Evidence Score: {evidence_score}")

        # Keep the strongest few chunks per source, best class first.
        priority_order = {
            "SUPPORT": 0, "CONTRADICTION": 1,
            "RELATED_CONTEXT": 2, "NON_EVIDENCE": 3,
        }
        source_results.sort(
            key=lambda r: (priority_order[r["evidence_class"]], -r["evidence_score"])
        )
        results.extend(source_results[:MAX_CHUNKS_PER_SOURCE])

    results.sort(key=lambda item: item["evidence_score"], reverse=True)

    return results

