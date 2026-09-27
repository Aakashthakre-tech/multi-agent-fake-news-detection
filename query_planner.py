"""
query_planner.py
================
Multi-angle query generation for a single claim.

A claim is never searched with one string. Every claim produces a small set of
deliberately different searches, each labelled with the intent it serves:

    exact       verbatim claim / headline phrasing
    date        exact event date in a different textual format
    location    claimed place + event wording
    keywords    claim-specific content words
    wording     the same event described with alternative wording
    factcheck   the claim phrased as a claim to be debunked
    official    official / primary-record phrasing (general web, not news)
    broad       year-level recall for undated or weakly dated claims

`topic` is honoured by the retriever: "news" for current reporting, "general"
for official/archival pages that a news-only search would never surface.

Alternative wording is generated from a compact incident synonym map plus
morphological rephrasing, so a claim about a "fire" also searches "blaze" and
an article phrased "broke out" is found by a claim phrased "erupted".
"""

from claim_profile import build_claim_profile

import re

MAX_QUERIES = 8

# Incident wording variants: (canonical trigger, [alternative phrasings])
WORDING_VARIANTS = {
    "fire": ["blaze", "inferno", "fire incident"],
    "fires": ["blazes", "fire incident"],
    "blaze": ["fire", "inferno"],
    "flood": ["flooding", "inundation", "waterlogging"],
    "flooded": ["flooding", "inundated", "waterlogged"],
    "flooding": ["flood", "inundation"],
    "explosion": ["blast", "explosion incident"],
    "blast": ["explosion"],
    "crash": ["collision", "crash incident"],
    "collision": ["crash"],
    "earthquake": ["quake", "seismic event"],
    "quake": ["earthquake", "seismic event"],
    "protest": ["demonstration", "rally", "agitation"],
    "protests": ["demonstrations", "rally"],
    "erupted": ["broke out", "flared up"],
    "broke": ["erupted", "flared"],
    "struck": ["hit", "attacked"],
    "killed": ["death toll", "casualties", " fatalities"],
    "injured": ["hurt", "casualties"],
    "derailed": ["train derailment"],
    "strike": ["action", "industrial action"],
    "resigns": ["resignation", "quit"],
    "elected": ["election", "vote"],
    "verdict": ["ruling", "judgment", "court ruling"],
    "acquitted": ["cleared of charges"],
    "convicted": ["found guilty", "sentenced"],
    "ban": ["prohibition", "restriction"],
    "summit": ["meeting", "conference"],
}

# Phrasings that surface fact-checks and corrections.
FACTCHECK_HINTS = [
    "fact check",
    "false viral",
    "did not happen",
    "misinformation",
    "myth",
]

# Phrasings that surface primary/official records.
OFFICIAL_HINTS = [
    "official statement",
    "press release",
    "ministry of",
    "government of",
    "district administration",
]


def _quote(text):
    text = (text or "").strip()
    if not text:
        return ""
    if " " in text and not text.startswith('"'):
        return f'"{text}"'
    return text


def _join(*parts):
    return " ".join(str(p).strip() for p in parts if p and str(p).strip())


def _wording_variants(claim, profile):
    """
    Build alternative-worded descriptions of the same event.

    Returns a list of (trigger, alternative, reworded_claim) triples.
    """
    lower_claim = (claim or "").lower()
    keywords = profile["keywords"]

    variants = []

    for keyword in keywords:
        for alternative in WORDING_VARIANTS.get(keyword, []):
            if alternative in lower_claim:
                continue
            replacement = _replace_token(claim, keyword, alternative)
            if replacement and replacement.lower() != lower_claim:
                variants.append((keyword, alternative, replacement))

    seen = set()
    unique = []
    for trigger, alternative, text in variants:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((trigger, alternative, text))

    return unique


def _replace_token(text, token, replacement):
    """Replace whole-word `token` occurrences, preserving original casing."""
    if not token:
        return text

    def substitute(match):
        found = match.group(0)
        if found[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return re.sub(rf"\b{re.escape(token)}\b", substitute, text)


def plan_queries(claim, max_queries=MAX_QUERIES):
    """
    Return an ordered, de-duplicated list of searches for one claim.

    Each entry:
        label   stable identifier (used for provenance in the evidence log)
        query   the actual search string
        intent  exact | date | location | keywords | wording | factcheck |
                official | broad
        topic   "news" or "general" (retrieval surface)
    """
    profile = build_claim_profile(claim)
    clean_claim = profile["clean_claim"]

    if not clean_claim:
        return []

    dates = profile["dates"]
    primary_date = profile["primary_date"]
    variants = profile["date_variants"]
    locations = profile["locations"]
    keywords = profile["keywords"]
    phrases = profile["phrases"]

    location_quoted = _quote(locations[0]) if locations else ""
    secondary_quoted = _quote(locations[1]) if len(locations) > 1 else ""
    location_plain = locations[0] if locations else ""

    keyword_terms = " ".join(_quote(k) for k in keywords[:3]) if keywords else ""
    phrase_quoted = _quote(phrases[0]) if phrases else ""
    second_phrase = _quote(phrases[1]) if len(phrases) > 1 else ""

    plans = []

    def add(label, intent, query, topic="news"):
        query = re.sub(r"\s+", " ", str(query or "")).strip()
        if not query:
            return
        plans.append({
            "label": label,
            "query": query,
            "intent": intent,
            "topic": topic,
        })

    # --------------------------------------------------------
    # 1. Verbatim claim — the highest-precision search
    # --------------------------------------------------------
    add("exact_claim", "exact", clean_claim)

    # --------------------------------------------------------
    # 2-3. Exact date, two textual formats (event date, not publication date)
    # --------------------------------------------------------
    if primary_date and variants:
        if location_quoted and keyword_terms:
            add(
                "date_loc_keywords",
                "date",
                _join(_quote(variants[0]), location_quoted, keyword_terms),
            )
        if location_quoted and phrase_quoted:
            add(
                "date_alt_format",
                "date",
                _join(_quote(variants[1]), location_quoted, phrase_quoted, second_phrase),
            )
        elif location_quoted and keyword_terms:
            add(
                "date_alt_format",
                "date",
                _join(_quote(variants[1]), location_quoted, keyword_terms),
            )
        elif keyword_terms:
            add(
                "date_alt_format",
                "date",
                _join(_quote(variants[1]), keyword_terms),
            )

    # --------------------------------------------------------
    # 4. Location + event phrasing (recovers articles without a written date)
    # --------------------------------------------------------
    if location_quoted and phrase_quoted:
        add(
            "location_phrase",
            "location",
            _join(location_quoted, phrase_quoted),
        )
    elif location_quoted and keyword_terms:
        add(
            "location_keywords",
            "location",
            _join(location_quoted, keyword_terms),
        )

    # --------------------------------------------------------
    # 5. Alternative wording for the same event
    # --------------------------------------------------------
    wording = _wording_variants(clean_claim, profile)
    if wording:
        trigger, alternative, reworded = wording[0]

        if location_quoted and primary_date:
            add(
                "alt_wording_date",
                "wording",
                _join(
                    _quote(variants[2] if len(variants) > 2 else variants[-1]),
                    location_quoted,
                    _quote(alternative),
                ),
            )

        add("alt_wording", "wording", reworded)

    # --------------------------------------------------------
    # 6. Fact-check / debunking angle
    # --------------------------------------------------------
    if location_quoted and keyword_terms:
        add(
            "factcheck",
            "factcheck",
            _join(
                _quote(clean_claim[:120]) if len(clean_claim) > 40 else "",
                FACTCHECK_HINTS[0],
            ),
        )
    elif keyword_terms:
        add("factcheck", "factcheck", _join(keyword_terms, FACTCHECK_HINTS[0]))

    # --------------------------------------------------------
    # 7. Official / primary record (general web, not news)
    # --------------------------------------------------------
    official_parts = [
        _quote(location_plain) if location_plain else "",
        _quote(keywords[0]) if keywords else "",
        OFFICIAL_HINTS[0],
    ]
    if primary_date:
        official_parts.insert(0, _quote(variants[-1]))
    add("official", "official", _join(*official_parts), topic="general")

    # --------------------------------------------------------
    # 8. Year-level recall (only when the claim is dated)
    # --------------------------------------------------------
    if primary_date and location_quoted and second_phrase:
        add(
            "year_recall",
            "broad",
            _join(
                str(primary_date["year"]),
                location_quoted,
                second_phrase,
            ),
        )
    elif not primary_date and len(keywords) >= 3:
        add("keyword_broad", "broad", " ".join(keywords[:8]))

    # Claims with very little keyword content still need a recall query that
    # uses the claim wording itself rather than a single stray verb.
    if len(keywords) < 3:
        add("broad_recall", "broad", profile["date_stripped"] or clean_claim)

    # --------------------------------------------------------
    # De-duplicate, cap, and restore deterministic ordering
    # --------------------------------------------------------
    seen_queries = set()
    unique = []
    for plan in plans:
        key = plan["query"].lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        unique.append(plan)

    return unique[:max_queries]


if __name__ == "__main__":

    for sample_claim in (
        "A major fire broke out at a commercial building in New Delhi, "
        "India on 12 Aug 2024.",
        "Heavy rains flood Mumbai on 08 July 2024, several areas "
        "waterlogged and traffic severely affected.",
    ):
        print("=" * 70)
        print("CLAIM:", sample_claim)
        for plan in plan_queries(sample_claim):
            print(f"  [{plan['intent']:>9}] ({plan['topic']:>7}) {plan['query']}")
        print()
