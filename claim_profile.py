"""
claim_profile.py
================
One dependency-free parser for a claim.

This replaces the three divergent copies of claim parsing that used to live in
`tools.py` (extract_claim_components) and `evidence_analyzer.py`
(extract_claim_attributes). Retrieval, evidence filtering and date handling now
all read the same structure, so a query cannot target a location that the
evidence layer then refuses to recognise.
"""

import re
from datetime import datetime

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_ABBR = [m[:3] for m in MONTH_NAMES]

_MONTH_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}
_MONTH_LOOKUP.update({a.lower(): i + 1 for i, a in enumerate(MONTH_ABBR)})

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
    "was", "were", "are", "has", "have", "had", "been", "broke", "out", "that",
    "this", "by", "from", "with", "as", "about", "into", "over", "after",
    "before", "reported", "allegedly", "claimed", "says", "said", "according",
    "news", "today", "yesterday", "recent", "major", "minor", "there", "their",
    "which", "who", "whom", "whose", "where", "when", "why", "how", "all", "any",
    "new", "old", "big", "small", "high", "low", "good", "bad", "first", "last",
    "next", "previous", "current", "latest", "early", "late", "main", "top",
    "it", "its", "he", "she", "they", "we", "you", "his", "her", "not", "no",
    "but", "if", "than", "then", "so", "such", "also", "will", "would", "can",
    "may", "might", "do", "does", "did", "has", "been", "during", "while",
    "here", "now", "one", "two", "three", "more", "most", "some", "only",
}

# Words that are usually part of a place name, not the event.
_GEO_HINT_WORDS = {
    "india", "pakistan", "bangladesh", "nepal", "sri", "lanka", "china",
    "japan", "korea", "brazil", "mexico", "canada", "australia", "france",
    "germany", "uk", "united", "kingdom", "states", "america", "usa", "russia",
    "ukraine", "egypt", "kenya", "nigeria", "indonesia", "thailand", "vietnam",
    "dubai", "delhi", "mumbai", "kolkata", "chennai", "bengaluru", "hyderabad",
    "pune", "ahmedabad", "jaipur", "lucknow", "karachi", "lahore", "dhaka",
    "kathmandu", "colombo", "beijing", "shanghai", "tokyo", "seoul",
    "london", "paris", "berlin", "madrid", "rome", "sydney", "toronto",
    "washington", "new", "york", "california", "texas", "florida", "chicago",
    "state", "city", "town", "village", "district", "province", "state",
    "county", "municipality", "country", "region", "area", "street", "road",
}


def _month_number(token):
    token = str(token or "").strip().lower().rstrip(".")
    if not token:
        return None
    if token in _MONTH_LOOKUP:
        return _MONTH_LOOKUP[token]
    for name, number in _MONTH_LOOKUP.items():
        if len(name) >= 3 and token.startswith(name[:3]):
            return number
    return None


# ============================================================
# DATES
# ============================================================

_DATE_PATTERNS = (
    # 12 Aug 2024 / 12th August, 2024
    (
        re.compile(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
            r"([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"
        ),
        ("day", "month", "year"),
    ),
    # August 12, 2024 / Aug 12 2024
    (
        re.compile(
            r"\b([A-Za-z]{3,9})\.?\s+"
            r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
        ),
        ("month", "day", "year"),
    ),
    # 2024-08-12 (also ISO timestamps such as 2024-08-12T08:01:45+05:30)
    (
        re.compile(
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})"
            r"(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:?\d{2})?)?"
        ),
        ("year", "month", "day"),
    ),
    # 12/08/2024 (ambiguous; both readings are stored)
    (
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
        ("a", "b", "year"),
    ),
)


def _make_date(day, month, year, raw=None, ambiguous=False):
    if not month or not day or not year:
        return None
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return None
    if not (1900 <= int(year) <= 2100):
        return None
    try:
        datetime(int(year), int(month), int(day))
    except ValueError:
        return None
    return {
        "day": int(day),
        "month": int(month),
        "year": int(year),
        "month_name": MONTH_NAMES[int(month) - 1],
        "raw": raw or "",
        "ambiguous": ambiguous,
    }


def extract_dates(text):
    """Return a de-duplicated list of parsed date dicts found in `text`."""
    if not text:
        return []

    found = []
    seen = set()

    for pattern, fields in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            values = {field: match.group(index + 1) for index, field in enumerate(fields)}

            if fields == ("a", "b", "year"):
                first, second, year = values["a"], values["b"], values["year"]
                if not (first.isdigit() and second.isdigit()):
                    continue
                # Prefer day-first; record the month-first reading as ambiguous.
                candidates = [
                    _make_date(int(first), int(second), int(year), match.group(0), True),
                    _make_date(int(second), int(first), int(year), match.group(0), True),
                ]
                for candidate in candidates:
                    if not candidate:
                        continue
                    key = (candidate["day"], candidate["month"], candidate["year"])
                    if key not in seen:
                        seen.add(key)
                        found.append(candidate)
                continue

            month_token = values["month"]
            month = (
                int(month_token)
                if month_token.isdigit()
                else _month_number(month_token)
            )
            candidate = _make_date(
                int(values["day"]), month, int(values["year"]), match.group(0)
            )
            if not candidate:
                continue
            key = (candidate["day"], candidate["month"], candidate["year"])
            if key not in seen:
                seen.add(key)
                found.append(candidate)

    return found


def date_variants(date_info):
    """All textual spellings of one parsed date (for exact-phrase queries)."""
    if not date_info:
        return []

    day = date_info["day"]
    month_full = date_info["month_name"]
    month_short = month_full[:3]
    year = date_info["year"]

    variants = [
        f"{day} {month_full} {year}",
        f"{month_full} {day}, {year}",
        f"{day} {month_short} {year}",
        f"{month_short} {day}, {year}",
        f"{month_short} {day} {year}",
        f"{year}-{date_info['month']:02d}-{day:02d}",
        f"{day:02d}/{date_info['month']:02d}/{year}",
    ]

    return list(dict.fromkeys(variants))


def extract_years(text):
    if not text:
        return set()
    return {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)}


_PUBLICATION_LABEL_RE = re.compile(
    r"^\s*(?:published date|publication date|published on|posted on|"
    r"last updated|updated on|published)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_publication_date(text):
    """
    Read a page's own publication date from its labelled metadata line.

    This is kept separate from the event date on purpose: a page's publication
    date is never treated as the date of the event it describes, it is only ever
    used when it is *equal* to the claimed date.
    """
    if not text:
        return None

    for match in _PUBLICATION_LABEL_RE.finditer(text):
        value = match.group(1).strip()
        dates = extract_dates(value)
        if dates:
            return dates[0]
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
        if year_match:
            return {"year": int(year_match.group(1))}

    return None


# ============================================================
# MORPHOLOGY
# ============================================================

_IRREGULAR_FORMS = {
    "died": "die", "killed": "kill", "injured": "injure", "broke": "break",
    "flooded": "flood", "flooding": "flood", "crashed": "crash",
    "struck": "strike", "elected": "elect", "banned": "ban",
    "resigned": "resign", "convicted": "convict", "acquitted": "acquit",
    "derailed": "derail", "erupted": "erupt", "collapsed": "collapse",
    "arrested": "arrest", "warned": "warn", "issued": "issue",
}


def keyword_forms(keyword):
    """
    Simple English forms of a content word.

    A claim about "flooding" must still match an article that says "flood", and a
    claim about "flooded" must match "flooded" or "flooding".
    """
    word = (keyword or "").lower().strip()
    if not word:
        return []

    forms = {word}

    if word in _IRREGULAR_FORMS:
        forms.add(_IRREGULAR_FORMS[word])

    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        forms.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        forms.add(word[:-1])
    if word.endswith("ed") and len(word) > 4:
        forms.add(word[:-2])
        forms.add(word[:-1])
    if word.endswith("ing") and len(word) > 5:
        forms.add(word[:-3])
        forms.add(word[:-3] + "e")

    return sorted(f for f in forms if f)


# Publication metadata must never be read as the date of the reported event.
_METADATA_LINE_RE = re.compile(
    r"^\s*(published(\s+date)?|publication\s+date|published\s+on|posted(\s+on)?|"
    r"last\s+updated|updated(\s+on)?|first\s+published|publish\s+date|"
    r"article\s+date|date\s+of\s+publication|created|modified)\s*[:\-–]?\s*",
    re.IGNORECASE,
)

# Same labels, unanchored, for metadata that trails a prose line.
_METADATA_INLINE_RE = re.compile(
    r"\b(published(\s+date)?|publication\s+date|posted(\s+on)?|last\s+updated|"
    r"updated(\s+on)?|first\s+published)\s*[:\-–]\s*",
    re.IGNORECASE,
)

_COPYRIGHT_RE = re.compile(r"^\s*(©|copyright)", re.IGNORECASE)

# Wire-service datelines ("NEW DELHI: ...") are location labels, not event
# dates. The label is removed; the sentence that follows is kept.
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z .]{1,24}:\s+")


def strip_publication_boilerplate(text):
    """
    Remove publication/update metadata and datelines from source text.

    A 2025 article about an August 2024 fire carries 2025 in its 'updated'
    line and in its copyright footer. Without stripping them, the article looks
    like a report of a 2025 event. Event sentences themselves are always kept —
    only labels, datelines and copyright notices are removed.
    """
    if not text:
        return ""

    kept = []

    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Metadata can also appear at the END of a prose line
        # ("...forced schools to close. Published Date: 2024-07-08").
        label = _METADATA_INLINE_RE.search(stripped)
        if label and label.start() > 0:
            prefix = stripped[:label.start()].strip()
            stripped = prefix if len(prefix) >= 60 else ""

        # A dateline may prefix a metadata line ("NEW DELI: Published ...").
        for _ in range(2):
            stripped = _DATELINE_RE.sub("", stripped).strip()
            match = _METADATA_LINE_RE.match(stripped)
            if not match:
                break
            remainder = stripped[match.end():].strip()
            # Keep the remainder only when it is real prose, not just a date.
            if len(remainder) >= 60 and re.search(r"[a-z]{3,}\s+[a-z]{3,}", remainder):
                stripped = remainder
            else:
                stripped = ""
                break

        if not stripped:
            continue
        if _COPYRIGHT_RE.match(stripped):
            continue

        kept.append(stripped)

    return "\n".join(kept)


# ============================================================
# ENTITIES / KEYWORDS / PHRASES
# ============================================================

_ENTITY_RE = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s*,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)*"
)


# Words that are capitalised only because they start a sentence, and that must
# never be mistaken for a place name or a named entity.
_SENTENCE_START_NOISE = {
    "heavy", "major", "minor", "several", "many", "multiple", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "dozens",
    "hundreds", "thousands", "police", "government", "officials", "authorities",
    "after", "before", "during", "following", "amid", "man", "woman", "child",
    "people", "local", "resident", "residents", "fire", "flood", "blast",
    "massive", "huge", "giant", "scores", "dozen", "several", "update",
    "exclusive", "breaking", "watch", "live", "report", "reports", "photo",
    "video", "images", "full", "latest", "source", "sources", "official",
}

# A capitalised phrase only counts as a place when one of these is true:
# it is a known place word, it follows a location preposition, it is the
# country half of a "Place, Country" pair, or it carries a geographic suffix.
# Prepositions that unambiguously introduce a place.
_LOCATION_PREPOSITIONS = {
    "in", "near", "from", "outside", "inside", "around", "across",
    "towards", "into", "beyond", "throughout",
}

# "at" and "on" also introduce times and positions ("at the top floor"), so
# they only introduce a place when the phrase itself is recognisably a place.
_WEAK_LOCATION_PREPOSITIONS = {"at", "on"}

_GEO_SUFFIXES = {
    "city", "town", "village", "district", "province", "state", "county",
    "island", "islands", "river", "lake", "hill", "hills", "mountain",
    "street", "road", "airport", "station", "port", "harbour", "bay",
    "valley", "desert", "beach", "park", "market", "bazaar", "municipality",
    "corporation", "council", "prefecture", "region", "territory",
}

_COUNTRIES = {
    "india", "usa", "uk", "united kingdom", "united states", "canada",
    "mexico", "brazil", "argentina", "chile", "peru", "colombia",
    "pakistan", "bangladesh", "sri lanka", "nepal", "china", "japan",
    "korea", "russia", "ukraine", "poland", "germany", "france", "spain",
    "italy", "greece", "egypt", "nigeria", "kenya", "ghana", "south africa",
    "australia", "indonesia", "malaysia", "singapore", "thailand", "vietnam",
    "philippines", "myanmar", "nepal", "iran", "iraq", "saudi arabia", "qatar",
    "uae", "turkey", "greece", "afghanistan", "ethiopia", "tanzania",
}


def _looks_like_place(candidate, text_lower, start_index):
    """
    Decide whether a capitalised phrase is genuinely a place.

    Requiring evidence (known place word, preposition, country pairing or a
    geographic suffix) is what stops "HEAVY RAINS FLOOD Mumbai" from being read
    as a single location.
    """
    if not candidate:
        return False

    candidate_lower = candidate.lower()

    if candidate_lower in _GEO_HINT_WORDS or candidate_lower in _COUNTRIES:
        return True

    words = candidate.split()
    if any(word.lower() in _GEO_SUFFIXES for word in words):
        return True

    # Preceded by a location preposition?
    if start_index > 0:
        preceding = text_lower[:start_index].split()
        if preceding:
            previous = preceding[-1]
            if previous in _LOCATION_PREPOSITIONS:
                return True
            if previous in _WEAK_LOCATION_PREPOSITIONS:
                # "at"/"on" need geographic evidence of their own.
                words = candidate.split()
                if (
                    candidate_lower in _GEO_HINT_WORDS
                    or candidate_lower in _COUNTRIES
                    or any(w.lower() in _GEO_SUFFIXES for w in words)
                ):
                    return True

    # Part of a "Place, Country" pair?
    if "," in text_lower[start_index:start_index + len(candidate) + 2]:
        tail = text_lower[start_index + len(candidate):][:60]
        for country in _COUNTRIES:
            if tail.strip(" ,").startswith(country):
                return True

    return False


def _trim_entity(part):
    """
    Strip leading/trailing function words from an entity candidate.

    Only genuine function words are removed: 'New Delhi' must survive intact,
    'The Taj Mahal' must lose its article.
    """
    trim_set = {"the", "a", "an", "of", "in", "at", "on", "to", "for", "and", "or"}

    tokens = part.split()
    while tokens and tokens[0].lower().strip(",.") in trim_set:
        tokens.pop(0)
    while tokens and tokens[-1].lower().strip(",.") in trim_set:
        tokens.pop()
    return " ".join(tokens)


def extract_entities(text):
    """
    Capitalised sequences that have evidence of being places.

    The country half of a "Place, Country" pair is always kept, so
    'New Delhi, India' yields ['New Delhi', 'India'].
    """
    if not text:
        return []

    text_lower = text.lower()
    first_token = re.match(r"\s*([A-Za-z']+)", text)
    first_word = first_token.group(1).lower() if first_token else ""

    entities = []

    for match in _ENTITY_RE.finditer(text):
        raw = re.sub(r"\s+", " ", match.group(0)).strip(" ,")
        if not raw:
            continue

        parts = [p.strip() for p in raw.split(",") if p.strip()]

        for part in parts:
            cleaned = _trim_entity(part.strip(" ."))
            if len(cleaned) < 3:
                continue
            if cleaned.lower() in STOPWORDS:
                continue
            if not any(ch.isalpha() for ch in cleaned):
                continue

            if " " not in cleaned and cleaned.lower() in _SENTENCE_START_NOISE:
                if cleaned.lower() not in _GEO_HINT_WORDS:
                    continue

            if not _looks_like_place(cleaned, text_lower, match.start()):
                # The run is not a place as a whole, but one word inside it
                # may still be a place ("FLOOD Mumbai" -> "Mumbai").
                for word in cleaned.split():
                    if word.lower() in STOPWORDS:
                        continue
                    if _looks_like_place(word, text_lower, match.start()):
                        if word not in entities:
                            entities.append(word)
                continue

            if cleaned not in entities:
                entities.append(cleaned)

    return entities


def extract_keywords(text, entity_words=None):
    """Content words of length >= 3 that are not stopwords or entity words."""
    if not text:
        return []

    entity_words = entity_words or set()

    words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text)

    keywords = []
    for word in words:
        lower = word.lower()
        if lower in STOPWORDS or lower in entity_words:
            continue
        if lower not in keywords:
            keywords.append(lower)

    return keywords


def entity_word_set(entities):
    words = set()
    for entity in entities:
        for token in re.findall(r"[A-Za-z]{2,}", entity):
            words.add(token.lower())
    return words


def extract_phrases(text, keywords, max_phrases=6):
    """
    Multi-word content phrases ('commercial building', 'heavy rains').

    Built from adjacent non-stopword tokens so they read as event phrases
    rather than bag-of-words noise.
    """
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)
    phrases = []

    for size in (3, 2):
        for start in range(len(tokens) - size + 1):
            window = tokens[start:start + size]
            lowered = [w.lower() for w in window]

            if any(w in STOPWORDS for w in lowered):
                continue
            if not all(w in keywords for w in lowered):
                continue

            phrase = " ".join(lowered)
            if phrase not in phrases:
                phrases.append(phrase)

    phrases.sort(key=lambda p: (-len(p.split()), -len(p)))

    return phrases[:max_phrases]


def strip_dates(text, dates):
    """Remove every detected date string from the claim text."""
    if not text or not dates:
        return text

    result = text
    for date_info in dates:
        raw = date_info.get("raw")
        if raw:
            result = re.sub(
                rf"\b(?:on|in|at|during|since)?\s*{re.escape(raw)}\b",
                " ",
                result,
                flags=re.IGNORECASE,
            )

    return re.sub(r"\s+", " ", result).strip()


# Acronyms that must stay upper-case; everything else in an all-caps headline
# is treated as ordinary capitalised words.
ACRONYMS = {
    "UN", "USA", "US", "UK", "EU", "GDP", "CPI", "PM", "CEO", "TV", "AI",
    "II", "III", "IV", "MP", "IAS", "IPS", "IIM", "IIIT", "NEET", "JEE",
    "GST", "IRCTC", "AIIMS", "NDRF", "ISRO", "BCCI", "ICC", "ATM", "OTP",
    "PDF", "WHO", "UNO", "NGO", "CBI", "ED", "FIR", "NGO", "HIV", "COVID",
}


def normalize_shouting(text):
    """
    Turn SHOUTED headline words into normal capitalised words.

    Viral images are written in capitals ("HEAVY RAINS FLOOD Mumbai"), and a
    capitalised-word parser reads that whole run as one place name and loses
    every content keyword. Mixed-case words are left untouched, and known
    acronyms keep their capitals.
    """
    if not text:
        return ""

    def fix(match):
        word = match.group(0)
        stripped = word.strip(".,:;!?\"'")

        if stripped != word:
            suffix = word[len(stripped):]
        else:
            suffix = ""

        if not stripped.isupper() or len(stripped) < 3:
            return word

        if stripped in ACRONYMS:
            return word

        if word.isupper():
            fixed = stripped.capitalize()
        else:
            fixed = stripped[0].upper() + stripped[1:].lower()

        return fixed + suffix

    return re.sub(r"\b[A-Za-z]{3,}\b", fix, text)


# ============================================================
# CLAIM PROFILE
# ============================================================

def build_claim_profile(claim):
    """
    Parse a claim into the structure shared by retrieval and analysis.

    Keys:
        claim             cleaned claim text
        dates             parsed date dicts
        primary_date      first parsed date or None
        date_variants     all spellings of primary_date
        years             years mentioned
        locations         entity/location strings
        location_terms    matchable location tokens (full + parts)
        keywords          content words
        phrases           multi-word event phrases
        event_text        date-free, keyword-led description
        has_date          bool
    """
    clean_claim = str(claim or "").strip().strip('"').strip("'").strip()
    clean_claim = re.sub(r"\s+", " ", clean_claim)

    # A shouted headline must still be parsed as ordinary words.
    parsed_text = normalize_shouting(clean_claim)

    dates = extract_dates(parsed_text)
    primary_date = dates[0] if dates else None

    date_stripped = strip_dates(parsed_text, dates)

    locations = extract_entities(date_stripped)
    location_terms = []
    for location in locations:
        if location.lower() not in [t.lower() for t in location_terms]:
            location_terms.append(location)
        for part in re.findall(r"[A-Za-z]+", location):
            # Short or generic fragments ("New", "City", "State") produce false
            # location matches in unrelated articles, so they are only kept
            # when they are recognised geography words.
            if len(part) < 4 and part.lower() not in _GEO_HINT_WORDS:
                continue
            if part.lower() in STOPWORDS:
                continue
            if part.lower() not in [t.lower() for t in location_terms]:
                location_terms.append(part)

    keywords = extract_keywords(date_stripped, entity_word_set(locations))
    phrases = extract_phrases(date_stripped, keywords)

    event_text = " ".join(keywords[:10])

    return {
        "claim": clean_claim,
        "clean_claim": clean_claim,
        "date_stripped": date_stripped,
        "dates": dates,
        "primary_date": primary_date,
        "date_variants": date_variants(primary_date),
        "has_date": bool(dates),
        "years": extract_years(clean_claim),
        "locations": locations,
        "location_terms": location_terms,
        "geo_terms": [
            term for term in location_terms
            if term.lower() in _GEO_HINT_WORDS
            or term.lower() in {loc.lower() for loc in locations}
        ],
        "keywords": keywords,
        "phrases": phrases,
        "event_text": event_text,
    }


if __name__ == "__main__":

    import json

    sample = (
        "A major fire broke out at a commercial building in New Delhi, "
        "India on 12 Aug 2024."
    )

    print(json.dumps(build_claim_profile(sample), indent=2, default=str))
