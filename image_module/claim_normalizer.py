import re
from datetime import datetime
from difflib import SequenceMatcher


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

_MONTH_ABBREVIATIONS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
)

_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
)

_DATE_PATTERNS = (
    (
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<month>{_MONTH_PATTERN})\.?\s*,?\s*"
        rf"(?P<year>(?:19|20)\d{{2}})\b",
        ("day", "month", "year")
    ),
    (
        rf"\b(?P<month>{_MONTH_PATTERN})\.?\s+"
        rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*"
        rf"(?P<year>(?:19|20)\d{{2}})\b",
        ("month", "day", "year")
    ),
    (
        r"\b(?P<year>(?:19|20)\d{2})-"
        r"(?P<month>\d{1,2})-(?P<day>\d{1,2})\b",
        ("year", "month", "day")
    )
)

_LOCATION_PATTERNS = (
    re.compile(
        r"(?<!\w)(?P<place>[A-Z][A-Za-z]*(?:[ -][A-Z][A-Za-z]*)*)"
        r"\s*[,;]\s*(?P<country>[A-Z][A-Za-z]+)\b"
    ),
    re.compile(
        r"(?<!\w)(?P<place>[A-Z][A-Za-z]+(?:[ -][A-Z][A-Za-z]+){0,2})"
        r"\s+(?P<country>India)\b"
    )
)


def _normalize_whitespace(value):
    text = str(value or "").replace("\n", " . ")
    return re.sub(r"[^\S\r\n]+", " ", text).strip()


def _find_date(text):
    for pattern, fields in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values = {field: match.group(field) for field in fields}
            month_token = values["month"]

            if month_token.isdigit():
                month = int(month_token)
            else:
                month = _MONTHS.get(month_token[:3].casefold(), 0)

            try:
                day = int(values["day"])
                year = int(values["year"])
                datetime(year, month, day)
            except (ValueError, KeyError):
                continue

            return {
                "value": (
                    f"{day:02d} {_MONTH_ABBREVIATIONS[month - 1]} {year}"
                ),
                "start": match.start(),
                "end": match.end()
            }

    return None


def _find_location(text):
    for pattern in _LOCATION_PATTERNS:
        for match in pattern.finditer(text):
            place = _normalize_whitespace(match.group("place"))
            country = match.group("country")

            if not place or not country:
                continue

            return {
                "value": f"{place}, {country}",
                "start": match.start(),
                "end": match.end()
            }

    return None


def _remove_span(text, start, end):
    return f"{text[:start]} {text[end:]}"


def _correct_location_spelling(text, location):
    if not location:
        return text

    city = location.split(",", 1)[0].strip()
    city_key = city.casefold()

    def replace_token(match):
        token = match.group(0)
        token_key = token.casefold()

        if token_key == city_key:
            return city

        collapsed_key = re.sub(r"(.)\1+", r"\1", token_key)
        similarity = SequenceMatcher(
            None,
            collapsed_key,
            city_key
        ).ratio()

        if similarity >= 0.80:
            return city

        return token

    return re.sub(r"\b[A-Za-z]+\b", replace_token, text)


def _append_claim_context(text, location, date):
    if not text:
        return text

    context = []

    if location:
        city = location.split(",", 1)[0].strip()

        if city.casefold() not in text.casefold():
            context.append(f"in {location}")

    if date:
        context.append(f"on {date}")

    if context:
        return f"{text.rstrip('.?!')} {' '.join(context)}."

    return text


def normalize_image_claim(
    ocr_text,
    detected_objects,
    headline_text=None
):

    cleaned_ocr = _normalize_whitespace(ocr_text)

    if headline_text is None:
        working_text = cleaned_ocr
    else:
        working_text = _normalize_whitespace(headline_text)

    date = None
    location = None

    date_match = _find_date(cleaned_ocr)

    if date_match:
        date = date_match["value"]

    location_match = _find_location(cleaned_ocr)

    if location_match:
        location = location_match["value"]

    if headline_text is None:
        if date_match:
            working_text = _remove_span(
                working_text,
                date_match["start"],
                date_match["end"]
            )

        if location_match:
            working_text = _remove_span(
                working_text,
                location_match["start"],
                location_match["end"]
            )

    working_text = _correct_location_spelling(
        working_text,
        location
    )

    working_text = re.sub(
        r"\bbreaking\s+news\b",
        " ",
        working_text,
        flags=re.IGNORECASE
    )
    working_text = re.sub(r"\s+", " ", working_text)
    working_text = re.sub(r"\s+([,.;:])", r"\1", working_text)
    working_text = re.sub(r"[:;]\s*\.", ".", working_text)
    working_text = re.sub(r"\.{2,}", ".", working_text)
    working_text = re.sub(r"([,.;:])(?=[A-Za-z])", r"\1 ", working_text)
    working_text = working_text.strip(" \t\r\n-|:;,.")

    if working_text:
        working_text = working_text[0].upper() + working_text[1:]
        working_text = _append_claim_context(
            working_text,
            location,
            date
        )

        if working_text[-1] not in ".?!":
            working_text += "."

    return {
        "claim": working_text,
        "cleaned_ocr": cleaned_ocr.casefold(),
        "location": location,
        "date": date,
        "event": working_text or None,
        "detected_objects": detected_objects
    }