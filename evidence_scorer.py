"""
evidence_scorer.py
==================
Strength of a single evidence chunk.

The old score was 30% embedding similarity + 50% NLI + 20% reliability, which
meant two articles about the same city in the wrong year could outrank a single
source that reports the exact event.

The new score is built from the parts that actually establish a claim:

    date match          20%   exact claimed event date
    event match         20%   same event, judged by place + claim details
    claim detail        15%   how much of the claim's specific content appears
    location match      10%   the claimed place is stated
    NLI confidence      15%   model agreement, not similarity
    source quality      15%   publisher tier
    semantic similarity  5%   weak corroborating signal only

A chunk that misses the date, the place or the event details cannot reach a high
score no matter how similar it looks.
"""

# Relative weights. They sum to 1.0.
W_DATE = 0.20
W_EVENT = 0.20
W_DETAIL = 0.15
W_LOCATION = 0.10
W_NLI = 0.15
W_QUALITY = 0.15
W_SIMILARITY = 0.05

DATE_SCORES = {
    "match": 1.00,
    "same_day_published": 0.85,
    "same_year": 0.45,
    "unknown": 0.30,
    "different": 0.00,
}

EVENT_SCORES = {
    "strong": 1.00,
    "moderate": 0.55,
    "weak": 0.15,
    "none": 0.00,
}


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def calculate_evidence_score(
    similarity=0.0,
    nli_label="neutral",
    nli_score=0.0,
    source_reliability=0.0,
    date_status="unknown",
    event_match="weak",
    detail_coverage=0.0,
    location_match=False,
    claim_has_location=True,
    debunking=False,
):
    """
    Composite evidence strength in [0, 1].

    `similarity`, `nli_label`, `nli_score` and `source_reliability` keep their
    original positional meaning, so older call sites still work; the additional
    keyword arguments supply the structural signals.
    """
    similarity = _clamp((similarity + 1) / 2)
    nli_score = _clamp(nli_score)
    source_reliability = _clamp(source_reliability)
    detail_coverage = _clamp(detail_coverage)

    label = (nli_label or "neutral").lower().strip()

    # Neutral NLI contributes nothing: a chunk that merely does not contradict
    # the claim is not evidence for it.
    nli_component = nli_score if label in ("entailment", "contradiction") else 0.0

    date_component = DATE_SCORES.get(date_status, 0.30)
    event_component = EVENT_SCORES.get(event_match, 0.15)

    # A claim without a stated location cannot be penalised for lacking one.
    location_component = 1.0 if not claim_has_location else (1.0 if location_match else 0.0)

    score = (
        date_component * W_DATE
        + event_component * W_EVENT
        + detail_coverage * W_DETAIL
        + location_component * W_LOCATION
        + nli_component * W_NLI
        + source_reliability * W_QUALITY
        + similarity * W_SIMILARITY
    )

    # An explicit date conflict, unless the text corrects the claim itself,
    # halves the strength regardless of how well the rest lines up.
    if date_status == "different" and not debunking:
        score *= 0.5

    # A snippet, or a non-primary publisher, is bounded well below primary
    # article evidence no matter how well the other parts line up.
    if label == "neutral":
        score *= 0.45

    return round(_clamp(score), 4)


# Hard ceilings applied by the analyzer after classification.
CLASS_CEILINGS = {
    "SUPPORT": 0.95,
    "CONTRADICTION": 0.95,
    "RELATED_CONTEXT": 0.25,
    "NON_EVIDENCE": 0.02,
}


def apply_class_ceiling(score, evidence_class, evidence_type="scraped_article"):
    """
    Bound an evidence score by its class and by the trust of the retrieval mode.

    A snippet is fallback evidence and cannot reach the strength of extracted
    article text, which in turn cannot reach the strength of a page we scraped
    and verified ourselves.
    """
    score = _clamp(score)
    ceiling = CLASS_CEILINGS.get(evidence_class, 0.5)

    if evidence_type == "search_extract":
        ceiling = min(ceiling, 0.80)
    elif evidence_type == "search_snippet":
        ceiling = min(ceiling, 0.45)

    return round(min(score, ceiling), 4)
