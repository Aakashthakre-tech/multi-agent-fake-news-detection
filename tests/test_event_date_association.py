"""
tests/test_event_date_association.py
=====================================
Regression tests for the date-aware evidence classification.

Root cause
----------
A passage can contain the claim's date while dating the CLAIMED event to a
different date. The UN's Apollo 11 Moon Rock page says:

    "…on their 20 July 1969 moon landing. They presented this gift to the
     United Nations … on 20 July 1970."

For the claim "The Apollo 11 mission landed humans on the Moon on July 20,
1970", the date check found "20 July 1970" somewhere in the passage and
reported a match, the NLI model reported entailment (0.85 - it judges the
topic, not the year), and the passage was classified SUPPORT. The 1970 date
belongs to the gift presentation; the landing is explicitly dated 1969.

The fix associates each date with the event it describes: a date is bound to
the event when it appears in a sentence that also mentions the claim's event
vocabulary. A passage that dates the claimed event differently can no longer
support the claim, and is a contradiction when the relationship is clear.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_analyzer import (  # noqa: E402
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
)
from evidence_fusion import calculate_source_agreement  # noqa: E402
from final_verdict import deterministic_gate  # noqa: E402
from source_reliability import get_source_profile  # noqa: E402

UN_PASSAGE = (
    "This Moon Rock was retrieved from the moon during the USA NASA Apollo 11 "
    "space mission, crewed by astronauts Neil A. Armstrong, Edwin E. Aldrin Jr., "
    "and Michael Collins, on their 20 July 1969 moon landing. They presented "
    "this gift to the United Nations at the one-year anniversary of that "
    "flight, on 20 July 1970."
)

CLAIM_1970 = "The Apollo 11 mission landed humans on the Moon on July 20, 1970."
CLAIM_1969 = "The Apollo 11 mission landed humans on the Moon on July 20, 1969."
CLAIM_1969_MONTH = "The Apollo 11 mission landed humans on the Moon in July 1969."

UN_URL = "https://www.un.org/ungifts/es/node/3986"


def _classify(claim, passage, url=UN_URL, nli_label="entailment",
               nli_score=0.85, similarity=0.85, evidence_type="scraped_article",
               primary=True, tier="established_wire"):
    attrs = extract_claim_attributes(claim)
    date_check = check_date_compatibility(claim, passage, attrs)
    level, signals = evaluate_event_match(
        claim, passage, attrs, date_check["status"], similarity, nli_entails=True,
        event_date_conflict=bool(date_check.get("event_date_conflict")),
    )
    merged = {
        "similarity": similarity,
        "date_status": date_check["status"],
        "event_match": level,
        "raw_nli_label": nli_label,
        "raw_nli_score": nli_score,
        "nli_status": "ok",
        "location_match": signals["location_match"],
        "claim_has_location": signals.get("claim_has_location", False),
        "detail_coverage": signals["detail_coverage"],
        "debunking": False,
        "passage_kind": "paragraph",
        "event_date_conflict": bool(date_check.get("event_date_conflict")),
        "event_date_mismatch": bool(date_check.get("event_date_mismatch")),
    }
    if evidence_type == "search_snippet":
        profile = {"is_primary": False, "tier": tier, "reliability": 0.4}
    elif primary:
        profile = get_source_profile(url, evidence_type)
    else:
        profile = {"is_primary": False, "tier": tier, "reliability": 0.4}
    klass = classify_evidence(merged, evidence_type, profile)
    return date_check, level, merged, klass


def _record(url, text, evidence_class, score=0.9, source_id=1):
    return {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2],
        "title": "t",
        "evidence_type": "scraped_article",
        "chunk": text,
        "similarity": 0.8,
        "nli_label": "entailment",
        "nli_score": 0.95,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.95,
        "source_reliability": 0.9,
        "source_tier": "official",
        "independence_group": url.split("/")[2].replace("www.", ""),
        "is_primary_source": True,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": "match",
        "event_match": "strong",
        "location_match": False,
        "detail_coverage": 0.9,
        "nli_status": "ok",
        "debunking": False,
    }


# ===========================================================================
# 1. Apollo 11 landing in 1969 - supported by the UN passage
# ===========================================================================

def test_1_apollo11_1969_is_supported_by_the_un_passage():
    date_check, level, signals, klass = _classify(CLAIM_1969, UN_PASSAGE)

    assert date_check["event_date_conflict"] is False
    assert date_check["event_date_mismatch"] is False
    assert date_check["status"] == "match"
    assert klass == "SUPPORT"


def test_1b_the_month_precision_claim_is_also_supported():
    date_check, level, signals, klass = _classify(CLAIM_1969_MONTH, UN_PASSAGE)

    assert date_check["event_date_conflict"] is False
    assert klass == "SUPPORT"


# ===========================================================================
# 2. Apollo 11 landing in 1970 - contradicted by the UN passage
# ===========================================================================

def test_2_apollo11_1970_is_contradicted_by_the_un_passage():
    date_check, level, signals, klass = _classify(CLAIM_1970, UN_PASSAGE)

    assert date_check["event_date_conflict"] is True
    assert klass == "CONTRADICTION", (
        "a passage that dates the claimed event to 1969 cannot support a "
        "claim that it happened in 1970"
    )


def test_2b_the_contradiction_is_not_merely_an_nli_effect():
    """
    The NLI model reports entailment for this passage. The contradiction must
    come from the date/event association, not from the model.
    """
    date_check, level, signals, klass = _classify(
        CLAIM_1970, UN_PASSAGE, nli_label="entailment", nli_score=0.85
    )
    assert signals["raw_nli_label"] == "entailment"
    assert klass == "CONTRADICTION"


# ===========================================================================
# 3. The 1970 presentation is not confused with the landing date
# ===========================================================================

def test_3_the_gift_presentation_date_is_not_the_landing_date():
    """
    The sentence carrying 20 July 1970 describes the gift presentation, not
    the landing, so it must not bind that date to the claimed event.
    """
    gift_only = (
        "They presented this gift to the United Nations at the one-year "
        "anniversary of that flight, on 20 July 1970."
    )
    date_check = check_date_compatibility(
        CLAIM_1970, gift_only, extract_claim_attributes(CLAIM_1970)
    )
    assert date_check["event_date_conflict"] is False
    assert date_check["event_date_mismatch"] is False


def test_3b_the_landing_sentence_alone_conflicts():
    landing_only = (
        "Aldrin Jr., and Michael Collins, on their 20 July 1969 moon landing."
    )
    date_check = check_date_compatibility(
        CLAIM_1970, landing_only, extract_claim_attributes(CLAIM_1970)
    )
    assert date_check["event_date_conflict"] is True


# ===========================================================================
# 4. Both dates in one passage, associated with different events
# ===========================================================================

def test_4_both_dates_present_but_bound_to_different_events():
    """
    The full UN passage carries both dates. Only the one bound to the landing
    decides the outcome.
    """
    date_check = check_date_compatibility(
        CLAIM_1970, UN_PASSAGE, extract_claim_attributes(CLAIM_1970)
    )
    assert date_check["status"] == "match", (
        "the claim's date IS in the passage, so the plain date status is a match"
    )
    assert date_check["event_date_conflict"] is True, (
        "but the event itself is dated 1969"
    )


# ===========================================================================
# 5. A passage with the year but no explicit event-date relationship
# ===========================================================================

def test_5_year_without_an_event_date_relationship_is_not_support():
    """
    A passage that gives the year but never ties it to the event stays
    context, not support.
    """
    passage = (
        "The Apollo 11 mission remains one of the most studied programmes in "
        "history, and its legacy was revisited in 1969 documentaries."
    )
    date_check, level, signals, klass = _classify(CLAIM_1969, passage)

    assert klass != "SUPPORT"
    assert klass in ("RELATED_CONTEXT", "NON_EVIDENCE")


def test_5b_a_passage_that_never_states_a_date_is_not_support():
    passage = (
        "The Apollo 11 mission landed humans on the Moon and changed what was "
        "thought possible, though it did not end the space race on its own."
    )
    date_check, level, signals, klass = _classify(CLAIM_1969, passage)
    assert klass != "SUPPORT"


# ===========================================================================
# 6. Duplicate passages from one publisher do not inflate the counts
# ===========================================================================

def test_6_duplicate_passages_from_one_publisher_are_one_source():
    records = [
        _record(UN_URL, UN_PASSAGE, "SUPPORT", source_id=1),
        _record(UN_URL, UN_PASSAGE, "SUPPORT", source_id=2),
    ]
    agreement = calculate_source_agreement(records)

    assert agreement["total_sources"] == 1
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1

    gate = deterministic_gate(agreement, records)
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}


# ===========================================================================
# 7. Insufficient independent corroboration still enforces UNVERIFIED
# ===========================================================================

def test_7_one_contradicting_publisher_is_not_enough_for_fake():
    records = [_record(UN_URL, UN_PASSAGE, "CONTRADICTION", source_id=1)]
    agreement = calculate_source_agreement(records)

    assert agreement["independent_contradicting"] == 1
    gate = deterministic_gate(agreement, records)

    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "Only one independent publisher contradicts" in gate["permitted_reason"]


def test_7b_two_independent_contradicting_publishers_can_be_fake():
    records = [
        _record(UN_URL, UN_PASSAGE, "CONTRADICTION", source_id=1),
        _record("https://www.nasa.gov/apollo-11", UN_PASSAGE, "CONTRADICTION",
                source_id=2),
    ]
    agreement = calculate_source_agreement(records)
    gate = deterministic_gate(agreement, records)

    assert agreement["independent_contradicting"] == 2
    assert "FAKE" in gate["permitted_verdicts"]


# ===========================================================================
# 8. The event-date conflict never fabricates a refutation on its own
# ===========================================================================

def test_8a_a_snippet_cannot_become_a_contradiction():
    date_check, level, signals, klass = _classify(
        CLAIM_1970, UN_PASSAGE, evidence_type="search_snippet", primary=False
    )
    assert klass != "CONTRADICTION"


def test_8b_a_non_primary_source_cannot_become_a_contradiction():
    date_check, level, signals, klass = _classify(
        CLAIM_1970, UN_PASSAGE, tier="weak/blog", primary=False
    )
    assert klass != "CONTRADICTION"


def test_8c_a_window_cannot_become_a_contradiction():
    attrs = extract_claim_attributes(CLAIM_1970)
    date_check = check_date_compatibility(CLAIM_1970, UN_PASSAGE, attrs)
    signals = {
        "similarity": 0.85,
        "date_status": date_check["status"],
        "event_match": "strong",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.85,
        "nli_status": "ok",
        "location_match": False,
        "claim_has_location": False,
        "detail_coverage": 0.9,
        "debunking": False,
        "passage_kind": "window",
        "event_date_conflict": bool(date_check.get("event_date_conflict")),
        "event_date_mismatch": bool(date_check.get("event_date_mismatch")),
    }
    klass = classify_evidence(
        signals, "scraped_article",
        get_source_profile(UN_URL, "scraped_article"),
    )
    assert klass != "CONTRADICTION"


def test_8d_the_deterministic_gate_is_unchanged():
    """
    The gate still requires two independent publishers and a primary
    article; the date fix does not relax it.
    """
    records = [
        _record(UN_URL, UN_PASSAGE, "CONTRADICTION", source_id=1),
        _record("https://www.nasa.gov/apollo-11", UN_PASSAGE, "CONTRADICTION",
                source_id=2),
    ]
    agreement = calculate_source_agreement(records)
    gate = deterministic_gate(agreement, records)

    assert agreement["primary_contradicting_groups"] >= 1
    assert agreement["article_contradicting_groups"] >= 1
    assert gate["permitted_verdicts"] == {"FAKE", "UNVERIFIED"}


def test_8e_a_claim_without_a_date_is_unaffected():
    """
    The new association must not change the behaviour of undated claims.
    """
    claim = "The Apollo 11 mission landed humans on the Moon."
    date_check, level, signals, klass = _classify(claim, UN_PASSAGE)

    assert date_check["event_date_conflict"] is False
    assert date_check["event_date_mismatch"] is False


# ===========================================================================
# 9. A passage that explicitly dates the claimed event to a conflicting date
# ===========================================================================

ASTRONAUT_FOODS_SENTENCE = (
    "Fifty-Two Years ago, on July 20th, 1969, two astronauts set foot on "
    "the moon for the first time in human history!"
)


def test_9_astronautfoods_1969_landing_contradicts_1970_claim():
    """
    The Astronaut Foods passage states the Apollo 11 landing happened on
    July 20, 1969. For a claim that it happened on July 20, 1970, this is an
    explicit event-date conflict and must classify as CONTRADICTION, even
    without debunking language.
    """
    date_check, level, signals, klass = _classify(
        CLAIM_1970, ASTRONAUT_FOODS_SENTENCE,
        nli_label="contradiction", nli_score=0.90, similarity=0.70,
    )
    assert date_check["event_date_conflict"] is True
    assert date_check["status"] == "different"
    assert level == "strong", f"expected strong event match, got {level}"
    assert klass == "CONTRADICTION", (
        "a passage that explicitly dates the claimed event to a conflicting "
        "date must be a contradiction"
    )


def test_9b_unrelated_date_in_passage_is_not_a_contradiction():
    """
    A date that refers to a separate event (not bound to the claimed event)
    must not become an event-date conflict, even when the date differs from
    the claim.
    """
    passage = (
        "The Apollo 11 mission was a historic achievement. The magazine was "
        "printed on July 20, 1971. It discusses its legacy in a separate "
        "column."
    )
    date_check, level, signals, klass = _classify(
        CLAIM_1970, passage,
        nli_label="neutral", nli_score=0.90, similarity=0.70,
    )
    # The 1971 date is in a sentence with no event vocabulary, so it is not
    # bound to the claimed event.
    assert date_check["event_date_conflict"] is False
    assert date_check["status"] == "different"
    assert level != "strong", (
        "an unrelated date must not produce a strong event match"
    )
    assert klass != "CONTRADICTION", (
        "an unrelated date must not become a contradiction"
    )
