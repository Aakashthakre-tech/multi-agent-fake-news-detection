"""
tests/test_apollo11_claim_evidence.py
=====================================
Regression tests for the Apollo 11 investigation.

Root cause
----------
The claim "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
states a MONTH and a YEAR but no day. The claim parser only recognised full
day/month/year dates, so:

  * the claim had no date at all (`primary_date` was None), which left
    `check_date_compatibility` able to return only "same_year";
  * the month name survived date-stripping and was extracted as a claim
    "location", so `has_claim_location` was true for a claim that names no
    place;
  * with `date_status == "same_year"` the event matcher took a branch that
    requires an *incident* marker (a disaster vocabulary: "broke out", "fire",
    "killed"), so `event_match` could never be "strong".

`classify_evidence` requires `date_status in ("match", "same_day_published")`
and `event_match == "strong"` for SUPPORT, so no passage could ever support
this claim - including NASA's own Apollo 11 text, which the NLI model entailed
at 0.997.

These tests are all fixture-based. No network call and no LLM call is made.
They prove the decision logic, not real-world retrieval quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claim_profile import build_claim_profile, extract_dates  # noqa: E402
from evidence_analyzer import (  # noqa: E402
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
)
from source_reliability import get_source_profile  # noqa: E402

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

NASA_TEXT = (
    "Apollo 11 was the mission that landed the first humans on the Moon. "
    "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the "
    "Sea of Tranquility on July 20, 1969, while Michael Collins remained in "
    "lunar orbit aboard the Columbia."
)

SMITHSONIAN_TEXT = (
    "On July 20, 1969, Apollo 11 landed on the Moon and Neil Armstrong and "
    "Edwin 'Buzz' Aldrin became the first humans to step onto the lunar "
    "surface."
)

IRRELEVANT_TEXT = (
    "The Saturn V rocket was used for the Skylab and Apollo-Soyuz missions. "
    "The Apollo Applications Program was cancelled in 1968."
)


def _classify(text, url="https://www.nasa.gov/apollo-11", similarity=0.7,
              nli_label="entailment", nli_score=0.99):
    attrs = extract_claim_attributes(CLAIM)
    date_check = check_date_compatibility(CLAIM, text, attrs)
    level, signals = evaluate_event_match(
        CLAIM, text, attrs, date_check["status"], similarity, nli_entails=True
    )
    klass = classify_evidence(
        {
            "date_status": date_check["status"],
            "event_match": level,
            "similarity": similarity,
            "raw_nli_label": nli_label,
            "raw_nli_score": nli_score,
            "nli_status": "ok",
            "location_match": signals["location_match"],
            "claim_has_location": signals.get("claim_has_location", True),
            "detail_coverage": signals["detail_coverage"],
            "debunking": False,
        },
        "scraped_article",
        get_source_profile(url, "scraped_article"),
    )
    return date_check, level, signals, klass


# ===========================================================================
# 1. The claim's month+year is recognised as a date
# ===========================================================================

def test_month_year_claim_is_parsed_as_a_date():
    profile = build_claim_profile(CLAIM)
    assert profile["primary_date"] is not None, (
        "a month+year is a date claim and must be parsed as one"
    )
    assert profile["primary_date"]["month"] == 7
    assert profile["primary_date"]["year"] == 1969
    assert profile["primary_date"]["day"] is None
    assert profile["primary_date"]["precision"] == "month"
    assert profile["has_date"] is True


def test_month_year_claim_has_no_phantom_location():
    """
    The month name must not be extracted as a place.

    A claim that names no place must not be held to a location requirement.
    """
    profile = build_claim_profile(CLAIM)
    assert profile["locations"] == []
    assert profile["location_terms"] == []


def test_a_month_name_lookalike_is_not_a_date():
    """"Mayflower 1620" must not be parsed as "May 1620"."""
    dates = extract_dates("The Mayflower 1620 crossed the Atlantic.")
    assert dates == []


def test_a_full_date_is_not_also_parsed_as_a_month_year():
    dates = extract_dates("The launch was on 12 Aug 2024.")
    assert len(dates) == 1
    assert dates[0]["day"] == 12
    assert dates[0]["month"] == 8
    assert dates[0]["year"] == 2024


def test_extract_dates_handles_whitespace_separated_ordinal():
    """Ordinal suffix separated by whitespace from the day: "July 20 th , 1969"."""
    dates = extract_dates("July 20 th , 1969")
    assert len(dates) == 1
    assert dates[0]["day"] == 20
    assert dates[0]["month"] == 7
    assert dates[0]["year"] == 1969
    assert dates[0]["precision"] == "day"


def test_extract_dates_handles_whitespace_separated_ordinal_in_sentence():
    """The Astronaut Foods sentence format from the live trace."""
    dates = extract_dates(
        "Fifty-Two Years ago, on July 20 th , 1969, two astronauts set foot on the moon."
    )
    assert len(dates) == 1
    assert dates[0]["day"] == 20
    assert dates[0]["month"] == 7
    assert dates[0]["year"] == 1969


def test_whitespace_separated_ordinal_unrelated_date_not_event_conflict():
    """
    A date with separated ordinal that refers to a separate event must not
    be bound to the claimed event. The sentence has no event vocabulary.
    """
    from evidence_analyzer import check_date_compatibility, extract_claim_attributes

    claim = "The Apollo 11 mission landed humans on the Moon on July 20, 1970."
    attrs = extract_claim_attributes(claim)
    passage = (
        "The magazine was printed on July 20 th , 1971, and discusses its "
        "editorial legacy."
    )
    date_check = check_date_compatibility(claim, passage, attrs)
    assert date_check["event_date_conflict"] is False
    assert date_check["event_date_mismatch"] is False


# ===========================================================================
# 2. Relevant NASA article text can be classified as supporting evidence
# ===========================================================================

def test_relevant_nasa_article_text_is_supporting_evidence():
    date_check, level, signals, klass = _classify(NASA_TEXT)

    assert date_check["status"] == "match", date_check["reason"]
    assert level == "strong"
    assert signals["claim_has_location"] is False
    assert signals["location_ok"] is True
    assert klass == "SUPPORT"


def test_the_smithsonian_article_is_evaluated_as_an_independent_source():
    """
    The Smithsonian Apollo 11 page must be evaluated on its own merits, as its
    own publisher group.
    """
    date_check, level, signals, klass = _classify(
        SMITHSONIAN_TEXT, url="https://www.si.edu/spotlight/apollo-11"
    )

    assert date_check["status"] == "match"
    assert level == "strong"
    assert klass == "SUPPORT"

    # It is a different publisher, so it is a different independence group.
    from evidence_fusion import calculate_source_agreement

    def _chunk(source_id, url, text):
        return {
            "source_id": source_id,
            "url": url,
            "domain": url.split("/")[2],
            "title": "t",
            "evidence_type": "scraped_article",
            "chunk": text,
            "similarity": 0.7,
            "nli_label": "entailment",
            "nli_score": 0.99,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.99,
            "source_reliability": 0.95,
            "source_tier": "official",
            "independence_group": url.split("/")[2].replace("www.", ""),
            "is_primary_source": True,
            "evidence_score": 0.9,
            "evidence_class": "SUPPORT",
            "date_status": "match",
            "event_match": "strong",
            "location_match": False,
            "detail_coverage": 0.9,
            "nli_status": "ok",
            "debunking": False,
        }

    agreement = calculate_source_agreement([
        _chunk(1, "https://www.nasa.gov/apollo-11", NASA_TEXT),
        _chunk(2, "https://www.si.edu/spotlight/apollo-11", SMITHSONIAN_TEXT),
    ])

    assert agreement["independence_groups"] == 2
    assert agreement["independent_supporting"] == 2
    assert agreement["primary_supporting_groups"] == 2
    assert agreement["article_supporting_groups"] == 2


# ===========================================================================
# 3. Search snippets alone cannot satisfy the primary-article requirement
# ===========================================================================

def test_snippet_only_support_cannot_establish_a_verdict():
    from evidence_fusion import calculate_source_agreement
    from final_verdict import deterministic_gate

    def _snippet(source_id, url):
        return {
            "source_id": source_id,
            "url": url,
            "domain": url.split("/")[2],
            "title": "t",
            "evidence_type": "search_snippet",
            "chunk": "Apollo 11 landed humans on the Moon in July 1969.",
            "similarity": 0.8,
            "nli_label": "entailment",
            "nli_score": 0.99,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.99,
            "source_reliability": 0.9,
            "source_tier": "official",
            "independence_group": url.split("/")[2].replace("www.", ""),
            "is_primary_source": True,
            "evidence_score": 0.4,
            "evidence_class": "SUPPORT",
            "date_status": "match",
            "event_match": "strong",
            "location_match": False,
            "detail_coverage": 0.9,
            "nli_status": "ok",
            "debunking": False,
        }

    results = [
        _snippet(1, "https://www.nasa.gov/apollo-11"),
        _snippet(2, "https://www.si.edu/spotlight/apollo-11"),
    ]
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)

    assert agreement["snippet_only_sources"] == 2
    assert agreement["article_supporting_groups"] == 0
    assert "REAL" not in gate["permitted_verdicts"]
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}


def test_a_snippet_is_never_classified_as_support():
    """
    The analyzer itself must refuse to classify a search snippet as support.
    """
    evidence_class = classify_evidence(
        {
            "date_status": "match",
            "event_match": "strong",
            "similarity": 0.9,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.99,
            "nli_status": "ok",
            "location_match": False,
            "claim_has_location": False,
            "detail_coverage": 0.95,
            "debunking": False,
        },
        "search_snippet",
        get_source_profile("https://www.nasa.gov/apollo-11", "search_snippet"),
    )
    assert evidence_class not in ("SUPPORT", "CONTRADICTION")


# ===========================================================================
# 4. Irrelevant Apollo context is not counted as claim support
# ===========================================================================

def test_irrelevant_apollo_context_is_not_support():
    """
    Text about the Apollo programme that does not assert the claimed event
    must not be counted as support, even though it is on-topic and mentions
    the same year.
    """
    date_check, level, signals, klass = _classify(
        IRRELEVANT_TEXT, nli_label="neutral", nli_score=0.95
    )

    assert klass != "SUPPORT"
    assert klass in ("RELATED_CONTEXT", "NON_EVIDENCE")


def test_a_different_month_of_the_same_year_is_a_conflict():
    """
    "August 1969" does not satisfy a claim about "July 1969".
    """
    text = (
        "Apollo 11 landed the first humans on the Moon on August 20, 1969, "
        "according to the mission log."
    )
    date_check = check_date_compatibility(CLAIM, text, extract_claim_attributes(CLAIM))
    assert date_check["status"] == "different"


def test_a_year_only_passage_does_not_confirm_the_month():
    """
    Evidence that only says "1969" does not confirm "July 1969", so it stays
    at "same_year" and cannot become SUPPORT.
    """
    text = "Apollo 11 landed the first humans on the Moon in 1969."
    date_check = check_date_compatibility(CLAIM, text, extract_claim_attributes(CLAIM))
    assert date_check["status"] == "same_year"


def test_a_passage_with_no_date_at_all_is_not_direct_evidence():
    text = (
        "Apollo 11 landed the first humans on the Moon, and Neil Armstrong "
        "and Buzz Aldrin walked on the lunar surface."
    )
    date_check = check_date_compatibility(CLAIM, text, extract_claim_attributes(CLAIM))
    assert date_check["status"] == "unknown"

    _, level, _, klass = _classify(text)
    assert klass != "SUPPORT"


# ===========================================================================
# 5. The deterministic evidence floor still blocks unsupported verdicts
# ===========================================================================

def test_the_gate_still_blocks_a_single_supporting_publisher():
    from evidence_fusion import calculate_source_agreement
    from final_verdict import deterministic_gate

    results = [{
        "source_id": 1,
        "url": "https://www.nasa.gov/apollo-11",
        "domain": "nasa.gov",
        "title": "t",
        "evidence_type": "scraped_article",
        "chunk": NASA_TEXT,
        "similarity": 0.7,
        "nli_label": "entailment",
        "nli_score": 0.99,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.99,
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "nasa.gov",
        "is_primary_source": True,
        "evidence_score": 0.9,
        "evidence_class": "SUPPORT",
        "date_status": "match",
        "event_match": "strong",
        "location_match": False,
        "detail_coverage": 0.9,
        "nli_status": "ok",
        "debunking": False,
    }]

    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)

    assert agreement["independent_supporting"] == 1
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "Only one independent publisher" in gate["permitted_reason"]


def test_the_existing_policy_guards_are_untouched():
    """
    The date fix must not weaken the snippet, non-primary, event-match,
    coverage or NLI requirements.
    """
    base = {
        "date_status": "match",
        "event_match": "strong",
        "similarity": 0.7,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.99,
        "nli_status": "ok",
        "location_match": False,
        "claim_has_location": False,
        "detail_coverage": 0.9,
        "debunking": False,
    }
    profile = get_source_profile("https://www.nasa.gov/apollo-11", "scraped_article")

    # A snippet is never SUPPORT.
    assert classify_evidence(base, "search_snippet", profile) != "SUPPORT"

    # A non-primary source is never SUPPORT on its own.
    weak = get_source_profile("https://blog.example.com/x", "scraped_article")
    assert classify_evidence(base, "scraped_article", weak) != "SUPPORT"

    # A weak event match is never SUPPORT.
    assert classify_evidence({**base, "event_match": "weak"}, "scraped_article", profile) != "SUPPORT"

    # A low claim-detail coverage is never SUPPORT.
    assert classify_evidence({**base, "detail_coverage": 0.1}, "scraped_article", profile) != "SUPPORT"

    # A non-entailing NLI is never SUPPORT.
    assert classify_evidence({**base, "raw_nli_label": "neutral"}, "scraped_article", profile) != "SUPPORT"

    # A low NLI score is never SUPPORT.
    assert classify_evidence({**base, "raw_nli_score": 0.5}, "scraped_article", profile) != "SUPPORT"

    # A date that is not a match is never SUPPORT.
    assert classify_evidence({**base, "date_status": "same_year"}, "scraped_article", profile) != "SUPPORT"
    assert classify_evidence({**base, "date_status": "different"}, "scraped_article", profile) != "SUPPORT"


# ===========================================================================
# 6. Verdict, evidence counts and confidence labels stay consistent
# ===========================================================================

def test_the_nli_label_is_the_models_own_label():
    """
    The record's `nli_label` must be the model's label, so it can be paired
    with the model's score. The evidence CLASS has its own field.
    """
    from evidence_analyzer import analyze_evidence

    sources = [{
        "url": "https://www.nasa.gov/apollo-11",
        "domain": "nasa.gov",
        "title": "Apollo 11",
        "content": NASA_TEXT,
        "evidence_type": "scraped_article",
        "scrape_status": "SUCCESS",
    }]

    results = analyze_evidence(CLAIM, sources, verbose=False)
    assert results, "the analyzer must produce at least one record"

    for record in results:
        # The label and the score must come from the same source: the model.
        assert record["nli_label"] == record["raw_nli_label"]
        assert record["nli_score"] == record["raw_nli_score"]
        # The class label is reported separately and is never the NLI label.
        assert record["evidence_class_label"] in (
            "entailment", "contradiction", "neutral"
        )

    # A passage the model entailed must not be reported as "neutral".
    entailed = [r for r in results if r["raw_nli_label"] == "entailment"]
    for record in entailed:
        assert record["nli_label"] == "entailment", (
            "a passage the model entailed must not carry a 'neutral' NLI label"
        )


def test_the_evidence_record_pairs_the_nli_label_with_the_nli_score():
    """
    The specific defect: a chunk the model entailed at 0.9976 was reported as
    "nli_label: neutral, nli_score: 0.9976" - a neutral label attached to an
    entailment score.
    """
    from evidence_analyzer import analyze_evidence

    sources = [{
        "url": "https://www.nasa.gov/apollo-11",
        "domain": "nasa.gov",
        "title": "Apollo 11",
        "content": NASA_TEXT,
        "evidence_type": "scraped_article",
        "scrape_status": "SUCCESS",
    }]

    results = analyze_evidence(CLAIM, sources, verbose=False)

    for record in results:
        if record["raw_nli_label"] == "entailment" and record["raw_nli_score"] >= 0.9:
            assert record["nli_label"] == "entailment"
            assert record["nli_score"] >= 0.9
        if record["raw_nli_label"] == "neutral":
            assert record["nli_label"] == "neutral"

    # The class label may differ from the NLI label, and is reported
    # separately rather than under the NLI label's name.
    for record in results:
        if record["evidence_class"] == "RELATED_CONTEXT":
            assert record["evidence_class_label"] == "neutral"


def test_verdict_counts_and_confidence_stay_consistent():
    """
    A REAL verdict must have the supporting counts that justify it, and its
    confidence must be labelled as evidence strength.
    """
    from evidence_fusion import calculate_source_agreement, calculate_fusion_score
    from final_verdict import generate_final_verdict

    def _chunk(source_id, url, text):
        return {
            "source_id": source_id,
            "url": url,
            "domain": url.split("/")[2],
            "title": "t",
            "evidence_type": "scraped_article",
            "chunk": text,
            "similarity": 0.7,
            "nli_label": "entailment",
            "nli_score": 0.99,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.99,
            "source_reliability": 0.95,
            "source_tier": "official",
            "independence_group": url.split("/")[2].replace("www.", ""),
            "is_primary_source": True,
            "evidence_score": 0.9,
            "evidence_class": "SUPPORT",
            "date_status": "match",
            "event_match": "strong",
            "location_match": False,
            "detail_coverage": 0.9,
            "nli_status": "ok",
            "debunking": False,
        }

    results = [
        _chunk(1, "https://www.nasa.gov/apollo-11", NASA_TEXT),
        _chunk(2, "https://www.si.edu/spotlight/apollo-11", SMITHSONIAN_TEXT),
    ]
    agreement = calculate_source_agreement(results)
    verdict = generate_final_verdict(
        results, agreement, calculate_fusion_score(results, agreement),
        "Verdict:\nREAL\n\nConfidence:\n95%\n\nReasoning:\nConfirmed.",
        "Verdict:\nREAL\n\nConfidence:\n93%\n\nReasoning:\nConfirmed.",
    )

    assert verdict["verdict"] == "REAL"
    assert verdict["independent_supporting"] == 2
    assert verdict["primary_supporting_groups"] == 2
    assert verdict["article_supporting_groups"] == 2
    assert verdict["confidence_basis"] == "evidence_strength"
    assert verdict["permitted_verdicts"] == ["REAL", "UNVERIFIED"]
    # The supporting passages are reported as SUPPORT, not as contradictions.
    assert all(
        e["url"].endswith(("nasa.gov/apollo-11", "si.edu/spotlight/apollo-11"))
        for e in verdict["supporting_evidence"]
    )
    assert verdict["contradicting_evidence"] == []
