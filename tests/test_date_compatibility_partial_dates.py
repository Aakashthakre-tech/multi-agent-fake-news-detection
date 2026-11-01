"""
tests/test_date_compatibility_partial_dates.py
==============================================
Regression tests for the evidence-analysis crash in
``evidence_analyzer.check_date_compatibility``.

Root cause
----------
The conflict collection was built as a set of raw ``(day, month, year)``
tuples and then sorted directly. A date that does not name a day - a
month-precision date such as "July 2024" - has ``day = None``, so the sort
compared ``None`` with an ``int`` and raised

    TypeError: '<' not supported between instances of 'int' and 'NoneType'

whenever such a date and an integer-day date were both in conflict. The
rendering of the conflicting dates (``f"{d['day']:02d}"``) had the same
problem one line later.

The fix compares only the fields that BOTH dates actually specify, and sorts
on a type-safe key. A month-precision date is the same month stated less
precisely, so it is not a conflict with a date that names a day; a
month-precision date in a genuinely different month or year IS still a
conflict, and is still reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_analyzer import (  # noqa: E402
    check_date_compatibility,
    extract_claim_attributes,
)

DAY_CLAIM = "The council approved the project on 12 Aug 2024."
MONTH_CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."


def _attrs(claim):
    return extract_claim_attributes(claim)


# ===========================================================================
# 1. Mixed integer and None values in the conflict structure
# ===========================================================================

def test_mixed_int_and_none_days_do_not_crash():
    """
    The reported crash: a day-precision claim, evidence containing a
    month-precision date in the same year (a conflict in its own right) plus
    an integer-day conflict.
    """
    evidence = (
        "The council approved the project on 15 Aug 2024. The original "
        "programme was approved in July 2024."
    )
    result = check_date_compatibility(DAY_CLAIM, evidence, _attrs(DAY_CLAIM))

    assert result["status"] == "different"
    # The conflict information is preserved, including the month-precision one.
    assert "Jul" in result["reason"] or "July" in result["reason"]
    assert "Aug" in result["reason"]


def test_a_month_precision_date_alone_does_not_crash():
    """A single month-precision date must not reach the sort."""
    evidence = "The programme was approved in July 2024."
    result = check_date_compatibility(DAY_CLAIM, evidence, _attrs(DAY_CLAIM))

    assert result["status"] in ("different", "same_year", "unknown")


def test_only_month_precision_dates_in_the_conflict_set():
    """Several month-precision conflicts must still be ordered safely."""
    evidence = (
        "It was approved in July 2024, and again in June 2024, and the "
        "original plan was from May 2024."
    )
    result = check_date_compatibility(DAY_CLAIM, evidence, _attrs(DAY_CLAIM))

    assert result["status"] == "different"
    assert result["reason"]


def test_source_level_month_precision_dates_do_not_crash():
    """
    The source-level date context can also carry a month-precision date.
    """
    evidence = "The council approved the project on 15 Aug 2024."
    result = check_date_compatibility(
        DAY_CLAIM,
        evidence,
        _attrs(DAY_CLAIM),
        source_dates=[{"day": None, "month": 7, "year": 2024,
                       "month_name": "July", "precision": "month"}],
    )
    assert result["status"] in ("different", "same_year", "match")


# ===========================================================================
# 2. Missing or partially parsed dates
# ===========================================================================

def test_a_claim_without_a_date_reports_unknown():
    claim = "The council approved the project."
    result = check_date_compatibility(claim, "Some text.", _attrs(claim))
    assert result["status"] == "unknown"
    assert "Claim does not contain a detectable date" in result["reason"]


def test_evidence_without_a_date_reports_unknown():
    result = check_date_compatibility(
        DAY_CLAIM, "A passage with no date in it at all.", _attrs(DAY_CLAIM)
    )
    assert result["status"] == "unknown"
    assert "Evidence does not contain a detectable date" in result["reason"]


def test_a_partially_parsed_date_is_not_fabricated():
    """
    A month-precision claim is reported at month precision. The day is not
    invented, and the evidence may be more specific without conflicting.
    """
    attrs = _attrs(MONTH_CLAIM)
    assert attrs["claim_date"]["day"] is None
    assert attrs["claim_date"]["precision"] == "month"

    result = check_date_compatibility(
        MONTH_CLAIM,
        "On 20 July 1969, Apollo 11 landed on the Moon.",
        attrs,
    )
    assert result["status"] == "match"
    assert "July 1969" in result["reason"]


def test_a_less_precise_date_is_not_a_conflict():
    """
    "July 1969" against a claim about "20 July 1969" is the same month
    stated less precisely. It must not be reported as a conflicting date.
    """
    result = check_date_compatibility(
        "The council approved the project on 20 July 1969.",
        "The council approved the project in July 1969.",
        _attrs("The council approved the project on 20 July 1969."),
    )
    assert result["status"] != "different", result["reason"]


# ===========================================================================
# 3. Valid conflicting dates still conflict
# ===========================================================================

def test_a_different_day_in_the_same_month_is_a_conflict():
    result = check_date_compatibility(
        DAY_CLAIM,
        "The council approved the project on 15 Aug 2024.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] == "different"
    assert "15 Aug 2024" in result["reason"]


def test_a_different_month_in_the_same_year_is_a_conflict():
    result = check_date_compatibility(
        DAY_CLAIM,
        "The council approved the project in July 2024.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] == "different"
    assert "Jul" in result["reason"]


def test_a_different_year_is_a_conflict_when_the_claimed_year_is_absent():
    result = check_date_compatibility(
        DAY_CLAIM,
        "The council approved the project on 15 Aug 2019.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] == "different"
    assert result["claim_year_absent"] is True


def test_a_different_year_is_not_a_conflict_when_the_claimed_year_appears():
    """
    The existing policy: a 2024 article that says "since 1916" is quoting
    history, not reporting a 1916 event.
    """
    result = check_date_compatibility(
        DAY_CLAIM,
        "The 2024 decision revived a programme first approved on 15 Aug 1916.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] != "different", result["reason"]


# ===========================================================================
# 4. Valid dates with no conflict
# ===========================================================================

def test_an_exact_date_match_is_a_match():
    result = check_date_compatibility(
        DAY_CLAIM,
        "The council approved the project on 12 Aug 2024.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] == "match"
    assert "12 August 2024" in result["reason"]


def test_the_same_year_without_the_exact_date_is_same_year():
    result = check_date_compatibility(
        DAY_CLAIM,
        "The council met during 2024 and discussed the project.",
        _attrs(DAY_CLAIM),
    )
    assert result["status"] == "same_year"


def test_an_exact_month_precision_match_is_a_match():
    result = check_date_compatibility(
        MONTH_CLAIM,
        "In July 1969 the Apollo 11 mission landed humans on the Moon.",
        _attrs(MONTH_CLAIM),
    )
    assert result["status"] == "match"


def test_a_different_month_for_a_month_precision_claim_is_a_conflict():
    result = check_date_compatibility(
        MONTH_CLAIM,
        "In August 1969 the Apollo 11 mission landed humans on the Moon.",
        _attrs(MONTH_CLAIM),
    )
    assert result["status"] == "different"
    assert "Aug" in result["reason"]


# ===========================================================================
# 5. The Apollo 11 claim completes evidence analysis
# ===========================================================================

def test_apollo11_claim_completes_evidence_analysis():
    """
    The original Apollo 11 claim, with article text that contains a
    month-precision date, must reach the normal decision stage.
    """
    from evidence_analyzer import analyze_evidence

    article = (
        "TITLE: July 20, 1969: One Giant Leap For Mankind - NASA\n\n"
        "They leave behind an American flag, a patch honoring the fallen "
        "Apollo 1 crew, and a plaque on one of Eagle's legs. It reads, “Here "
        "men from the planet Earth first set foot upon the moon. July 1969 A.D. "
        "We came in peace and for all mankind.”\n\n"
        "Armstrong and Aldrin blast off and dock with Collins in Columbia."
    )

    results = analyze_evidence(
        MONTH_CLAIM,
        [{
            "url": "https://www.nasa.gov/history/july-20-1969",
            "domain": "nasa.gov",
            "title": "July 20, 1969: One Giant Leap For Mankind - NASA",
            "content": article,
            "evidence_type": "scraped_article",
            "scrape_status": "SUCCESS",
        }],
        verbose=False,
    )

    # The analysis completed and produced records.
    assert results
    for record in results:
        assert record["date_status"] in (
            "match", "same_day_published", "same_year", "different", "unknown"
        )


def test_apollo11_claim_with_a_day_precision_claim_does_not_crash():
    """
    The crash shape seen with the Apollo material: a day-precision claim and
    an article whose text contains a month-precision date.
    """
    from evidence_analyzer import analyze_evidence

    claim = "The Apollo 11 mission landed humans on the Moon on 20 July 1969."
    article = (
        "TITLE: Apollo 11 - NASA\n\n"
        "The mission was prepared in July 1969 and launched on 16 July 1969, "
        "landing on 20 July 1969."
    )

    results = analyze_evidence(
        claim,
        [{
            "url": "https://www.nasa.gov/apollo-11",
            "domain": "nasa.gov",
            "title": "Apollo 11 - NASA",
            "content": article,
            "evidence_type": "scraped_article",
            "scrape_status": "SUCCESS",
        }],
        verbose=False,
    )

    assert results
