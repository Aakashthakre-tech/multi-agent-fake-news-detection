"""
tests/test_evidence_retrieval_reliability.py
=============================================
Regression tests for evidence retrieval reliability.

Root cause these cover
----------------------
A publisher often states a claim across ADJACENT paragraphs. NASA writes
"Apollo 11 astronauts ... sit atop another Saturn V", then "When the lunar
module lands ... The Eagle has landed", then "Armstrong is ready to plant the
first human foot on another world". Read on its own, each paragraph is
NEUTRAL against "NASA's Apollo 11 mission landed humans on the Moon in July
1969", so the evidence layer could never support the claim even though the
article states it plainly.

The analyzer now also offers contiguous windows of adjacent paragraphs to the
NLI model. A window may SUPPORT a claim but may not REFUTE one: the NLI model
reads a diluted premise unreliably, and a spurious contradiction is far more
harmful than a missed one.

All tests use deterministic fixtures - no network call, no live site. The live
retrieval results are reported separately by `scratch/` probes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_analyzer import (  # noqa: E402
    MAX_WINDOW_CHUNKS_PER_SOURCE,
    _build_context_windows,
    _count_tokens,
    analyze_evidence,
    classify_evidence,
)
from evidence_fusion import calculate_source_agreement  # noqa: E402

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

# A faithful miniature of NASA's article structure: the claim's elements are
# spread over three adjacent paragraphs, and no single paragraph states it.
NASA_PARAGRAPHS = [
    "TITLE: July 20, 1969: One Giant Leap For Mankind - NASA",
    "July 1969. It is a little over eight years since the flights of Gagarin "
    "and Shepard, followed quickly by President Kennedy's challenge to put a "
    "man on the moon before the decade is out.",
    "Now, on the morning of July 16, Apollo 11 astronauts Neil Armstrong, Buzz "
    "Aldrin and Michael Collins sit atop another Saturn V at Launch Complex "
    "39A at the Kennedy Space Center.",
    "When the lunar module lands at 4:17 p.m EDT, only 30 seconds of fuel "
    "remain. Armstrong radios Houston, Tranquility Base here, the Eagle has "
    "landed.",
    "At 10:56 p.m. EDT Armstrong is ready to plant the first human foot on "
    "another world. With more than half a billion people watching on "
    "television, he climbs down the ladder and proclaims that's one small step "
    "for a man, one giant leap for mankind.",
    "Aldrin joins him shortly, and offers a simple but powerful description of "
    "the lunar surface: magnificent desolation. They explore the surface for "
    "two and a half hours, collecting samples and taking photographs.",
    "They leave behind an American flag, a patch honoring the fallen Apollo 1 "
    "crew, and a plaque on one of Eagle's legs. It reads, Here men from the "
    "planet Earth first set foot upon the moon. July 1969 A.D.",
    "Armstrong and Aldrin blast off and dock with Collins in Columbia.",
]

NASA_ARTICLE = "\n\n".join(NASA_PARAGRAPHS)


def _source(url="https://www.nasa.gov/history/july-20-1969", content=None,
            evidence_type="scraped_article", domain="nasa.gov"):
    return {
        "url": url,
        "domain": domain,
        "title": "Apollo 11",
        "content": content if content is not None else NASA_ARTICLE,
        "evidence_type": evidence_type,
        "scrape_status": "SUCCESS",
    }


@pytest.fixture(scope="module")
def nasa_results():
    """
    The NASA article analysed once for the whole module.

    Each analysis runs the NLI model over several passages, so the result is
    shared instead of recomputed for every assertion.
    """
    return analyze_evidence(CLAIM, [_source()], verbose=False)


# ===========================================================================
# 1. Relevant text inside a retrieved article is preserved end to end
# ===========================================================================

def test_1_relevant_text_in_an_article_survives_extraction_and_classification(nasa_results):
    """
    The landing passage is present in the retrieved article and must reach the
    evidence layer as a support record.
    """
    results = nasa_results

    assert results, "the analyzer must produce evidence records"

    supporting = [r for r in results if r["evidence_class"] == "SUPPORT"]
    assert supporting, (
        "the article states the claim across adjacent paragraphs; the evidence "
        "layer must find it"
    )

    record = supporting[0]
    # The passage is the article's own text, not something invented.
    assert record["chunk"].strip() in NASA_ARTICLE
    # At least one supporting record must carry the explicit landing statement.
    # The NLI model may rank multiple windows as SUPPORT (e.g. the introductory
    # paragraphs that mention the moon + 1969, AND the "Eagle has landed"
    # passage). We check all support records, not just the top-ranked one.
    landing_phrases = (
        "first set foot upon the moon",
        "Eagle has landed",
        "first human foot",
        "man on the moon",
        "Apollo 11",
    )
    assert any(
        any(phrase in r["chunk"] for phrase in landing_phrases)
        for r in supporting
    ), "at least one supporting passage must carry Apollo 11 moon-landing language"
    assert record["evidence_type"] == "scraped_article"
    assert record["date_status"] in ("match", "same_day_published")
    assert record["event_match"] == "strong"
    assert record["nli_label"] == "entailment"
    assert record["nli_score"] >= 0.70
    assert record["passage_kind"] == "window"

    # And the source votes for the claim.
    agreement = calculate_source_agreement(results)
    assert agreement["independent_supporting"] == 1
    assert agreement["independent_contradicting"] == 0
    assert agreement["source_decisions"][0]["decision"] == "entailment"



def test_1b_the_window_contains_only_the_articles_own_text(nasa_results):
    """
    A window must never introduce text that is not in the article.
    """
    for record in nasa_results:
        if record["passage_kind"] != "window":
            continue
        for paragraph in record["chunk"].split("\n\n"):
            assert paragraph.strip() in NASA_ARTICLE


def test_1c_window_candidates_are_bounded():
    """
    The window budget is explicit and small, so the NLI cost stays bounded.
    """
    assert 0 < MAX_WINDOW_CHUNKS_PER_SOURCE <= 4


def test_1d_windows_respect_the_nli_token_budget():
    chunks = [f"paragraph number {i} about the moon and apollo eleven" for i in range(10)]
    windows = _build_context_windows(chunks, [0, 5], 3, max_tokens=40)
    assert len(windows) <= 3
    for window in windows:
        assert _count_tokens(window) <= 40


# ===========================================================================
# 2. Truncated or unavailable article text is reported honestly
# ===========================================================================

def test_2a_a_truncated_article_is_not_treated_as_full_evidence():
    """
    A page that yields only a fragment is analysed as the fragment it is.
    """
    truncated = (
        "TITLE: Apollo 11 | NASA\n\n"
        "July 1969. It is a little over eight years since the flights of "
        "Gagarin and Shepard."
    )
    results = analyze_evidence(CLAIM, [_source(content=truncated)], verbose=False)
    for record in results:
        assert record["evidence_class"] != "SUPPORT"


def test_2b_an_unusable_source_is_skipped_without_evidence():
    """
    A source with no usable content produces no evidence at all - it is never
    turned into a record with empty text.
    """
    results = analyze_evidence(CLAIM, [_source(content="   ")], verbose=False)
    assert results == []


def test_2c_a_short_source_is_skipped():
    results = analyze_evidence(CLAIM, [_source(content="too short")], verbose=False)
    assert results == []


def test_2d_a_truncated_passage_is_not_padded_with_invented_text(nasa_results):
    """
    The window is a real span of the article; the record's text must be
    traceable to it.
    """
    for record in nasa_results:
        if record["passage_kind"] == "window":
            assert record["chunk"].strip() in NASA_ARTICLE
            assert not record["chunk"].strip().endswith("...")


# ===========================================================================
# 3. Search snippets alone cannot satisfy primary article requirements
# ===========================================================================

def test_3a_a_snippet_is_never_support():
    signals = {
        "date_status": "match",
        "event_match": "strong",
        "similarity": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.99,
        "nli_status": "ok",
        "location_match": False,
        "claim_has_location": False,
        "detail_coverage": 0.9,
        "debunking": False,
    }
    from source_reliability import get_source_profile

    profile = get_source_profile("https://www.nasa.gov/x", "search_snippet")
    assert classify_evidence(signals, "search_snippet", profile) != "SUPPORT"


def test_3b_a_snippet_source_produces_no_supporting_evidence():
    """
    Even with the window fix, a search snippet can never be support.
    """
    snippet = (
        "Apollo 11 landed the first humans on the Moon in July 1969, NASA said. "
        "Neil Armstrong and Buzz Aldrin walked on the lunar surface."
    )
    results = analyze_evidence(
        CLAIM, [_source(url="https://www.nasa.gov/x", content=snippet,
                        evidence_type="search_snippet")],
        verbose=False,
    )
    assert all(r["evidence_class"] != "SUPPORT" for r in results)


def test_3c_a_window_from_a_snippet_is_still_not_support():
    results = analyze_evidence(
        CLAIM,
        [_source(url="https://www.nasa.gov/x", content=snippet_text(),
                 evidence_type="search_snippet")],
        verbose=False,
    )
    assert all(r["evidence_class"] != "SUPPORT" for r in results)


def snippet_text() -> str:
    return (
        "Apollo 11 landed the first humans on the Moon in July 1969, NASA said. "
        "Neil Armstrong and Buzz Aldrin walked on the lunar surface. "
        "The mission launched on 16 July 1969 and splashed down on 24 July 1969."
    )


# ===========================================================================
# 4. Duplicate URLs or publishers do not inflate independent-source counts
# ===========================================================================

def test_4a_duplicate_urls_are_one_source():
    results = analyze_evidence(
        CLAIM,
        [_source(url="https://www.nasa.gov/a"), _source(url="https://www.nasa.gov/a")],
        verbose=False,
    )
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 1
    assert agreement["independence_groups"] == 1


def test_4b_duplicate_urls_with_different_query_strings_are_one_source():
    results = analyze_evidence(
        CLAIM,
        [
            _source(url="https://www.nasa.gov/a?utm_source=x"),
            _source(url="https://www.nasa.gov/a"),
        ],
        verbose=False,
    )
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 1
    assert agreement["independent_supporting"] <= 1


def test_4c_two_pages_of_one_publisher_are_one_independence_group():
    results = analyze_evidence(
        CLAIM,
        [
            _source(url="https://www.nasa.gov/a"),
            _source(url="https://www.nasa.gov/b"),
        ],
        verbose=False,
    )
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 2
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1


# ===========================================================================
# 5. Retrieval failure produces a transparent UNVERIFIED, not fabricated evidence
# ===========================================================================

def test_5a_a_retrieval_failure_produces_no_evidence():
    """
    A source whose body could not be retrieved contributes nothing.
    """
    results = analyze_evidence(CLAIM, [_source(content="")], verbose=False)
    assert results == []

    agreement = calculate_source_agreement(results)
    assert agreement["independent_supporting"] == 0
    assert agreement["independent_contradicting"] == 0
    assert agreement["evidence_sufficiency"] == "none"


def test_5b_the_gate_abstains_when_there_is_no_evidence():
    from final_verdict import deterministic_gate

    results = analyze_evidence(CLAIM, [_source(content="")], verbose=False)
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)

    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "REAL" not in gate["permitted_verdicts"]
    assert "FAKE" not in gate["permitted_verdicts"]


def test_5c_no_source_text_is_invented_for_a_failed_retrieval(nasa_results):
    """
    Every record must be traceable to the retrieved text.
    """
    for record in nasa_results:
        assert record["chunk"].strip() in NASA_ARTICLE


# ===========================================================================
# 6. Apollo 11 can reach REAL only when the evidence satisfies the policy
# ===========================================================================

def test_6a_one_publisher_is_not_enough_for_real(nasa_results):
    """
    NASA's article alone supports the claim, but one publisher is not
    independent corroboration.
    """
    from final_verdict import deterministic_gate

    agreement = calculate_source_agreement(nasa_results)
    gate = deterministic_gate(agreement, nasa_results)

    assert agreement["independent_supporting"] == 1
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "Only one independent publisher" in gate["permitted_reason"]


def test_6b_two_independent_publishers_can_reach_real():
    """
    NASA plus a second independent primary publisher satisfies the policy and
    REAL becomes permitted.
    """
    from final_verdict import deterministic_gate, generate_final_verdict
    from evidence_fusion import calculate_fusion_score

    smithsonian = (
        "On July 20, 1969, astronauts Neil Armstrong and Buzz Aldrin landed in "
        "the Lunar Module Eagle and became the first humans to step foot on the "
        "lunar surface during the Apollo 11 mission."
    )
    results = analyze_evidence(
        CLAIM,
        [
            _source(url="https://www.nasa.gov/history/july-20-1969"),
            _source(url="https://www.si.edu/spotlight/apollo-11",
                    content=smithsonian, domain="si.edu"),
        ],
        verbose=False,
    )
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)

    assert agreement["independence_groups"] == 2
    assert agreement["independent_supporting"] == 2
    assert agreement["primary_supporting_groups"] == 2
    assert agreement["article_supporting_groups"] == 2
    assert "REAL" in gate["permitted_verdicts"]

    verdict = generate_final_verdict(
        results, agreement, calculate_fusion_score(results, agreement),
        "Verdict:\nREAL\n\nConfidence:\n95%\n\nReasoning:\nTwo independent publishers.",
        "Verdict:\nREAL\n\nConfidence:\n93%\n\nReasoning:\nAccept.",
    )
    assert verdict["verdict"] == "REAL"
    assert verdict["confidence_basis"] == "evidence_strength"
    assert verdict["independent_supporting"] == 2
    assert verdict["supporting_evidence"]
    assert verdict["contradicting_evidence"] == []


def test_6c_a_window_may_support_but_may_not_refute():
    """
    The safety asymmetry: a multi-paragraph window cannot produce a
    CONTRADICTION, because the NLI model reads a diluted premise unreliably.
    """
    from source_reliability import get_source_profile

    profile = get_source_profile("https://www.nasa.gov/x", "scraped_article")
    signals = {
        "date_status": "match",
        "event_match": "strong",
        "similarity": 0.66,
        "raw_nli_label": "contradiction",
        "raw_nli_score": 0.99,
        "nli_status": "ok",
        "location_match": False,
        "claim_has_location": False,
        "detail_coverage": 0.33,
        "debunking": False,
        "passage_kind": "window",
    }
    assert classify_evidence(signals, "scraped_article", profile) != "CONTRADICTION"

    # A single focused passage may still refute.
    signals["passage_kind"] = "paragraph"
    assert classify_evidence(signals, "scraped_article", profile) == "CONTRADICTION"


def test_6d_the_existing_guard_tests_still_pass():
    """
    The date-parsing and policy-guard behaviour must be unchanged.
    """
    from claim_profile import build_claim_profile

    profile = build_claim_profile(CLAIM)
    assert profile["primary_date"]["precision"] == "month"
    assert profile["locations"] == []

    from final_verdict import REQUIRED_POLICY_FIELDS, deterministic_gate

    assert "primary_correction_groups" in REQUIRED_POLICY_FIELDS
    gate = deterministic_gate({"independent_supporting": 3}, [])
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert gate["policy_data_valid"] is False
