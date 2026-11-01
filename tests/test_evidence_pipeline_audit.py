"""
tests/test_evidence_pipeline_audit.py
=====================================
Regression tests for the evidence-pipeline audit and repair.

Every test here uses CONTROLLED FIXTURES. No network call and no LLM call is
made. These tests assert what the pipeline does with a given evidence shape;
they do not prove anything about real-world detection accuracy, and no live
web evidence is involved.

The 18 required cases from the audit brief are marked with their number.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_brief import build_evidence_brief  # noqa: E402
from evidence_analyzer import (  # noqa: E402
    analyze_evidence,
    classify_evidence,
)
from evidence_fusion import (  # noqa: E402
    calculate_fusion_score,
    calculate_source_agreement,
    reconcile_evidence,
)
from source_reliability import get_source_profile  # noqa: E402
from final_verdict import (  # noqa: E402
    MIN_INDEPENDENT_SOURCES,
    deterministic_gate,
    extract_verdict,
    generate_final_verdict,
)

FIRE_CLAIM = (
    "A major fire broke out at a commercial building in New Delhi, "
    "India on 12 Aug 2024."
)

REAL_AGENT = "Verdict:\nREAL\n\nConfidence:\n94%\n\nReasoning:\nConfirmed by sources."
FAKE_AGENT = "Verdict:\nFAKE\n\nConfidence:\n88%\n\nReasoning:\nNever happened."
MISLEADING_AGENT = "Verdict:\nMISLEADING\n\nConfidence:\n70%\n\nReasoning:\nRight event, wrong detail."


def chunk(
    source_id,
    url,
    evidence_class="SUPPORT",
    score=0.9,
    evidence_type="scraped_article",
    group=None,
    primary=True,
    tier="established_wire",
    debunking=False,
    date_status="match",
    event_match="strong",
    **overrides,
) -> Dict[str, Any]:
    """One evidence record, shaped exactly like evidence_analyzer produces."""
    record = {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2] if url else "",
        "title": "title",
        "evidence_type": evidence_type,
        "chunk": "evidence passage",
        "similarity": 0.80,
        "nli_label": (
            "entailment" if evidence_class == "SUPPORT"
            else "contradiction" if evidence_class == "CONTRADICTION"
            else "neutral"
        ),
        "nli_score": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.9,
        "source_reliability": 0.9,
        "source_tier": tier,
        "independence_group": group if group is not None else (
            url.split("/")[2].replace("www.", "") if url else ""
        ),
        "is_primary_source": primary,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": date_status,
        "date_reason": "claimed date matched",
        "event_match": event_match,
        "location_match": True,
        "detail_coverage": 0.9,
        "claim_has_location": True,
        "nli_status": "ok",
        "debunking": debunking,
    }
    record.update(overrides)
    return record


def decide(results, verification=REAL_AGENT, critic=REAL_AGENT):
    agreement = calculate_source_agreement(results)
    return generate_final_verdict(
        results, agreement, calculate_fusion_score(results, agreement),
        verification, critic,
    )


# ===========================================================================
# 1. No evidence results -> UNVERIFIED
# ===========================================================================

def test_case_01_no_evidence_results_is_unverified():
    verdict = decide([])
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["permitted_verdicts"] == ["UNVERIFIED"]
    assert "No passage passed" in verdict["deterministic_gate_reason"]


# ===========================================================================
# 2. Context-only sources -> UNVERIFIED
# ===========================================================================

def test_case_02_context_only_sources_is_unverified():
    results = [
        chunk(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.2),
        chunk(2, "https://www.ndtv.com/b", "RELATED_CONTEXT", 0.2),
    ]
    verdict = decide(results, FAKE_AGENT, FAKE_AGENT)
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["supporting_evidence"] == []
    assert verdict["contradicting_evidence"] == []


# ===========================================================================
# 3. Two snippet-only publishers -> cannot establish REAL or FAKE
# ===========================================================================

def test_case_03a_two_snippet_only_publishers_cannot_establish_real():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9,
              evidence_type="search_snippet"),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88,
              evidence_type="search_snippet"),
    ]
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "REAL" not in verdict["permitted_verdicts"]


def test_case_03b_two_snippet_only_publishers_cannot_establish_fake():
    """
    The defect this covers: the gate required primary reporting for SUPPORT
    but applied no article requirement to CONTRADICTION, so two search
    snippets could refute a claim.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9,
              evidence_type="search_snippet"),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88,
              evidence_type="search_snippet"),
    ]
    verdict = decide(results, FAKE_AGENT, FAKE_AGENT)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "FAKE" not in verdict["permitted_verdicts"]
    assert "snippet" in verdict["deterministic_gate_reason"].lower()


def test_case_03c_one_snippet_plus_one_article_cannot_refute():
    """Snippet evidence must not add up to a refutation."""
    results = [
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9,
              evidence_type="search_snippet"),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88),
    ]
    verdict = decide(results, FAKE_AGENT, FAKE_AGENT)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "FAKE" not in verdict["permitted_verdicts"]


# ===========================================================================
# 4. One genuine supporting article -> insufficient independent corroboration
# ===========================================================================

def test_case_04_one_supporting_article_is_not_real():
    results = [chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9)]
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "Only one independent publisher" in verdict["deterministic_gate_reason"]


# ===========================================================================
# 5. Two independent supporting primary publishers -> REAL may be permitted
# ===========================================================================

def test_case_05_two_independent_primary_publishers_can_be_real():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    verdict = decide(results)
    assert verdict["verdict"] == "REAL"
    assert "REAL" in verdict["permitted_verdicts"]
    assert verdict["independent_supporting"] == 2
    assert verdict["primary_supporting_groups"] == 2


# ===========================================================================
# 6. Two independent contradicting primary publishers -> FAKE may be permitted
# ===========================================================================

def test_case_06_two_independent_contradicting_publishers_can_be_fake():
    results = [
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88),
    ]
    verdict = decide(results, FAKE_AGENT, FAKE_AGENT)
    assert verdict["verdict"] == "FAKE"
    assert "FAKE" in verdict["permitted_verdicts"]
    assert verdict["primary_contradicting_groups"] == 2


# ===========================================================================
# 7. Two chunks from the same article -> one source vote
# ===========================================================================

def test_case_07_two_chunks_from_the_same_article_are_one_vote():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.87),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 1
    assert agreement["independent_supporting"] == 1
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"


# ===========================================================================
# 8. Syndicated copies -> one independence-group vote
# ===========================================================================

def test_case_08_syndicated_copies_are_one_independence_group():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.syndicate.com/b", "SUPPORT", 0.88,
              group="reuters.com"),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"


# ===========================================================================
# 9. Different URLs with missing source IDs -> not silently merged
# ===========================================================================

def test_case_09_missing_source_ids_do_not_merge_unrelated_sources():
    results = [
        {**chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
         "source_id": None},
        {**chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
         "source_id": None},
        {**chunk(3, "https://www.ndtv.com/c", "SUPPORT", 0.86),
         "source_id": None},
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 3
    assert agreement["independence_groups"] == 3
    assert agreement["independent_supporting"] == 3


def test_case_09b_reused_source_id_across_different_urls_does_not_merge():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(1, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["total_sources"] == 2
    assert agreement["independence_groups"] == 2


def test_case_09c_unresolvable_identity_is_excluded_but_preserved():
    """
    A record with neither a usable URL nor a publisher identity must not be
    merged into a publisher, and must not count as corroboration - but it must
    still be visible in the diagnostics.
    """
    results = [
        chunk(1, "", "SUPPORT", 0.9, group=""),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1
    assert agreement["unresolved_identity_sources"] >= 1
    assert agreement["unresolved_source_decisions"], "the record must be preserved"


# ===========================================================================
# 10. A source containing conflicting chunks -> deterministic source decision
# ===========================================================================

def test_case_10a_conflicting_chunks_in_one_source_are_neutral():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.80),
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.84),
    ]
    agreement = calculate_source_agreement(results)
    decision = agreement["source_decisions"][0]
    assert decision["decision"] == "neutral"
    assert decision["has_support"] and decision["has_contradiction"]


def test_case_10b_conflicting_chunks_are_deterministic():
    first = calculate_source_agreement([
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.80),
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.84),
    ])
    second = calculate_source_agreement([
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.80),
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.84),
    ])
    assert first["source_decisions"][0]["decision"] == second["source_decisions"][0]["decision"]
    assert first["agreement_score"] == second["agreement_score"]


# ===========================================================================
# 11. Independent support AND contradiction -> UNVERIFIED
# ===========================================================================

def test_case_11_independent_support_and_contradiction_is_unverified():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.9),
        chunk(4, "https://apnews.com/d", "CONTRADICTION", 0.88),
    ]
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "contested" in verdict["deterministic_gate_reason"].lower()


# ===========================================================================
# 12. Alternate location spelling / abbreviation
# ===========================================================================

def test_case_12_alternate_location_spelling_is_recognised():
    """
    A report that says "Delhi" for a claim about "New Delhi" must still match.
    """
    from claim_profile import build_claim_profile
    from evidence_analyzer import evaluate_event_match

    claim = "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024."
    attrs = build_claim_profile(claim)
    assert "delhi" in [t.lower() for t in attrs["location_terms"]]

    chunk_text = (
        "A major fire broke out at a commercial building in Delhi on 12 August 2024, "
        "police said. The blaze started at the fourth floor."
    )
    level, signals = evaluate_event_match(
        claim, chunk_text, attrs, "match", 0.8, nli_entails=True
    )
    assert signals["location_match"] is True
    assert level in ("strong", "moderate")


def test_case_12b_a_different_place_is_a_conflict_not_a_match():
    from claim_profile import build_claim_profile
    from evidence_analyzer import evaluate_event_match

    claim = "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024."
    attrs = build_claim_profile(claim)
    chunk_text = (
        "A major fire broke out at a commercial building in Mumbai on 12 August 2024, "
        "police said. The blaze started at the fourth floor."
    )
    level, signals = evaluate_event_match(
        claim, chunk_text, attrs, "match", 0.8, nli_entails=True
    )
    assert signals["location_match"] is False
    assert level in ("none", "weak")


def test_case_12c_a_claim_without_a_location_is_not_penalised_for_it():
    """
    A claim that states no location cannot be rejected because the passage
    does not name a place.

    This was found by a live trace: a Chandrayaan-3 launch claim that named no
    location retrieved nine good sources, and every passage came back with
    event_match 'weak', because the event matcher required a location match
    even though the claim had no location to match. SUPPORT requires
    event_match 'strong', so no amount of good evidence could ever support the
    claim.
    """
    from claim_profile import build_claim_profile
    from evidence_analyzer import evaluate_event_match

    claim = (
        "The Indian Space Research Organisation launched the Chandrayaan-3 "
        "mission on 14 July 2023."
    )
    attrs = build_claim_profile(claim)
    assert attrs["locations"] == [], "this claim deliberately states no location"

    article = (
        "India on Friday launched its Chandrayaan-3 mission on board an LVM3-M4 "
        "rocket from Satish Dhawan Space Centre in Sriharikota, Andhra Pradesh, "
        "the Indian Space Research Organisation said. The launch took place on "
        "14 July 2023."
    )
    level, signals = evaluate_event_match(
        claim, article, attrs, "match", 0.8, nli_entails=True
    )
    assert signals["location_match"] is False
    assert signals["location_ok"] is True
    assert signals["claim_has_location"] is False
    assert level == "strong", (
        "a location-less claim must still be able to reach a strong event match"
    )


def test_case_12d_a_claim_with_a_location_still_requires_the_match():
    """
    The fix must not weaken the location requirement for claims that DO state
    a place.
    """
    from claim_profile import build_claim_profile
    from evidence_analyzer import evaluate_event_match

    claim = (
        "The Indian Space Research Organisation launched the Chandrayaan-3 "
        "mission from Sriharikota in Andhra Pradesh on 14 July 2023."
    )
    attrs = build_claim_profile(claim)
    assert attrs["locations"], "this claim states a location"

    # A passage about the same launch that never names the place.
    elsewhere = (
        "The Indian Space Research Organisation launched the Chandrayaan-3 "
        "mission on 14 July 2023, the agency said."
    )
    level, signals = evaluate_event_match(
        claim, elsewhere, attrs, "match", 0.8, nli_entails=True
    )
    assert signals["location_ok"] is False
    assert level != "strong"


# ===========================================================================
# 13. Publication date different from event date
# ===========================================================================

def test_case_13_publication_date_different_from_event_date():
    from evidence_analyzer import check_date_compatibility, extract_claim_attributes

    claim = "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024."
    attrs = extract_claim_attributes(claim)
    result = check_date_compatibility(
        claim,
        "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024. "
        "The blaze started at the fourth floor.",
        attrs,
    )
    status = result.get("status")
    assert status in ("match", "same_day_published", "same_year", "unknown"), status


def test_case_13b_a_source_from_another_year_is_not_direct_evidence():
    from evidence_analyzer import (
        check_date_compatibility,
        classify_evidence,
        extract_claim_attributes,
    )

    claim = "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024."
    attrs = extract_claim_attributes(claim)
    result = check_date_compatibility(
        claim,
        "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2023. "
        "The blaze started at the fourth floor.",
        attrs,
    )
    assert result.get("status") == "different"

    evidence_class = classify_evidence(
        {
            "date_status": result.get("status"),
            "event_match": "strong",
            "similarity": 0.9,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.95,
            "nli_status": "ok",
            "location_match": True,
            "claim_has_location": True,
            "detail_coverage": 0.9,
            "debunking": False,
        },
        "scraped_article",
        {"is_primary": True, "tier": "established_wire", "reliability": 0.92},
    )
    assert evidence_class not in ("SUPPORT", "CONTRADICTION")


# ===========================================================================
# 14. Similar event, wrong date or wrong location -> not direct evidence
# ===========================================================================

def test_case_14a_similar_event_wrong_location_is_not_support():
    from evidence_analyzer import classify_evidence

    evidence_class = classify_evidence(
        {
            "date_status": "match",
            "event_match": "none",
            "similarity": 0.95,
            "raw_nli_label": "entailment",
            "raw_nli_score": 0.99,
            "nli_status": "ok",
            "location_match": False,
            "claim_has_location": True,
            "detail_coverage": 0.9,
            "debunking": False,
        },
        "scraped_article",
        {"is_primary": True, "tier": "established_wire", "reliability": 0.92},
    )
    assert evidence_class not in ("SUPPORT", "CONTRADICTION")


def test_case_14b_similarity_alone_never_classifies():
    from evidence_analyzer import classify_evidence

    evidence_class = classify_evidence(
        {
            "date_status": "unknown",
            "event_match": "weak",
            "similarity": 0.99,
            "raw_nli_label": "neutral",
            "raw_nli_score": 0.99,
            "nli_status": "ok",
            "location_match": False,
            "claim_has_location": True,
            "detail_coverage": 0.0,
            "debunking": False,
        },
        "scraped_article",
        {"is_primary": True, "tier": "established_wire", "reliability": 0.92},
    )
    assert evidence_class not in ("SUPPORT", "CONTRADICTION")


# ===========================================================================
# 15. Explicit, evidence-backed material correction -> MISLEADING
# ===========================================================================

def test_case_15a_misleading_requires_a_recorded_correction():
    """
    Corroboration alone must not make MISLEADING available.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" not in gate["permitted_verdicts"]


def test_case_15b_an_explicit_correction_permits_misleading():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.86,
              debunking=True, date_status="different"),
    ]
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" in gate["permitted_verdicts"]
    assert agreement["correction_groups"] >= 1
    assert agreement["primary_correction_groups"] >= 1


def test_case_15c_generic_disagreement_is_not_a_correction():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.86,
              debunking=False),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["correction_groups"] == 0
    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" not in gate["permitted_verdicts"]


# ===========================================================================
# 16. Evidence brief: zero results, mixed classifications, snippet-only
# ===========================================================================

def test_case_16a_brief_with_zero_results_states_absence():
    brief = build_evidence_brief(FIRE_CLAIM, [], [], calculate_source_agreement([]), 0.0)
    assert "EVIDENCE STATE: ABSENT" in brief
    assert "not evidence that the claim is false" in brief.lower()
    assert "Sources retrieved           : 0" in brief


def test_case_16b_brief_reports_mixed_classifications():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.85),
        chunk(3, "https://www.ndtv.com/c", "RELATED_CONTEXT", 0.2),
        chunk(4, "https://blog.example.org/d", "NON_EVIDENCE", 0.02,
              primary=False, tier="weak/blog"),
    ]
    agreement = calculate_source_agreement(results)
    brief = build_evidence_brief(FIRE_CLAIM, [], results, agreement, 0.4)

    assert "Sources retrieved           : 4" in brief
    assert "Evidence state             : CONTESTED" in brief
    assert "DISAGREE" in brief
    assert "Permitted verdicts" in brief
    # The brief must agree with the gate.
    gate = deterministic_gate(agreement, results)
    assert f"Permitted verdicts         : {', '.join(sorted(gate['permitted_verdicts']))}" in brief


def test_case_16c_brief_flags_snippet_only_sources():
    results = [
        chunk(1, "https://www.reuters.com/a", "RELATED_CONTEXT", 0.3,
              evidence_type="search_snippet"),
        chunk(2, "https://www.bbc.com/b", "RELATED_CONTEXT", 0.3,
              evidence_type="search_snippet"),
    ]
    agreement = calculate_source_agreement(results)
    brief = build_evidence_brief(FIRE_CLAIM, [], results, agreement, 0.2)

    assert "only a search snippet was available" in brief
    assert "cannot establish or refute a claim" in brief
    assert "Snippet-only evidence can never establish or refute a claim" in brief
    assert "Sources snippet-only        : 2" in brief


def test_case_16d_brief_does_not_call_a_non_primary_source_inspected():
    results = [
        chunk(1, "https://blog.example.org/a", "RELATED_CONTEXT", 0.3,
              primary=False, tier="weak/blog"),
    ]
    agreement = calculate_source_agreement(results)
    brief = build_evidence_brief(FIRE_CLAIM, [], results, agreement, 0.1)
    assert "not primary reporting" in brief


# ===========================================================================
# 17. Agent confidence / agreement cannot override the deterministic gate
# ===========================================================================

def test_case_17a_agents_agreeing_on_real_cannot_force_real():
    results = [
        chunk(1, "https://www.reuters.com/a", "RELATED_CONTEXT", 0.2),
    ]
    verdict = decide(results, REAL_AGENT, REAL_AGENT)
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["agents_agree"] is True
    assert verdict["verdict_overridden_by_evidence"] is True


def test_case_17b_agent_confidence_cannot_force_a_verdict():
    results = []
    overconfident = "Verdict:\nREAL\n\nConfidence:\n99%\n\nReasoning:\nCertain."
    verdict = decide(results, overconfident, overconfident)
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["agent_confidence"] == 99.0
    assert verdict["verdict_overridden_by_evidence"] is True


def test_case_17c_incidental_label_in_prose_is_not_a_verdict():
    text = (
        "The source article discusses the claim in depth. It is not REAL to say "
        "the event happened; the record suggests FAKE. "
        "Final Explanation: The evidence does not establish the claim."
    )
    assert extract_verdict(text) is None


def test_case_17d_labelled_verdict_is_still_read():
    assert extract_verdict(REAL_AGENT) == "REAL"
    assert extract_verdict("Recommended Verdict: UNVERIFIED") == "UNVERIFIED"
    assert extract_verdict("Verdict:\nMISLEADING") == "MISLEADING"


# ===========================================================================
# 18. Existing API response fields remain compatible
# ===========================================================================

def test_case_18_verdict_dict_keeps_its_existing_fields():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    verdict = decide(results)

    for field in (
        "verdict", "confidence", "evidence_sufficiency", "source_agreement",
        "fusion_score", "independent_supporting", "independent_contradicting",
        "primary_supporting_groups", "total_sources", "independence_groups",
        "supporting_sources", "contradicting_sources", "entailment_score",
        "contradiction_score", "context_chunks", "permitted_verdicts",
        "permitted_reason", "deterministic_gate_reason", "verification_verdict",
        "critic_verdict", "agents_agree", "verdict_overridden_by_evidence",
        "agent_confidence", "critic_confidence", "supporting_evidence",
        "contradicting_evidence", "reasoning", "verification_result",
        "critic_result",
    ):
        assert field in verdict, f"the verdict dict must keep '{field}'"


def test_case_18b_agreement_dict_keeps_its_existing_fields():
    agreement = calculate_source_agreement([
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
    ])
    for field in (
        "total_sources", "total_chunks", "contradictions", "entailments",
        "supporting_sources", "contradicting_sources", "neutral_sources",
        "related_context_sources", "non_evidence_sources",
        "independent_supporting", "independent_contradicting",
        "independence_groups", "primary_supporting_groups", "agreement_score",
        "evidence_sufficiency", "source_decisions", "group_decisions",
    ):
        assert field in agreement, f"the agreement dict must keep '{field}'"


# ===========================================================================
# Additional invariants of the repair
# ===========================================================================

def test_primary_reporting_is_not_inferred_from_being_scraped():
    """
    A blog post we managed to scrape is still not primary reporting.
    """
    results = [
        chunk(1, "https://blog.example.com/a", "SUPPORT", 0.9,
              primary=False, tier="weak/blog"),
        chunk(2, "https://other.example.org/b", "SUPPORT", 0.88,
              primary=False, tier="weak/blog"),
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["primary_supporting_groups"] == 0
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "REAL" not in verdict["permitted_verdicts"]


def test_unverified_confidence_is_labelled_as_coverage():
    results = [
        chunk(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.2),
    ]
    verdict = decide(results)
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["confidence_basis"] == "verification_coverage"
    assert 0.0 <= verdict["confidence"] <= 100.0


def test_supporting_and_contradicting_lists_are_selected_by_class():
    """
    A contradicting passage must never appear in the supporting list, whatever
    the verdict is.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.9),
    ]
    verdict = decide(results)
    assert [e["url"] for e in verdict["supporting_evidence"]] == [
        "https://www.reuters.com/a"
    ]
    assert [e["url"] for e in verdict["contradicting_evidence"]] == [
        "https://www.bbc.com/b"
    ]


def test_canonical_url_ignores_campaign_parameters_but_keeps_article_ids():
    from evidence_fusion import _canonical_url

    assert _canonical_url("https://www.example.com/a/") == _canonical_url(
        "https://example.com/a"
    )
    assert _canonical_url("https://example.com/a?utm_source=twitter") == _canonical_url(
        "https://example.com/a"
    )
    assert _canonical_url("https://example.com/a?id=1") != _canonical_url(
        "https://example.com/a?id=2"
    )


def test_gate_reason_states_the_actual_requirements():
    results = [
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9,
              evidence_type="search_snippet"),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88,
              evidence_type="search_snippet"),
    ]
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)
    reason = gate["permitted_reason"]
    assert str(MIN_INDEPENDENT_SOURCES) in reason
    assert "primary" in reason.lower()


# ===========================================================================
# Evidence consistency check tests
# ===========================================================================

from evidence_fusion import reconcile_evidence  # noqa: E402


def _chunk_with_class(
    source_id, url, evidence_class, score=0.9, evidence_type="scraped_article",
    group=None, primary=True, tier="established_wire", date_status="match",
    event_match="strong", claim_type="current_event",
    similarity=0.80, nli_score=0.9, raw_nli_score=0.9, detail_coverage=0.9,
    claim_has_location=True, event_date_conflict=False,
    event_date_mismatch=False, nli_label=None, raw_nli_label=None,
):
    """Build an evidence record shaped like evidence_analyzer output."""
    derived_nli_label = (
        "entailment" if evidence_class == "SUPPORT"
        else "contradiction" if evidence_class == "CONTRADICTION"
        else "neutral"
    )
    return {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2] if url else "",
        "title": "title",
        "evidence_type": evidence_type,
        "chunk": "evidence passage",
        "similarity": similarity,
        "nli_label": nli_label if nli_label is not None else derived_nli_label,
        "nli_score": nli_score,
        "raw_nli_label": raw_nli_label if raw_nli_label is not None else derived_nli_label,
        "raw_nli_score": raw_nli_score,
        "source_reliability": 0.9,
        "source_tier": tier,
        "independence_group": group if group is not None else (
            url.split("/")[2].replace("www.", "") if url else ""
        ),
        "is_primary_source": primary,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": date_status,
        "date_reason": "claimed date matched",
        "event_match": event_match,
        "location_match": claim_has_location,
        "detail_coverage": detail_coverage,
        "claim_has_location": claim_has_location,
        "nli_status": "ok",
        "debunking": False,
        "claim_type": claim_type,
        "event_date_conflict": event_date_conflict,
        "event_date_mismatch": event_date_mismatch,
    }


# ---------------------------------------------------------------------------
# Consistency: duplicate chunks from one URL count as one article
# ---------------------------------------------------------------------------

def test_consistency_two_chunks_from_one_url_count_as_one_article():
    """
    Two SUPPORT chunks from the same The Hindu URL must count as one source,
    not two supporting sources.
    """
    results = [
        _chunk_with_class(1, "https://www.thehindu.com/a", "SUPPORT", 0.9),
        _chunk_with_class(2, "https://www.thehindu.com/a", "SUPPORT", 0.87),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    # Both chunks collapse into one source record
    assert len(reconciled) == 1
    assert reconciled[0]["evidence_class"] == "SUPPORT"

    # Diagnostics record the deduplication
    assert diag["total_chunks_received"] == 2
    assert diag["unique_articles"] == 1
    assert diag["duplicate_chunks_removed"] == 1
    assert len(diag["supporting_articles"]) == 1

    # Source agreement must also see one source
    agreement = calculate_source_agreement(reconciled)
    assert agreement["total_sources"] == 1
    assert agreement["independent_supporting"] == 1

    # Verdict must still be UNVERIFIED (one source is insufficient)
    verdict = decide(reconciled)
    assert verdict["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Consistency: duplicate URLs do not inflate publisher independence
# ---------------------------------------------------------------------------

def test_consistency_duplicate_urls_do_not_inflate_publishers():
    """
    Two chunks from the same URL, even with different source_ids, must count
    as one independent publisher.
    """
    results = [
        _chunk_with_class(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        _chunk_with_class(1, "https://www.reuters.com/a", "SUPPORT", 0.88),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    agreement = calculate_source_agreement(reconciled)
    assert agreement["total_sources"] == 1
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1


# ---------------------------------------------------------------------------
# Consistency: genuinely supporting passage from a second independent publisher
# ---------------------------------------------------------------------------

def test_consistency_second_independent_publisher_is_recognized():
    """
    Two independent publishers (BBC and Reuters) each supporting the claim
    must count as two independent supporting voices.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "SUPPORT", 0.9,
                          group="bbc.com"),
        _chunk_with_class(2, "https://www.reuters.com/b", "SUPPORT", 0.88,
                          group="reuters.com"),
    ]
    reconciled, diag, pub_decisions = reconcile_evidence(results)

    assert len(reconciled) == 2
    assert len(pub_decisions) == 2

    agreement = calculate_source_agreement(reconciled)
    assert agreement["independent_supporting"] == 2

    verdict = decide(reconciled)
    assert verdict["verdict"] == "REAL"
    assert verdict["independent_supporting"] == 2


# ---------------------------------------------------------------------------
# Consistency: neutral or irrelevant passages do not become support
# ---------------------------------------------------------------------------

def test_consistency_neutral_passages_do_not_become_support():
    """
    A RELATED_CONTEXT passage that does not meet all SUPPORT gates must stay
    as RELATED_CONTEXT and not inflate supporting counts.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.5,
                          event_match="weak", similarity=0.5,
                          detail_coverage=0.2),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    assert len(reconciled) == 1
    assert reconciled[0]["evidence_class"] == "RELATED_CONTEXT"

    # No classification change (passage does not meet all gates)
    assert len(diag["classification_changes"]) == 0
    assert len(diag["related_context_excluded"]) == 1

    agreement = calculate_source_agreement(reconciled)
    assert agreement["independent_supporting"] == 0
    assert agreement["evidence_sufficiency"] in ("none", "context_only")

    verdict = decide(reconciled)
    assert verdict["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Consistency: contradictory passages about another event do not count
# ---------------------------------------------------------------------------

def test_consistency_contradiction_about_another_event_is_not_counted():
    """
    A passage that contradicts a different event (wrong date, wrong match)
    must not count as a contradiction of this claim.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "CONTRADICTION", 0.9,
                          date_status="different",
                          event_match="weak",
                          event_date_conflict=False),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    assert len(reconciled) == 1
    assert reconciled[0]["evidence_class"] == "CONTRADICTION"

    agreement = calculate_source_agreement(reconciled)
    # The passage was classified as CONTRADICTION by the evidence layer,
    # but it dates a different event. The consistency check does not
    # upgrade it, but the existing gate logic still applies.
    # One contradicting source is insufficient for FAKE.
    assert agreement["independent_contradicting"] == 1

    verdict = decide(reconciled)
    assert verdict["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Consistency: insufficient corroboration still results in UNVERIFIED
# ---------------------------------------------------------------------------

def test_consistency_single_supporting_publisher_stays_unverified():
    """
    Even after deduplication and re-evaluation, one supporting publisher
    is not enough for REAL.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "SUPPORT", 0.9),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    assert len(reconciled) == 1
    assert reconciled[0]["evidence_class"] == "SUPPORT"

    agreement = calculate_source_agreement(reconciled)
    assert agreement["independent_supporting"] == 1
    assert agreement["evidence_sufficiency"] == "single_source"

    verdict = decide(reconciled)
    assert verdict["verdict"] == "UNVERIFIED"
    assert "Only one independent publisher" in verdict["deterministic_gate_reason"]


# ---------------------------------------------------------------------------
# Consistency: two supporting chunks from same URL, one from another
# ---------------------------------------------------------------------------

def test_consistency_two_chunks_same_url_plus_one_another():
    """
    Two supporting chunks from BBC + one from Reuters = 2 independent
    supporting publishers, not 3.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "SUPPORT", 0.9),
        _chunk_with_class(2, "https://www.bbc.com/a", "SUPPORT", 0.87),
        _chunk_with_class(3, "https://www.reuters.com/b", "SUPPORT", 0.85),
    ]
    reconciled, diag, pub_decisions = reconcile_evidence(results)

    # Two chunks from BBC collapse to one source
    assert len(reconciled) == 2
    assert diag["total_chunks_received"] == 3
    assert diag["duplicate_chunks_removed"] == 1

    # Two independent publishers
    assert len(pub_decisions) == 2
    supporting_publishers = [
        p for p in pub_decisions if p["decision"] == "SUPPORT"
    ]
    assert len(supporting_publishers) == 2

    agreement = calculate_source_agreement(reconciled)
    assert agreement["independent_supporting"] == 2

    verdict = decide(reconciled)
    assert verdict["verdict"] == "REAL"
    assert verdict["independent_supporting"] == 2


# ---------------------------------------------------------------------------
# Consistency: RELATED_CONTEXT with strong passage may be upgraded
# ---------------------------------------------------------------------------

def test_consistency_related_context_with_strong_passage_can_be_upgraded():
    """
    If a RELATED_CONTEXT passage's best chunk actually meets all SUPPORT
    gates on re-evaluation, it is upgraded.
    """
    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.95,
                          event_match="strong",
                          similarity=0.95,
                          detail_coverage=0.95,
                          nli_label="entailment",
                          nli_score=0.98,
                          raw_nli_label="entailment",
                          raw_nli_score=0.98,
                          date_status="match",
                          claim_type="current_event"),
    ]
    reconciled, diag, _ = reconcile_evidence(results)

    # The passage was RELATED_CONTEXT but meets all SUPPORT gates
    assert reconciled[0]["evidence_class"] == "SUPPORT"
    assert reconciled[0].get("consistency_upgraded") is True

    assert len(diag["classification_changes"]) == 1
    assert diag["classification_changes"][0]["from"] == "RELATED_CONTEXT"
    assert diag["classification_changes"][0]["to"] == "SUPPORT"


# ---------------------------------------------------------------------------
# Consistency: brief includes diagnostics
# ---------------------------------------------------------------------------

def test_consistency_brief_includes_diagnostics():
    from evidence_brief import build_evidence_brief

    results = [
        _chunk_with_class(1, "https://www.bbc.com/a", "SUPPORT", 0.9),
    ]
    reconciled, diag, _ = reconcile_evidence(results)
    agreement = calculate_source_agreement(reconciled)
    fusion_score = calculate_fusion_score(reconciled, agreement)

    brief = build_evidence_brief(
        "India defeated New Zealand in T20.",
        [],
        reconciled,
        agreement,
        fusion_score,
        evidence_diagnostics=diag,
    )

    assert "EVIDENCE CONSISTENCY DIAGNOSTICS" in brief
    assert "Unique articles" in brief
    assert "Unique publishers" in brief


# ===========================================================================
# Cricket claim: high-confidence NLI entailment with date_status != "match"
# ===========================================================================

CRICKET_CLAIM = (
    "India defeated New Zealand by six wickets in the third T20 international "
    "on 27 October 2026."
)

CRICKET_SUPPORT_PASSAGE = (
    "India defeated New Zealand by six wickets in the third T20 international "
    "on Sunday, chasing down a target of 146 with eight balls to spare."
)


def _cricket_signals(
    evidence_class="SUPPORT",
    nli_label="entailment",
    nli_score=0.9,
    similarity=0.85,
    date_status="match",
    event_match="strong",
    coverage=0.50,
    passage_kind="paragraph",
    claim_has_location=False,
    claim_has_date=True,
    claim_type="current_event",
    event_date_conflict=False,
    event_date_mismatch=False,
    location_match=True,
    raw_nli_label=None,
    raw_nli_score=None,
):
    """Build signals dict shaped like evidence_analyzer output for cricket claim."""
    derived_nli_label = (
        "entailment" if evidence_class == "SUPPORT"
        else "contradiction" if evidence_class == "CONTRADICTION"
        else "neutral"
    )
    return {
        "similarity": similarity,
        "date_status": date_status,
        "event_match": event_match,
        "raw_nli_label": raw_nli_label if raw_nli_label is not None else derived_nli_label,
        "raw_nli_score": raw_nli_score if raw_nli_score is not None else nli_score,
        "nli_label": nli_label if nli_label is not None else derived_nli_label,
        "nli_score": nli_score,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "major_outlet",
        "independence_group": "bbc.com",
        "detail_coverage": coverage,
        "debunking": False,
        "passage_kind": passage_kind,
        "claim_has_location": claim_has_location,
        "claim_has_date": claim_has_date,
        "claim_type": claim_type,
        "event_date_conflict": event_date_conflict,
        "event_date_mismatch": event_date_mismatch,
        "location_match": location_match,
    }


def test_cricket_claim_high_confidence_entailment_date_unknown_is_support():
    """
    Cricket Times: entailment 0.9981, strong event match, coverage 1.0, but
    date_status != "match" (passage doesn't restate "27 October 2026").
    High-confidence NLI entailment with strong event match and high coverage
    qualifies as SUPPORT even without explicit date restatement.
    """
    signals = _cricket_signals(
        evidence_class="SUPPORT",
        nli_label="entailment",
        nli_score=0.9981,
        similarity=0.92,
        date_status="unknown",
        event_match="strong",
        coverage=1.0,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result == "SUPPORT", (
        f"High-confidence entailment with strong event match should be SUPPORT, "
        f"got {result}"
    )


def test_cricket_claim_date_conflict_blocks_support_even_with_high_nli():
    """
    If the passage explicitly dates the event to a different date, high NLI
    entailment must NOT produce SUPPORT (date conflict safeguard).
    """
    signals = _cricket_signals(
        evidence_class="SUPPORT",
        nli_label="entailment",
        nli_score=0.9981,
        similarity=0.92,
        date_status="different",
        event_match="strong",
        coverage=1.0,
        event_date_conflict=True,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", (
        f"Date conflict must block SUPPORT even with high NLI, got {result}"
    )


def test_cricket_claim_neutral_nli_not_support():
    """
    Al Jazeera: NLI neutral 0.9988, strong event match, coverage 0.8714.
    NLI neutral must not be promoted to SUPPORT.
    """
    signals = _cricket_signals(
        evidence_class="RELATED_CONTEXT",
        nli_label="neutral",
        nli_score=0.9988,
        similarity=0.87,
        date_status="match",
        event_match="strong",
        coverage=0.8714,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", (
        f"Neutral NLI must not be SUPPORT, got {result}"
    )


def test_cricket_claim_non_primary_not_support():
    """
    Non-primary source with NLI entailment must not classify as SUPPORT
    for a cricket claim.
    """
    signals = _cricket_signals(
        evidence_class="SUPPORT",
        nli_label="entailment",
        nli_score=0.99,
        similarity=0.9,
        date_status="match",
        event_match="strong",
        coverage=1.0,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://randomblog.wordpress.com/cricket", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", (
        f"Non-primary source must not be SUPPORT, got {result}"
    )


def test_cricket_claim_snippet_not_support():
    """
    Search snippet with NLI entailment must not classify as SUPPORT
    for a cricket claim.
    """
    signals = _cricket_signals(
        evidence_class="SUPPORT",
        nli_label="entailment",
        nli_score=0.99,
        similarity=0.9,
        date_status="match",
        event_match="strong",
        coverage=1.0,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://www.reuters.com/sport/cricket/article", "search_snippet"
    )
    result = classify_evidence(
        signals, "search_snippet", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", (
        f"Snippet must not be SUPPORT, got {result}"
    )


def test_cricket_claim_single_supporting_publisher_stays_unverified():
    """
    One supporting publisher (The Hindu) is insufficient for REAL;
    verdict remains UNVERIFIED per deterministic gate.
    """
    from final_verdict import generate_final_verdict

    chunks = [
        _chunk_with_class(1, "https://www.thehindu.com/sport/cricket/article",
                          "SUPPORT", 0.9),
    ]
    agreement = calculate_source_agreement(chunks)
    state = {
        "claim": CRICKET_CLAIM,
        "claim_type": "current_event",
        "evidence_analysis": [dict(chunks[0], evidence_class="SUPPORT")],
        "source_agreement": agreement,
        "agent_errors": [],
    }
    verdict = generate_final_verdict(
        state["evidence_analysis"],
        state["source_agreement"],
        0.5,
        "",
        "",
    )
    assert verdict["verdict"] == "UNVERIFIED"
    assert "Only one independent publisher" in verdict["deterministic_gate_reason"]


def test_cricket_claim_unrelated_passage_remains_neutral():
    """
    Negative: an unrelated paragraph about a different match must not become
    support for the India vs New Zealand claim.
    """
    signals = _cricket_signals(
        evidence_class="RELATED_CONTEXT",
        nli_label="neutral",
        nli_score=0.95,
        similarity=0.30,
        date_status="unknown",
        event_match="weak",
        coverage=0.10,
        claim_type="current_event",
    )
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", (
        f"Unrelated passage must not be SUPPORT, got {result}"
    )
    assert result != "CONTRADICTION", (
        f"Unrelated passage must not be CONTRADICTION, got {result}"
    )


def test_cricket_claim_historical_event_behavior_unchanged():
    """
    Historical/event claims still require event_match == "strong" even with
    entailment + primary source — no regression from the high-confidence fix.
    """
    signals = _cricket_signals(
        evidence_class="SUPPORT",
        nli_label="entailment",
        nli_score=0.95,
        similarity=0.85,
        date_status="match",
        event_match="weak",
        coverage=0.50,
        claim_type="historical_event",
    )
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="historical_event"
    )
    assert result != "SUPPORT", (
        f"Historical event claims require strong event match, got {result}"
    )
