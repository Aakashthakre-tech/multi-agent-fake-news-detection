"""
tests/test_policy_diagnostics_propagation.py
==============================================
Regression tests for the propagation of the deterministic gate's policy
diagnostics from ``final_verdict`` through the research report and the API
response.

What must hold:

* ``policy_data_valid`` / ``policy_data_errors`` reach the API response;
* the research report carries a policy-diagnostics block that states whether
  the policy data was valid, names the unusable fields, and explains that the
  restriction to UNVERIFIED is an internal evidence-accounting issue and not
  proof that the claim is false;
* a healthy run is unaffected - the diagnostics report a valid policy and the
  existing response fields keep their values.

Every test uses controlled fixtures. No network call and no LLM call is made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from final_verdict import generate_final_verdict  # noqa: E402
from research_report import build_research_report  # noqa: E402

REAL_AGENT = "Verdict:\nREAL\n\nConfidence:\n94%\n\nReasoning:\nConfirmed."


def chunk(source_id, url, evidence_class="SUPPORT", score=0.9, **overrides) -> Dict[str, Any]:
    record = {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2],
        "title": "title",
        "evidence_type": "scraped_article",
        "chunk": "evidence passage",
        "similarity": 0.8,
        "nli_label": "entailment",
        "nli_score": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.9,
        "source_reliability": 0.9,
        "source_tier": "established_wire",
        "independence_group": url.split("/")[2].replace("www.", ""),
        "is_primary_source": True,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": "match",
        "event_match": "strong",
        "location_match": True,
        "detail_coverage": 0.9,
        "nli_status": "ok",
        "debunking": False,
    }
    record.update(overrides)
    return record


def _research_state(final_verdict: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal text-pipeline state carrying the given final verdict."""
    return {
        "claim": "The claim under test.",
        "search_results": "SOURCE 1\nURL:\nhttps://www.reuters.com/a\n",
        "search_plans": [{"label": "exact", "query": "the claim", "intent": "exact"}],
        "search_errors": [],
        "scrape_failures": [],
        "source_data": [
            {
                "url": "https://www.reuters.com/a",
                "domain": "reuters.com",
                "title": "A",
                "scrape_status": "SUCCESS",
                "search_query": "the claim",
            }
        ],
        "evidence_analysis": [chunk(1, "https://www.reuters.com/a")],
        "source_agreement": {
            "total_sources": 1,
            "independence_groups": 1,
            "independent_supporting": 1,
            "independent_contradicting": 0,
            "primary_supporting_groups": 1,
            "primary_contradicting_groups": 0,
            "article_supporting_groups": 1,
            "article_contradicting_groups": 0,
            "correction_groups": 0,
            "primary_correction_groups": 0,
            "snippet_only_sources": 0,
            "article_sources": 1,
            "unresolved_identity_sources": 0,
            "verification_coverage": 1.0,
            "agreement_score": 0.4,
            "evidence_sufficiency": "single_source",
        },
        "verification": REAL_AGENT,
        "feedback": REAL_AGENT,
        "final_verdict": final_verdict,
    }


def _broken_policy_state() -> Dict[str, Any]:
    """A state whose final verdict was produced with unusable policy data."""
    results = [chunk(1, "https://www.reuters.com/a"), chunk(2, "https://www.bbc.com/b")]
    agreement = {
        "independent_supporting": 2,
        "independent_contradicting": 0,
        "primary_supporting_groups": 2,
        "primary_contradicting_groups": 0,
        "article_supporting_groups": 2,
        "article_contradicting_groups": 0,
        "correction_groups": 0,
        "primary_correction_groups": 0,
    }
    # Drop a required policy field so the gate must abstain.
    del agreement["primary_correction_groups"]

    verdict = generate_final_verdict(results, agreement, 0.8, REAL_AGENT, REAL_AGENT)

    state = _research_state(verdict)
    state["evidence_analysis"] = results
    state["source_agreement"] = agreement
    return state


# ===========================================================================
# 1. The diagnostics reach the report
# ===========================================================================

def test_policy_diagnostics_report_a_valid_policy():
    state = _research_state({
        "verdict": "REAL",
        "confidence": 70.0,
        "confidence_basis": "evidence_strength",
        "policy_data_valid": True,
        "policy_data_errors": [],
    })

    report = build_research_report(state, "text", state["claim"])
    diagnostics = report["executive_summary"]["policy_diagnostics"]

    assert diagnostics["gate_ran"] is True
    assert diagnostics["policy_data_valid"] is True
    assert diagnostics["policy_data_errors"] == []
    assert diagnostics["verdict_restricted"] is False
    assert "complete and well-formed" in diagnostics["explanation"]


def test_policy_diagnostics_report_an_invalid_policy():
    state = _broken_policy_state()

    report = build_research_report(state, "text", state["claim"])
    diagnostics = report["executive_summary"]["policy_diagnostics"]

    assert diagnostics["gate_ran"] is True
    assert diagnostics["policy_data_valid"] is False
    assert diagnostics["verdict_restricted"] is True
    assert diagnostics["policy_data_errors"], "the unusable field must be named"
    assert "primary_correction_groups" in " ".join(diagnostics["policy_data_errors"])

    explanation = diagnostics["explanation"]
    assert "restricted the result to UNVERIFIED" in explanation
    assert "NOT proof that the claim is false" in explanation
    assert "primary_correction_groups" in explanation


def test_policy_diagnostics_report_when_the_gate_never_ran():
    """
    A modality with no verifiable claim has no final verdict from the gate at
    all. The report must say so rather than claiming the policy was checked.
    """
    state = _research_state({
        "verdict": "INSUFFICIENT_VERIFIABLE_CLAIMS",
        "evidence_sufficiency": "INSUFFICIENT",
    })

    report = build_research_report(state, "text", state["claim"])
    diagnostics = report["executive_summary"]["policy_diagnostics"]

    assert diagnostics["gate_ran"] is False
    assert diagnostics["policy_data_valid"] is None
    assert diagnostics["verdict_restricted"] is False
    assert "did not run" in diagnostics["explanation"]


def test_policy_diagnostics_are_always_present():
    """
    The block must exist for every report, so the UI never has to guess.
    """
    for verdict in (
        {"verdict": "REAL", "policy_data_valid": True, "policy_data_errors": []},
        {"verdict": "UNVERIFIED", "policy_data_valid": False,
         "policy_data_errors": ["missing 'primary_supporting_groups'"]},
        {"verdict": "INSUFFICIENT_VERIFIABLE_CLAIMS"},
    ):
        state = _research_state(verdict)
        report = build_research_report(state, "text", state["claim"])
        assert "policy_diagnostics" in report["executive_summary"]


def test_broken_policy_still_yields_unverified_and_says_why():
    state = _broken_policy_state()

    report = build_research_report(state, "text", state["claim"])
    summary = report["executive_summary"]

    assert summary["verdict"] == "UNVERIFIED"
    assert "UNVERIFIED" in summary["permitted_verdicts"] if "permitted_verdicts" in summary else True
    # The gate reason explains the restriction.
    assert "policy" in (summary["why_this_verdict"] or "").lower()


# ===========================================================================
# 2. The diagnostics reach the API response
# ===========================================================================

def test_api_response_carries_the_policy_diagnostics():
    """
    The API response model must accept and expose the new fields.
    """
    from api.server import FactualVerificationDetails, ResearchExecutiveSummary

    fact = FactualVerificationDetails(
        verdict="UNVERIFIED",
        confidence=40.0,
        confidence_basis="verification_coverage",
        policy_data_valid=False,
        policy_data_errors=["missing 'primary_correction_groups'"],
    )
    assert fact.policy_data_valid is False
    assert fact.policy_data_errors == ["missing 'primary_correction_groups'"]

    summary = ResearchExecutiveSummary(
        verdict="UNVERIFIED",
        policy_diagnostics={
            "gate_ran": True,
            "policy_data_valid": False,
            "policy_data_errors": ["missing 'primary_correction_groups'"],
            "verdict_restricted": True,
            "explanation": "The evidence policy data was incomplete or invalid.",
        },
    )
    assert summary.policy_diagnostics is not None
    assert summary.policy_diagnostics.verdict_restricted is True
    assert summary.policy_diagnostics.policy_data_errors == [
        "missing 'primary_correction_groups'"
    ]


def test_api_response_keeps_existing_fields_compatible():
    """
    The new fields are additive: every field that existed before must still
    be accepted and keep its value.
    """
    from api.server import FactualVerificationDetails

    fact = FactualVerificationDetails(
        verdict="REAL",
        confidence=0.7,
        evidence_sufficiency="strong_support",
        independent_supporting=2,
        independent_contradicting=0,
        contradiction_score=0.0,
        source_agreement=0.5,
        fusion_score=0.6,
        reasoning="because",
        permitted_verdicts=["REAL", "UNVERIFIED"],
        deterministic_gate_reason="because",
        total_sources=4,
        supporting_sources=2,
        contradicting_sources=0,
        context_chunks=1,
        agent_verdict="REAL",
        critic_verdict="REAL",
        agents_agree=True,
        verdict_overridden_by_evidence=False,
        agent_confidence=94.0,
        critic_confidence=92.0,
        primary_supporting_groups=2,
    )
    assert fact.verdict == "REAL"
    assert fact.confidence == 0.7
    assert fact.permitted_verdicts == ["REAL", "UNVERIFIED"]
    assert fact.primary_supporting_groups == 2
    # The new fields default to "not reported", not to a fabricated state.
    assert fact.policy_data_valid is None
    assert fact.policy_data_errors == []


def test_router_carries_the_policy_fields_into_fact_data():
    """
    The router's shared helper must forward the diagnostics, so all four
    modalities report the same state.
    """
    from multimodal_router import _evidence_policy_fields

    fields = _evidence_policy_fields({
        "policy_data_valid": False,
        "policy_data_errors": ["missing 'article_supporting_groups'"],
    })
    assert fields["policy_data_valid"] is False
    assert fields["policy_data_errors"] == ["missing 'article_supporting_groups'"]

    # A verdict that never ran reports "unknown", not "valid".
    fields = _evidence_policy_fields({})
    assert fields["policy_data_valid"] is None
    assert fields["policy_data_errors"] == []


# ===========================================================================
# 3. No secret or internal detail leaks into the report
# ===========================================================================

def test_policy_diagnostics_expose_no_stack_traces_or_credentials():
    state = _broken_policy_state()
    report = build_research_report(state, "text", state["claim"])
    diagnostics = report["executive_summary"]["policy_diagnostics"]

    blob = (diagnostics["explanation"] + " " + " ".join(diagnostics["policy_data_errors"])).lower()
    for forbidden in (
        "traceback", "file \"", "line ", "api_key", "grok", "secret",
        "password", "token", "os.environ",
    ):
        assert forbidden not in blob, f"the diagnostic must not leak {forbidden!r}"


def test_policy_diagnostics_only_name_policy_field_names():
    """
    The errors are produced by the validator and name policy counters only.
    """
    state = _broken_policy_state()
    report = build_research_report(state, "text", state["claim"])
    errors = report["executive_summary"]["policy_diagnostics"]["policy_data_errors"]

    for error in errors:
        assert error.startswith("missing '") or error.startswith("'")
        assert any(
            token in error
            for token in (
                "independent_supporting", "independent_contradicting",
                "primary_supporting_groups", "primary_contradicting_groups",
                "article_supporting_groups", "article_contradicting_groups",
                "correction_groups", "primary_correction_groups",
            )
        ), f"unexpected error content: {error!r}"


# ===========================================================================
# 4. Healthy runs are unaffected
# ===========================================================================

def test_a_healthy_run_is_unchanged():
    """
    The diagnostics are additive. A normal report keeps every value it had.
    """
    state = _research_state({
        "verdict": "REAL",
        "confidence": 70.0,
        "confidence_basis": "evidence_strength",
        "policy_data_valid": True,
        "policy_data_errors": [],
    })

    report = build_research_report(state, "text", state["claim"])
    summary = report["executive_summary"]

    assert summary["verdict"] == "REAL"
    assert summary["confidence"] == pytest.approx(0.7)
    assert summary["input_claim"] == "The claim under test."
    assert report["methodology"]["search_queries"]
    assert report["methodology"]["sources_retrieved"] == 1
    assert report["findings"]["supporting"]


# ===========================================================================
# 5. An error path must still return a valid response
# ===========================================================================

def test_error_path_returns_a_well_formed_report():
    """
    A pipeline failure must produce a valid research report.

    An empty report object fails response validation, which turned any
    pipeline failure (for example a provider rate limit) into a 500 instead
    of a graceful error result.
    """
    from multimodal_router import build_unified_response

    response = build_unified_response(
        modality="text",
        status="error",
        error="provider rate limit reached for org_01EXAMPLEPLACEHOLDER00",
    )

    report = response["research_report"]
    assert report["modality"] == "text"
    assert report["executive_summary"]["policy_diagnostics"] is not None
    assert "did not complete" in report["methodology"]["research_skipped_reason"]


@pytest.mark.parametrize("modality", ["text", "image", "video", "audio"])
def test_error_path_report_validates_against_the_api_schema(modality):
    """
    Every modality's error path must produce a report the API model accepts.
    """
    from api.server import ResearchReport
    from multimodal_router import build_unified_response

    response = build_unified_response(
        modality=modality, status="error", error="boom"
    )
    validated = ResearchReport(**response["research_report"])
    assert validated.modality == modality


def test_error_text_is_redacted_of_provider_identifiers():
    """
    A provider error can carry the account's organisation id. That must not
    reach the API response.
    """
    from multimodal_router import build_unified_response

    response = build_unified_response(
        modality="text",
        status="error",
        error="Rate limit reached in organization org_01EXAMPLEPLACEHOLDER00",
    )
    assert "org_01EXAMPLEPLACEHOLDER00" not in (response["error"] or "")
    assert "org_01EXAMPLEPLACEHOLDER00" not in (
        response["research_report"]["methodology"]["research_skipped_reason"]
    )


def test_redaction_removes_keys_but_keeps_the_message():
    from research_report import redact_sensitive

    redacted = redact_sensitive(
        "Invalid key gsk_PLACEHOLDERKEYPLACEHOLDER sent to api_key=supersecretvalue"
    )
    assert "gsk_PLACEHOLDERKEYPLACEHOLDER" not in redacted
    assert "supersecretvalue" not in redacted
    assert "Invalid key" in redacted


def test_redaction_leaves_ordinary_messages_untouched():
    from research_report import redact_sensitive

    for message in (
        "Claim text cannot be empty.",
        "Image file not found: /tmp/x.png",
        "The evidence policy data was incomplete or invalid.",
    ):
        assert redact_sensitive(message) == message


def test_a_failing_pipeline_returns_200_with_an_error_status():
    """
    A provider failure (for example a rate limit) must produce a graceful
    error result, not a 500 caused by an invalid response body.
    """
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)

    with patch("pipeline.run_research_pipeline", side_effect=RuntimeError(
        "Rate limit reached in organization org_01EXAMPLEPLACEHOLDER00"
    )):
        response = client.post("/verify/text", json={"claim": "A test claim."})

    assert response.status_code == 200, (
        "a pipeline failure must not become a 500: "
        + response.text[:400]
    )
    data = response.json()
    assert data["status"] == "error"
    assert data["research_report"]["modality"] == "text"
    assert "org_01EXAMPLEPLACEHOLDER00" not in (data.get("error") or "")
