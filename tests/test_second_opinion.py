"""
tests/test_second_opinion.py
=============================
Tests for the independent LLM second-opinion verification stage
(second_opinion.py).

Covers:
  * agreement between the LLM opinion and the system verdict
  * disagreement between the LLM opinion and the system verdict
  * no usable evidence (short-circuit: inconclusive UNVERIFIED, no LLM call)
  * LLM timeout and API failure
  * invalid LLM verdicts (not in REAL/FAKE/MISLEADING/UNVERIFIED)
  * deterministic evidence gate enforcement (LLM cannot promote a
    gate-restricted UNVERIFIED to REAL or FAKE)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import second_opinion as so


def test_llm_output_parsed_and_agrees():
    raw = """
VERDICT: REAL
CONFIDENCE: 85%
CLAIM_SUMMARY: The sky is blue.
REASONING: Two independent publishers support the claim.
SUPPORTING_EVIDENCE:
- https://example.com/a | example.com | The sky is blue.
CONTRADICTING_EVIDENCE:
none
SOURCE_CREDIBILITY: Two primary publishers with retrieved article text.
EVIDENCE_SUFFICIENCY: sufficient
MISSING_EVIDENCE: Exact timestamp not stated.
UNCERTAINTY: Location not confirmed in all sources.
LIMITATIONS:
- Only two publishers retrieved.
AGREEMENT_WITH_SYSTEM: agree - both verdicts are REAL.
"""
    with patch.object(so, "invoke_agent", return_value=raw):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "The sky is blue.", "evidence_class": "SUPPORT", "evidence_score": 0.9}],
            source_agreement={"independent_supporting": 2, "independent_contradicting": 0},
            existing_verdict="REAL",
            permitted_verdicts=["REAL", "UNVERIFIED"],
        )
    assert result["verdict"] == "REAL"
    assert result["confidence"] == 85.0
    assert result["claim_summary"] == "The sky is blue."
    assert result["comparison"]["agreement"] is True
    assert result["comparison"]["disagreement"] is False
    assert result["comparison"]["gate_enforced"] is False
    assert "agrees" in result["comparison"]["comparison_summary"].lower()
    assert len(result["supporting_evidence"]) == 1
    assert result["supporting_evidence"][0]["url"] == "https://example.com/a"


def test_disagreement_when_verdicts_differ():
    raw = "VERDICT: FAKE\nCONFIDENCE: 70%\nCLAIM_SUMMARY: x\nREASONING: Sources refute the claim.\nSUPPORTING_EVIDENCE:\nnone\nCONTRADICTING_EVIDENCE:\nnone\nSOURCE_CREDIBILITY:\nnone\nEVIDENCE_SUFFICIENCY:\nnone\nMISSING_EVIDENCE:\nnone\nUNCERTAINTY:\nnone\nLIMITATIONS:\nnone\nAGREEMENT_WITH_SYSTEM: disagree - system says REAL."
    with patch.object(so, "invoke_agent", return_value=raw):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "Sources refute the claim.", "evidence_class": "CONTRADICTION", "evidence_score": 0.8}],
            source_agreement={"independent_supporting": 2, "independent_contradicting": 0},
            existing_verdict="REAL",
            permitted_verdicts=["REAL", "UNVERIFIED"],
        )
    assert result["verdict"] == "FAKE"
    assert result["comparison"]["agreement"] is False
    assert result["comparison"]["disagreement"] is True
    assert "disagrees" in result["comparison"]["comparison_summary"].lower()


def test_no_evidence_short_circuit():
    """No evidence passages -> inconclusive UNVERIFIED without an LLM call."""
    with patch.object(so, "invoke_agent", return_value="") as m:
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[],
            source_agreement={"independent_supporting": 0, "independent_contradicting": 0},
            existing_verdict="UNVERIFIED",
            permitted_verdicts=["UNVERIFIED"],
        )
    # invoke_agent must NOT be called when there is no evidence.
    m.assert_not_called()
    assert result["no_evidence"] is True
    assert result["verdict"] == "UNVERIFIED"
    assert result["comparison"]["agreement"] is True  # UNVERIFIED == UNVERIFIED
    assert "inconclusive" in result["explanation"].lower()


def test_llm_timeout():
    with patch.object(so, "invoke_agent", side_effect=TimeoutError("request timed out")):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "x", "evidence_class": "SUPPORT", "evidence_score": 0.5}],
            source_agreement={"independent_supporting": 0, "independent_contradicting": 0},
            existing_verdict="UNVERIFIED",
            permitted_verdicts=["UNVERIFIED"],
        )
    assert result["timeout"] is True
    assert result["error"] is not None
    assert result["verdict"] is None


def test_llm_api_failure():
    with patch.object(so, "invoke_agent", side_effect=Exception("429 rate limit exceeded")):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "x", "evidence_class": "SUPPORT", "evidence_score": 0.5}],
            source_agreement={"independent_supporting": 0, "independent_contradicting": 0},
            existing_verdict="UNVERIFIED",
            permitted_verdicts=["UNVERIFIED"],
        )
    assert result["api_failure"] is True
    assert result["error"] is not None
    assert result["verdict"] is None


def test_invalid_verdict_treated_as_missing():
    raw = "VERDICT: PROBABLY_TRUE\nCONFIDENCE: 90%\nCLAIM_SUMMARY: x\nREASONING: y\nSUPPORTING_EVIDENCE:\nnone\nCONTRADICTING_EVIDENCE:\nnone\nSOURCE_CREDIBILITY:\nnone\nEVIDENCE_SUFFICIENCY:\nnone\nMISSING_EVIDENCE:\nnone\nUNCERTAINTY:\nnone\nLIMITATIONS:\nnone\nAGREEMENT_WITH_SYSTEM:\nnone"
    with patch.object(so, "invoke_agent", return_value=raw):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "x", "evidence_class": "SUPPORT", "evidence_score": 0.5}],
            source_agreement={"independent_supporting": 0, "independent_contradicting": 0},
            existing_verdict="UNVERIFIED",
            permitted_verdicts=["UNVERIFIED"],
        )
    assert result["verdict"] is None
    assert result["comparison"]["agreement"] is False


def test_deterministic_gate_enforcement_blocks_llm_promotion():
    """
    The LLM proposes REAL, but the system is UNVERIFIED and the gate does not
    permit REAL or FAKE. The second opinion must be reported as UNVERIFIED
    with gate_enforced True.
    """
    raw = "VERDICT: REAL\nCONFIDENCE: 90%\nCLAIM_SUMMARY: x\nREASONING: Looks real.\nSUPPORTING_EVIDENCE:\nnone\nCONTRADICTING_EVIDENCE:\nnone\nSOURCE_CREDIBILITY:\nnone\nEVIDENCE_SUFFICIENCY:\nnone\nMISSING_EVIDENCE:\nnone\nUNCERTAINTY:\nnone\nLIMITATIONS:\nnone\nAGREEMENT_WITH_SYSTEM:\nnone"
    with patch.object(so, "invoke_agent", return_value=raw):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "Looks real.", "evidence_class": "SUPPORT", "evidence_score": 0.9}],
            source_agreement={"independent_supporting": 0, "independent_contradicting": 0},
            existing_verdict="UNVERIFIED",
            permitted_verdicts=["UNVERIFIED"],
        )
    assert result["comparison"]["gate_enforced"] is True
    assert result["verdict"] == "UNVERIFIED"
    assert result["comparison"]["llm_verdict"] == "UNVERIFIED"
    assert result["comparison"]["agreement"] is True


def test_deterministic_gate_allows_agreement_when_permitted():
    """
    When the gate permits REAL and the LLM agrees with REAL, no enforcement
    is needed and agreement is recorded normally.
    """
    raw = "VERDICT: REAL\nCONFIDENCE: 80%\nCLAIM_SUMMARY: x\nREASONING: Supported.\nSUPPORTING_EVIDENCE:\nnone\nCONTRADICTING_EVIDENCE:\nnone\nSOURCE_CREDIBILITY:\nnone\nEVIDENCE_SUFFICIENCY:\nnone\nMISSING_EVIDENCE:\nnone\nUNCERTAINTY:\nnone\nLIMITATIONS:\nnone\nAGREEMENT_WITH_SYSTEM:\nnone"
    with patch.object(so, "invoke_agent", return_value=raw):
        result = so.build_second_opinion(
            claim="The sky is blue.",
            evidence_analysis=[{"url": "https://example.com/a", "domain": "example.com", "chunk": "Supported.", "evidence_class": "SUPPORT", "evidence_score": 0.9}],
            source_agreement={"independent_supporting": 2, "independent_contradicting": 0},
            existing_verdict="REAL",
            permitted_verdicts=["REAL", "UNVERIFIED"],
        )
    assert result["comparison"]["gate_enforced"] is False
    assert result["comparison"]["agreement"] is True
    assert result["verdict"] == "REAL"
