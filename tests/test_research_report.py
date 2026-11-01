"""
tests/test_research_report.py
=============================
Regression tests for the point-wise, evidence-grounded research report
(`research_report.py`) and for the report's exposure through the API.

These are FIXTURE tests: they build pipeline states by hand and assert that
the report reproduces exactly what the backend recorded - no more, no less.
The live end-to-end runs are covered separately in
`tests/test_end_to_end_report.py`.

The most important property asserted throughout is that the report never
fabricates: no source, URL, score or conclusion appears that the backend did
not record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from research_report import build_research_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO_RUN = PROJECT_ROOT / "video_module" / "outputs" / "tmpx191xwhp"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _evidence(
    evidence_class: str,
    *,
    url: str = "https://example.org/article",
    domain: str = "example.org",
    title: str = "An article",
    chunk: str = "The passage that was retrieved.",
    score: float = 0.8,
    reliability: float = 0.9,
    tier: str = "major_outlet",
    nli_label: str = "entailment",
    nli_score: float = 0.91,
    date_status: str = "match",
    event_match: str = "strong",
    group: str = "example.org",
    search_query: str = "the query that found it",
) -> Dict[str, Any]:
    return {
        "source_id": 1,
        "url": url,
        "domain": domain,
        "title": title,
        "evidence_type": "scraped_article",
        "chunk": chunk,
        "similarity": 0.8,
        "nli_label": nli_label,
        "nli_score": nli_score,
        "nli_status": "ok",
        "raw_nli_label": nli_label,
        "raw_nli_score": nli_score,
        "source_reliability": reliability,
        "source_tier": tier,
        "source_tier_reason": "established outlet",
        "independence_group": group,
        "is_primary_source": True,
        "is_rolling_page": False,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": date_status,
        "date_reason": "publication date matches the claimed date",
        "publication_date": {"day": 1, "month": 3, "year": 2020},
        "event_match": event_match,
        "location_match": True,
        "location_conflict": False,
        "detail_coverage": 0.8,
        "debunking": False,
        "event_signals": ["event"],
        "search_query": search_query,
    }


def _state(
    *,
    claim: str = "The claim under test.",
    evidence: list | None = None,
    source_data: list | None = None,
    final_verdict: dict | None = None,
    search_plans: list | None = None,
    search_errors: list | None = None,
    scrape_failures: list | None = None,
    verification: str = "",
    feedback: str = "",
    source_agreement: dict | None = None,
) -> Dict[str, Any]:
    return {
        "claim": claim,
        "search_results": "SOURCE 1\nURL:\nhttps://example.org/a\n",
        "search_plans": search_plans if search_plans is not None else [
            {"label": "exact_claim", "query": claim, "intent": "exact", "topic": "news"},
        ],
        "search_errors": search_errors or [],
        "scrape_failures": scrape_failures or [],
        "source_data": source_data if source_data is not None else [],
        "evidence_analysis": evidence if evidence is not None else [],
        "source_agreement": source_agreement if source_agreement is not None else {},
        "verification": verification,
        "feedback": feedback,
        "final_verdict": final_verdict if final_verdict is not None else {
            "verdict": "UNVERIFIED",
            "confidence": 55,
            "evidence_sufficiency": "context_only",
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "total_sources": 1,
            "permitted_verdicts": ["UNVERIFIED"],
            "permitted_reason": "No independent corroboration was found.",
            "reasoning": "No independent corroboration was found.",
            "deterministic_gate_reason": "No independent corroboration was found.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
        },
    }


# ---------------------------------------------------------------------------
# 1. Text with supporting evidence
# ---------------------------------------------------------------------------

def test_supporting_evidence_produces_supporting_findings():
    state = _state(
        claim="City council approved the new transit plan on 4 March 2020.",
        evidence=[_evidence("SUPPORT", score=0.88)],
        final_verdict={
            "verdict": "REAL",
            "confidence": 82.0,
            "evidence_sufficiency": "strong_support",
            "independent_supporting": 2,
            "independent_contradicting": 0,
            "total_sources": 2,
            "permitted_verdicts": ["MISLEADING", "REAL", "UNVERIFIED"],
            "permitted_reason": "Two independent publishers corroborate the claim.",
            "reasoning": "Two independent publishers corroborate the claim.",
            "supporting_evidence": [{
                "url": "https://example.org/article",
                "domain": "example.org",
                "evidence_type": "scraped_article",
                "source_tier": "major_outlet",
                "date_status": "match",
                "event_match": "strong",
                "evidence_score": 0.88,
                "chunk": "The passage that was retrieved.",
            }],
            "contradicting_evidence": [],
        },
    )

    report = build_research_report(state, "text", state["claim"])

    assert report["findings"]["supporting"], "a SUPPORT passage must yield a finding"
    finding = report["findings"]["supporting"][0]
    assert finding["source_url"] == "https://example.org/article"
    assert finding["source_name"] == "An article"
    assert finding["status"] == "SUPPORTED"
    assert finding["evidence_strength"] == 0.88
    assert finding["strength_label"] == "Strong"
    assert finding["excerpt"] == "The passage that was retrieved."
    assert report["executive_summary"]["verdict"] == "REAL"
    assert report["executive_summary"]["confidence"] == pytest.approx(0.82)
    assert report["findings"]["contradicting"] == []


# ---------------------------------------------------------------------------
# 2. Text with contradicting evidence
# ---------------------------------------------------------------------------

def test_contradicting_evidence_produces_contradicting_findings():
    state = _state(
        evidence=[
            _evidence("CONTRADICTION", nli_label="contradiction", nli_score=0.88, score=0.74),
        ],
        final_verdict={
            "verdict": "FAKE",
            "confidence": 71.5,
            "evidence_sufficiency": "strong_contradiction",
            "independent_supporting": 0,
            "independent_contradicting": 2,
            "total_sources": 2,
            "permitted_verdicts": ["FAKE", "UNVERIFIED"],
            "permitted_reason": "Two independent publishers contradict the claim.",
            "reasoning": "Two independent publishers contradict the claim.",
            "supporting_evidence": [],
            "contradicting_evidence": [{
                "url": "https://example.org/factcheck",
                "domain": "example.org",
                "evidence_type": "scraped_article",
                "source_tier": "major_outlet",
                "date_status": "match",
                "event_match": "strong",
                "evidence_score": 0.74,
                "chunk": "The passage that was retrieved.",
            }],
        },
    )

    report = build_research_report(state, "text", state["claim"])

    assert report["findings"]["contradicting"], "a CONTRADICTION passage must yield a finding"
    finding = report["findings"]["contradicting"][0]
    assert finding["status"] == "CONTRADICTED"
    assert finding["nli_label"] == "contradiction"
    assert report["executive_summary"]["verdict"] == "FAKE"
    assert report["executive_summary"]["confidence"] == pytest.approx(0.715)


# ---------------------------------------------------------------------------
# 3. Text with insufficient evidence
# ---------------------------------------------------------------------------

def test_insufficient_evidence_is_reported_as_unverified_not_false():
    state = _state(
        evidence=[_evidence("RELATED_CONTEXT", nli_label="neutral", nli_score=0.4, score=0.2)],
    )

    report = build_research_report(state, "text", state["claim"])
    summary = report["executive_summary"]

    assert summary["verdict"] == "UNVERIFIED"
    assert report["findings"]["supporting"] == []
    assert report["findings"]["contradicting"] == []
    assert report["findings"]["contextual"], "context passages must still be reported"

    # The report must state that UNVERIFIED is not a finding of falsity.
    limitation_text = " ".join(summary["limitations"]).lower()
    assert "not a finding of falsity" in limitation_text

    # An evidence gap must be recorded.
    assert report["evidence_gaps"], "an inconclusive run must record an evidence gap"
    gap = report["evidence_gaps"][0]
    assert gap["status"] == "NOT_ESTABLISHED"
    assert gap["what_would_help"]


# ---------------------------------------------------------------------------
# 4. Empty evidence collections and null optional scores
# ---------------------------------------------------------------------------

def test_empty_evidence_produces_empty_findings_without_inventing_them():
    state = _state(evidence=[])

    report = build_research_report(state, "text", state["claim"])

    for bucket in ("supporting", "contradicting", "contextual", "unresolved"):
        assert report["findings"][bucket] == []
    assert report["methodology"]["sources_retrieved"] == 0
    assert report["methodology"]["articles_scraped"] == 0
    assert report["methodology"]["evidence_passages"] == 0
    assert report["executive_summary"]["confidence"] == pytest.approx(0.55)


def test_null_optional_scores_are_not_rendered_as_numbers():
    state = _state(
        evidence=[_evidence("SUPPORT", score=None, reliability=None, nli_score=None)],
        final_verdict={
            "verdict": "UNVERIFIED",
            "confidence": None,
            "evidence_sufficiency": None,
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "total_sources": 0,
            "permitted_verdicts": ["UNVERIFIED"],
            "permitted_reason": "No evidence.",
            "reasoning": "No evidence.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
        },
    )

    report = build_research_report(state, "text", state["claim"])
    finding = report["findings"]["supporting"][0]

    assert finding["evidence_strength"] is None
    assert finding["strength_label"] == "Not scored"
    assert finding["source_reliability"] is None
    assert finding["nli_score"] is None
    assert report["executive_summary"]["confidence"] is None


# ---------------------------------------------------------------------------
# 5. Search and scraping failure reporting
# ---------------------------------------------------------------------------

def test_search_queries_and_failures_are_reported_verbatim():
    state = _state(
        search_plans=[
            {"label": "exact_claim", "query": "the exact claim", "intent": "exact", "topic": "news"},
            {"label": "factcheck", "query": "factcheck the claim", "intent": "factcheck", "topic": "news"},
        ],
        search_errors=["factcheck: rate limited"],
        scrape_failures=[
            {
                "url": "https://blocked.example/page",
                "domain": "blocked.example",
                "title": "Blocked page",
                "reason": "SCRAPE_FAILED: HTTP 403 after 2 attempts.",
                "fallback_used": "search_snippet",
            },
            {
                "url": "https://dead.example/page",
                "domain": "dead.example",
                "title": "Dead page",
                "reason": "SCRAPE_FAILED: Request timed out.",
                "fallback_used": False,
            },
        ],
    )

    report = build_research_report(state, "text", state["claim"])
    method = report["methodology"]

    assert [q["query"] for q in method["search_queries"]] == [
        "the exact claim",
        "factcheck the claim",
    ]
    assert method["search_errors"] == ["factcheck: rate limited"]
    assert len(method["scrapes_failed"]) == 2
    assert method["scrapes_failed"][0]["reason"].startswith("SCRAPE_FAILED")
    assert method["scrapes_failed"][1]["fallback_used"] is False

    # The failures must also surface as a limitation and an evidence gap.
    assert any("could not be scraped" in lim for lim in report["executive_summary"]["limitations"])
    assert any(gap["status"] == "PARTIAL" for gap in report["evidence_gaps"])


# ---------------------------------------------------------------------------
# 6. Non-http URLs must never become clickable links
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_url", [
    "javascript:alert(1)",
    "data:text/html;base64,PHNjcmlwdD4=",
    "file:///etc/passwd",
    "not a url",
    "",
])
def test_unsafe_urls_are_not_exposed_as_links(bad_url):
    state = _state(evidence=[_evidence("SUPPORT", url=bad_url)])

    report = build_research_report(state, "text", state["claim"])
    finding = report["findings"]["supporting"][0]

    assert finding["source_url"] is None


# ---------------------------------------------------------------------------
# 7. The report never invents a source that the backend did not record
# ---------------------------------------------------------------------------

def test_report_contains_no_source_absent_from_the_evidence_layer():
    recorded_urls = {"https://a.example/1", "https://b.example/2"}
    state = _state(
        evidence=[
            _evidence("SUPPORT", url="https://a.example/1", domain="a.example"),
            _evidence("CONTRADICTION", url="https://b.example/2", domain="b.example",
                      nli_label="contradiction"),
        ],
    )

    report = build_research_report(state, "text", state["claim"])
    found = {
        f["source_url"]
        for bucket in report["findings"].values()
        for f in bucket
        if f.get("source_url")
    }

    assert found == recorded_urls
    assert found <= recorded_urls


# ---------------------------------------------------------------------------
# 8. Agent output is quoted, not paraphrased into a new conclusion
# ---------------------------------------------------------------------------

def test_agent_narrative_and_critic_review_are_quoted_verbatim():
    agent_text = (
        "Verdict: REAL\n"
        "Confidence: 80%\n"
        "Reasoning: Because two outlets reported it.\n"
        "Final Explanation: The claim is corroborated by two independent outlets."
    )
    critic_text = (
        "Review: Sound.\n"
        "Problems: Minor date wording issue.\n"
        "Recommended Verdict: REAL\n"
        "Recommended Confidence: 78%\n"
        "Final Recommendation: Accept the corroboration."
    )

    state = _state(evidence=[_evidence("SUPPORT")], verification=agent_text, feedback=critic_text)
    report = build_research_report(state, "text", state["claim"])

    assert report["executive_summary"]["agent_narrative"] == (
        "The claim is corroborated by two independent outlets."
    )
    assert report["executive_summary"]["critic_review"] == (
        "Accept the corroboration."
    )


# ---------------------------------------------------------------------------
# 9. Video: research skipped must be explained, never faked
# ---------------------------------------------------------------------------

def test_video_with_no_verifiable_claims_reports_skipped_research():
    state = {
        "research_skipped_reason": (
            "No verifiable factual claims passed filtering. All candidates were "
            "conversational filler, greetings, apologies, or noise. Web research skipped."
        ),
        "candidate_claims_count": 32,
        "selected_claims_count": 0,
        "rejected_claims_count": 32,
        "verified_claims": [],
        "transcription_segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "..."}],
        "transcription_language": "mr",
    }

    report = build_research_report(state, "video", None)
    method = report["methodology"]
    summary = report["executive_summary"]

    assert method["research_skipped"] is True
    assert "No verifiable factual claims passed filtering" in method["research_skipped_reason"]
    assert "32 candidate claim(s)" in method["research_skipped_reason"]
    assert method["search_queries"] == []

    # The report must explain the situation, not invent findings.
    assert summary["verdict"] is None
    assert all(bucket == [] for bucket in report["findings"].values())
    assert report["evidence_gaps"]
    assert report["evidence_gaps"][0]["status"] == "NOT_ESTABLISHED"

    # The extraction stages must be reported with their real counters.
    stage_names = {s["name"]: s for s in method["stages"]}
    assert "Speech transcription" in stage_names
    assert stage_names["Speech transcription"]["detail"].startswith("1 Whisper segment")
    assert stage_names["Factual claim filtering"]["detail"] == (
        "32 candidate claim(s) extracted, 0 passed factual-claim filtering, 32 rejected"
    )


# ---------------------------------------------------------------------------
# 10. Video: real saved artifacts from the run that produced zero claims
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_VIDEO_RUN.exists(), reason="saved video artifacts not present")
def test_real_video_run_artifact_reproduces_zero_claims_without_fabrication():
    """
    The run in video_module/outputs/tmpx191xwhp produced 32 candidate claims and
    zero accepted ones. The report must present that honestly: research
    skipped, no fabricated findings.
    """
    candidate_claims = json.loads(
        (REAL_VIDEO_RUN / "candidate_claims.json").read_text(encoding="utf-8")
    )
    claims = candidate_claims["candidate_claims"]
    assert candidate_claims["candidate_claims_count"] == 32
    assert candidate_claims["selected_claims_count"] == 0

    state = {
        "research_skipped_reason": (
            "No verifiable factual claims passed filtering. All candidates were "
            "conversational filler, greetings, apologies, or noise. Web research skipped."
        ),
        "candidate_claims_count": candidate_claims["candidate_claims_count"],
        "selected_claims_count": candidate_claims["selected_claims_count"],
        "rejected_claims_count": candidate_claims["rejected_claims_count"],
        "verified_claims": [],
        "candidate_claims": claims,
    }

    report = build_research_report(state, "video", None)

    assert report["methodology"]["research_skipped"] is True
    assert all(bucket == [] for bucket in report["findings"].values())
    assert report["claims"] == []
    assert report["evidence_gaps"], "a skipped-research run must record why"

    # Every rejected candidate in the artifact must have a recorded reason.
    assert all(c.get("selection_reason") for c in claims)
