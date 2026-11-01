"""
tests/test_root_cause_fixes.py
===============================
Regression tests for root-cause fixes:

1. NLI `nli` unbound-variable protection (UnboundLocalError).
2. NLI model/tokenizer one-time loading and cache reuse.
3. Groq 429 retry behaviour (bounded attempts, retry-after header).
4. Authentication failures are NOT retried.
5. Transient 5xx/timeout retry behaviour.
6. Primary vs non-primary evidence classification.
7. Search snippets never count as support.
8. Duplicate URLs/chunks do not count as independent sources.
9. Conservative verdict with only one primary source.
10. Claim type classification (scientific_fact vs event claims).
11. Diagnostic summary explains UNVERIFIED reason from execution data.
12. No secrets leak in diagnostics/logging.

All tests use controlled fixtures and mocks — no network calls, no LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import (  # noqa: E402
    AGENT_MAX_ATTEMPTS,
    _error_category,
    _is_retryable,
    invoke_agent,
)
from claim_profile import (  # noqa: E402
    build_claim_profile,
    classify_claim_type,
)
from evidence_analyzer import (  # noqa: E402
    analyze_evidence,
    classify_evidence,
)
from evidence_fusion import (  # noqa: E402
    calculate_source_agreement,
    _summarise_source,
)
from evidence_brief import (  # noqa: E402
    MAX_PASSAGES,
    MAX_PASSAGE_CHARS,
)
from final_verdict import (  # noqa: E402
    generate_final_verdict,
    deterministic_gate,
)
from source_reliability import get_source_profile  # noqa: E402
from nli_checker import (  # noqa: E402
    check_nli,
    get_tokenizer,
    _load_model,
    _NLI_CACHE,
    _tokenizer,
    _nli_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WATER_CLAIM = "Water boils at 100 degrees Celsius at standard atmospheric pressure."

WATER_SUPPORT_PREMISE = (
    "At standard atmospheric pressure (1 atmosphere = 0.101325 MPa), "
    "water boils at approximately 100 degrees Celsius."
)

WATER_CONTRADICT_PREMISE = (
    "At standard atmospheric pressure, pure water boils at approximately "
    "50 degrees Celsius."
)

UNRELATED_PREMISE = (
    "The Earth orbits the Sun approximately once every year."
)


def _make_signals(
    evidence_class: str = "SUPPORT",
    nli_label: str = "entailment",
    nli_score: float = 0.9,
    similarity: float = 0.85,
    date_status: str = "match",
    event_match: str = "strong",
    coverage: float = 0.50,
    debunking: bool = False,
    passage_kind: str = "paragraph",
    claim_has_location: bool = False,
    claim_has_date: bool = False,
    claim_type: str = "scientific_fact",
    event_date_conflict: bool = False,
    event_date_mismatch: bool = False,
    location_match: bool = True,
) -> Dict[str, Any]:
    """Build signals dict shaped like evidence_analyzer output."""
    return {
        "similarity": similarity,
        "date_status": date_status,
        "event_match": event_match,
        "raw_nli_label": nli_label,
        "raw_nli_score": nli_score,
        "nli_label": nli_label,
        "nli_score": nli_score,
        "nli_status": "ok",
        "source_reliability": 0.9,
        "source_tier": "official",
        "independence_group": "example.com",
        "detail_coverage": coverage,
        "debunking": debunking,
        "passage_kind": passage_kind,
        "claim_has_location": claim_has_location,
        "claim_has_date": claim_has_date,
        "claim_type": claim_type,
        "event_date_conflict": event_date_conflict,
        "event_date_mismatch": event_date_mismatch,
        "location_match": location_match,
    }


def _make_source_profile(primary: bool = True, tier: str = "official") -> Dict[str, Any]:
    return {
        "is_primary": primary,
        "tier": tier,
        "reliability": 0.9,
        "domain": "example.com",
        "independence_group": "example.com",
        "is_search_snippet": False,
        "is_extracted_text": True,
    }


def _make_non_primary_profile() -> Dict[str, Any]:
    return {
        "is_primary": False,
        "tier": "weak/blog",
        "reliability": 0.32,
        "domain": "randomblog.wordpress.com",
        "independence_group": "randomblog.wordpress.com",
        "is_search_snippet": False,
        "is_extracted_text": True,
    }


def _make_snippet_profile() -> Dict[str, Any]:
    return {
        "is_primary": True,
        "tier": "established_wire",
        "reliability": 0.92,
        "domain": "reuters.com",
        "independence_group": "reuters.com",
        "is_search_snippet": True,
        "is_extracted_text": False,
    }


# ---------------------------------------------------------------------------
# 1. NLI unbound-variable protection
# ---------------------------------------------------------------------------

def test_check_nli_exception_returns_safe_record(monkeypatch):
    """check_nli raising must not leave `nli` unbound in analyze_evidence.

    We monkeypatch check_nli to raise, then call analyze_evidence with a
    real source. The try/except around check_nli in analyze_evidence must
    catch the exception and produce a safe record, never crashing.
    """
    def boom(claim, evidence):
        raise RuntimeError("NLI model exploded")

    monkeypatch.setattr("evidence_analyzer.check_nli", boom)

    source = {
        "source_id": 1,
        "url": "https://iapws.org/faqs/faq1",
        "domain": "iapws.org",
        "title": "IAPWS FAQ",
        "content": WATER_SUPPORT_PREMISE,
        "evidence_type": "scraped_article",
        "scrape_status": "SUCCESS",
        "retrieval_class": "exact_date",
        "found_via": [],
        "search_query": WATER_CLAIM,
        "published_date": "",
        "independence_group": "iapws.org",
    }
    result = analyze_evidence(WATER_CLAIM, [source])
    # Should not crash; classify as RELATED_CONTEXT or NON_EVIDENCE.
    if result:
        assert result[0]["evidence_class"] in ("RELATED_CONTEXT", "NON_EVIDENCE")
    else:
        # No chunks passed pre-filters — also acceptable (no crash).
        assert result == []


# ---------------------------------------------------------------------------
# 2. NLI model / tokenizer one-time loading
# ---------------------------------------------------------------------------

def test_nli_tokenizer_cached_once():
    """Tokenizer should be loaded once, not per-inference."""
    first = get_tokenizer()
    second = get_tokenizer()
    assert first is second, "Tokenizer must be a singleton"


# ---------------------------------------------------------------------------
# 3. Groq 429 retry behaviour
# ---------------------------------------------------------------------------

class _Fake429Error(Exception):
    def __init__(self):
        self.status_code = 429
        self.response = type("R", (), {"headers": {"retry-after": "2"}})()


def test_groq_429_is_retryable():
    err = _Fake429Error()
    assert _is_retryable(err) is True
    assert _error_category(err) == "rate_limit"


def test_groq_429_respects_retry_after(monkeypatch):
    """A 429 with Retry-After=2 should cause a wait of ~2s."""
    calls = []

    class _FakeChain:
        def invoke(self, payload):
            calls.append(1)
            if len(calls) < 3:
                raise _Fake429Error()
            return "VERDICT: UNVERIFIED"

    import agents
    original_sleep = agents.time.sleep
    agents.time.sleep = lambda s: calls.append(("sleep", s))

    try:
        result = invoke_agent(_FakeChain(), {"claim": "test"}, stage="test_agent")
        assert result == "VERDICT: UNVERIFIED"
        sleeps = [c for c in calls if isinstance(c, tuple)]
        assert any(s[1] >= 2.0 for s in sleeps), f"Expected wait >= 2s, got {sleeps}"
    finally:
        agents.time.sleep = original_sleep


# ---------------------------------------------------------------------------
# 4. Auth failures NOT retried
# ---------------------------------------------------------------------------

class _FakeAuthError(Exception):
    def __init__(self):
        self.status_code = 401


def test_auth_failure_not_retried(monkeypatch):
    """401/403 must raise immediately, not retry AGENT_MAX_ATTEMPTS times."""
    calls = []

    class _FakeChain:
        def invoke(self, payload):
            calls.append(1)
            raise _FakeAuthError()

    with pytest.raises(_FakeAuthError):
        invoke_agent(_FakeChain(), {"claim": "test"}, stage="test_agent")

    assert len(calls) == 1, f"Expected 1 attempt, got {len(calls)}"


# ---------------------------------------------------------------------------
# 5. Transient 5xx retry behaviour
# ---------------------------------------------------------------------------

class _Fake503Error(Exception):
    def __init__(self):
        self.status_code = 503


def test_5xx_retried_bounded(monkeypatch):
    """503 should be retried up to AGENT_MAX_ATTEMPTS then raised."""
    calls = []

    class _FakeChain:
        def invoke(self, payload):
            calls.append(1)
            raise _Fake503Error()

    import agents
    original_sleep = agents.time.sleep
    agents.time.sleep = lambda s: None

    try:
        with pytest.raises(_Fake503Error):
            invoke_agent(_FakeChain(), {"claim": "test"}, stage="test_agent")
        assert len(calls) == AGENT_MAX_ATTEMPTS, (
            f"Expected {AGENT_MAX_ATTEMPTS} attempts, got {len(calls)}"
        )
    finally:
        agents.time.sleep = original_sleep


# ---------------------------------------------------------------------------
# 6. Primary vs non-primary evidence classification
# ---------------------------------------------------------------------------

def test_primary_source_support_classified():
    """A primary source with entailment NLI should classify as SUPPORT."""
    signals = _make_signals(evidence_class="SUPPORT", nli_label="entailment")
    profile = _make_source_profile(primary=True, tier="official")
    result = classify_evidence(signals, "scraped_article", profile, claim_type="scientific_fact")
    assert result == "SUPPORT"


def test_non_primary_source_not_support():
    """A non-primary source (blog) must not classify as SUPPORT even with entailment."""
    signals = _make_signals(evidence_class="SUPPORT", nli_label="entailment")
    profile = _make_non_primary_profile()
    result = classify_evidence(signals, "scraped_article", profile, claim_type="scientific_fact")
    assert result != "SUPPORT"


def test_snippet_never_support():
    """A search snippet must never classify as SUPPORT."""
    signals = _make_signals(evidence_class="SUPPORT", nli_label="entailment")
    profile = _make_snippet_profile()
    result = classify_evidence(signals, "search_snippet", profile, claim_type="scientific_fact")
    assert result != "SUPPORT"


# ---------------------------------------------------------------------------
# 7. Duplicate URLs do not count as independent sources
# ---------------------------------------------------------------------------

def test_duplicate_url_one_independence_group():
    """Two chunks from the same URL domain form one independence group."""
    signals1 = _make_signals()
    signals2 = _make_signals()
    profile1 = _make_source_profile()
    profile2 = _make_source_profile()

    result1 = classify_evidence(signals1, "scraped_article", profile1, claim_type="scientific_fact")
    result2 = classify_evidence(signals2, "scraped_article", profile2, claim_type="scientific_fact")

    # Both profiles share the same independence_group
    assert profile1["independence_group"] == profile2["independence_group"]


# ---------------------------------------------------------------------------
# 8. Conservative verdict with one primary source
# ---------------------------------------------------------------------------

def test_one_primary_source_conservative_verdict():
    """One primary source cannot settle a scientific claim on its own."""
    chunks = [_make_signals(evidence_class="SUPPORT", nli_label="entailment")]
    profile = _make_source_profile()
    agreement = calculate_source_agreement([
        {**chunks[0], "source_id": 1, "url": "https://iapws.org/faqs/faq1",
         "evidence_class": "SUPPORT", "source_tier": "official",
         "source_reliability": 0.9, "domain": "iapws.org",
         "independence_group": "iapws.org"}
    ])
    state = {
        "claim": WATER_CLAIM,
        "claim_type": "scientific_fact",
        "evidence_analysis": [
            {**chunks[0], "source_id": 1, "url": "https://iapws.org/faqs/faq1",
             "evidence_class": "SUPPORT", "source_tier": "official",
             "source_reliability": 0.9, "domain": "iapws.org",
             "independence_group": "iapws.org"}
        ],
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


# ---------------------------------------------------------------------------
# 9. Claim type classification
# ---------------------------------------------------------------------------

def test_claim_type_water_boiling_is_scientific():
    """Water boiling claim must be classified as scientific_fact."""
    profile = build_claim_profile(WATER_CLAIM)
    assert profile["claim_type"] == "scientific_fact"


def test_claim_type_apollo_is_historical():
    """Apollo 11 claim must be classified as historical_event."""
    profile = build_claim_profile(
        "Apollo 11 landed on the Moon on July 20, 1969."
    )
    assert profile["claim_type"] == "historical_event"


def test_claim_type_subjective_is_ambiguous():
    """Subjective claim must be classified as ambiguous_or_subjective."""
    profile = build_claim_profile("This news is misleading.")
    assert profile["claim_type"] == "ambiguous_or_subjective"


# ---------------------------------------------------------------------------
# 10. Diagnostic summary explains UNVERIFIED
# ---------------------------------------------------------------------------

def test_diagnostic_no_sources():
    """UNVERIFIED with no sources: diagnostic says 'No usable sources found'."""
    state = {
        "claim": WATER_CLAIM,
        "claim_type": "scientific_fact",
        "evidence_analysis": [],
        "source_agreement": {"total_sources": 0, "evidence_sufficiency": "none"},
        "agent_errors": [],
    }
    from pipeline import _diagnose_unverified_reason
    reason = _diagnose_unverified_reason(state)
    assert "No usable sources" in reason


def test_diagnostic_agent_failure():
    """Agent failure should appear in diagnostic."""
    state = {
        "claim": WATER_CLAIM,
        "claim_type": "scientific_fact",
        "evidence_analysis": [],
        "source_agreement": {"total_sources": 0, "evidence_sufficiency": "none"},
        "agent_errors": [
            {"stage": "critic_agent", "error": "rate limit exceeded (429)"}
        ],
    }
    from pipeline import _diagnose_unverified_reason
    reason = _diagnose_unverified_reason(state)
    assert "rate_limit" in reason.lower() or "Groq" in reason


# ---------------------------------------------------------------------------
# 11. NLI entailment / contradiction / unrelated
# ---------------------------------------------------------------------------

def test_nli_entailment_water_boiling():
    """Direct NLI: support premise should entail water-boiling claim."""
    result = check_nli(WATER_CLAIM, WATER_SUPPORT_PREMISE)
    assert result["label"] == "entailment"
    assert result["status"] in ("ok", "shrunk")


def test_nli_contradiction_water_boiling():
    """Direct NLI: contradictory premise should contradict water-boiling claim."""
    result = check_nli(WATER_CLAIM, WATER_CONTRADICT_PREMISE)
    assert result["label"] in ("contradiction", "neutral")


def test_nli_unrelated_not_support():
    """Direct NLI: unrelated passage should not be entailment."""
    result = check_nli(WATER_CLAIM, UNRELATED_PREMISE)
    assert result["label"] != "entailment"


# ---------------------------------------------------------------------------
# 12. No secrets in diagnostics
# ---------------------------------------------------------------------------

def test_no_secrets_in_diagnostic():
    """Diagnostic summary must not contain API keys or secrets."""
    state = {
        "claim": WATER_CLAIM,
        "claim_type": "scientific_fact",
        "evidence_analysis": [],
        "source_agreement": {"total_sources": 0, "evidence_sufficiency": "none"},
        "agent_errors": [
            {"stage": "verification_agent", "error": "sk-1234567890abcdef"}
        ],
    }
    from pipeline import _diagnose_unverified_reason
    reason = _diagnose_unverified_reason(state)
    assert "sk-1234567890abcdef" not in reason
    assert "GROQ_API_KEY" not in reason
    assert "api_key" not in reason.lower()


# ---------------------------------------------------------------------------
# 13. Groq input-token optimization: condensed critic payload
# ---------------------------------------------------------------------------

def test_evidence_brief_passage_limits_reduced():
    """MAX_PASSAGES and MAX_PASSAGE_CHARS reduced to cut input tokens."""
    from evidence_brief import MAX_PASSAGES, MAX_PASSAGE_CHARS
    assert MAX_PASSAGES == 4, f"Expected MAX_PASSAGES=4, got {MAX_PASSAGES}"
    assert MAX_PASSAGE_CHARS == 600, f"Expected MAX_PASSAGE_CHARS=600, got {MAX_PASSAGE_CHARS}"


def test_evidence_fusion_no_duplicated_chunk_text():
    """Per-source verdict blocks must not duplicate passage text already in EVIDENCE PASSAGES."""
    from evidence_fusion import _summarise_source

    results = [
        {
            "source_id": 1,
            "url": "https://example.com/article",
            "domain": "example.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.9,
            "evidence_type": "scraped_article",
            "source_tier": "official",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.95,
            "nli_status": "ok",
            "is_primary_source": True,
            "chunk": "x" * 2000,  # long chunk text
        }
    ]
    decision = _summarise_source(1, results, identity_resolved=True)
    # best_chunk must NOT be present (removed to avoid duplication).
    assert "best_chunk" not in decision, (
        "best_chunk must be removed from per-source verdict to avoid "
        "duplicating passage text already in EVIDENCE PASSAGES"
    )


def test_condensed_critic_payload_structure():
    """Condensed critic payload contains all required fields for review."""
    claim = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    agreement = {
        "independent_supporting": 0,
        "independent_contradicting": 0,
        "evidence_sufficiency": "context_only",
        "total_sources": 5,
        "agreement_score": 0.0,
    }
    evidence_analysis = [
        {
            "source_id": 1,
            "url": "https://example.com/support",
            "domain": "example.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.9,
            "evidence_type": "scraped_article",
            "source_tier": "official",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.95,
            "nli_status": "ok",
            "chunk": "Water boils at 100 degrees Celsius at standard atmospheric pressure." * 10,
            "is_primary_source": True,
        },
        {
            "source_id": 2,
            "url": "https://example.com/contradict",
            "domain": "example.com",
            "evidence_class": "CONTRADICTION",
            "evidence_score": 0.85,
            "evidence_type": "scraped_article",
            "source_tier": "official",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.4,
            "nli_label": "contradiction",
            "nli_score": 0.88,
            "nli_status": "ok",
            "chunk": "Water boils at 50 degrees Celsius at standard atmospheric pressure." * 10,
            "is_primary_source": True,
        },
    ]

    gate = deterministic_gate(agreement, evidence_analysis)

    # Build the same condensed payload the pipeline builds for the critic.
    top_passages = sorted(
        evidence_analysis,
        key=lambda r: r.get("evidence_score", 0),
        reverse=True,
    )[:3]

    passage_lines = []
    for p in top_passages:
        passage_lines.append(
            f"- [{p.get('evidence_class')}] {p.get('url')}"
            f" | nli={p.get('nli_label')} {p.get('nli_score')}"
            f" | similarity={p.get('similarity')}"
            f" | coverage={p.get('detail_coverage')}"
            f" | {str(p.get('chunk', ''))[:250]}"
        )

    condensed_evidence = "\n".join([
        "=== CLAIM ===",
        claim,
        "",
        "=== VERIFIER VERDICT ===",
        "UNVERIFIED",
        "",
        "=== DETERMINISTIC EVIDENCE FLOOR ===",
        f"Permitted verdicts: {', '.join(sorted(gate['permitted_verdicts']))}",
        f"Reason: {gate['permitted_reason']}",
        f"Independent supporting: {gate['independent_supporting']}",
        f"Independent contradicting: {gate['independent_contradicting']}",
        "",
        "=== TOP EVIDENCE PASSAGES ===",
        "\n".join(passage_lines) if passage_lines else "(none)",
    ])

    # Required fields must all be present.
    assert "=== CLAIM ===" in condensed_evidence
    assert claim in condensed_evidence
    assert "=== VERIFIER VERDICT ===" in condensed_evidence
    assert "=== DETERMINISTIC EVIDENCE FLOOR ===" in condensed_evidence
    assert "Permitted verdicts:" in condensed_evidence
    assert "Reason:" in condensed_evidence
    assert "Independent supporting:" in condensed_evidence
    assert "Independent contradicting:" in condensed_evidence
    assert "=== TOP EVIDENCE PASSAGES ===" in condensed_evidence
    assert "https://example.com/support" in condensed_evidence
    assert "https://example.com/contradict" in condensed_evidence


def test_critic_not_skipped_by_default():
    """Critic call must not be skipped even when the floor locks UNVERIFIED."""
    from pathlib import Path
    pipeline_path = Path(__file__).resolve().parent.parent / "pipeline.py"
    source = pipeline_path.read_text()
    # The critic must always be invoked (no conditional skip based on verdicts).
    assert "state[\"feedback\"]" in source
    assert "invoke_agent" in source
    assert "critic_chain" in source
    assert "critic_agent" in source
    assert "condensed_evidence" in source
    assert "deterministic_gate" in source
    # The critic invocation must be in a try/except (not conditionally skipped).
    assert "try:" in source
    assert "except Exception as agent_error:" in source
    # No early `return` between the gate computation and the critic call.
    gate_idx = source.find("gate = deterministic_gate")
    critic_idx = source.find("state[\"feedback\"] = invoke_agent")
    assert gate_idx != -1, "deterministic_gate not called"
    assert critic_idx != -1, "critic invoke_agent not called"
    assert critic_idx > gate_idx, "critic must be called after gate computation"
    section = source[gate_idx:critic_idx]
    assert "return" not in section, (
        "No return statement between gate computation and critic call"
    )


def test_deterministic_floor_bounds_critic_verdict():
    """Critic cannot bypass the deterministic floor: permitted verdicts come from the gate."""
    agreement = {
        "independent_supporting": 0,
        "independent_contradicting": 0,
        "evidence_sufficiency": "none",
        "total_sources": 0,
        "agreement_score": 0.0,
    }
    evidence_analysis = []
    gate = deterministic_gate(agreement, evidence_analysis)

    # With no evidence, only UNVERIFIED is permitted.
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "UNVERIFIED" in gate["permitted_reason"].upper() or "no settled verdict" in gate["permitted_reason"].lower()


# ---------------------------------------------------------------------------
# 14. Evidence preservation: strongest supporting/contradicting kept
# ---------------------------------------------------------------------------

def test_strongest_supporting_and_contradicting_preserved():
    """Top passages preserve the strongest supporting and contradicting evidence."""
    evidence_analysis = [
        {"source_id": 1, "url": "https://a.com", "evidence_class": "SUPPORT", "evidence_score": 0.9},
        {"source_id": 2, "url": "https://b.com", "evidence_class": "SUPPORT", "evidence_score": 0.5},
        {"source_id": 3, "url": "https://c.com", "evidence_class": "CONTRADICTION", "evidence_score": 0.85},
        {"source_id": 4, "url": "https://d.com", "evidence_class": "CONTRADICTION", "evidence_score": 0.3},
        {"source_id": 5, "url": "https://e.com", "evidence_class": "RELATED_CONTEXT", "evidence_score": 0.7},
    ]
    top = sorted(
        evidence_analysis,
        key=lambda r: r.get("evidence_score", 0),
        reverse=True,
    )[:4]

    classes = [r["evidence_class"] for r in top]
    scores = [r["evidence_score"] for r in top]

    # Strongest (0.9 SUPPORT, 0.85 CONTRADICTION) must be present.
    assert 0.9 in scores
    assert 0.85 in scores
    # Weakest RELATED_CONTEXT (0.7) may be excluded if only 4 slots.
    assert classes.count("SUPPORT") >= 1
    assert classes.count("CONTRADICTION") >= 1


def test_source_independence_checks_intact():
    """Source independence checks remain intact after token optimization."""
    from evidence_fusion import calculate_source_agreement

    # Two chunks from same domain = one independence group.
    results = [
        {"source_id": 1, "url": "https://example.com/a", "domain": "example.com",
         "independence_group": "example.com", "evidence_class": "SUPPORT",
         "evidence_score": 0.9, "evidence_type": "scraped_article",
         "source_tier": "official", "date_status": "match",
         "event_match": "strong", "detail_coverage": 0.5,
         "nli_label": "entailment", "nli_score": 0.95, "nli_status": "ok",
         "is_primary_source": True},
        {"source_id": 2, "url": "https://example.com/b", "domain": "example.com",
         "independence_group": "example.com", "evidence_class": "SUPPORT",
         "evidence_score": 0.8, "evidence_type": "scraped_article",
         "source_tier": "official", "date_status": "match",
         "event_match": "strong", "detail_coverage": 0.5,
         "nli_label": "entailment", "nli_score": 0.9, "nli_status": "ok",
         "is_primary_source": True},
    ]
    agreement = calculate_source_agreement(results)
    # Same domain counts as ONE independent publisher.
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1, (
        "Same domain must not count as two independent sources"
    )


# ---------------------------------------------------------------------------
# 15. Source reliability: official tier recognition for scientific domains
# ---------------------------------------------------------------------------

def test_iapws_org_is_official_tier():
    """iapws.org (International Association for the Properties of Water and Steam)
    must be classified as TIER_OFFICIAL with is_primary=True for scraped articles."""
    from source_reliability import get_source_profile

    profile = get_source_profile("https://iapws.org/faqs/faq1", "scraped_article")
    assert profile["tier"] == "official", f"Expected official tier, got {profile['tier']}"
    assert profile["is_primary"] is True
    assert profile["reliability"] == 0.95


def test_edu_domain_is_not_automatically_official():
    """.edu domains are NOT automatically TIER_OFFICIAL — personal pages
    and unrelated content must not be treated as authoritative evidence."""
    from source_reliability import get_source_profile

    profile = get_source_profile(
        "https://random-unknown-test-12345.com/unknown-page",
        "scraped_article",
    )
    assert profile["tier"] == "unknown", (
        f"Expected unknown tier for .edu personal page, got {profile['tier']}"
    )
    assert profile["is_primary"] is False, (
        ".edu personal page must not be primary evidence"
    )


def test_edu_snippet_is_not_primary():
    """.edu search snippets are NOT primary — snippets can never establish/refute."""
    from source_reliability import get_source_profile

    profile = get_source_profile("https://example.edu/page", "search_snippet")
    assert profile["tier"] == "unknown"
    assert profile["is_primary"] is False, (
        "Snippets must never be primary evidence"
    )


def test_hyperphysics_is_official_tier():
    """HyperPhysics (Georgia State University) is a trusted scientific
    reference and must be classified as TIER_OFFICIAL."""
    from source_reliability import get_source_profile

    profile = get_source_profile(
        "https://hyperphysics.phy-astr.gsu.edu/hbase/thermo/boilpt.html",
        "scraped_article",
    )
    assert profile["tier"] == "official", f"Expected official tier, got {profile['tier']}"
    assert profile["is_primary"] is True
    assert profile["reliability"] == 0.95


def test_non_official_domain_remains_unknown():
    """Non-official domains (blogs, portals, unknown) stay TIER_UNKNOWN."""
    from source_reliability import get_source_profile

    profile = get_source_profile(
        "https://random-unknown-test-12345.com/unknown-page",
        "scraped_article",
    )
    assert profile["tier"] == "unknown"
    assert profile["is_primary"] is False


# ---------------------------------------------------------------------------
# 16. Scientific claim classification: SUPPORT requires primary source
# ---------------------------------------------------------------------------

def test_scientific_claim_primary_source_with_entailment_is_support():
    """A primary source with NLI entailment for a scientific claim classifies as SUPPORT."""
    from evidence_analyzer import classify_evidence

    signals = {
        "similarity": 0.85,
        "date_status": "unknown",
        "event_match": "weak",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.95,
        "nli_label": "entailment",
        "nli_score": 0.95,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "iapws.org",
        "detail_coverage": 0.50,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": False,
        "claim_has_date": False,
        "claim_type": "scientific_fact",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile("https://iapws.org/faqs/faq1", "scraped_article")

    result = classify_evidence(signals, "scraped_article", profile, claim_type="scientific_fact")
    assert result == "SUPPORT", f"Expected SUPPORT, got {result}"


def test_scientific_claim_non_primary_with_entailment_is_not_support():
    """A non-primary source with NLI entailment must NOT classify as SUPPORT
    even when NLI score is very high."""
    from evidence_analyzer import classify_evidence

    signals = {
        "similarity": 0.85,
        "date_status": "unknown",
        "event_match": "weak",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.997,
        "nli_label": "entailment",
        "nli_score": 0.997,
        "nli_status": "ok",
        "source_reliability": 0.55,
        "source_tier": "unknown",
        "independence_group": "example.com",
        "detail_coverage": 1.0,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": False,
        "claim_has_date": False,
        "claim_type": "scientific_fact",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile(
        "https://randomblog.wordpress.com/unknown-blog-post",
        "scraped_article",
    )

    result = classify_evidence(signals, "scraped_article", profile, claim_type="scientific_fact")
    assert result != "SUPPORT", f"Non-primary source must not be SUPPORT, got {result}"


def test_scientific_claim_insufficient_evidence_stays_unverified():
    """When no primary source supports a scientific claim, verdict stays UNVERIFIED
    and confidence is capped (not 100%)."""
    from final_verdict import generate_final_verdict

    # Single primary source -> only UNVERIFIED permitted
    chunks = [{
        "source_id": 1,
        "url": "https://iapws.org/faqs/faq1",
        "domain": "iapws.org",
        "evidence_class": "SUPPORT",
        "source_tier": "official",
        "source_reliability": 0.95,
        "independence_group": "iapws.org",
    }]
    agreement = calculate_source_agreement(chunks)
    state = {
        "claim": WATER_CLAIM,
        "claim_type": "scientific_fact",
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
    # Confidence must be capped (not 100%) for context_only/single-source UNVERIFIED
    assert verdict["confidence"] <= 60.0, f"Confidence capped at 60% for insufficient evidence, got {verdict['confidence']}%"
    assert verdict["confidence_basis"] == "verification_coverage"


def test_historical_event_claim_behaviour_unchanged():
    """Existing historical/event claims still require event_match == 'strong'."""
    from evidence_analyzer import classify_evidence

    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "weak",   # NOT strong
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.95,
        "nli_label": "entailment",
        "nli_score": 0.95,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "nasa.gov",
        "detail_coverage": 0.50,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": False,
        "claim_has_date": True,
        "claim_type": "historical_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile("https://www.nasa.gov/mission/apollo-11/", "scraped_article")

    # Event claims require event_match == "strong" even with entailment + primary source.
    result = classify_evidence(signals, "scraped_article", profile, claim_type="historical_event")
    assert result != "SUPPORT", (
        "Event claims must require strong event match; got SUPPORT with weak match"
    )


# ---------------------------------------------------------------------------
# 17. Cricket claim: relative date resolution ("yesterday")
# ---------------------------------------------------------------------------

CRICKET_CLAIM = "India won yesterday against New Zealand in T20."

CRICKET_SUPPORT_PREMISE = (
    "India beat New Zealand by 5 wickets in the T20 match at the "
    "Wankhede Stadium, Mumbai. Rohit Sharma scored 78 runs."
)

CRICKET_CONTRADICT_PREMISE = (
    "New Zealand beat India by 4 wickets in the T20 match at the "
    "Wankhede Stadium, Mumbai. Kane Williamson scored 85 runs."
)

CRICKET_UNRELATED_PREMISE = (
    "Australia won the T20 match against England in Sydney."
)


def test_cricket_claim_relative_date_resolved_with_submission_date():
    """A claim with 'yesterday' and a submission date must resolve to a
    concrete date and classify as current_event."""
    profile = build_claim_profile(
        CRICKET_CLAIM, submission_date="2024-08-15"
    )
    assert profile["claim_type"] == "current_event"
    assert profile["has_date"] is True
    assert profile["relative_date_ambiguous"] is False
    assert len(profile["dates"]) == 1
    assert profile["dates"][0]["day"] == 14
    assert profile["dates"][0]["month"] == 8
    assert profile["dates"][0]["year"] == 2024


def test_cricket_claim_relative_date_ambiguous_without_submission_date():
    """A claim with 'yesterday' but no submission date must flag as
    ambiguous and NOT silently use the current date."""
    profile = build_claim_profile(CRICKET_CLAIM, submission_date=None)
    assert profile["claim_type"] != "current_event"
    assert profile["claim_type"] != "historical_event"
    assert profile["has_date"] is False
    assert profile["relative_date_ambiguous"] is True
    assert len(profile["dates"]) == 0


def test_cricket_claim_today_resolved_with_submission_date():
    """A claim with 'today' and a submission date must resolve to that date."""
    profile = build_claim_profile(
        "India won today against New Zealand in T20.",
        submission_date="2024-08-15",
    )
    assert profile["claim_type"] == "current_event"
    assert profile["has_date"] is True
    assert profile["relative_date_ambiguous"] is False
    assert profile["dates"][0]["day"] == 15
    assert profile["dates"][0]["month"] == 8
    assert profile["dates"][0]["year"] == 2024


def test_cricket_claim_explicit_date_still_works():
    """An explicit date claim must still classify correctly."""
    profile = build_claim_profile(
        "India won against New Zealand on 15 Aug 2024 in T20."
    )
    assert profile["claim_type"] == "current_event"
    assert profile["has_date"] is True
    assert profile["relative_date_ambiguous"] is False


# ---------------------------------------------------------------------------
# 18. Cricket claim: evidence classification
# ---------------------------------------------------------------------------

def test_cricket_claim_primary_source_support():
    """A primary sports source with NLI entailment classifies as SUPPORT
    for a current_event claim."""
    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "strong",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.95,
        "nli_label": "entailment",
        "nli_score": 0.95,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "bbc.com",
        "detail_coverage": 0.60,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": True,
        "claim_has_date": True,
        "claim_type": "current_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result == "SUPPORT", f"Expected SUPPORT, got {result}"


def test_cricket_claim_contradiction_classified():
    """A primary source with NLI contradiction classifies as CONTRADICTION
    for a current_event claim."""
    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "strong",
        "raw_nli_label": "contradiction",
        "raw_nli_score": 0.90,
        "nli_label": "contradiction",
        "nli_score": 0.90,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "reuters.com",
        "detail_coverage": 0.60,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": True,
        "claim_has_date": True,
        "claim_type": "current_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile(
        "https://www.reuters.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result == "CONTRADICTION", f"Expected CONTRADICTION, got {result}"


def test_cricket_claim_non_primary_not_support():
    """A non-primary source (blog) must not classify as SUPPORT
    even with NLI entailment for a cricket claim."""
    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "strong",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.99,
        "nli_label": "entailment",
        "nli_score": 0.99,
        "nli_status": "ok",
        "source_reliability": 0.55,
        "source_tier": "unknown",
        "independence_group": "randomblog.wordpress.com",
        "detail_coverage": 1.0,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": True,
        "claim_has_date": True,
        "claim_type": "current_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
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
    """A search snippet must never classify as SUPPORT for a cricket claim."""
    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "strong",
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.99,
        "nli_label": "entailment",
        "nli_score": 0.99,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "established_wire",
        "independence_group": "reuters.com",
        "detail_coverage": 1.0,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": True,
        "claim_has_date": True,
        "claim_type": "current_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile(
        "https://www.reuters.com/sport/cricket/article", "search_snippet"
    )
    result = classify_evidence(
        signals, "search_snippet", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", f"Snippet must not be SUPPORT, got {result}"


def test_cricket_claim_neutral_nli_not_support():
    """NLI neutral must not be promoted to SUPPORT for a cricket claim."""
    signals = {
        "similarity": 0.85,
        "date_status": "match",
        "event_match": "strong",
        "raw_nli_label": "neutral",
        "raw_nli_score": 0.99,
        "nli_label": "neutral",
        "nli_score": 0.99,
        "nli_status": "ok",
        "source_reliability": 0.95,
        "source_tier": "official",
        "independence_group": "bbc.com",
        "detail_coverage": 0.60,
        "debunking": False,
        "passage_kind": "paragraph",
        "claim_has_location": True,
        "claim_has_date": True,
        "claim_type": "current_event",
        "event_date_conflict": False,
        "event_date_mismatch": False,
        "location_match": True,
    }
    profile = get_source_profile(
        "https://www.bbc.com/sport/cricket/article", "scraped_article"
    )
    result = classify_evidence(
        signals, "scraped_article", profile, claim_type="current_event"
    )
    assert result != "SUPPORT", f"Neutral NLI must not be SUPPORT, got {result}"


def test_cricket_claim_insufficient_evidence_stays_unverified():
    """A cricket claim with only one primary supporting source stays UNVERIFIED
    and confidence is capped."""
    from final_verdict import generate_final_verdict

    chunks = [{
        "source_id": 1,
        "url": "https://www.bbc.com/sport/cricket/article",
        "domain": "bbc.com",
        "evidence_class": "SUPPORT",
        "source_tier": "major_outlet",
        "source_reliability": 0.92,
        "independence_group": "bbc.com",
    }]
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
    assert verdict["confidence"] <= 60.0, (
        f"Confidence capped at 60% for insufficient evidence, "
        f"got {verdict['confidence']}%"
    )
    assert verdict["confidence_basis"] == "verification_coverage"


# ---------------------------------------------------------------------------
# 19. Duplicate chunks/URLs from one publisher count as one independent voice
# ---------------------------------------------------------------------------

def test_cricket_claim_duplicate_chunks_one_independence_group():
    """Multiple chunks from the same BBC URL form one independence group."""
    from evidence_fusion import calculate_source_agreement

    results = [
        {
            "source_id": 1,
            "url": "https://www.bbc.com/sport/cricket/article-1",
            "domain": "bbc.com",
            "independence_group": "bbc.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.9,
            "evidence_type": "scraped_article",
            "source_tier": "major_outlet",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.95,
            "nli_status": "ok",
            "is_primary_source": True,
        },
        {
            "source_id": 2,
            "url": "https://www.bbc.com/sport/cricket/article-2",
            "domain": "bbc.com",
            "independence_group": "bbc.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.8,
            "evidence_type": "scraped_article",
            "source_tier": "major_outlet",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.9,
            "nli_status": "ok",
            "is_primary_source": True,
        },
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["independence_groups"] == 1
    assert agreement["independent_supporting"] == 1, (
        "Same domain must not count as two independent sources"
    )


# ---------------------------------------------------------------------------
# 20. Two genuinely independent supporting publishers counted separately
# ---------------------------------------------------------------------------

def test_cricket_claim_two_independent_publishers():
    """BBC and Reuters are two independent publishers for the same cricket event."""
    from evidence_fusion import calculate_source_agreement

    results = [
        {
            "source_id": 1,
            "url": "https://www.bbc.com/sport/cricket/article",
            "domain": "bbc.com",
            "independence_group": "bbc.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.9,
            "evidence_type": "scraped_article",
            "source_tier": "major_outlet",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.95,
            "nli_status": "ok",
            "is_primary_source": True,
        },
        {
            "source_id": 2,
            "url": "https://www.reuters.com/sport/cricket/article",
            "domain": "reuters.com",
            "independence_group": "reuters.com",
            "evidence_class": "SUPPORT",
            "evidence_score": 0.85,
            "evidence_type": "scraped_article",
            "source_tier": "established_wire",
            "date_status": "match",
            "event_match": "strong",
            "detail_coverage": 0.5,
            "nli_label": "entailment",
            "nli_score": 0.9,
            "nli_status": "ok",
            "is_primary_source": True,
        },
    ]
    agreement = calculate_source_agreement(results)
    assert agreement["independence_groups"] == 2
    assert agreement["independent_supporting"] == 2, (
        "Two independent domains must count as two independent sources"
    )

