"""
tests/test_text_evaluation_set.py
=================================
A deterministic, repeatable evaluation set for the text-verification pipeline.

Everything external is mocked:
  * the search provider returns a FIXED candidate set per claim;
  * the scraper returns a FIXED article body (or a fixed failure);
  * the NLI model is replaced by a stub that returns the label the fixture
    declares for that article, so the suite never depends on a live site or on
    a model's mood.

What is NOT mocked is the part that matters: query planning, claim parsing,
date and event matching, chunking, window construction, classification, source
independence, fusion, the deterministic gate, the verdict and the confidence
semantics.

Scenario list (from the readiness brief):
  1  well-established supported claim
  2  clearly contradicted claim
  3  materially misleading claim
  4  conflicting reliable evidence
  5  recent claim needing fresh sources
  6  vague / unverifiable claim
  7  month-and-year date claim
  8  evidence spanning adjacent paragraphs
  9  ambiguous word retrieving an unrelated event
  10 all article scrapes fail
  11 only snippets available
  12 provider rate limit after the evidence was collected
  13 incomplete / invalid policy payload
  14 duplicate sources from the same publisher
  15 valid claim with a missing agent or critic response
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_fusion import calculate_source_agreement  # noqa: E402
from final_verdict import (  # noqa: E402
    deterministic_gate,
    generate_final_verdict,
)
from research_report import build_research_report  # noqa: E402


# ===========================================================================
# Fixture corpus
# ===========================================================================

def _article(key, title, domain, paragraphs, label, window_label=None):
    return {
        "key": key,
        "title": title,
        "domain": domain,
        "url": f"https://www.{domain}/fixtures/{key}",
        "paragraphs": paragraphs,
        "label": label,
        "window_label": window_label or label,
    }


ARTICLES: Dict[str, Dict[str, Any]] = {
    # 1 / 7 / 8 - a well-established claim, month-and-year date, and evidence
    # that is split across adjacent paragraphs.
    #
    # The wording is taken from the passages a live run actually retrieved, so
    # the similarity and coverage signals behave as they do in production.
    "nasa-apollo11": _article(
        "nasa-apollo11",
        "July 20, 1969: One Giant Leap For Mankind - NASA (nasa-apollo11)",
        "nasa.gov",
        [
            "They leave behind an American flag, a patch honoring the fallen "
            "Apollo 1 crew, and a plaque on one of Eagle's legs. It reads, "
            "“Here men from the planet Earth first set foot upon the moon. "
            "July 1969 A.D. We came in peace and for all mankind.”",
            "Armstrong and Aldrin blast off and dock with Collins in Columbia . "
            "Collins later says that “for the first time,” he “really felt that "
            "we were going to carry this thing off.”",
            "When the lunar module lands at 4:17 p.m EDT, only 30 seconds of fuel "
            "remain. Armstrong radios “Houston, Tranquility Base here. The Eagle "
            "has landed.” Mission control erupts in celebration.",
            "The crew splashes down off Hawaii on July 24. Kennedy's challenge "
            "has been met. Men from Earth have walked on the moon and returned "
            "safely home.",
        ],
        "neutral",          # no single paragraph states the whole claim
        "entailment",       # the adjacent window does
    ),
    "smithsonian-apollo11": _article(
        "smithsonian-apollo11",
        "Apollo 11 (smithsonian-apollo11)",
        "si.edu",
        [
            "The National Air and Space Museum holds approximately 17,000 space "
            "artifacts in its collection, with 400 objects related specifically "
            "to the first successful lunar landing mission, Apollo 11. On July "
            "20, 1969, astronauts Neil Armstrong and Buzz Aldrin landed in the "
            "Lunar Module _Eagle_ and became the first humans to step foot on "
            "the lunar surface.",
        ],
        "entailment",
    ),
    # 2 - a source that directly contradicts the claim.
    "nasa-mars": _article(
        "nasa-mars",
        "Mars Exploration Rovers (nasa-mars)",
        "nasa.gov",
        [
            "No human being has ever landed on Mars. Every mission to the "
            "surface of Mars has been robotic, and the crew that first walked "
            "on another world did so on the Moon in July 1969, not on Mars. "
            "Armstrong's first step was taken on the lunar surface, and the "
            "Apollo 11 mission landed humans on the Moon.",
        ],
        "contradiction",
    ),
    # 2 - a second, independent publisher that contradicts the same claim.
    "smithsonian-mars": _article(
        "smithsonian-mars",
        "Apollo 11 and the Moon landing (smithsonian-mars)",
        "si.edu",
        [
            "Neil Armstrong was the first human to walk on the Moon, and in "
            "July 1969 the Apollo 11 mission landed humans on the Moon. No "
            "human being has ever landed on Mars; every mission to the red "
            "planet has been robotic.",
        ],
        "contradiction",
    ),
    # 3 - a source that corrects a material detail of the claim.
    "nasa-apollo-duration": _article(
        "nasa-apollo-duration",
        "Apollo 11 mission duration (nasa-apollo-duration)",
        "nasa.gov",
        [
            "Armstrong and Aldrin spent about 21.5 hours on the lunar surface "
            "after the Apollo 11 mission landed humans on the Moon in July 1969, "
            "of which roughly two and a half hours were spent outside the lunar "
            "module. They did not stay on the surface for three days; the whole "
            "mission lasted eight days.",
        ],
        "contradiction",
    ),
    # 4 - two credible sources that disagree.
    "source-a-conflict": _article(
        "source-a-conflict",
        "Origin assessment A (source-a-conflict)",
        "who.int",
        [
            "Available evidence indicates that the virus most likely arose from "
            "an animal host, and that COVID-19 was not caused by a laboratory "
            "engineered pathogen. The pandemic was not the result of a "
            "laboratory engineered virus.",
        ],
        "contradiction",
    ),
    "source-b-conflict": _article(
        "source-b-conflict",
        "Origin assessment B (source-b-conflict)",
        "odni.gov",
        [
            "The origin of the virus was not conclusively determined. Some "
            "intelligence agencies believe the lab origin hypothesis cannot be "
            "ruled out, and no specific foreign government or entity was "
            "supported as the cause. Whether COVID-19 was engineered in a "
            "laboratory remains open.",
        ],
        "neutral",
    ),
    # 5 - a recent event with a fresh source.
    "whitehouse-biden": _article(
        "whitehouse-biden",
        "President Biden to Step Down (whitehouse-biden)",
        "whitehouse.gov",
        [
            "On 21 July 2024 President Biden announced that he would not seek a "
            "second term and would withdraw from the 2024 United States "
            "presidential race, saying the party must move forward with a "
            "different nominee.",
        ],
        "entailment",
    ),
    "ap-biden": _article(
        "ap-biden",
        "Biden withdraws from the 2024 race (ap-biden)",
        "apnews.com",
        [
            "President Joe Biden withdrew from the 2024 United States "
            "presidential race on 21 July 2024, saying the party needed to "
            "focus on the nominee who would beat Donald Trump in the election.",
        ],
        "entailment",
    ),
    # 9 - an unrelated event that shares the key word "withdrawal".
    "iraq-withdrawal": _article(
        "iraq-withdrawal",
        "US forces leave Iraq (iraq-withdrawal)",
        "cbc.ca",
        [
            "United States forces completed their withdrawal from Iraq in "
            "September 2026, ending a two-decade military presence in the "
            "country and prompting fears of an Iranian-backed militia vacuum. "
            "The troop withdrawal was announced by the defence ministry in "
            "Baghdad, which said the last American soldiers had left the base "
            "near Baghdad. Officials said the withdrawal of forces from Iraq "
            "was agreed last year.",
        ],
        "neutral",
    ),
    # A vague, non-factual article.
    "opinion-space": _article(
        "opinion-space",
        "Space programmes in perspective (opinion-space)",
        "blog.example.org",
        [
            "Many people believe the Apollo programme was the most successful "
            "space programme in history, though of course this is a matter of "
            "opinion rather than a measurable fact. Whether NASA's Apollo "
            "programme was the most successful space program ever is therefore "
            "not something the record can settle.",
        ],
        "neutral",
    ),
}

# The claims each scenario verifies.
CLAIM_APOLLO = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
CLAIM_APOLLO_MARS = "Neil Armstrong was the first human to walk on Mars in July 1969."
CLAIM_APOLLO_THREE_DAYS = (
    "The Apollo 11 mission landed humans on the Moon in July 1969, and the "
    "astronauts stayed on the lunar surface for three days."
)
CLAIM_COVID = "COVID-19 was engineered in a laboratory and is not of natural origin."
CLAIM_BIDEN = "Joe Biden withdrew from the 2024 United States presidential race in July 2024."
CLAIM_VAGUE = "NASA's Apollo program was the most successful space program in history."


# ===========================================================================
# Deterministic doubles
# ===========================================================================

def _nli_label_for(evidence: str, claim: str) -> str:
    """The stub NLI model: it returns what the fixture declares."""
    for article in ARTICLES.values():
        present = sum(
            1 for paragraph in article["paragraphs"]
            if paragraph in evidence
        )
        if present > 1:
            return article["window_label"]
        if present == 1:
            return article["label"]
        if any(paragraph[:40] in evidence for paragraph in article["paragraphs"]):
            return article["label"]
    return "neutral"


def _stub_nli(claim, evidence):
    label = _nli_label_for(str(evidence), str(claim))
    scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    scores[label] = 0.95
    return {
        "label": label,
        "score": 0.95,
        "scores": scores,
        "truncated": False,
        "status": "ok",
        "reason": "fixture stub",
    }


def _candidate_block(index, article, query, snippet, tier="established_wire",
                     reliability=0.9, content_kind="search_extract",
                     score=0.8):
    return (
        f"SOURCE {index}\n"
        f"Title:\n{article['title']}\n\n"
        f"URL:\n{article['url']}\n\n"
        f"Domain:\n{article['domain']}\n\n"
        f"Published Date:\n2024-07-21T00:00:00Z\n\n"
        f"Search Query:\n{query}\n\n"
        f"Query Intent:\nexact\n\n"
        f"Found Via:\nexact\n\n"
        f"Source Tier:\n{tier} (fixture)\n\n"
        f"Reliability:\n{reliability}\n\n"
        f"Match Signals:\ndate=exact | tier={tier} | location=False | keywords=1 | phrase=0 | factcheck=False\n\n"
        f"Retrieval Class:\nexact_date\n\n"
        f"Content Kind:\n{content_kind} ({len(snippet)} chars)\n\n"
        f"Search Score:\n{score}\n\n"
        f"Search Snippet:\n{snippet}\n"
    )


def _search_stub(plan_map, snippets=None, snippet_only=False):
    """
    Build a `search_with_diagnostics` double.

    `plan_map` maps a query substring to the list of article keys it returns.
    """
    def _search(claim):
        plans = [{"label": "exact", "query": claim, "intent": "exact", "topic": "news"}]
        blocks = []
        index = 0
        for needle, keys in plan_map.items():
            if needle.lower() not in str(claim).lower():
                continue
            for key in keys:
                article = ARTICLES[key]
                index += 1
                if snippet_only:
                    # A real provider excerpt: short, and not article text.
                    snippet = " ".join(article["paragraphs"])[:220]
                else:
                    snippet = (snippets or {}).get(key) or " ".join(article["paragraphs"])[:600]
                blocks.append(_candidate_block(
                    index, article, claim, snippet,
                    content_kind="search_snippet" if snippet_only else "search_extract",
                    tier="major_outlet" if key.startswith(("nasa", "smithsonian")) else "established_wire",
                ))
        return {
            "output": "\n--------------------\n".join(blocks),
            "plans": plans,
            "search_errors": [],
            "candidates": [],
        }
    return _search


def _scrape_stub(key_map, fail_all=False, snippet_only=False):
    """Build a `scrape_url` double."""
    def _scrape(url):
        if fail_all:
            return "SCRAPE_FAILED: HTTP 403 Forbidden after 2 attempts."
        for key, article in ARTICLES.items():
            if key in url or article["url"] == url:
                if snippet_only:
                    return " ".join(article["paragraphs"])[:200]
                body = "\n\n".join(article["paragraphs"])
                return f"TITLE: {article['title']}\n\n{body}"
        return "SCRAPE_FAILED: HTTP 404 Not Found"
    return _scrape


class _ToolProxy:
    """Keeps the LangChain tool interface the pipeline calls (.invoke)."""

    def __init__(self, tool, fn):
        self._tool = tool
        self._fn = fn

    def invoke(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tool, name)


def run_pipeline(claim, plan_map, key_map, agent_verdict="REAL", **kwargs):
    """
    Run the real pipeline with every external dependency mocked.

    The agents are stubbed too: they are an LLM call, and the evaluation set
    must be deterministic and fast. Scenarios 12 and 15 replace them with
    their own failing agents.
    """
    import pipeline as pipeline_mod

    real_scrape = pipeline_mod.scrape_url
    with patch.object(pipeline_mod, "search_with_diagnostics",
                      _search_stub(plan_map,
                                   snippet_only=kwargs.get("snippet_only", False))), \
         patch.object(pipeline_mod, "scrape_url",
                      _ToolProxy(real_scrape, _scrape_stub(key_map, **kwargs))), \
         patch("evidence_analyzer.check_nli", _stub_nli), \
         patch.object(pipeline_mod, "verification_chain", stub_agents(agent_verdict)), \
         patch.object(pipeline_mod, "critic_chain", stub_agents(agent_verdict)):
        return pipeline_mod.run_research_pipeline(claim)


def summarise(state):
    """The same adjudication the API performs."""
    results = state.get("evidence_analysis") or []
    agreement = calculate_source_agreement(results)
    gate = deterministic_gate(agreement, results)
    verdict = generate_final_verdict(
        results, agreement, state.get("fusion_score", 0.0) or 0.0,
        state.get("verification", ""), state.get("feedback", ""),
    )
    return agreement, gate, verdict


def stub_agents(verdict="REAL", confidence="90%"):
    class _Chain:
        def invoke(self, _payload):
            return f"Verdict:\n{verdict}\n\nConfidence:\n{confidence}\n\nReasoning:\nfixture"

    return _Chain()


# ===========================================================================
# 1. Well-established, supported historical claim
# ===========================================================================

def test_scenario_01_well_established_claim_is_real():
    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "REAL"
    assert agreement["independent_supporting"] >= 2
    assert agreement["primary_supporting_groups"] >= 2
    assert verdict["confidence_basis"] == "evidence_strength"
    assert "REAL" in verdict["permitted_verdicts"]


# ===========================================================================
# 2. Clearly contradicted factual claim
# ===========================================================================

def test_scenario_02_contradicted_claim_is_fake():
    state = run_pipeline(
        CLAIM_APOLLO_MARS,
        {"armstrong": ["nasa-mars", "smithsonian-mars"]},
        {},
        agent_verdict="FAKE",
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "FAKE"
    assert agreement["independent_contradicting"] >= 2
    assert agreement["primary_contradicting_groups"] >= 1
    assert "FAKE" in verdict["permitted_verdicts"]


def test_scenario_02b_an_agent_cannot_widen_the_gate():
    """
    The agent asked for REAL; the evidence permits only FAKE. The gate may
    only narrow, so the result is UNVERIFIED - never the agent's REAL.
    """
    state = run_pipeline(
        CLAIM_APOLLO_MARS,
        {"armstrong": ["nasa-mars", "smithsonian-mars"]},
        {},
        agent_verdict="REAL",
    )
    agreement, gate, verdict = summarise(state)

    assert "REAL" not in gate["permitted_verdicts"]
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["verdict_overridden_by_evidence"] is True
    assert "REAL" not in verdict["permitted_verdicts"]


# ===========================================================================
# 3. Materially misleading claim
# ===========================================================================

def test_scenario_03_misleading_claim_never_returns_a_clean_real():
    state = run_pipeline(
        CLAIM_APOLLO_THREE_DAYS,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11", "nasa-apollo-duration"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    # The event is real but a material detail is wrong, so a clean REAL is not
    # an acceptable outcome.
    assert verdict["verdict"] != "REAL", verdict["reasoning"]
    assert verdict["verdict"] in ("MISLEADING", "FAKE", "UNVERIFIED")
    if verdict["verdict"] == "MISLEADING":
        assert agreement["correction_groups"] >= 1
        assert agreement["primary_correction_groups"] >= 1


# ===========================================================================
# 4. Conflicting reliable evidence
# ===========================================================================

def test_scenario_04_conflicting_evidence_is_unverified():
    state = run_pipeline(
        CLAIM_COVID,
        {"covid": ["source-a-conflict", "source-b-conflict"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["verdict"] not in ("REAL", "FAKE")
    assert verdict["confidence_basis"] == "verification_coverage"


# ===========================================================================
# 5. Recent claim needing fresh sources
# ===========================================================================

def test_scenario_05_recent_claim_is_real_with_fresh_sources():
    state = run_pipeline(
        CLAIM_BIDEN,
        {"biden": ["whitehouse-biden", "ap-biden"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "REAL"
    assert agreement["independent_supporting"] >= 2


# ===========================================================================
# 6. Vague or unverifiable claim
# ===========================================================================

def test_scenario_06_vague_claim_is_unverified():
    state = run_pipeline(
        CLAIM_VAGUE,
        {"apollo": ["opinion-space"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["verdict"] not in ("REAL", "FAKE")
    assert verdict["confidence_basis"] == "verification_coverage"


# ===========================================================================
# 7. Claim with a month-and-year date
# ===========================================================================

def test_scenario_07_month_and_year_date_claim_matches_evidence():
    from claim_profile import build_claim_profile

    profile = build_claim_profile(CLAIM_APOLLO)
    assert profile["primary_date"]["precision"] == "month"
    assert profile["primary_date"]["month"] == 7
    assert profile["primary_date"]["year"] == 1969
    # A month must not be mistaken for a place.
    assert profile["locations"] == []

    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
        {},
    )
    agreement, gate, verdict = summarise(state)
    assert verdict["verdict"] == "REAL"


# ===========================================================================
# 8. Evidence spanning adjacent paragraphs
# ===========================================================================

def test_scenario_08_adjacent_paragraph_window_supports_the_claim():
    """
    The NASA fixture states the claim across adjacent paragraphs: no single
    paragraph entails it, the window does.
    """
    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11"]},
        {},
    )
    results = state.get("evidence_analysis") or []

    windows = [r for r in results if r.get("passage_kind") == "window"]
    assert windows, "a window candidate must have been produced"
    assert any(r["evidence_class"] == "SUPPORT" for r in windows), (
        "the window that states the claim must be classified as support"
    )
    # The window must contain only the article's own text.
    article = ARTICLES["nasa-apollo11"]
    allowed = set(article["paragraphs"]) | {f"TITLE: {article['title']}"}
    for record in windows:
        for paragraph in record["chunk"].split("\n\n"):
            assert paragraph.strip() in allowed, (
                "a window may only contain the article's own paragraphs"
            )


# ===========================================================================
# 9. Ambiguous word retrieving an unrelated event
# ===========================================================================

def test_scenario_09_unrelated_event_is_not_support():
    """
    "withdrawal" also matches reporting about US troops leaving Iraq. That
    article must never become evidence for the claim about the 2024 race.
    """
    state = run_pipeline(
        CLAIM_BIDEN,
        {"biden": ["iraq-withdrawal", "whitehouse-biden", "ap-biden"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    # The unrelated article must appear in the source ledger...
    ledger = [s.get("url") for s in (state.get("source_data") or [])]
    assert any("iraq-withdrawal" in str(url) for url in ledger), (
        "the unrelated article must appear in the retrieved source ledger"
    )

    # ...and it must never contribute supporting evidence.
    iraq_records = [
        r for r in (state.get("evidence_analysis") or [])
        if "iraq-withdrawal" in str(r.get("url"))
    ]
    assert all(
        r["evidence_class"] not in ("SUPPORT", "CONTRADICTION")
        for r in iraq_records
    ), "an article about a different withdrawal must not support or refute the claim"

    iraq_groups = [
        d for d in agreement["source_decisions"]
        if d["domain"] == "cbc.ca"
    ]
    assert all(d["decision"] != "entailment" for d in iraq_groups)


# ===========================================================================
# 10. All article scrapes fail
# ===========================================================================

def test_scenario_10_all_scrapes_fail_is_unverified_not_fabricated():
    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
        {},
        fail_all=True,
    )
    agreement, gate, verdict = summarise(state)

    assert verdict["verdict"] == "UNVERIFIED"
    assert agreement["independent_supporting"] == 0
    assert agreement["independent_contradicting"] == 0

    # A failed scrape may still leave the provider's snippet in the ledger,
    # but a snippet is fallback evidence: it can never establish a claim.
    for record in state.get("evidence_analysis") or []:
        assert record["evidence_class"] not in ("SUPPORT", "CONTRADICTION"), (
            "snippet fallback must not establish or refute a claim"
        )

    # The failure must be reported, not hidden.
    failures = state.get("scrape_failures") or []
    assert failures
    assert "SCRAPE_FAILED" in failures[0]["reason"]


# ===========================================================================
# 11. Only snippets are available
# ===========================================================================

def test_scenario_11_snippet_only_evidence_cannot_establish_a_claim():
    """
    Only provider snippets are available: every scrape fails, so the pipeline
    falls back to the search excerpts. A snippet is fallback evidence and can
    neither establish nor refute a claim.
    """
    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
        {},
        fail_all=True,
        snippet_only=True,
    )
    agreement, gate, verdict = summarise(state)

    assert agreement["snippet_only_sources"] >= 1
    assert agreement["article_supporting_groups"] == 0
    assert agreement["article_contradicting_groups"] == 0
    assert agreement["independent_supporting"] == 0
    assert verdict["verdict"] == "UNVERIFIED"
    assert "REAL" not in verdict["permitted_verdicts"]
    assert "FAKE" not in verdict["permitted_verdicts"]


# ===========================================================================
# 12. Provider rate limit after the evidence was collected
# ===========================================================================

def test_scenario_12_agent_rate_limit_preserves_the_evidence():
    """
    A provider rate limit in the agent stage must not discard the retrieval,
    scraping and NLI work that already completed, and must not be presented
    as a finished verification.
    """
    import pipeline as pipeline_mod
    from groq import RateLimitError

    class _RateLimited:
        def invoke(self, _payload):
            raise RateLimitError(
                "Rate limit reached for model `qwen/qwen3.8-27b` in organization "
                "`org_01EXAMPLEPLACEHOLDER00` service tier `on_demand` on "
                "input tokens per minute (ITPM): Limit 7000, Used 6902. "
                "Please try again in 31.86s."
            )

    with patch.object(pipeline_mod, "search_with_diagnostics",
                      _search_stub({"apollo": ["nasa-apollo11", "smithsonian-apollo11"]})), \
         patch.object(pipeline_mod, "scrape_url",
                      _ToolProxy(pipeline_mod.scrape_url, _scrape_stub({}))), \
         patch("evidence_analyzer.check_nli", _stub_nli), \
         patch.object(pipeline_mod, "verification_chain", _RateLimited()), \
         patch.object(pipeline_mod, "critic_chain", _RateLimited()):
        # The retry must be bounded, so the test does not wait for it.
        with patch("agents.time.sleep", lambda _s: None):
            state = pipeline_mod.run_research_pipeline(CLAIM_APOLLO)

    # The evidence work survived.
    assert state.get("evidence_analysis"), "the collected evidence must be preserved"
    assert state.get("scrape_failures") is not None

    # The failure is recorded, and the run is marked partial.
    errors = state.get("agent_errors") or []
    assert errors, "the failed stage must be recorded"
    assert {e["stage"] for e in errors} == {"verification_agent", "critic_agent"}
    assert state.get("verification_status") == "failed"
    assert state.get("feedback_status") == "failed"

    # The provider's organisation id must not leak.
    blob = str(errors)
    assert "org_01EXAMPLEPLACEHOLDER00" not in blob

    # The report says the result is partial.
    report = build_research_report(state, "text", CLAIM_APOLLO)
    status = report["executive_summary"]["agent_status"]
    assert status["partial"] is True
    assert status["verification_agent"] == "failed"
    assert any("PARTIAL" in lim for lim in report["executive_summary"]["limitations"])


# ===========================================================================
# 13. Incomplete or invalid policy payload
# ===========================================================================

def test_scenario_13_invalid_policy_payload_fails_safe():
    results = [
        {
            "source_id": 1, "url": "https://www.nasa.gov/a", "domain": "nasa.gov",
            "title": "t", "evidence_type": "scraped_article", "chunk": "c",
            "similarity": 0.8, "nli_label": "entailment", "nli_score": 0.95,
            "raw_nli_label": "entailment", "raw_nli_score": 0.95,
            "source_reliability": 0.9, "source_tier": "established_wire",
            "independence_group": "nasa.gov", "is_primary_source": True,
            "evidence_score": 0.9, "evidence_class": "SUPPORT",
            "date_status": "match", "event_match": "strong",
            "location_match": False, "detail_coverage": 0.9,
            "nli_status": "ok", "debunking": False,
        }
    ]

    # A payload missing a required counter must abstain, not decide.
    gate = deterministic_gate({"independent_supporting": 3}, results)
    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}

    # A malformed counter must abstain too.
    gate = deterministic_gate(
        {
            "independent_supporting": 3, "independent_contradicting": 0,
            "primary_supporting_groups": "two", "primary_contradicting_groups": 0,
            "article_supporting_groups": 2, "article_contradicting_groups": 0,
            "correction_groups": 0, "primary_correction_groups": 0,
        },
        results,
    )
    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}


# ===========================================================================
# 14. Duplicate sources from the same publisher
# ===========================================================================

def test_scenario_14_duplicate_publisher_does_not_inflate_counts():
    state = run_pipeline(
        CLAIM_APOLLO,
        {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
        {},
    )
    agreement, gate, verdict = summarise(state)

    # The two articles are different publishers, so two is right. A third
    # article from the same publisher must not add a third vote.
    from evidence_fusion import calculate_source_agreement as _csa

    duplicated = list(state["evidence_analysis"]) + [
        dict(record, source_id=99)
        for record in state["evidence_analysis"]
        if record.get("domain") == "nasa.gov"
    ]
    agreement2 = _csa(duplicated)
    assert agreement2["independence_groups"] == agreement["independence_groups"]
    assert agreement2["independent_supporting"] == agreement["independent_supporting"]


# ===========================================================================
# 15. Valid claim with a missing agent or critic response
# ===========================================================================

def test_scenario_15_missing_agent_response_is_reported_as_partial():
    import pipeline as pipeline_mod

    class _Unavailable:
        def invoke(self, _payload):
            raise RuntimeError("agent endpoint unavailable")

    with patch.object(pipeline_mod, "search_with_diagnostics",
                      _search_stub({"apollo": ["nasa-apollo11", "smithsonian-apollo11"]})), \
         patch.object(pipeline_mod, "scrape_url",
                      _ToolProxy(pipeline_mod.scrape_url, _scrape_stub({}))), \
         patch("evidence_analyzer.check_nli", _stub_nli), \
         patch.object(pipeline_mod, "verification_chain", _Unavailable()), \
         patch.object(pipeline_mod, "critic_chain", _Unavailable()):
        state = pipeline_mod.run_research_pipeline(CLAIM_APOLLO)

    agreement, gate, verdict = summarise(state)

    # The evidence is intact and the gate still permits a settled verdict...
    assert state.get("evidence_analysis")
    assert agreement["independent_supporting"] >= 2
    assert "REAL" in gate["permitted_verdicts"]

    # ...but no agent label was available, so no settled verdict was accepted
    # and the result is reported as partial rather than as a verification.
    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["verdict_overridden_by_evidence"] is False

    report = build_research_report(state, "text", CLAIM_APOLLO)
    status = report["executive_summary"]["agent_status"]
    assert status["partial"] is True
    assert status["verification_agent"] == "failed"
    assert status["critic_agent"] == "failed"


# ===========================================================================
# 16. Query planning must preserve the claim's distinguishing qualifiers
# ===========================================================================

def test_scenario_16_every_planned_query_keeps_the_qualifier():
    """
    A query built from a location plus a couple of keywords loses the claim's
    meaning: "United States joe biden withdrew" matches reporting about US
    troops leaving Iraq. Every planned query must carry the event anchor.
    """
    from query_planner import plan_queries

    plans = plan_queries(CLAIM_BIDEN)
    assert plans

    for plan in plans:
        query = plan["query"].lower()
        if plan["label"] == "exact_claim":
            # The verbatim claim already carries everything.
            continue
        if plan["label"] == "year_recall":
            # The year-level recall query is deliberately broad, but it must
            # still carry the event phrase.
            assert "presidential race" in query, plan["query"]
            continue
        assert "presidential race" in query, (
            f"query {plan['label']} lost the claim's qualifier: {plan['query']}"
        )


def test_scenario_16b_the_event_anchor_carries_the_distinguishing_terms():
    from query_planner import build_event_anchor
    from claim_profile import build_claim_profile

    profile = build_claim_profile(CLAIM_BIDEN)
    anchor = build_event_anchor(profile).lower()
    assert "joe biden withdrew" in anchor
    assert "presidential race" in anchor


def test_scenario_16c_the_precision_retry_keeps_the_qualifier():
    """
    The retriever's precision retry is a second place where the claim's
    qualifiers used to be dropped.
    """
    import inspect

    import tools as tools_mod

    source = inspect.getsource(tools_mod.retrieve)
    assert "build_event_anchor" in source, (
        "the precision retry must use the shared event anchor"
    )


def test_scenario_16d_a_different_event_keeps_its_own_qualifier():
    """The anchor must not be a fixed phrase - each claim keeps its own."""
    from query_planner import build_event_anchor
    from claim_profile import build_claim_profile

    fire = build_event_anchor(build_claim_profile(
        "A major fire broke out at a commercial building in New Delhi, "
        "India on 12 Aug 2024."
    )).lower()
    assert "commercial building" in fire
    assert "fire" in fire


# ===========================================================================
# 17. The NLI cache must not change any decision
# ===========================================================================

def test_scenario_17_the_nli_cache_returns_identical_decisions():
    import nli_checker

    nli_checker._NLI_CACHE.clear()
    claim = "Water boils at 100 degrees Celsius at sea level."
    evidence = "At standard atmospheric pressure, water boils at 100 degrees Celsius."

    first = nli_checker.check_nli(claim, evidence)
    second = nli_checker.check_nli(claim, evidence)

    assert first["label"] == second["label"]
    assert first["score"] == second["score"]
    assert first["scores"] == second["scores"]
    assert first["status"] == second["status"]


def test_scenario_17b_the_cache_is_bounded():
    import nli_checker

    assert nli_checker._NLI_CACHE_MAX > 0
    assert nli_checker._NLI_CACHE_MAX <= 100000


def test_scenario_17c_the_label_aliases_match_the_model_configuration():
    """
    The generic label_N names are positional. They must agree with the
    model's own id2label, or every label would be inverted.
    """
    from nli_checker import MODEL_NAME, _label_aliases

    aliases = _label_aliases()
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_NAME)
        id2label = dict(getattr(config, "id2label", {}) or {})
    except Exception:
        pytest.skip("model configuration not reachable")

    if not id2label:
        pytest.skip("model configuration exposes no id2label")

    for index, label in id2label.items():
        canonical = aliases.get(f"label_{index}")
        assert canonical is not None, f"label_{index} is not mapped"
        expected = {
            "contradiction": "contradiction",
            "entailment": "entailment",
            "neutral": "neutral",
        }.get(str(label).lower())
        if expected:
            assert canonical == expected, (
                f"label_{index} maps to {canonical!r} but the model says {label!r}"
            )
