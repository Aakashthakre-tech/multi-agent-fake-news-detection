"""
tests/test_tavily_search_resilience.py
======================================
Regression tests for:
- Tavily API key configuration and error categorization
- Bounded retries on transient errors vs immediate abort on quota/auth errors
- Candidate parsing and URL normalization/preservation
- Web scraping robustness (handling encoding, JSON-LD fallbacks, no Brotli binary leaks)
- Query planning for historical vs contemporary events
- Deterministic gate verdict policy and confidence semantics
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_planner import plan_queries
from tools import (
    execute_tavily_search,
    get_tavily_client,
    normalize_url,
    extract_domain,
    scrape_url,
    _scrape_once,
)
from pipeline import parse_search_candidates
from final_verdict import deterministic_gate, calculate_final_confidence


def test_tavily_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = get_tavily_client()
    assert client is None

    res, err = execute_tavily_search({"query": "test"})
    assert res == {"results": []}
    assert err is not None
    assert "MISSING_CONFIGURATION" in err


def test_tavily_auth_error_no_retry(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "invalid_key")
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("401 Unauthorized: Invalid API key")

    with patch("tools.get_tavily_client", return_value=mock_client):
        res, err = execute_tavily_search({"query": "test"})
        assert res == {"results": []}
        assert err is not None
        assert "AUTH_ERROR" in err
        # Must not retry auth errors
        assert mock_client.search.call_count == 1


def test_tavily_usage_limit_no_retry(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test_key")
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Usage limit exceeded for current billing cycle")

    with patch("tools.get_tavily_client", return_value=mock_client):
        res, err = execute_tavily_search({"query": "test"})
        assert res == {"results": []}
        assert err is not None
        assert "USAGE_LIMIT_EXCEEDED" in err
        # Must not retry usage/quota exhaustion
        assert mock_client.search.call_count == 1


def test_tavily_transient_retry_success(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test_key")
    mock_client = MagicMock()
    # First attempt fails with 429, second succeeds
    mock_client.search.side_effect = [
        Exception("429 Too Many Requests"),
        {"results": [{"url": "https://example.com/article", "title": "Test"}]},
    ]

    with patch("tools.get_tavily_client", return_value=mock_client), patch("time.sleep"):
        res, err = execute_tavily_search({"query": "test"})
        assert err is None
        assert len(res["results"]) == 1
        assert mock_client.search.call_count == 2


def test_tavily_empty_results_distinguished():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}

    with patch("tools.get_tavily_client", return_value=mock_client):
        res, err = execute_tavily_search({"query": "nonexistent query xyz"})
        assert err is None
        assert res["results"] == []


def test_candidate_parsing_preserves_urls():
    mock_output = (
        "SOURCE 1\n"
        "URL: https://www.nasa.gov/mission/apollo-11/?utm_source=test\n"
        "Domain: nasa.gov\n"
        "Title: Apollo 11 Mission Overview\n"
        "Search Snippet: Apollo 11 landed on the Moon on July 20, 1969.\n"
        "Published Date: 2019-07-20\n"
        "Found Via: exact_claim\n"
        "Source Tier: Tier 1 (Official / Primary)\n"
        "Match Signals: date_match, location_match\n"
        "Retrieval Class: exact_date\n"
        "Content Kind: full_text\n"
        "Search Score: 0.95\n"
        "Reliability: 0.98\n"
        "Search Query: Apollo 11 landing\n"
    )
    candidates = parse_search_candidates(mock_output)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["url"] == "https://www.nasa.gov/mission/apollo-11/?utm_source=test"
    assert c["domain"] == "nasa.gov"
    assert c["title"] == "Apollo 11 Mission Overview"
    assert normalize_url(c["url"]) == "https://www.nasa.gov/mission/apollo-11"


def test_historical_query_planner_general_topic():
    claim = "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969."
    plans = plan_queries(claim)
    assert len(plans) >= 3
    # Historical events should search general topic rather than news
    for p in plans:
        assert p["topic"] == "general"
    
    # Official search should not use modern 'official statement'
    official_plans = [p for p in plans if p["intent"] == "official"]
    assert len(official_plans) == 1
    assert "mission history" in official_plans[0]["query"] or "history" in official_plans[0]["query"]


def test_contemporary_query_planner_preserves_context():
    claim = "Joe Biden withdrew from the 2024 United States presidential race in July 2024."
    plans = plan_queries(claim)
    assert len(plans) >= 3
    # Must preserve presidential race context across queries
    for p in plans:
        if p["intent"] in ("date", "location"):
            assert "presidential race" in p["query"].lower() or "withdrew" in p["query"].lower()


def test_deterministic_gate_policy():
    # 1 source only -> UNVERIFIED
    agreement = {
        "independent_supporting": 1,
        "independent_contradicting": 0,
        "primary_supporting_groups": 1,
        "primary_contradicting_groups": 0,
        "article_supporting_groups": 1,
        "article_contradicting_groups": 0,
        "correction_groups": 0,
        "primary_correction_groups": 0,
    }
    gate = deterministic_gate(agreement, [{"evidence_class": "SUPPORT"}])
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}

    # 2 independent sources with primary article -> REAL permitted
    agreement["independent_supporting"] = 2
    agreement["primary_supporting_groups"] = 1
    agreement["article_supporting_groups"] = 1
    gate = deterministic_gate(agreement, [{"evidence_class": "SUPPORT"}])
    assert "REAL" in gate["permitted_verdicts"]
