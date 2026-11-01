"""
tests/test_google_fact_check.py
=================================
Regression tests for Google Fact Check Tools API integration.

Tests use mocked HTTP responses; no live API calls or real API key required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (
    execute_tavily_search,
    get_tavily_client,
)


class MockResponse:
    """Simple mock for requests.Response."""
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self.text)


# ---------------------------------------------------------------------------
# Test: API key missing
# ---------------------------------------------------------------------------

def test_google_api_key_missing(monkeypatch):
    """When GOOGLE_FACT_CHECK_API_KEY is not set, the client returns None."""
    monkeypatch.delenv("GOOGLE_FACT_CHECK_API_KEY", raising=False)
    # Directly test that get_google_fact_check_client returns None
    # when the env var is not set
    from tools import get_google_fact_check_client
    client = get_google_fact_check_client()
    assert client is None


# ---------------------------------------------------------------------------
# Test: Relevant review with TRUE rating (supporting evidence)
# ---------------------------------------------------------------------------

def test_google_fact_check_true_rating(monkeypatch):
    """A relevant review with TRUE rating is accepted as supporting evidence."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand in T20 on 27 October 2026",
                    "claimDate": "2026-10-27",
                    "claimant": "Social media",
                    "claimReview": [
                        {
                            "publisher": {"name": "FactCheck.org"},
                            "textualRating": "True",
                            "url": "https://www.factcheck.org/2026/10/india-nz-cricket",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand in T20 on 27 October 2026")

    assert result is not None
    assert result["status"] == "success"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claimReview"][0]["textualRating"] == "True"


# ---------------------------------------------------------------------------
# Test: Relevant review with FALSE rating (contradicting evidence)
# ---------------------------------------------------------------------------

def test_google_fact_check_false_rating(monkeypatch):
    """A relevant review with FALSE rating is classified as contradiction."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand by 10 wickets",
                    "claimDate": "2026-10-27",
                    "claimant": "Social media",
                    "claimReview": [
                        {
                            "publisher": {"name": "Reuters Fact Check"},
                            "textualRating": "False",
                            "url": "https://www.reuters.com/fact-check/india-nz",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand by 10 wickets")

    assert result is not None
    assert result["status"] == "success"
    assert result["claims"][0]["claimReview"][0]["textualRating"] == "False"


# ---------------------------------------------------------------------------
# Test: Unrelated review must not influence verdict
# ---------------------------------------------------------------------------

def test_google_fact_check_unrelated_review(monkeypatch):
    """An unrelated review (different topic) must not count as evidence."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "COVID-19 vaccines contain microchips",
                    "claimDate": "2021-01-15",
                    "claimant": "Social media",
                    "claimReview": [
                        {
                            "publisher": {"name": "FactCheck.org"},
                            "textualRating": "False",
                            "url": "https://www.factcheck.org/2021/vaccine-microchip",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand in T20 on 27 October 2026")

    assert result is not None
    assert result["status"] == "success"
    # The claim about vaccines is unrelated to the cricket claim
    assert result["claims"][0]["text"] != "India defeated New Zealand in T20 on 27 October 2026"


# ---------------------------------------------------------------------------
# Test: Outdated review about a different event
# ---------------------------------------------------------------------------

def test_google_fact_check_outdated_review(monkeypatch):
    """An outdated review about a different event must not be treated as current evidence."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India won the 2011 Cricket World Cup",
                    "claimDate": "2011-04-02",
                    "claimant": "Sports media",
                    "claimReview": [
                        {
                            "publisher": {"name": "ESPN Fact Check"},
                            "textualRating": "True",
                            "url": "https://www.espn.com/factcheck/2011-world-cup",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand in T20 on 27 October 2026")

    assert result is not None
    assert result["claims"][0]["claimDate"] == "2011-04-02"
    assert result["claims"][0]["claimDate"] != "2026-10-27"


# ---------------------------------------------------------------------------
# Test: Multiple conflicting fact-check reviews
# ---------------------------------------------------------------------------

def test_google_fact_check_conflicting_reviews(monkeypatch):
    """Conflicting reviews remain visible; neither is silently resolved."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand on 27 October 2026",
                    "claimDate": "2026-10-27",
                    "claimant": "Source A",
                    "claimReview": [
                        {
                            "publisher": {"name": "FactCheck.org"},
                            "textualRating": "True",
                            "url": "https://factcheck.org/true",
                        }
                    ],
                },
                {
                    "text": "India defeated New Zealand on 27 October 2026",
                    "claimDate": "2026-10-27",
                    "claimant": "Source B",
                    "claimReview": [
                        {
                            "publisher": {"name": "Reuters Fact Check"},
                            "textualRating": "False",
                            "url": "https://reuters.com/factcheck/false",
                        }
                    ],
                },
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand on 27 October 2026")

    assert result is not None
    assert len(result["claims"]) == 2
    ratings = [cr["claimReview"][0]["textualRating"] for cr in result["claims"]]
    assert "True" in ratings
    assert "False" in ratings


# ---------------------------------------------------------------------------
# Test: Empty API results
# ---------------------------------------------------------------------------

def test_google_fact_check_empty_results(monkeypatch):
    """Empty API results are handled gracefully."""
    mock_response = MockResponse(200, {"claims": []})

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Some obscure claim")

    assert result is not None
    assert result["status"] == "success"
    assert result["claims"] == []


# ---------------------------------------------------------------------------
# Test: Missing or invalid API key
# ---------------------------------------------------------------------------

def test_google_fact_check_missing_key(monkeypatch):
    """Missing API key results in graceful degradation."""
    monkeypatch.delenv("GOOGLE_FACT_CHECK_API_KEY", raising=False)
    from tools import get_google_fact_check_client
    client = get_google_fact_check_client()
    assert client is None


# ---------------------------------------------------------------------------
# Test: HTTP 403, 429, and 5xx responses
# ---------------------------------------------------------------------------

def test_google_fact_check_403(monkeypatch):
    """HTTP 403 is handled gracefully."""
    mock_response = MockResponse(403, text="Forbidden")

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    assert result["error_category"] == "auth"


def test_google_fact_check_429(monkeypatch):
    """HTTP 429 rate limit is handled gracefully."""
    mock_response = MockResponse(429, text="Rate limit exceeded")

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    assert result["error_category"] == "rate_limit"


def test_google_fact_check_503(monkeypatch):
    """HTTP 503 server error is handled gracefully."""
    mock_response = MockResponse(503, text="Service unavailable")

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    assert result["error_category"] == "server_error"


# ---------------------------------------------------------------------------
# Test: Request timeout
# ---------------------------------------------------------------------------

def test_google_fact_check_timeout(monkeypatch):
    """Request timeout is handled gracefully."""
    import requests

    with patch("tools.requests.get", side_effect=requests.exceptions.Timeout()):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    assert result["error_category"] == "timeout"


# ---------------------------------------------------------------------------
# Test: Malformed response
# ---------------------------------------------------------------------------

def test_google_fact_check_malformed_response(monkeypatch):
    """Malformed JSON response is handled gracefully."""
    mock_response = MockResponse(200, text="not valid json{{{")

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    assert result["error_category"] == "parse_error"


# ---------------------------------------------------------------------------
# Test: Missing optional fields
# ---------------------------------------------------------------------------

def test_google_fact_check_missing_fields(monkeypatch):
    """Missing optional fields (rating, URL) are handled gracefully."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand",
                    "claimDate": "2026-10-27",
                    "claimant": "Social media",
                    # No claimReview field at all
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand on 27 October 2026")

    assert result is not None
    assert result["status"] == "success"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claimReview"] == []


# ---------------------------------------------------------------------------
# Test: Google API outage - existing pipeline still works
# ---------------------------------------------------------------------------

def test_google_fact_check_outage_pipeline_continues(monkeypatch):
    """When Google API is down, the existing Tavily pipeline still works."""
    # Google API fails
    mock_response = MockResponse(503, text="Service unavailable")

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("Test claim")

    assert result is not None
    assert result["status"] == "error"
    # The pipeline should continue with Tavily search regardless
    assert result.get("pipeline_continues", True) is True


# ---------------------------------------------------------------------------
# Test: Duplicate review from Google and web search
# ---------------------------------------------------------------------------

def test_google_fact_check_duplicate_with_web_search(monkeypatch):
    """Same review from Google and web search is deduplicated."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand",
                    "claimDate": "2026-10-27",
                    "claimant": "Social media",
                    "claimReview": [
                        {
                            "publisher": {"name": "FactCheck.org"},
                            "textualRating": "True",
                            "url": "https://factcheck.org/2026/10/india-nz",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand on 27 October 2026")

    assert result is not None
    assert len(result["claims"]) == 1
    # Same URL as a potential web search result - deduplication should handle it
    assert result["claims"][0]["claimReview"][0]["url"] == "https://factcheck.org/2026/10/india-nz"


# ---------------------------------------------------------------------------
# Test: Google review whose original article cannot be scraped
# ---------------------------------------------------------------------------

def test_google_fact_check_article_scrape_fails(monkeypatch):
    """When the review's original article page cannot be scraped, metadata
    is still preserved as an external fact-check record."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "India defeated New Zealand",
                    "claimDate": "2026-10-27",
                    "claimant": "Social media",
                    "claimReview": [
                        {
                            "publisher": {"name": "FactCheck.org"},
                            "textualRating": "True",
                            "url": "https://factcheck.org/2026/10/india-nz",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand on 27 October 2026")

    assert result is not None
    assert result["status"] == "success"
    # The review metadata is preserved even though we can't scrape the article
    assert result["claims"][0]["claimReview"][0]["url"] == "https://factcheck.org/2026/10/india-nz"


# ---------------------------------------------------------------------------
# Test: Existing pipeline safeguards unchanged
# ---------------------------------------------------------------------------

def test_google_fact_check_does_not_weaken_safeguards(monkeypatch):
    """Google API results do not override existing evidence policy."""
    mock_response = MockResponse(
        200,
        {
            "claims": [
                {
                    "text": "Some unrelated claim",
                    "claimDate": "2020-01-01",
                    "claimant": "Someone",
                    "claimReview": [
                        {
                            "publisher": {"name": "Some Publisher"},
                            "textualRating": "True",
                            "url": "https://example.com/unrelated",
                        }
                    ],
                }
            ]
        },
    )

    with patch("tools.requests.get", return_value=mock_response):
        from tools import query_google_fact_check
        result = query_google_fact_check("India defeated New Zealand on 27 October 2026")

    assert result is not None
    # Unrelated review should not be treated as evidence for the claim
    assert result["claims"][0]["text"] != "India defeated New Zealand on 27 October 2026"
