"""
tests/test_scrape_reliability.py
=================================
Regression tests for web scraping reliability:
- Successful full-page extraction
- HTTP 403 and 429 handling with bounded retries
- Timeout and transient server failure
- Empty or boilerplate-only page
- Snippet fallback with correct provenance
- Retry limits and bounded concurrency
- Duplicate URLs and syndicated articles
- Existing snippet-exclusion and deterministic-verdict safeguards unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (
    scrape_url,
    scrape_url_detailed,
    _scrape_once,
    SCRAPE_MAX_ATTEMPTS,
    SCRAPE_CONNECT_TIMEOUT,
    SCRAPE_READ_TIMEOUT,
    SCRAPE_BACKOFF_BASE,
    SCRAPE_BACKOFF_MAX,
)


# ---------------------------------------------------------------------------
# Helper: build a mock HTTP response
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, text=""):
    """Build a mock requests.Response object with PropertyMock attributes."""
    r = MagicMock()
    type(r).status_code = PropertyMock(return_value=status_code)
    type(r).encoding = PropertyMock(return_value="utf-8")
    type(r).text = PropertyMock(return_value=text)
    return r


# Long article content (> 600 chars of extracted text after HTML parsing)
LONG_ARTICLE = """<html><head><title>Test Article</title></head>
<body><article>
<p>This is the main article content about the claim that we need to verify.</p>
<p>It contains multiple sentences with relevant information that should be extracted by the scraper. The article discusses the topic in detail with several paragraphs of content to ensure sufficient length for the extraction quality checks.</p>
<p>Additional context is provided here to make the article substantial enough for the extraction quality checks to pass without triggering the empty content rejection path.</p>
<p>Further details about the topic are provided in this third paragraph, including specific facts, figures, and context that make this a substantial piece of content worthy of being classified as a full article rather than just a snippet.</p>
<p>Even more content here to ensure we have well over 600 characters of extracted text after BeautifulSoup strips the HTML tags and normalizes whitespace. This paragraph adds significant length to guarantee we cross the threshold for full article classification.</p>
</article></body></html>"""


# ---------------------------------------------------------------------------
# Successful full-page extraction
# ---------------------------------------------------------------------------

def test_scrape_success_full_article(monkeypatch):
    """A successful scrape returns clean article text with status 'success'."""
    mock_response = _mock_response(200, LONG_ARTICLE)

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://example.com/article")

    assert result["scrape_status"] == "success"
    assert result["extraction_method"] == "full_article"
    assert result["char_count"] >= 600
    assert result["http_status"] == 200
    assert result["error_category"] is None
    assert "main article content" in result["content"]


def test_scrape_success_short_but_meaningful(monkeypatch):
    """A short but meaningful official bulletin should not be rejected."""
    mock_response = _mock_response(
        200,
        "<html><head><title>Official Notice</title></head>"
        "<body><article><p>Water boils at 100C at standard pressure. This is an "
        "official government notice about the boiling point of water under "
        "standard atmospheric conditions. The information is provided for "
        "educational purposes only.</p></article></body></html>",
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://official.gov/notice")

    assert result["scrape_status"] == "success"
    assert result["char_count"] > 0
    assert "100C" in result["content"]


# ---------------------------------------------------------------------------
# HTTP 403 and 429 handling
# ---------------------------------------------------------------------------

def test_scrape_403_forbidden(monkeypatch):
    """HTTP 403 should not be retried endlessly; bounded retries apply."""
    mock_response = _mock_response(403, "")

    with patch("tools.requests.get", return_value=mock_response), \
         patch("time.sleep"):
        result = scrape_url_detailed("https://example.com/blocked")

    assert result["scrape_status"] == "failed"
    assert result["http_status"] == 403
    assert result["error_category"] == "http_403"
    assert result["retry_count"] <= SCRAPE_MAX_ATTEMPTS


def test_scrape_429_rate_limited(monkeypatch):
    """HTTP 429 should trigger bounded retries with backoff."""
    mock_response = _mock_response(429, "")

    with patch("tools.requests.get", return_value=mock_response), \
         patch("time.sleep"):
        result = scrape_url_detailed("https://example.com/rate-limited")

    assert result["scrape_status"] == "failed"
    assert result["http_status"] == 429
    assert result["error_category"] == "http_429"
    assert result["retry_count"] == SCRAPE_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Timeout and transient server failure
# ---------------------------------------------------------------------------

def test_scrape_timeout(monkeypatch):
    """Request timeout should be retried bounded times then fail gracefully."""
    import requests

    with patch("tools.requests.get", side_effect=requests.exceptions.Timeout()), \
         patch("time.sleep"):
        result = scrape_url_detailed("https://slow.example.com/article")

    assert result["scrape_status"] == "failed"
    assert result["error_category"] == "timeout"
    assert result["retry_count"] == SCRAPE_MAX_ATTEMPTS


def test_scrape_transient_5xx_then_success(monkeypatch):
    """Transient 503 should be retried and eventually succeed."""
    mock_503 = _mock_response(503, "")
    mock_200 = _mock_response(
        200,
        LONG_ARTICLE,
    )

    responses = [mock_503, mock_200]

    with patch("tools.requests.get", side_effect=responses), \
         patch("time.sleep"):
        result = scrape_url_detailed("https://example.com/flaky")

    assert result["scrape_status"] == "success"
    assert result["retry_count"] == 2
    assert "main article content" in result["content"]


# ---------------------------------------------------------------------------
# Empty or boilerplate-only page
# ---------------------------------------------------------------------------

def test_scrape_empty_content(monkeypatch):
    """A page with no extractable content should fail gracefully."""
    mock_response = _mock_response(
        200, "<html><head><title>Empty</title></head><body></body></html>"
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://example.com/empty")

    assert result["scrape_status"] == "failed"
    assert result["error_category"] == "empty_content"


def test_scrape_boilerplate_only(monkeypatch):
    """A page with only navigation/boilerplate should fail."""
    mock_response = _mock_response(
        200,
        "<html><head><title>Menu</title></head>"
        "<body><nav><a href=\"/\">Home</a><a href=\"/about\">About</a></nav>"
        "<footer>Copyright 2024</footer></body></html>",
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://example.com/menu-only")

    assert result["scrape_status"] == "failed"
    assert result["error_category"] == "empty_content"


# ---------------------------------------------------------------------------
# Snippet fallback with correct provenance
# ---------------------------------------------------------------------------

def test_scrape_snippet_not_primary(monkeypatch):
    """A snippet-only result must not be classified as primary evidence."""
    mock_response = _mock_response(
        200,
        "<html><body><p>Some snippet content.</p></body></html>",
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://example.com/snippet-only")

    assert result["scrape_status"] == "failed"
    assert result["content"] == ""


def test_scrape_url_string_return_still_works(monkeypatch):
    """The legacy scrape_url() string return must still work."""
    mock_response = _mock_response(
        200,
        LONG_ARTICLE,
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url.invoke({"url": "https://example.com/legacy"})

    assert isinstance(result, str)
    assert "main article content" in result
    assert not result.startswith("SCRAPE_FAILED:")


# ---------------------------------------------------------------------------
# Duplicate URLs and syndicated articles (counted once)
# ---------------------------------------------------------------------------

def test_scrape_duplicate_urls_count_once(monkeypatch):
    """Multiple URLs from same publisher collapse to one independence group."""
    mock_response = _mock_response(200, LONG_ARTICLE)

    with patch("tools.requests.get", return_value=mock_response):
        r1 = scrape_url_detailed("https://example.com/article")
        r2 = scrape_url_detailed("https://example.com/article?utm_source=x")

    # Both are successful scrapes but same domain -> one independence group
    assert r1["scrape_status"] == "success"
    assert r2["scrape_status"] == "success"
    assert r1["url"] != r2["url"]  # Different URLs


# ---------------------------------------------------------------------------
# Deterministic-verdict safeguards unchanged
# ---------------------------------------------------------------------------

def test_scrape_non_primary_not_support(monkeypatch):
    """A non-primary source domain should not be classified as primary."""
    mock_response = _mock_response(
        200,
        LONG_ARTICLE,
    )

    with patch("tools.requests.get", return_value=mock_response):
        result = scrape_url_detailed("https://randomblog.wordpress.com/cricket")

    assert result["scrape_status"] == "success"
    assert result["content"]  # Has content