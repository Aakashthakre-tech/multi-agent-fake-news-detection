"""
scratch/probe_nasa_extraction.py
================================
Fetches NASA's Apollo 11 history page with the REAL scraper and reports:

  1. what the current extractor returns (length + head + tail);
  2. every candidate container the selector list could match, with its
     length and a text sample, so the correct container can be identified;
  3. whether the landing sentence ("Apollo 11 landed ... humans ... Moon")
     is present in the extracted text.

This is a live network probe. It is NOT part of the automated test suite.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from bs4 import BeautifulSoup

URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

LANDING_MARKERS = [
    "landed the first humans",
    "first humans to step",
    "first humans to set foot",
    "first humans on the moon",
    "first humans to walk",
    "landed humans",
]


def main():
    from tools import scrape_url

    print("=" * 96)
    print("1. THE REAL SCRAPER OUTPUT")
    print("=" * 96)
    text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
    if text.startswith("SCRAPE_FAILED"):
        print(f"  SCRAPE FAILED: {text}")
        return
    print(f"  length: {len(text)} chars")
    print("  ---- head (600) ----")
    print(text[:600])
    print("  ---- tail (600) ----")
    print(text[-600:])

    print()
    print("=" * 96)
    print("2. IS THE LANDING SENTENCE PRESENT?")
    print("=" * 96)
    low = text.lower()
    for marker in LANDING_MARKERS:
        print(f"  {marker!r:<32} -> {'FOUND' if marker in low else 'missing'}")

    print()
    print("=" * 96)
    print("3. EVERY CANDIDATE CONTAINER THE SELECTORS COULD MATCH")
    print("=" * 96)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(URL, timeout=25, headers=headers)
    print(f"  HTTP {response.status_code}, {len(response.text)} chars of HTML")
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in ("script", "style", "nav", "footer", "header", "aside", "form",
                "noscript", "iframe", "svg", "button", "input", "select",
                "textarea", "menu", "dialog"):
        for element in soup(tag):
            element.decompose()

    selectors = [
        "article", '[itemprop="articleBody"]', '[role="main"]', "main",
        ".article-content", ".article-body", ".article__body", ".story-content",
        ".story-body", ".article-body-content", ".content-body", ".post-content",
        ".entry-content", "#article-body", ".td-post-content", ".post-body",
        ".single-content", ".news-body", ".text-body",
    ]

    rows = []
    for selector in selectors:
        for candidate in soup.select(selector):
            body = candidate.get_text(separator=" ", strip=True)
            if len(body) > 300:
                paras = [
                    p.get_text(" ", strip=True) for p in candidate.find_all(["p", "h2", "h3"])
                ]
                paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
                paras = [p for p in paras if len(p) >= 25]
                rows.append((selector, len(body), len(paras), paras))
    for selector, length, nparas, paras in rows:
        print(f"  {selector:<24} text={length:<8} paras={nparas}")
        if paras:
            print(f"      first para: {paras[0][:150]}")
    if not rows:
        print("  (no container produced > 300 chars)")

    print()
    print("=" * 96)
    print("4. WHAT THE FULL PAGE BODY CONTAINS (for comparison)")
    print("=" * 96)
    body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
    print(f"  <body> text length: {len(body_text)}")
    for marker in LANDING_MARKERS:
        print(f"  {marker!r:<32} -> {'FOUND' if marker in body_text.lower() else 'missing'}")


if __name__ == "__main__":
    main()
