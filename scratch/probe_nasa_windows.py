"""
scratch/probe_nasa_windows.py
==============================
Feasibility test for the candidate fix.

The analyzer offers the NLI model only single-paragraph chunks. NASA states
the claim across ADJACENT paragraphs, so no single paragraph entails it.

This builds contiguous multi-paragraph windows (within the NLI token budget)
and asks the NLI model what it says about them.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    _count_tokens,
    _resolve_chunk_budget,
    check_date_compatibility,
    evaluate_event_match,
    extract_claim_attributes,
)
from nli_checker import check_nli
from tools import scrape_url

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
if text.startswith("SCRAPE_FAILED"):
    print(text)
    raise SystemExit(0)

budget = _resolve_chunk_budget(CLAIM)
paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n+", text) if p.strip()]
print("=" * 100)
print(f"article {len(text)} chars, {len(paragraphs)} paragraphs, NLI chunk budget {budget} tokens")
print("=" * 100)

attrs = extract_claim_attributes(CLAIM)
profile = build_claim_profile(CLAIM)

print()
print("=" * 100)
print("CONTIGUOUS WINDOWS: what the NLI model says")
print("=" * 100)
best = None
for size in (1, 2, 3, 4, 5, 6):
    for start in range(0, len(paragraphs) - size + 1):
        window = "\n\n".join(paragraphs[start:start + size])
        tokens = _count_tokens(window)
        if tokens > budget + 40:
            continue
        r = check_nli(CLAIM, window)
        if r["label"] == "entailment" or (best is None or r["score"] > best[0]):
            if r["label"] == "entailment":
                print(f"  size={size} start={start:<3} tokens={tokens:<4} "
                      f"nli={r['label']}/{r['score']}  <== ENTAILMENT")
                print(f"      {window[:220]}")
                if best is None:
                    best = (r["score"], window, start, size)

if best is None:
    print("  (no entailment window found in this probe)")

print()
print("=" * 100)
print("BEST WINDOW - full classification")
print("=" * 100)
if best:
    score, window, start, size = best
    date_check = check_date_compatibility(CLAIM, window, attrs)
    level, signals = evaluate_event_match(
        CLAIM, window, attrs, date_check["status"], 0.5, nli_entails=True
    )
    print(f"  nli        : {score}")
    print(f"  date       : {date_check['status']} - {date_check['reason']}")
    print(f"  event      : {level}")
    print(f"  coverage   : {signals['detail_coverage']}")
    print(f"  window ({size} paragraphs from {start}):")
    print(f"      {window[:700]}")
else:
    print("  none")
