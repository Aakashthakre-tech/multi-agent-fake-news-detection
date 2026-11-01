"""
scratch/verify_window_fix.py
============================
Runs the REAL analyzer on NASA's Apollo 11 article with the window fix in
place, and reports what the evidence layer now produces - including the
source-level decision (the fusion layer collapses a source's chunks into one
vote, so a spurious contradiction inside one source cannot refute a claim).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_analyzer import analyze_evidence
from evidence_fusion import calculate_source_agreement
from tools import scrape_url

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
if text.startswith("SCRAPE_FAILED"):
    print(text)
    raise SystemExit(0)

source = {
    "url": URL, "domain": "nasa.gov", "title": "July 20, 1969: One Giant Leap For Mankind",
    "content": text, "evidence_type": "scraped_article", "scrape_status": "SUCCESS",
}

results = analyze_evidence(CLAIM, [source], verbose=False)

print("=" * 112)
print(f"NASA article: {len(text)} chars -> {len(results)} evidence record(s)")
print("=" * 112)
for r in results:
    print(
        f"  class={r.get('evidence_class'):<15} kind={r.get('passage_kind'):<9} "
        f"date={r.get('date_status'):<8} evt={r.get('event_match'):<8} "
        f"nli={r.get('nli_label')}/{r.get('nli_score'):<7} "
        f"sim={r.get('similarity'):<6} score={r.get('evidence_score'):<6} "
        f"det={r.get('detail_coverage')}"
    )
    print(f"      {str(r.get('chunk'))[:200]}")

print()
print("=" * 112)
print("SOURCE-LEVEL DECISION (what the fusion layer sees)")
print("=" * 112)
agreement = calculate_source_agreement(results)
for d in agreement["source_decisions"]:
    print(f"  decision          : {d['decision']}")
    print(f"  has_support       : {d['has_support']}")
    print(f"  has_contradiction : {d['has_contradiction']}")
    print(f"  source_score      : {d['source_score']}")
print(f"  independent_supporting    : {agreement['independent_supporting']}")
print(f"  independent_contradicting : {agreement['independent_contradicting']}")
print(f"  evidence_sufficiency      : {agreement['evidence_sufficiency']}")
