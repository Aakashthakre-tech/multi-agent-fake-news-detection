"""
scratch/probe_apollo11_chunks.py
================================
Takes the real chunks captured from the live run and asks, for the most
relevant ones:

  1. what does the chunk actually say?
  2. what does the NLI model say about THAT EXACT chunk against the claim?
  3. what would the classification be if the date gate were satisfied?
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nli_checker import check_nli
from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
)
from source_reliability import get_source_profile

d = json.loads((ROOT / "scratch" / "apollo11_trace.json").read_text(encoding="utf-8"))
CLAIM = d["claim"]

full = json.loads((ROOT / "scratch" / "apollo11_trace.json").read_text(encoding="utf-8"))
# the captured chunks are truncated to 600 chars; re-read the full evidence from
# the source ledger instead
chunks = full["chunks"]

attrs = extract_claim_attributes(CLAIM)
profile = build_claim_profile(CLAIM)

# Re-derive the chunk text from the saved trace (600 chars each).
best = sorted(chunks, key=lambda c: -(c.get("detail_coverage") or 0))[:5]

print("=" * 100)
print("TOP CHUNKS BY CLAIM-DETAIL COVERAGE")
print("=" * 100)
for c in best:
    print()
    print(f"  url   : {c.get('url')}")
    print(f"  group : {c.get('independence_group')}")
    print(f"  type  : {c.get('evidence_type')}   class={c.get('evidence_class')}")
    print(f"  detail_coverage={c.get('detail_coverage')} similarity={c.get('similarity')}")
    print(f"  recorded nli: label={c.get('nli_label')} score={c.get('nli_score')} "
          f"raw_label={c.get('raw_nli_label')} raw_score={c.get('raw_nli_score')}")
    print(f"  text  : {c.get('chunk')[:400]}")
    r = check_nli(CLAIM, c.get("chunk", ""))
    print(f"  RE-RUN NLI on this chunk: label={r['label']} score={r['score']} "
          f"scores={r.get('scores')}")

print()
print("=" * 100)
print("WHAT WOULD HAPPEN IF THE DATE GATE WERE SATISFIED?")
print("=" * 100)
src_profile = get_source_profile("https://www.rocketstem.org/apollo-50th-celebration", "scraped_article")
for c in best:
    text = c.get("chunk", "")
    for assumed_date in ("same_year", "match"):
        level, signals = evaluate_event_match(
            CLAIM, text, attrs, assumed_date, c.get("similarity") or 0.7, nli_entails=True
        )
        klass = classify_evidence(
            {
                "date_status": assumed_date,
                "event_match": level,
                "similarity": c.get("similarity"),
                "raw_nli_label": c.get("raw_nli_label"),
                "raw_nli_score": c.get("raw_nli_score"),
                "nli_status": c.get("nli_status"),
                "location_match": signals["location_match"],
                "claim_has_location": signals.get("claim_has_location", True),
                "detail_coverage": c.get("detail_coverage"),
                "debunking": False,
            },
            c.get("evidence_type") or "scraped_article",
            src_profile,
        )
        print(f"  {str(c.get('domain'))[:22]:<22} date={assumed_date:<10} "
              f"evt={level:<8} -> {klass}")
