"""scratch/build_matrix.py - build the validation test matrix from the measurements."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from claim_matrix import MATRIX

MEAS = ROOT / "scratch" / "measurements"

rows = []
for entry in MATRIX:
    path = MEAS / f"{entry['key']}.json"
    if not path.exists():
        rows.append((entry, None))
        continue
    rows.append((entry, json.loads(path.read_text(encoding="utf-8"))))

W = 118
print("=" * W)
print("TRUTHLENSAI TEXT-VERIFICATION TEST MATRIX")
print("=" * W)

for entry, m in rows:
    print()
    print("-" * W)
    print(f"CLAIM : {entry['claim']}")
    print(f"TYPE  : {entry['key']}   reference outcome: {entry['reference_outcome']}")
    print(f"REF   : {entry['reference_note']}")
    for s in entry["reference_sources"]:
        print(f"        - {s}")
    print(f"EXPECT: {entry['expected_shape']}")
    print("-" * W)
    if m is None:
        print("  NOT MEASURED")
        continue
    print(f"  actual verdict      : {m['verdict']}   confidence {m['confidence']} "
          f"({m['confidence_basis']})")
    print(f"  sufficiency        : {m['evidence_sufficiency']}   permitted {m['permitted_verdicts']}")
    print(f"  publishers         : {m['independence_groups']} groups / {m['total_sources']} sources")
    print(f"  independent sup/con: {m['independent_supporting']} / {m['independent_contradicting']}")
    print(f"  primary sup/con    : {m['primary_supporting_groups']} / {m['primary_contradicting_groups']}")
    print(f"  article sources    : {m['article_sources']}  snippet-only {m['snippet_only_sources']}")
    print(f"  evidence classes   : {m['class_counts']}")
    print(f"  passage kinds      : {m['passage_kinds']}")
    print(f"  supporting sources :")
    for s in m["supporting_passages"]:
        print(f"      {s['domain']:<28} score={s['score']}  {s['url'][:70]}")
        print(f"          {s['nli'][:150]}")
    print(f"  contradicting sources :")
    for s in m["contradicting_passages"]:
        print(f"      {s['domain']:<28} score={s['score']}  {s['url'][:70]}")
        print(f"          {s['nli'][:200]}")
    if m["supporting_passages"] or m["contradicting_passages"]:
        print(f"  per-source decisions:")
        for d in m["source_decisions"]:
            print(f"      [{d['decision']:<14}] {d['group']:<26} {d['retrieval']:<14} "
                  f"primary={str(d['primary']):<5} snippet_only={str(d['snippet_only']):<5} "
                  f"score={d['score']}")
    print(f"  TIMING total {m['total_sec']}s | retrieval {m['retrieval_calls']}c/"
          f"{m['retrieval_sec']}s | scrape {m['scrape_calls']}c/{m['scrape_sec']}s "
          f"(fail {m['scrape_failures']}) | NLI {m['nli_calls']}c/{m['nli_sec']}s")
    print(f"  policy: valid={m['policy_data_valid']} errors={m['policy_data_errors']}")
    print(f"  gate  : {m['gate_reason'][:200]}")

# ---- aggregate performance ----
print()
print("=" * W)
print("PERFORMANCE AGGREGATE")
print("=" * W)
tot = ret = scr = nli = 0.0
rc = sc = nc = sf = 0
for entry, m in rows:
    if not m:
        continue
    tot += m["total_sec"]; ret += m["retrieval_sec"]; scr += m["scrape_sec"]; nli += m["nli_sec"]
    rc += m["retrieval_calls"]; sc += m["scrape_calls"]; nc += m["nli_calls"]; sf += m["scrape_failures"]
print(f"  runs               : {len([1 for _, m in rows if m])}")
print(f"  total wall clock   : {tot:.1f}s  (mean {tot/max(1,len([1 for _,m in rows if m])):.1f}s)")
print(f"  retrieval          : {rc} calls / {ret:.1f}s  ({ret/tot*100:.1f}% of total)")
print(f"  scraping           : {sc} calls / {scr:.1f}s  ({scr/tot*100:.1f}%)  failures {sf}")
print(f"  NLI                : {nc} calls / {nli:.1f}s  ({nli/tot*100:.1f}% of total)")
print(f"  NLI mean latency   : {nli/max(1,nc):.2f}s per call")
