"""scratch/print_run.py - print a saved measured run."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
key = sys.argv[1] if len(sys.argv) > 1 else "biden_after"
d = json.loads((ROOT / "scratch" / "measurements" / f"{key}.json").read_text(encoding="utf-8"))

print("=" * 96)
print(f"[{key}] {d['claim']}")
print("=" * 96)
print(f"  VERDICT     : {d['verdict']}  confidence {d['confidence']} ({d['confidence_basis']})")
print(f"  sufficiency : {d['evidence_sufficiency']}  permitted {d['permitted_verdicts']}")
print(f"  publishers  : {d['independence_groups']} groups / {d['total_sources']} sources "
      f"(primary sup {d['primary_supporting_groups']}, primary contra {d['primary_contradicting_groups']})")
print(f"  article src : {d['article_sources']}  snippet-only {d['snippet_only_sources']}")
print(f"  classes     : {d['class_counts']}  kinds {d['passage_kinds']}")
print(f"  TIME        : total {d['total_sec']}s | retrieval {d['retrieval_calls']}c/{d['retrieval_sec']}s "
      f"| scrape {d['scrape_calls']}c/{d['scrape_sec']}s (fail {d['scrape_failures']}) "
      f"| NLI {d['nli_calls']}c/{d['nli_sec']}s")
print("  QUERIES:")
for q in d["search_queries"]:
    print("     ", q)
print("  SOURCES:")
for s in d["source_decisions"]:
    print(f"     [{s['decision']:<15}] {s['group']:<26} {s['retrieval']:<15} "
          f"primary={str(s['primary']):<5} score={s['score']}")
    print(f"          {s['url'][:96]}")
print("  SUPPORTING:")
for s in d["supporting_passages"]:
    print(f"     {s['domain']:<26} score={s['score']}  {s['url'][:80]}")
    print(f"        {s['nli'][:170]}")
print("  CONTRADICTING:")
for s in d["contradicting_passages"]:
    print(f"     {s['domain']:<26} score={s['score']}  {s['url'][:80]}")
    print(f"        {s['nli'][:170]}")
print("  GATE:", d["gate_reason"][:300])
