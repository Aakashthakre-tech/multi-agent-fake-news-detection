"""scratch/inspect_biden.py - why did the Biden claim get no SUPPORT?"""
import json
from pathlib import Path

d = json.loads(Path("scratch/measurements/biden_withdrawal_recent.json").read_text(encoding="utf-8"))
print("QUERIES:")
for q in d["search_queries"]:
    print("   ", q)
print()
print("SOURCES:")
for s in d["source_decisions"]:
    print(f"  [{s['decision']:<15}] {s['domain']:<30} {s['retrieval']:<15} "
          f"primary={s['primary']} score={s['score']}")
    print(f"      {s['url']}")
print()
print("gate:", d["gate_reason"])
