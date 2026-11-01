"""scratch/compare_queries.py - before vs after search queries."""
import json
from pathlib import Path

after = json.loads(Path("scratch/apollo11_api_response.json").read_text(encoding="utf-8"))
before = json.loads(Path("scratch/apollo11_trace.json").read_text(encoding="utf-8"))

print("=== BEFORE (old parser) ===")
for p in before.get("search_plans") or []:
    print(f"  [{p.get('label')}] intent={p.get('intent')} :: {p.get('query')}")

print()
print("=== AFTER (month+year parsed as a date) ===")
for p in (after.get("raw") or {}).get("search_plans") or []:
    print(f"  [{p.get('label')}] intent={p.get('intent')} :: {p.get('query')}")
