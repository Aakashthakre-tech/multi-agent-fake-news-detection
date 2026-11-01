"""scratch/inspect_after_chunks.py - per-chunk diagnostics of the AFTER run."""
import json
from pathlib import Path

d = json.loads(Path("scratch/apollo11_api_response.json").read_text(encoding="utf-8"))
ev = (d.get("raw") or {}).get("evidence_analysis") or []

print("=" * 118)
print(f"{'domain':<24} {'type':<15} {'date':<10} {'evt':<8} {'class':<15} {'nli':<13} {'nliS':<7} {'raw':<13} {'rawS':<7} {'sim':<6} {'score':<6} {'det'}")
print("-" * 118)
for c in ev:
    print(
        f"{str(c.get('domain'))[:23]:<24} "
        f"{str(c.get('evidence_type'))[:14]:<15} "
        f"{str(c.get('date_status'))[:9]:<10} "
        f"{str(c.get('event_match'))[:7]:<8} "
        f"{str(c.get('evidence_class'))[:14]:<15} "
        f"{str(c.get('nli_label'))[:12]:<13} "
        f"{str(c.get('nli_score'))[:6]:<7} "
        f"{str(c.get('raw_nli_label'))[:12]:<13} "
        f"{str(c.get('raw_nli_score'))[:6]:<7} "
        f"{str(c.get('similarity'))[:5]:<6} "
        f"{str(c.get('evidence_score'))[:5]:<6} "
        f"{c.get('detail_coverage')}"
    )

print()
print("=" * 118)
print("NASA chunks in full")
print("=" * 118)
for c in ev:
    if "nasa.gov" in str(c.get("url")):
        print()
        print(f"  {c.get('url')}")
        print(f"  class={c.get('evidence_class')} date={c.get('date_status')} "
              f"evt={c.get('event_match')} sim={c.get('similarity')} "
              f"det={c.get('detail_coverage')} nli={c.get('nli_label')}/{c.get('nli_score')}")
        print(f"  reason: {str(c.get('date_reason'))[:130]}")
        print(f"  text: {str(c.get('chunk'))[:300]}")

print()
print("=" * 118)
print("SOURCE LEDGER")
print("=" * 118)
for s in (d.get("raw") or {}).get("source_data") or []:
    print(f"  {str(s.get('scrape_status'))[:26]:<28} {str(s.get('evidence_type'))[:15]:<16} {s.get('url')}")
