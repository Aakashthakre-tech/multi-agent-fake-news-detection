"""scratch/print_apollo11_api.py - print the live API response summary."""
import json
from pathlib import Path

d = json.loads(Path("scratch/apollo11_api_response.json").read_text(encoding="utf-8"))
fact = d["factual_verification"]

print("=" * 92)
print("LIVE API - factual_verification (AFTER the fix)")
print("=" * 92)
for k in sorted(fact.keys()):
    v = fact[k]
    if isinstance(v, str) and len(v) > 130:
        v = v[:130] + "..."
    print(f"  {k:<34} {v}")

es = (d.get("research_report") or {}).get("executive_summary") or {}
print()
print("=" * 92)
print("RESEARCH REPORT - executive summary")
print("=" * 92)
for k in ("verdict", "confidence", "confidence_basis", "key_findings",
          "why_this_verdict", "agent_narrative", "critic_review", "limitations"):
    v = es.get(k)
    if isinstance(v, str) and len(v) > 200:
        v = v[:200] + "..."
    if isinstance(v, list):
        v = [str(x)[:150] for x in v]
    print(f"  {k:<22} {v}")

print()
print("=" * 92)
print("FINDINGS")
print("=" * 92)
for bucket in ("supporting", "contradicting", "contextual", "unresolved"):
    items = ((d.get("research_report") or {}).get("findings") or {}).get(bucket) or []
    print(f"  {bucket}: {len(items)}")
    for f in items[:3]:
        print(f"     {f.get('source_name')} | {f.get('source_url')}")
        print(f"       strength={f.get('strength_label')} ({f.get('evidence_strength')}) "
              f"nli={f.get('nli_label')}/{f.get('nli_score')} "
              f"class={f.get('evidence_class_label') or f.get('evidence_class')}")
        print(f"       {str(f.get('excerpt'))[:120]}")

m = (d.get("research_report") or {}).get("methodology") or {}
print()
print("  methodology policy counters:")
for k in ("independent_publishers", "primary_supporting_groups",
          "primary_contradicting_groups", "article_supporting_groups",
          "snippet_only_sources", "article_sources", "verification_coverage"):
    print(f"    {k:<30} {m.get(k)}")
