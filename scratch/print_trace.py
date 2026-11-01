"""scratch/print_trace.py - print the saved real-evidence trace."""
import json

d = json.load(open("scratch/trace_real_evidence.json", encoding="utf-8"))
a = d["agreement"]
v = d["verdict"]
g = d["gate"]

print("=== CLASSIFICATION PER CHUNK ===")
for c in d["chunks"]:
    print(
        f"  {str(c['domain'])[:26]:26s} class={c['evidence_class']:14s} "
        f"score={c['evidence_score']} nli={c['nli_label']}/{c['nli_score']} "
        f"date={c['date_status']:20s} event={c['event_match']:9s} "
        f"loc={c['location_match']} detail={c['detail_coverage']} "
        f"type={c['evidence_type']} primary={c['is_primary_source']}"
    )

print()
print("=== AGGREGATION ===")
for k in [
    "total_sources", "snippet_only_sources", "article_sources",
    "unresolved_identity_sources", "independence_groups",
    "independent_supporting", "independent_contradicting",
    "primary_supporting_groups", "primary_contradicting_groups",
    "article_supporting_groups", "article_contradicting_groups",
    "correction_groups", "verification_coverage", "evidence_sufficiency",
    "agreement_score",
]:
    print(f"  {k:32s} {a.get(k)}")

print()
print("=== GATE ===")
print("  permitted:", sorted(g["permitted_verdicts"]))
print("  reason  :", g["permitted_reason"])

print()
print("=== VERDICT ===")
for k in [
    "verdict", "confidence", "confidence_basis", "verification_verdict",
    "critic_verdict", "verdict_overridden_by_evidence", "evidence_sufficiency",
]:
    print(f"  {k:32s} {v.get(k)}")
print("  reasoning:", v.get("reasoning"))
