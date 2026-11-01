"""scratch/debug_mars2.py - trace the Mars scenario through the pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_text_evaluation_set import CLAIM_APOLLO_MARS, run_pipeline, summarise

state = run_pipeline(
    CLAIM_APOLLO_MARS,
    {"armstrong": ["nasa-mars", "smithsonian-mars"]},
    {},
)

print("=" * 92)
print("EVIDENCE RECORDS")
print("=" * 92)
for r in state.get("evidence_analysis") or []:
    print(f"  class={r['evidence_class']:<16} kind={r.get('passage_kind'):<9} "
          f"type={r.get('evidence_type'):<15} date={r['date_status']:<10} "
          f"evt={r['event_match']:<8} nli={r['nli_label']}/{r['nli_score']} "
          f"score={r['evidence_score']} primary={r.get('is_primary_source')}")
    print(f"      {str(r.get('chunk'))[:100]}")

agreement, gate, verdict = summarise(state)
print()
print("=" * 92)
print("AGGREGATION")
print("=" * 92)
for d in agreement["source_decisions"]:
    print(f"  [{d['decision']:<15}] {d['domain']:<20} score={d['source_score']:<8} "
          f"primary={d['is_primary']} has_contra={d['has_contradiction']}")
print()
print(f"  independent_contradicting   : {agreement['independent_contradicting']}")
print(f"  primary_contradicting_groups: {agreement['primary_contradicting_groups']}")
print(f"  article_contradicting_groups : {agreement['article_contradicting_groups']}")
print(f"  correction_groups            : {agreement['correction_groups']}")
print()
print("  GATE permitted :", sorted(gate["permitted_verdicts"]))
print("  GATE reason   :", gate["permitted_reason"][:300])
print("  VERDICT       :", verdict["verdict"], "|", verdict.get("reasoning", "")[:200])
