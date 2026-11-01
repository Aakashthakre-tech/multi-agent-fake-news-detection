import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import run_research_pipeline

claim = "Water boils at 100°C at standard atmospheric pressure."
state = run_research_pipeline(claim)

fv = state.get("final_verdict", {})
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)
print(f"Verdict: {fv.get('verdict')}")
print(f"Confidence: {fv.get('confidence')}% (basis: {fv.get('confidence_basis')})")
print(f"Evidence Sufficiency: {fv.get('evidence_sufficiency')}")
print(f"Independent Supporting: {fv.get('independent_supporting')}")
print(f"Independent Contradicting: {fv.get('independent_contradicting')}")
print(f"Primary Supporting Groups: {fv.get('primary_supporting_groups')}")
print(f"Total Sources: {fv.get('total_sources')}")
print(f"Permitted Verdicts: {fv.get('permitted_verdicts')}")
print(f"Reason: {fv.get('reasoning')}")
print(f"Agents Agree: {fv.get('agents_agree')}")
print(f"Verification Verdict: {fv.get('verification_verdict')}")
print(f"Critic Verdict: {fv.get('critic_verdict')}")
print(f"Override: {fv.get('verdict_overridden_by_evidence')}")

print("\n" + "=" * 60)
print("EVIDENCE ANALYSIS")
print("=" * 60)
for r in state.get("evidence_analysis", []):
    print(f"  Source {r.get('source_id')}: {r.get('domain') or r.get('url')}")
    print(f"    Class: {r.get('evidence_class')} | Event Match: {r.get('event_match')}")
    print(f"    Date Status: {r.get('date_status')} | NLI: {r.get('nli_label')} {r.get('nli_score')}")
    print(f"    Similarity: {r.get('similarity')} | Coverage: {r.get('detail_coverage')}")
    print(f"    Chunk: {r.get('chunk', '')[:100]}")

print("\n" + "=" * 60)
print("SOURCE AGREEMENT")
print("=" * 60)
agreement = state.get("source_agreement", {})
print(f"  Total Sources: {agreement.get('total_sources')}")
print(f"  Independence Groups: {agreement.get('independence_groups')}")
print(f"  Independent Supporting: {agreement.get('independent_supporting')}")
print(f"  Independent Contradicting: {agreement.get('independent_contradicting')}")
print(f"  Agreement Score: {agreement.get('agreement_score')}")
print(f"  Evidence Sufficiency: {agreement.get('evidence_sufficiency')}")
print(f"  Verification Coverage: {agreement.get('verification_coverage')}")
print(f"  Source Decisions: {agreement.get('source_decisions', [])}")

print("\n" + "=" * 60)
print("AGENT ERRORS")
print("=" * 60)
for e in state.get("agent_errors", []):
    print(f"  {e.get('stage')}: {e.get('error')}")

print("\n" + "=" * 60)
print("SEARCH PLANS")
print("=" * 60)
for p in state.get("search_plans", []):
    print(f"  {p.get('label')}: {p.get('intent')} | {p.get('topic')} | {p.get('query')}")

print("\n" + "=" * 60)
print("SEARCH ERRORS")
print("=" * 60)
for e in state.get("search_errors", []):
    print(f"  {e}")
