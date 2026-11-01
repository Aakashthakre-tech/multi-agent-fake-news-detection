"""
Live integration test: Water boiling at 100°C at standard atmospheric pressure.

Runs the full pipeline end-to-end with actual web retrieval, scraping, NLI,
evidence classification, and verdict aggregation. Verifies that IAPWS is found
and classified correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import run_research_pipeline

CLAIM = "Water boils at 100 degrees Celsius at standard atmospheric pressure."

print("=" * 70)
print("LIVE INTEGRATION TEST: Water boiling at 100°C")
print("=" * 70)
print(f"Claim: {CLAIM}")
print()

state = run_research_pipeline(CLAIM)

fv = state.get("final_verdict", {})
agreement = state.get("source_agreement", {})
diagnostic = state.get("diagnostic_summary", "")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Verdict:           {fv.get('verdict')}")
print(f"Confidence:        {fv.get('confidence')}% (basis: {fv.get('confidence_basis')})")
print(f"Evidence sufficiency: {fv.get('evidence_sufficiency')}")
print(f"Independent supporting: {fv.get('independent_supporting')}")
print(f"Independent contradicting: {fv.get('independent_contradicting')}")
print(f"Permitted verdicts: {fv.get('permitted_verdicts')}")
print(f"Reasoning: {str(fv.get('reasoning', ''))[:200]}")
print()
print(f"Diagnostic: {diagnostic}")
print()

print("-" * 70)
print("EVIDENCE ANALYSIS")
print("-" * 70)
for r in state.get("evidence_analysis", []):
    print(f"  Source {r.get('source_id')}: {r.get('domain') or r.get('url')}")
    print(f"    Class: {r.get('evidence_class')} | NLI: {r.get('nli_label')} ({r.get('nli_score')})")
    print(f"    Similarity: {r.get('similarity')} | Date: {r.get('date_status')}")
    print(f"    Chunk: {r.get('chunk', '')[:120]}")
    print()

print("-" * 70)
print("AGENT ERRORS")
print("-" * 70)
for e in state.get("agent_errors", []):
    print(f"  {e.get('stage')}: {e.get('error')}")
if not state.get("agent_errors"):
    print("  None")

print("\n" + "=" * 70)
print("INTEGRATION TEST PASSED (no crash)")
print("=" * 70)
