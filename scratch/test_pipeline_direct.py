"""scratch/test_pipeline_direct.py - Direct pipeline test"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import run_research_pipeline

claim = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
print(f"Running pipeline for: {claim}")
state = run_research_pipeline(claim)

print(f"\n=== STATE KEYS: {list(state.keys())}")
print(f"search_results length: {len(state.get('search_results', ''))}")
print(f"search_plans count: {len(state.get('search_plans', []))}")
print(f"search_errors: {state.get('search_errors', [])}")
print(f"source_data count: {len(state.get('source_data', []))}")
print(f"evidence_analysis count: {len(state.get('evidence_analysis', []))}")

print(f"\nFinal verdict: {state.get('final_verdict', {})}")
print(f"Case ID: {state.get('case_summary', {}).get('case_id')}")

# Print search results
print(f"\n=== Search Results (first 2000 chars) ===")
print(state.get("search_results", "")[:2000])

# Print candidate details
print(f"\n=== CANDIDATES ({len(state.get('source_data', []))}) ===")
for i, s in enumerate(state.get("source_data", [])):
    print(f"\n--- Source {i}: {s.get('domain')} ({s.get('source_tier')})")
    print(f"   URL: {s.get('url')}")
    print(f"   Title: {s.get('title')}")
    print(f"   Type: {s.get('evidence_type')}")
    print(f"   Reason: {s.get('selection_reason')}")
    print(f"   Score: {s.get('evidence_score')}")
