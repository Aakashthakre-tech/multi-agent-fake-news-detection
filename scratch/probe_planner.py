"""scratch/probe_planner.py - what the query planner sees for the failing claim."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from query_planner import plan_queries

CLAIMS = [
    "Joe Biden withdrew from the 2024 United States presidential race in July 2024.",
    "NASA's Apollo 11 mission landed humans on the Moon in July 1969.",
    "A major fire broke out at a commercial building in New Delhi, India on 12 Aug 2024.",
]

for claim in CLAIMS:
    print("=" * 92)
    print("CLAIM:", claim)
    p = build_claim_profile(claim)
    print(f"  locations  : {p['locations']}")
    print(f"  keywords   : {p['keywords']}")
    print(f"  phrases    : {p['phrases']}")
    print(f"  dates      : {p['dates']}")
    print("  QUERIES:")
    for plan in plan_queries(claim):
        print(f"    [{plan['intent']:>9}] ({plan['topic']:>7}) {plan['query']}")
    print()
