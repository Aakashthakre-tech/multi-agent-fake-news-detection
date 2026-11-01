import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_planner import plan_queries

claims = [
    "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
    "The Apollo 11 mission landed humans on the Moon on July 20, 1970.",
    "Joe Biden withdrew from the 2024 United States presidential race in July 2024.",
    "This news is misleading.",
]

for c in claims:
    print("=" * 70)
    print("CLAIM:", c)
    plans = plan_queries(c)
    for p in plans:
        print(f"  [{p['intent']:>9}] ({p['topic']:>7}) {p['query']}")
