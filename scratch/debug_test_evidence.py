import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_analyzer import analyze_evidence
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_evidence_retrieval_reliability import CLAIM, _source

results = analyze_evidence(CLAIM, [_source()])
print("Total results:", len(results))
for i, r in enumerate(results):
    print(f"[{i}] class={r['evidence_class']} score={r['evidence_score']} date={r['date_status']}")
    print(r['chunk'][:180].replace('\n', ' '))
