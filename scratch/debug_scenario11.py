"""scratch/debug_scenario11.py - why is snippet_only_sources zero?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_text_evaluation_set import (
    ARTICLES, CLAIM_APOLLO, run_pipeline, summarise,
)

state = run_pipeline(
    CLAIM_APOLLO,
    {"apollo": ["nasa-apollo11", "smithsonian-apollo11"]},
    {},
    snippet_only=True,
)

print("=" * 92)
print("SOURCE LEDGER")
print("=" * 92)
for s in state.get("source_data") or []:
    print(f"  evidence_type={s.get('evidence_type')!r} status={str(s.get('scrape_status'))[:40]}")
    print(f"    url={s.get('url')}")
    print(f"    snippet_len={len(s.get('snippet') or '')} content_len={len(s.get('content') or '')}")

print()
print("=" * 92)
print("EVIDENCE RECORDS")
print("=" * 92)
for r in state.get("evidence_analysis") or []:
    print(f"  class={r['evidence_class']:<16} type={r.get('evidence_type'):<16} "
          f"score={r.get('evidence_score')} domain={r.get('domain')}")

agreement, gate, verdict = summarise(state)
print()
print("snippet_only_sources :", agreement.get("snippet_only_sources"))
print("article_sources      :", agreement.get("article_sources"))
print("total_sources        :", agreement.get("total_sources"))
print("verdict              :", verdict["verdict"])
