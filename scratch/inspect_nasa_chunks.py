"""
scratch/inspect_nasa_chunks.py
==============================
Runs the analyzer on the NASA source from the captured live response and
lists EVERY chunk it produced, with its class, before the pipeline's
"strongest few chunks per source" selection is applied.

This distinguishes two possibilities:
  (a) the NASA landing sentence was never classified SUPPORT, or
  (b) it WAS classified SUPPORT but was dropped by the selection step.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

d = json.loads((ROOT / "scratch" / "apollo11_api_response.json").read_text(encoding="utf-8"))
CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

sources = (d.get("raw") or {}).get("source_data") or []
print("source_data keys:", sorted(sources[0].keys()) if sources else "none")

nasa = [s for s in sources if "nasa.gov" in str(s.get("url")) and s.get("content")]
if not nasa:
    print("No NASA source with content in the response payload.")
    raise SystemExit(0)

from evidence_analyzer import analyze_evidence

for source in nasa:
    print()
    print("=" * 112)
    print(f"SOURCE {source.get('url')}  ({source.get('evidence_type')}, {len(source.get('content') or '')} chars)")
    print("=" * 112)
    results = analyze_evidence(CLAIM, [source], verbose=False)
    for r in results:
        print(
            f"  class={r.get('evidence_class'):<15} date={r.get('date_status'):<10} "
            f"evt={r.get('event_match'):<8} nli={r.get('nli_label')}/{r.get('nli_score'):<7} "
            f"sim={r.get('similarity'):<6} score={r.get('evidence_score'):<6} "
            f"det={r.get('detail_coverage')}"
        )
        print(f"      {str(r.get('chunk'))[:190]}")
