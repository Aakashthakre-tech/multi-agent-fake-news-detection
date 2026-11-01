"""scratch/summarise_e2e.py - summarise the saved live e2e results."""
import json
from pathlib import Path

p = Path("scratch/e2e_results.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(f"fixture file: {p}  ({p.stat().st_size} bytes)")
for modality, payload in data.items():
    report = payload.get("report") or {}
    summary = report.get("executive_summary") or {}
    method = report.get("methodology") or {}
    findings = report.get("findings") or {}
    counts = {k: len(v or []) for k, v in findings.items()}
    print()
    print("=" * 74)
    print(f"{modality.upper()}  status={payload.get('status')}  "
          f"verdict={payload.get('verdict')}")
    print("=" * 74)
    print(f"  report verdict     : {summary.get('verdict')}")
    print(f"  confidence         : {summary.get('confidence')} "
          f"({summary.get('confidence_basis')})")
    print(f"  research skipped   : {method.get('research_skipped')}")
    if method.get("research_skipped_reason"):
        print(f"  skip reason        : {method['research_skipped_reason'][:150]}")
    print(f"  search queries     : {len(method.get('search_queries') or [])}")
    print(f"  sources retrieved  : {method.get('sources_retrieved')}")
    print(f"  articles scraped   : {method.get('articles_scraped')}")
    print(f"  snippet-only srcs  : {method.get('snippet_only_sources')}")
    print(f"  independent pubs   : {method.get('independent_publishers')}")
    print(f"  primary supporting : {method.get('primary_supporting_groups')}")
    print(f"  primary contradict : {method.get('primary_contradicting_groups')}")
    print(f"  unresolved identity: {method.get('unresolved_identity_sources')}")
    print(f"  coverage           : {method.get('verification_coverage')}")
    print(f"  evidence passages  : {method.get('evidence_passages')}")
    print(f"  findings           : {counts}")
    print(f"  claim results      : {len(report.get('claims') or [])}")
    print(f"  evidence gaps      : {len(report.get('evidence_gaps') or [])}")
    for st in method.get("stages") or []:
        print(f"    stage: {st.get('name')} [{st.get('status')}] {str(st.get('detail'))[:120]}")
