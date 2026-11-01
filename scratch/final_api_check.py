"""scratch/final_api_check.py - Live API verification of Apollo 11 claim."""
import json
import urllib.request

BASE = "http://127.0.0.1:8001"
CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

req = urllib.request.Request(
    BASE + "/verify/text",
    data=json.dumps({"claim": CLAIM}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
except Exception as e:
    print(f"REQUEST FAILED: {e}")
    raise

fact = d.get("factual_verification") or {}
report = d.get("research_report") or {}
summary = report.get("executive_summary") or {}
methodology = report.get("methodology") or {}
findings = report.get("findings") or {}

print("=" * 70)
print("API RESPONSE - Factual Verification")
print("=" * 70)
print(f"  modality         : {d.get('modality')}")
print(f"  status           : {d.get('status')}")
print(f"  verdict          : {fact.get('verdict')}")
print(f"  confidence       : {fact.get('confidence')} ({fact.get('confidence_basis')})")
print(f"  evidence_sufficiency: {fact.get('evidence_sufficiency')}")
print(f"  independent_supporting : {fact.get('independent_supporting')}")
print(f"  independent_contradicting : {fact.get('independent_contradicting')}")
print(f"  primary_supporting_groups : {fact.get('primary_supporting_groups')}")
print(f"  primary_contradicting_groups : {fact.get('primary_contradicting_groups')}")
print(f"  article_sources : {fact.get('article_supporting_groups')}")
print(f"  total_sources : {fact.get('total_sources')}")
print(f"  entailment_score  : {fact.get('entailment_score')}")
print(f"  contradiction_score : {fact.get('contradiction_score')}")
print(f"  source_agreement : {fact.get('source_agreement')}")
print(f"  fusion_score : {fact.get('fusion_score')}")
print(f"  permitted_verdicts : {fact.get('permitted_verdicts')}")
print(f"  deterministic_gate_reason : {fact.get('deterministic_gate_reason')}")
print(f"  reasoning : {fact.get('reasoning')}")

print()
print("METHODOLOGY:")
print(f"  search_queries: {[q.get('query') for q in (methodology.get('search_queries') or [])]}")
print(f"  search_errors : {methodology.get('search_errors')}")
print(f"  sources_retrieved : {methodology.get('sources_retrieved')}")
print(f"  articles_scraped : {methodology.get('articles_scraped')}")
print(f"  scrapes_failed : {len(methodology.get('scrapes_failures') or [])}")
print(f"  evidence_passages : {methodology.get('evidence_passages')}")
print(f"  nli_evaluated : {methodology.get('nli_evaluated')}")

print()
print("FINDINGS:")
print(f"  supporting: {len(findings.get('supporting') or [])}")
for f in (findings.get("supporting") or []):
    print(f"    - {f.get('source_name')} | {f.get('source_url')}")
    print(f"      excerpt: {str(f.get('excerpt'))[:150]}")
print(f"  contradicting: {len(findings.get('contradicting') or [])}")
for f in (findings.get("contradicting") or []):
    print(f"    - {f.get('source_name')} | {f.get('source_url')}")
    print(f"      excerpt: {str(f.get('excerpt'))[:150]}")

print()
print("EVIDENCE GAPS:")
for g in (report.get("evidence_gaps") or []):
    print(f"  - {g.get('question')}: {g.get('detail')}")
