"""scratch/final_api_verify.py — Live end-to-end verification of three claims via /verify/text endpoint."""
import json
import urllib.request
import time

BASE = "http://127.0.0.1:8001"
CLAIMS = [
    ("apollo_1969", "NASA's Apollo 11 mission landed humans on the Moon in July 1969."),
    ("apollo_1970", "The Apollo 11 mission landed humans on the Moon on July 20, 1970."),
    ("biden_2024", "Joe Biden withdrew from the 2024 United States presidential race in July 2024."),
]

def verify(claim):
    req = urllib.request.Request(
        BASE + "/verify/text",
        data=json.dumps({"claim": claim}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), time.time() - t0

results = []
for key, claim in CLAIMS:
    print("=" * 80)
    print(f"CLAI DENT: {key}")
    print(f"Claim: {claim}")
    print("=" * 80)
    try:
        result, elapsed = verify(claim)
    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        results.append((key, {"error": str(e)}))
        print()
        continue

    fact = result.get("factual_verification") or {}
    report = result.get("research_report") or {}
    summary = report.get("executive_summary") or {}
    methodology = report.get("methodology") or {}
    findings = report.get("findings") or {}

    print(f"  status : {result.get('status')}")
    print(f"  verdict: {fact.get('verdict')}")
    print(f"  confidence: {fact.get('confidence')} ({fact.get('confidence_basis')})")
    print(f"  evidence_sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"  independent_supporting: {fact.get('independent_supporting')}")
    print(f"  independent_contradicting: {fact.get('independent_contradicting')}")
    print(f"  article_supporting_groups: {fact.get('primary_supporting_groups')}")
    print(f"  primary_contradicting_groups: {fact.get('primary_contradicting_groups')}")
    print(f"  article_sources: {fact.get('article_supporting_groups')}")
    print(f"  total_sources: {fact.get('total_sources')}")
    print(f"  search_queries: {[q.get('query') for q in (methodology.get('search_queries') or [])]}")
    print(f"  search_errors: {methodology.get('search_errors')}")
    print(f"  scrapes_failed: {len(methodology.get('scrapes_failures') or [])}")
    print(f"  sources_retrieved: {methodology.get('sources_retrieved')}")
    print(f"  articles_scraped: {methodology.get('articles_scraped')}")
    print(f"  gate_reason: {fact.get('deterministic_gate_reason')}")
    print(f"  elapsed: {elapsed:.2f}s")
    print()
    print(f"  SUPPORTING findings ({len(findings.get('supporting') or [])}):")
    for f in (findings.get("supporting") or []):
        print(f"    - {f.get('source_name')} | {f.get('source_url')}")
        print(f"      excerpt: {str(f.get('excerpt'))[:150]}")
    print(f"  CONTRADICTING findings ({len(findings.get('contradicting') or [])}):")
    for f in (findings.get("contradicting") or []):
        print(f"    - {f.get('source_name')} | {f.get('source_url')}")
        print(f"      excerpt: {str(f.get('excerpt'))[:150]}")
    results.append((key, result))
