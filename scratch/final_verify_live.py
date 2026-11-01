"""
scratch/final_e2e_verify.py
=============================
Genuine end-to-end test against the live API.
Does NOT modify any source code. Only reads from the running API.
"""
import json
import urllib.request
import urllib.error
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
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "reason": str(e)}, None
    except Exception as e:
        return {"error": str(e)}, None

results = []
for key, claim in CLAIMS:
    print(f"=" * 80)
    print(f"CLAIM: {claim}")
    print("=" * 80)
    try:
        result, _ = verify(claim)
    except Exception as e:
        print(f"  REQUEST FAILED: {e}")
        results.append((key, {"error": str(e)}))
        continue

    fact = result.get("factual_verification") or {}
    report = d.get("research_report") or {} if isinstance(d, dict) else {}
    
    print(f"  status: {data.get('status')}")
    print(f"  verdict : {fact.get('verdict')}")
    print(f"  confidence: {fact.get('confidence')} ({fact.get('confidence_basis')})")
    print(f"  sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"  independent_supporting: {fact.get('independent_supporting')}")
    print(f"  independent_contradicting: {fact.get('independent_contradicting')}")
    print(f"  article_sources: {fact.get('article_supporting_groups')}")
    print(f"  total_sources: {fact.get('total_sources')}")
    print(f"  search_queries: {[q.get('query') for q in (methodology.get('search_queries') or [])]}")
    print(f"  search_errors: {methodology.get('search_errors')}")
    print(f"  scrapes_failed: {len(methodology.get('scrapes_failures') or [])}")
    print(f"  gate_reason: {fact.get('deterministic_gate_reason')}")
    
    findings = report.get("findings") or {}
    print(f"  supporting findings: {len(findings.get('supporting') or [])}")
    print(f"  contradicting findings: {len(findings.get('contradicting') or [])}")
    for f in (findings.get("supporting") or []):
        print(f"    SUPPORT: {f.get('source_name')} | {f.get('source_url')}")
    for f in (findings.get("contradicting") or []):
        print(f"    CONTRADICT: {f.get('source_name')} | {f.get('source_url')}")
    
    results.append((key, data))
