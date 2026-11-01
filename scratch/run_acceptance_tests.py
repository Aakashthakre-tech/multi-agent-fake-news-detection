import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multimodal_router import verify_text

def run_test(claim_label, claim_text):
    print("=" * 80)
    print(f"RUNNING ACCEPTANCE TEST: {claim_label}")
    print(f"Claim: {claim_text}")
    print("=" * 80)
    t0 = time.time()
    res = verify_text(claim_text)
    elapsed = time.time() - t0

    fact = res.get("factual_verification", {}) or {}
    report = res.get("research_report", {}) or {}
    summary = report.get("executive_summary", {}) or {}
    method = report.get("methodology", {}) or {}
    findings = report.get("findings", []) or []

    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {res.get('status')}")
    print(f"Verdict: {fact.get('verdict')}")
    print(f"Confidence: {fact.get('confidence')} (Basis: {fact.get('confidence_basis')})")
    print(f"Sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"Independent Supporting Publishers: {fact.get('independent_supporting')}")
    print(f"Independent Contradicting Publishers: {fact.get('independent_contradicting')}")
    print(f"Primary Supporting Groups: {fact.get('primary_supporting_groups')}")
    print(f"Deterministic Gate Reason: {fact.get('deterministic_gate_reason')}")
    print(f"Sources Discovered/Retrieved: {method.get('sources_discovered', 0)} / {method.get('sources_retrieved', 0)}")
    print(f"Articles Scraped: {method.get('articles_scraped', 0)}")
    print(f"Scrape Failures: {len(method.get('scrapes_failed', []))}")
    print(f"Search Errors: {method.get('search_errors', [])}")
    print("\nFindings:")
    if isinstance(findings, dict):
        for category, items in findings.items():
            print(f"  Category: {category} ({len(items)} items)")
            for item in items[:2]:
                print(f"    - [{item.get('status')}] {item.get('source_name')} ({item.get('source_domain')}): {item.get('title')}")
                print(f"      URL: {item.get('source_url')}")
                print(f"      Excerpt: {str(item.get('excerpt'))[:150]}...")
    elif isinstance(findings, list):
        for f in findings[:4]:
            print(f"  - [{f.get('status')}] {f.get('source_name')} ({f.get('source_domain')}): {f.get('title')}")
            print(f"    URL: {f.get('source_url')}")
            print(f"    Excerpt: {str(f.get('excerpt'))[:150]}...")
    
    out_file = ROOT / "scratch" / f"acceptance_{claim_label.lower().replace(' ', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2, default=str)
    print(f"\nSaved full response to {out_file}")
    return res

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    claims = {
        "A": ("Test_A", "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969."),
        "B": ("Test_B", "The Apollo 11 mission landed humans on the Moon on July 20, 1970."),
        "C": ("Test_C", "Joe Biden withdrew from the 2024 United States presidential race in July 2024."),
        "D": ("Test_D", "This news is misleading."),
    }
    label, claim = claims[which]
    run_test(label, claim)
