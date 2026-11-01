import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import run_research_pipeline

claim = "Water boils at 100 degrees Celsius at standard atmospheric pressure."

state = run_research_pipeline(claim)

fv = state.get("final_verdict", {})
agreement = state.get("source_agreement", {})
agreement_data = state.get("source_agreement", {})

print("=" * 60)
print("PIPELINE ACCEPTANCE TRACE")
print("=" * 60)
print(f"\nCLAIM: {claim}")
print(f"\nFINAL VERDICT: {fv.get('verdict')}")
print(f"Confidence: {fv.get('confidence')}%")
print(f"Confidence Basis: {fv.get('confidence_basis')}")
print(f"Evidence Sufficiency: {fv.get('evidence_sufficiency')}")
print(f"Independent Supporting: {fv.get('independent_supporting')}")
print(f"Independent Contradicting: {fv.get('independent_contradicting')}")
print(f"Permitted Verdicts: {fv.get('permitted_verdicts')}")
print(f"Reasoning: {fv.get('reasoning', 'N/A')}")

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)
search_results = state.get("search_results", [])
print(f"Total search results: {len(search_results)}")
for i, s in enumerate(search_results[:10], 1):
    print(f"  {i}. {s.get('url', 'N/A')} - {s.get('title', 'N/A')[:60]} - {s.get('scrape_status', 'N/A')}")

print("\n" + "=" * 60)
print("SOURCE DATA (after scraping)")
print("=" * 60)
source_data = state.get("source_data", [])
print(f"Total sources: {len(source_data)}")
successful_scrapes = 0
failed_scrapes = 0
snippet_only = 0
full_scrapes = 0
for s in source_data:
    scrape_status = s.get("scrape_status", "unknown")
    evidence_type = s.get("evidence_type", "unknown")
    if scrape_status == "SUCCESS" and evidence_type == "scraped_article":
        successful_scrapes += 1
        full_scrapes += 1
    elif scrape_status == "FAILED":
        failed_scrapes += 1
    elif evidence_type == "search_extract":
        snippet_only += 1
    print(f"  {source_data.index(s)+1}. {s.get('url', 'N/A')} - {evidence_type} - {scrape_status} - {s.get('scrape_diag', {}).get('error_category', 'N/A')}")

print(f"\n  Full scrapes: {full_scrapes}")
print(f"  Failed scrapes: {failed_scrapes}")
print(f"  Snippet-only: {snippet_only}")

print("\n" + "=" * 60)
print("EVIDENCE ANALYSIS")
print("=" * 60)
evidence_analysis = state.get("evidence_analysis", [])
print(f"Total evidence chunks: {len(evidence_analysis)}")
support_count = sum(1 for e in evidence_analysis if e.get("evidence_class") == "SUPPORT")
contra_count = sum(1 for e in evidence_analysis if e.get("evidence_class") == "CONTRADICTION")
related_count = sum(1 for e in evidence_analysis if e.get("evidence_class") == "RELATED_CONTEXT")
non_evidence_count = sum(1 for e in evidence_analysis if e.get("evidence_class") == "NON_EVIDENCE")
print(f"  SUPPORT: {support_count}")
print(f"  CONTRADICTION: {contra_count}")
print(f"  RELATED_CONTEXT: {related_count}")
print(f"  NON_EVIDENCE: {non_evidence_count}")

print("\n" + "=" * 60)
print("SOURCE AGREEMENT")
print("=" * 60)
print(f"  Independent Supporting: {agreement_data.get('independent_supporting')}")
print(f"  Independent Contradicting: {agreement_data.get('independent_contradicting')}")
print(f"  Independent Groups: {agreement_data.get('independence_groups')}")
print(f"  Total Sources: {agreement_data.get('total_sources')}")

print("\n" + "=" * 60)
print("GOOGLE FACT CHECK")
print("=" * 60)
gf = state.get("google_fact_check")
if gf:
    print(f"  Status: {gf.get('status')}")
    print(f"  Error Category: {gf.get('error_category')}")
    print(f"  Error Message: {gf.get('error_message')}")
    claims = gf.get("claims", [])
    print(f"  Claims Returned: {len(claims)}")
    for c in claims[:5]:
        print(f"    - {c.get('text', 'N/A')[:80]}")
        for r in c.get("claimReview", []):
            print(f"      Publisher: {r.get('publisher', {}).get('name', 'N/A')}")
            print(f"        Rating: {r.get('textualRating', 'N/A')}")
            print(f"        URL: {r.get('url', 'N/A')}")
else:
    print("  No Google Fact Check result found in state")

print("\n" + "=" * 60)
print("FINAL VERDICT REASONING")
print("=" * 60)
print(f"Reasoning: {fv.get('reasoning', 'N/A')}")