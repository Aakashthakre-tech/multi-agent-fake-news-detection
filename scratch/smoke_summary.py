"""Print the saved smoke test results."""
import json
from pathlib import Path

path = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System/scratch/smoke_final_text.json")
with open(path, encoding="utf-8") as f:
    data = json.load(f)

for key, d in data.items():
    print("=" * 80)
    print(f"MODALITY: {d.get('modality')}")
    print(f"Claim: {d.get('primary_claim', 'N/A')}")
    print("-" * 80)
    fact = d.get("factual_verification") or {}
    print(f"  Verdict: {fact.get('verdict')}")
    print(f"  Confidence: {fact.get('confidence')} ({fact.get('confidence_basis')})")
    print(f"  Evidence sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"  Independent supporting: {fact.get('independent_supporting')}")
    print(f"  Independent contradicting: {fact.get('independent_contradicting')}")
    print(f"  Primary supporting groups: {fact.get('primary_supporting_groups')}")
    print(f"  Primary contradicting groups: {fact.get('primary_contradicting_groups')}")
    print(f"  Search queries: {fact.get('search_queries')}")
    print(f"  Search errors: {fact.get('search_errors')}")
    print(f"  Scrape failures: {fact.get('scrapes_failed')}")
    report = d.get("research_report") or {}
    exec_summary = report.get("executive_summary") or {}
    methodology = report.get("methodology") or {}
    findings = report.get("findings") or {}
    print(f"  Reports - supporting: {len(findings.get('supporting') or [])}")
    print(f"  Reports - contradicting: {len(findings.get('contradicting') or [])}")
    print(f"  findings - supporting sources: {[f.get('source_url') for f in (findings.get('supporting') or [])]}")
    print(f"  findings - contradicting sources: {[f.get('source_url') for f in (findings.get('contradicting') or [])]}")
    print(f"  methodology.search_queries: {len(methodology.get('search_queries') or [])}")
    print(f"  methodology.search_errors: {methodology.get('search_errors')}")
    print(f"  methodology.scrapes_failed: {len(methodology.get('scrapes_failed') or [])}")
    print(f"  methodology.sources_retrieved: {methodology.get('sources_retrieved')}")
    print(f"  methodology.articles_scraped: {methodology.get('articles_scraped')}")
    print(f"  executive_summary.limitations: {exec_summary.get('limitations')}")
    print(f"  executive_summary.agent_narrative: {str(exec_summary.get('agent_narrative'))[:200]}")
    print(f"  executive_summary.critic_review: {str(exec_summary.get('critic_review'))[:200]}")
    print()
