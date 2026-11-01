import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time, json
from pipeline import run_research_pipeline

claim = "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969."
t0 = time.time()
state = run_research_pipeline(claim)
elapsed = time.time() - t0
fv = state.get("final_verdict", {})
agreement = state.get("source_agreement", {})
print("=== FINAL VERDICT ===")
print("verdict:", fv.get("verdict"))
print("confidence:", fv.get("confidence"), "% basis:", fv.get("confidence_basis"))
print("evidence_sufficiency:", fv.get("evidence_sufficiency"))
print("independent_supporting:", fv.get("independent_supporting"))
print("independent_contradicting:", fv.get("independent_contradicting"))
print("primary_supporting_groups:", fv.get("primary_supporting_groups"))
print("article_supporting_groups:", fv.get("article_supporting_groups"))
print("total_sources:", fv.get("total_sources"))
print("fusion_score:", fv.get("fusion_score"))
print("source_agreement:", fv.get("source_agreement"))
print("entailment_score:", fv.get("entailment_score"))
print("contradiction_score:", fv.get("contradiction_score"))
print("verification_coverage:", fv.get("verification_coverage"))
print("permitted_verdicts:", fv.get("permitted_verdicts"))
print("reason:", fv.get("reasoning"))
print("agent_verdict:", fv.get("verification_verdict"), "critic:", fv.get("critic_verdict"), "agents_agree:", fv.get("agents_agree"))
print("elapsed_sec:", round(elapsed, 1))
print("=== SEARCH ERRORS ===")
for e in state.get("search_errors", []):
    print(" -", e)
print("=== SEARCH PLANS ===")
for p in state.get("search_plans", []):
    print(" -", p["label"], p["intent"], p["topic"], p["query"])
print("=== SCRAPE FAILURES ===")
for f in state.get("scrape_failures", []):
    print(" -", f.get("url"), "|", f.get("reason"), "| fallback:", f.get("fallback_used"))
print("=== SUPPORTING EVIDENCE ===")
for e in fv.get("supporting_evidence", []):
    print(" -", e.get("url"), "|", e.get("domain"), "| score", e.get("evidence_score"), "|", e.get("chunk", "")[:120])
print("=== CONTRADICTING EVIDENCE ===")
for e in fv.get("contradicting_evidence", []):
    print(" -", e.get("url"), "|", e.get("domain"), "| score", e.get("evidence_score"), "|", e.get("chunk", "")[:120])
