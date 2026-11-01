"""scratch/final_policy_diag_check.py - live check of the policy diagnostics."""
import json
import urllib.request

BASE = "http://127.0.0.1:8001"

req = urllib.request.Request(
    BASE + "/verify/text",
    data=json.dumps({
        "claim": "The Reserve Bank of India raised the repo rate by 25 basis points "
                 "on 6 February 2024."
    }).encode(),
    headers={"Content-Type": "application/json"},
)
d = json.loads(urllib.request.urlopen(req, timeout=900).read())

print("=== factual_verification (top level) ===")
fact = d["factual_verification"]
for key in ("verdict", "confidence", "confidence_basis", "policy_data_valid",
            "policy_data_errors", "permitted_verdicts"):
    print(f"  {key:26s} {fact.get(key)}")

print()
print("=== research_report.executive_summary.policy_diagnostics ===")
pd = (d["research_report"].get("executive_summary") or {}).get("policy_diagnostics")
for key, value in (pd or {}).items():
    print(f"  {key:26s} {value}")

print()
print("=== raw.final_verdict.policy_data_* ===")
raw_fv = (d.get("raw") or {}).get("final_verdict") or {}
print("  policy_data_valid           ", raw_fv.get("policy_data_valid"))
print("  policy_data_errors          ", raw_fv.get("policy_data_errors"))
print()
print("NOTE: the response schema is additive - all pre-existing "
      "factual_verification keys are still present:")
print("  ", sorted(fact.keys()))
