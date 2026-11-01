"""scratch/verify_api_final_fields.py - confirm the API exposes the final-assessment fields."""
import json
import urllib.request

BASE = "http://127.0.0.1:8001"
CLAIM = "The Apollo 11 mission landed humans on the Moon on July 20, 1969."

req = urllib.request.Request(
    BASE + "/verify/text",
    data=json.dumps({"claim": CLAIM}).encode(),
    headers={"Content-Type": "application/json"},
)
d = json.loads(urllib.request.urlopen(req, timeout=1800).read())

fact = d["factual_verification"]
print("=== factual_verification (live API) ===")
for k in sorted(fact.keys()):
    v = fact[k]
    if isinstance(v, str) and len(v) > 100:
        v = v[:100] + "..."
    print(f"  {k:<34} {v}")

print()
print("=== final-assessment fields required by the UI ===")
required = [
    "verdict", "confidence", "confidence_basis", "evidence_sufficiency",
    "independent_supporting", "independent_contradicting",
    "contradiction_score", "entailment_score", "source_agreement",
    "fusion_score", "total_sources", "permitted_verdicts",
    "deterministic_gate_reason", "reasoning",
]
missing = [k for k in required if k not in fact]
print("  missing:", missing or "none")

print()
print("=== status ===")
print("  result.status:", d.get("status"))
print("  raw.final_verdict keys present:",
      sorted((d.get('raw', {}).get('final_verdict') or {}).keys())[:8], "...")
