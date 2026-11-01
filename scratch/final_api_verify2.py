import json
import urllib.request

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
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

results = {}
for key, claim in CLAIMS:
    print(f"=== {key}: {claim}")
    try:
        result, = verify(claim), None
    except Exception as e:
        print(f"  REQUEST FAILED: {e}")
        results[key] = {"error": str(e)}
        continue
    fact = result.get("factual_verification") or {}
    print(f"  verdict: {fact.get('verdict')}")
    print(f"  confidence: {fact.get('confidence')} ({fact.get('confidence_basis')})")
    print(f"  evidence_sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"  independent_supporting: {fact.get('independent_supporting')}")
    print(f"  independent_contradicting: {fact.get('independent_contradicting')}")
    print(f"  gate_reason: {fact.get('deterministic_gate_reason')}")
    results[key] = result

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 70)
for key, result in results.items():
    if "error" in result:
        print(f"{key}: FAILED - {result['error']}")
    else:
        fv = result.get("factual_verification", {})
        print(f"{key}: verdict={fv.get('verdict')}, confidence={fv.get('confidence')} ({fv.get('confidence_basis')}), sufficiency={fv.get('evidence_sufficiency')}, supporting={fv.get('independent_supporting')}, contradicting={fv.get('independent_contradicting')}")
