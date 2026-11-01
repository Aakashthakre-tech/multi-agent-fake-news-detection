"""scratch/verify_claim_ranker_fixes.py
Check the three claim-ranker fixes in both directions:
  1. the rejection reason must name the real blocker;
  2. the selection reason must name a verb that matches its category;
  3. Devanagari entities must be recognised, without letting noise in.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video_module.claim_ranker import (  # noqa: E402
    evaluate_candidate_claim,
    extract_potential_entities,
)

print("=" * 78)
print("FIX 3: Devanagari entity recognition")
print("=" * 78)
marathi = "राज्य सरकार ने 25 मार्च 2023 को नई नीति की घोषणा की।"
print("  entities:", extract_potential_entities(marathi))
print("  english  :", extract_potential_entities("The Maharashtra government announced a new policy on 25 March 2023."))
print()

print("=" * 78)
print("NOISE MUST STILL BE REJECTED")
print("=" * 78)
noise = [
    "Hello everyone, welcome to the channel.",
    "Sorry, my microphone was not working.",
    "Thank you very much, and we will see you in the next video.",
    "um, uh, you know, like, I mean...",
    "Please subscribe to my channel and hit the bell icon.",
    "राज्य सरकार ने 25 मार्च 2023 को नई नीति की घोषणा की।",  # real Marathi claim -> should PASS
]
for text in noise:
    r = evaluate_candidate_claim(text, source="whisper")
    print(f"  [{r['selection_status']:8s}] {text[:55]!r}")
    print(f"              reason: {r['selection_reason'][:120]}")
print()

print("=" * 78)
print("FIX 1: rejection reason must not misdescribe the evidence")
print("=" * 78)
anchor_only = "Suhana 25 March 2023 abp Hf"  # entities + date, no predicate
r = evaluate_candidate_claim(anchor_only, source="ocr")
print("  claim   :", anchor_only)
print("  entities:", r.get("entities"))
print("  dates   :", r.get("dates"))
print("  status  :", r["selection_status"])
print("  reason  :", r["selection_reason"])
assert "lacks specific entities" not in r["selection_reason"], "reason still lies about the anchors"
print("  OK: reason names the real blocker")
print()

print("=" * 78)
print("FIX 2: the reported verb must belong to the reported category")
print("=" * 78)
from video_module.claim_ranker import (  # noqa: E402
    _ACTION_POLICY_VERBS,
    _INCIDENT_VERBS,
    _PREDICATE_VERBS,
)

candidates = [
    "The new policy is expected to take effect next month in Maharashtra.",
    "Police reported that two people were injured during the protest in Delhi.",
]
for text in candidates:
    r = evaluate_candidate_claim(text, source="whisper")
    print(f"  {r['selection_status']:8s} | {r['selection_reason'][:150]}")
