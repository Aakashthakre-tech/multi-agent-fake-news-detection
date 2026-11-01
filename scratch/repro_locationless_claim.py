"""
scratch/repro_locationless_claim.py
===================================
Isolates the false negative found by the real Chandrayaan-3 trace.

The real run retrieved 9 sources (all with article text) and 26 passages, and
NOT ONE of them reached SUPPORT. Every chunk came back with
``event_match = weak`` even though the claim states no location at all.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import evaluate_event_match, extract_claim_attributes

ARTICLE = (
    "India on Friday launched its Chandrayaan-3 mission on board an LVM3-M4 "
    "rocket from Satish Dhawan Space Centre in Sriharikota, Andhra Pradesh, "
    "the Indian Space Research Organisation said. The launch took place on "
    "14 July 2023."
)


def trace(claim, chunk, label):
    from evidence_analyzer import extract_claim_attributes as _eca

    attrs = build_claim_profile(claim)
    aattrs = _eca(claim)
    level, signals = evaluate_event_match(
        claim, chunk, attrs, "match", 0.8, nli_entails=True
    )
    print(f"--- {label} ---")
    print(f"  claim              : {claim}")
    print(f"  claim locations    : {attrs['locations']}")
    print(f"  claim_has_location : {aattrs.get('claim_has_location')}")
    print(f"  location_match     : {signals['location_match']}")
    print(f"  detail_coverage    : {signals['detail_coverage']}")
    print(f"  event_match        : {level}")
    print()
    return level, signals


print("=" * 78)
print("A claim that states NO location")
print("=" * 78)
no_location = (
    "The Indian Space Research Organisation launched the Chandrayaan-3 "
    "mission on 14 July 2023."
)
level_a, signals_a = trace(no_location, ARTICLE, "no location in the claim")

print("=" * 78)
print("The same claim WITH a location")
print("=" * 78)
with_location = (
    "The Indian Space Research Organisation launched the Chandrayaan-3 "
    "mission from Sriharikota in Andhra Pradesh on 14 July 2023."
)
level_b, signals_b = trace(with_location, ARTICLE, "location in the claim")

print("=" * 78)
print("VERDICT")
print("=" * 78)
print(f"  location-less claim -> event_match = {level_a}")
print(f"  located claim       -> event_match = {level_b}")
if level_a != "strong" and level_b == "strong":
    print("  >>> A claim that states no location can never reach event_match='strong',")
    print("      and SUPPORT requires event_match='strong'. Legitimate evidence is")
    print("      discarded for a reason that has nothing to do with the evidence.")
