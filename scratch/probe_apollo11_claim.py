"""scratch/probe_apollo11_claim.py - local (no network) probes of the claim parser."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility,
    extract_claim_attributes,
    extract_dates,
    extract_years,
)

print("=" * 90)
print("CLAIM PARSE")
print("=" * 90)
profile = build_claim_profile(CLAIM)
for key in ("clean_claim", "dates", "years", "primary_date", "has_date",
            "locations", "location_terms", "keywords", "phrases"):
    print(f"  {key:<18} {profile.get(key)}")

attrs = extract_claim_attributes(CLAIM)
print()
print(f"  claim_date (day/month/year) : {attrs.get('claim_date')}")
print(f"  claim_years                 : {attrs.get('claim_years')}")
print(f"  has_date                    : {attrs.get('has_date')}")
print(f"  locations                   : {attrs.get('locations')}")
print(f"  location_terms              : {attrs.get('location_terms')}")
print(f"  keywords                    : {attrs.get('keywords')}")
print(f"  phrases                     : {attrs.get('phrases')}")

print()
print("=" * 90)
print("DATE COMPATIBILITY against a NASA-style passage")
print("=" * 90)
passages = [
    "Apollo 11 launched on July 16, 1969, and landed the first humans on the "
    "Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar "
    "surface while Michael Collins remained in orbit.",
    "The Apollo 11 mission took place in July 1969, and the first humans "
    "walked on the Moon.",
    "NASA's Apollo 11 mission landed humans on the Moon in July 1969.",
]
for text in passages:
    result = check_date_compatibility(CLAIM, text, attrs)
    dates = extract_dates(text)
    years = extract_years(text)
    print()
    print(f"  passage: {text[:90]}...")
    print(f"    extracted dates: {[(d['day'], d['month'], d['year']) for d in dates]}")
    print(f"    extracted years: {sorted(years)}")
    print(f"    -> status={result.get('status')}  reason={result.get('reason')}")
