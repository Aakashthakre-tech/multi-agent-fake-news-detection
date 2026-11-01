"""scratch/probe_location_month.py - why did "July" become a location?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import _looks_like_place, _GEO_HINT_WORDS, _COUNTRIES
from claim_profile import extract_entities, strip_dates, extract_dates

TEXT = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

print("=" * 84)
print("Why 'July' is extracted as an entity/location")
print("=" * 84)
dates = extract_dates(TEXT)
print(f"  extract_dates(claim)      -> {dates}")
stripped = strip_dates(TEXT, dates)
print(f"  strip_dates(claim)        -> {stripped!r}")
print(f"  extract_entities(stripped)-> {extract_entities(stripped)}")
print()
print(f"  'july' in _GEO_HINT_WORDS : {'july' in _GEO_HINT_WORDS}")
print(f"  'july' in _COUNTRIES      : {'july' in _COUNTRIES}")
print(f"  _looks_like_place('July') : {_looks_like_place('July', TEXT.lower(), 48)}")
print()
print("  -> the month name survives strip_dates (it is not a parsed date) and")
print("     _looks_like_place accepts it, so it becomes a claim 'location'.")

print()
print("=" * 84)
print("Consequence: has_claim_location is True for a claim with no place in it")
print("=" * 84)
print("  evaluate_event_match uses `location_ok = location_match or not claim_has_location`.")
print("  With 'July' as the only location, a passage that does not literally")
print("  contain 'July' fails the location requirement and can never reach")
print("  event_match='strong', which SUPPORT requires.")
print()
for passage, label in [
    ("Apollo 11 landed the first humans on the Moon in 1969.", "no 'July'"),
    ("Apollo 11 landed the first humans on the Moon in July 1969.", "has 'July'"),
]:
    print(f"  [{label}] {passage}")
    print(f"      contains 'july': {'july' in passage.lower()}")
