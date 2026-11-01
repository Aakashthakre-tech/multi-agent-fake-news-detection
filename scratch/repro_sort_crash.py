"""scratch/repro_sort_crash.py - prove the crash and verify the fix."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import check_date_compatibility, extract_claim_attributes

# The failing shape: a DAY-precision claim, and evidence containing a
# MONTH-precision date ("July 2024", no day) in the SAME year - so it is a
# conflict in its own right - alongside an integer-day conflict.
CLAIM = "The council approved the project on 12 Aug 2024."
EVIDENCE = (
    "The council approved the project on 15 Aug 2024. The original programme "
    "was approved in July 2024."
)

attrs = extract_claim_attributes(CLAIM)
print("claim_date :", attrs.get("claim_date"))
print()

# ---- the original expression -------------------------------------------
all_dates = [
    {"day": 15, "month": 8, "year": 2024, "month_name": "August"},
    {"day": None, "month": 7, "year": 2024, "month_name": "July"},
]
claim_key = (attrs["claim_date"]["day"], attrs["claim_date"]["month"],
             attrs["claim_date"]["year"])
print("--- original expression ---")
try:
    sorted(
        key for key in {(d["day"], d["month"], d["year"]) for d in all_dates}
        if key != claim_key
        and (
            (key[2] == claim_key[2] and key[:2] != claim_key[:2])
            or (key[2] != claim_key[2] and not False)
        )
    )
    print("  no crash (unexpected)")
except TypeError as exc:
    print(f"  TypeError: {exc}")

print()
print("--- fixed function ---")
result = check_date_compatibility(CLAIM, EVIDENCE, attrs)
print("  status :", result["status"])
print("  reason :", result["reason"][:200])
print("  no crash")
