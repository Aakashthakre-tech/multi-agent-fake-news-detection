"""scratch/verify_apollo1970_fix.py - does the fix classify the UN passage correctly?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility, classify_evidence, evaluate_event_match,
    extract_claim_attributes,
)
from source_reliability import get_source_profile

PASSAGE = (
    "This Moon Rock was retrieved from the moon during the USA NASA Apollo 11 "
    "space mission, crewed by astronauts Neil A. Armstrong, Edwin E. Aldrin Jr., "
    "and Michael Collins, on their 20 July 1969 moon landing. They presented "
    "this gift to the United Nations at the one-year anniversary of that "
    "flight, on 20 July 1970."
)

CLAIMS = {
    "1970 (the bug)": "The Apollo 11 mission landed humans on the Moon on July 20, 1970.",
    "1969 (correct)": "The Apollo 11 mission landed humans on the Moon on July 20, 1969.",
    "1969 month-only": "The Apollo 11 mission landed humans on the Moon in July 1969.",
}

for label, claim in CLAIMS.items():
    attrs = extract_claim_attributes(claim)
    profile = build_claim_profile(claim)
    dc = check_date_compatibility(claim, PASSAGE, attrs)
    level, sig = evaluate_event_match(
        claim, PASSAGE, attrs, dc["status"], 0.7, nli_entails=True
    )
    signals = {
        "similarity": 0.85,
        "date_status": dc["status"],
        "event_match": level,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.85,
        "nli_status": "ok",
        "location_match": sig["location_match"],
        "claim_has_location": sig.get("claim_has_location", False),
        "detail_coverage": sig["detail_coverage"],
        "debunking": False,
        "passage_kind": "paragraph",
        "event_date_conflict": dc.get("event_date_conflict"),
        "event_date_mismatch": dc.get("event_date_mismatch"),
    }
    klass = classify_evidence(
        signals, "scraped_article",
        get_source_profile("https://www.un.org/ungifts/es/node/3986", "scraped_article"),
    )
    print("=" * 92)
    print(f"{label}: {claim}")
    print("=" * 92)
    print(f"  date_status          : {dc['status']}")
    print(f"  event_date_conflict  : {dc.get('event_date_conflict')}")
    print(f"  event_date_mismatch  : {dc.get('event_date_mismatch')}")
    print(f"  event_match          : {level}")
    print(f"  detail_coverage      : {sig['detail_coverage']}")
    print(f"  CLASSIFICATION       : {klass}")
    print(f"  reason               : {dc['reason'][:170]}")
    print()
