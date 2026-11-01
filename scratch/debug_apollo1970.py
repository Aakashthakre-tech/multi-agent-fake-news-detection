import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_analyzer import (
    extract_claim_attributes, check_date_compatibility,
    evaluate_event_match, classify_evidence,
)
from source_reliability import get_source_profile

claim = "The Apollo 11 mission landed humans on the Moon on July 20, 1970."
passage = "Fifty-Two Years ago, on July 20th, 1969, two astronauts set foot on the moon for the first time in human history!"

attrs = extract_claim_attributes(claim)
date_check = check_date_compatibility(claim, passage, attrs)
print("date_check status:", date_check["status"])
print("date_check event_date_conflict:", date_check.get("event_date_conflict"))
print("date_check event_date_mismatch:", date_check.get("event_date_mismatch"))
print("date_check reason:", date_check.get("reason"))

level, signals = evaluate_event_match(
    claim, passage, attrs, date_check["status"], 0.70,
    nli_entails=False,
    event_date_conflict=bool(date_check.get("event_date_conflict")),
)
print("event_match:", level)

merged = {
    "similarity": 0.70,
    "date_status": date_check["status"],
    "event_match": level,
    "raw_nli_label": "contradiction",
    "raw_nli_score": 0.90,
    "nli_status": "ok",
    "location_match": signals["location_match"],
    "claim_has_location": signals.get("claim_has_location", False),
    "detail_coverage": signals["detail_coverage"],
    "debunking": False,
    "passage_kind": "paragraph",
    "event_date_conflict": bool(date_check.get("event_date_conflict")),
    "event_date_mismatch": bool(date_check.get("event_date_mismatch")),
}
profile = get_source_profile(
    "https://astronautfoods.com/blogs/news/15-facts-about-apollo-11-you-didnt-know",
    "scraped_article",
)
print("is_primary:", profile.get("is_primary"))
klass = classify_evidence(merged, "scraped_article", profile)
print("evidence_class:", klass)
