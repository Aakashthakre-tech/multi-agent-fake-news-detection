"""
scratch/probe_apollo11_classification.py
=========================================
Reproduces every step of the classification chain for a relevant Apollo 11
passage, calling the responsible functions directly.

For each step it prints the real inputs and outputs, and which precondition of
`classify_evidence` fails.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
    _detail_coverage,
)
from source_reliability import get_source_profile

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

PASSAGES = {
    "NASA-style (contains 'July')": (
        "Apollo 11 was the mission that landed the first humans on the Moon. "
        "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the "
        "Sea of Tranquility on July 20, 1969, while Michael Collins remained "
        "in lunar orbit aboard the Columbia."
    ),
    "Smithsonian-style (contains 'July')": (
        "On July 20, 1969, Apollo 11 landed on the Moon and Neil Armstrong "
        "and Edwin 'Buzz' Aldrin became the first humans to walk on the "
        "lunar surface."
    ),
    "NASA-style WITHOUT 'July'": (
        "Apollo 11 was the mission that landed the first humans on the Moon. "
        "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the "
        "Sea of Tranquility while Michael Collins remained in lunar orbit."
    ),
}

attrs = extract_claim_attributes(CLAIM)
profile = build_claim_profile(CLAIM)

print("=" * 96)
print("CLAIM ATTRIBUTES (as the matcher sees them)")
print("=" * 96)
print(f"  primary_date          : {profile['primary_date']}")
print(f"  has_date              : {profile['has_date']}")
print(f"  years                 : {profile['years']}")
print(f"  locations             : {profile['locations']}")
print(f"  location_terms        : {profile['location_terms']}")
print(f"  keywords              : {profile['keywords']}")
print(f"  phrases               : {profile['phrases']}")

for label, text in PASSAGES.items():
    print()
    print("=" * 96)
    print(f"PASSAGE: {label}")
    print("=" * 96)

    date_result = check_date_compatibility(CLAIM, text, attrs)
    date_status = date_result.get("status")
    print(f"  1. check_date_compatibility -> status={date_status}")
    print(f"       reason: {date_result.get('reason')}")

    similarity = 0.75  # a relevant passage against this claim
    event_level, signals = evaluate_event_match(
        CLAIM, text, attrs, date_status, similarity, nli_entails=True
    )
    print(f"  2. evaluate_event_match    -> level={event_level}")
    print(f"       location_match={signals['location_match']} "
          f"location_ok={signals.get('location_ok')} "
          f"claim_has_location={signals.get('claim_has_location')}")
    print(f"       detail_coverage={signals['detail_coverage']} "
          f"similarity={signals.get('similarity')}")

    coverage, detail = _detail_coverage(
        {"keywords": profile["keywords"], "phrases": profile["phrases"]},
        text, text.lower(),
    )
    print(f"  3. _detail_coverage        -> coverage={coverage}")
    print(f"       detail: {detail}")

    source_profile = get_source_profile("https://www.nasa.gov/apollo-11", "scraped_article")
    nli = {"raw_nli_label": "entailment", "raw_nli_score": 0.997,
           "nli_status": "ok", "similarity": similarity}
    evidence_class = classify_evidence(
        {
            "date_status": date_status,
            "event_match": event_level,
            "similarity": similarity,
            "raw_nli_label": nli["raw_nli_label"],
            "raw_nli_score": nli["raw_nli_score"],
            "nli_status": nli["nli_status"],
            "location_match": signals["location_match"],
            "claim_has_location": signals.get("claim_has_location", True),
            "detail_coverage": coverage,
            "debunking": False,
        },
        "scraped_article",
        source_profile,
    )
    print(f"  4. classify_evidence       -> {evidence_class}")
    print(f"       source_profile: is_primary={source_profile['is_primary']} "
          f"tier={source_profile['tier']} reliability={source_profile['reliability']}")

    print("       SUPPORT preconditions:")
    checks = [
        ("not is_snippet", True),
        ("is_primary", source_profile["is_primary"]),
        ("nli usable", nli["nli_status"] not in ("error", "unavailable")),
        ("nli_label == entailment", nli["raw_nli_label"] == "entailment"),
        ("nli_score >= 0.70", nli["raw_nli_score"] >= 0.70),
        ("similarity >= 0.40", similarity >= 0.40),
        ("event_match == strong", event_level == "strong"),
        ("location_ok", signals.get("location_ok")),
        ("coverage >= 0.30", coverage >= 0.30),
        ("date_status in (match, same_day_published)",
         date_status in ("match", "same_day_published")),
    ]
    for name, ok in checks:
        print(f"         {'PASS' if ok else 'FAIL'}  {name}")
