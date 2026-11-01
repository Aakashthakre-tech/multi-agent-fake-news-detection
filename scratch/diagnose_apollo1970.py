"""
scratch/diagnose_apollo1970.py
=============================
Diagnose why the UN Moon Rock passage was classified as SUPPORT for a claim
asserting the Apollo 11 landing happened on 20 July 1970.

Measures the real signals: NLI label/score, date compatibility, event match,
detail coverage, and the per-sentence date/event association.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility, evaluate_event_match, extract_claim_attributes,
    extract_dates,
)
from nli_checker import check_nli

CLAIM_1970 = "The Apollo 11 mission landed humans on the Moon on July 20, 1970."
CLAIM_1969 = "The Apollo 11 mission landed humans on the Moon on July 20, 1969."

PASSAGE = (
    "This Moon Rock was retrieved from the moon during the USA NASA Apollo 11 "
    "space mission, crewed by astronauts Neil A. Armstrong, Edwin E. Aldrin Jr., "
    "and Michael Collins, on their 20 July 1969 moon landing. They presented "
    "this gift to the United Nations at the one-year anniversary of that "
    "flight, on 20 July 1970."
)

for claim in (CLAIM_1970, CLAIM_1969):
    print("=" * 94)
    print(f"CLAIM: {claim}")
    print("=" * 94)
    attrs = extract_claim_attributes(claim)
    profile = build_claim_profile(claim)
    print(f"  claim_date   : {attrs.get('claim_date')}")
    print(f"  keywords    : {profile.get('keywords')}")
    print(f"  phrases     : {profile.get('phrases')}")

    dc = check_date_compatibility(claim, PASSAGE, attrs)
    print(f"  date_status : {dc['status']}  | {dc['reason'][:120]}")

    nli = check_nli(claim, PASSAGE)
    print(f"  NLI         : {nli['label']}/{nli['score']}  scores={nli.get('scores')}")

    level, sig = evaluate_event_match(
        claim, PASSAGE, attrs, dc["status"], 0.7, nli_entails=True
    )
    print(f"  event_match : {level}   coverage={sig.get('detail_coverage')} "
          f"location_ok={sig.get('location_ok')}")
    print()

print("=" * 94)
print("PER-SENTENCE DATE / EVENT ASSOCIATION")
print("=" * 94)
sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", PASSAGE) if s.strip()]
attrs = extract_claim_attributes(CLAIM_1970)
for i, sentence in enumerate(sentences, 1):
    dates = extract_dates(sentence)
    low = sentence.lower()
    print(f"  S{i}: {sentence[:110]}")
    print(f"      dates in sentence : {[(d['day'], d['month'], d['year']) for d in dates]}")
    print(f"      has 'apollo 11'   : {'apollo 11' in low}")
    print(f"      has 'moon landing' : {'moon landing' in low}")
    print(f"      has 'landed'       : {'landed' in low}")
