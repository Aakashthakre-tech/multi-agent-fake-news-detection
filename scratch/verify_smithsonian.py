"""
scratch/verify_smithsonian.py
=============================
The BEFORE run retrieved the Smithsonian's Apollo 11 page (si.edu) with a
chunk that explicitly states the landing. That chunk was classified
RELATED_CONTEXT because of the date/event defect.

This re-runs the analyzer on that SAME captured chunk with the fix in place,
to show what the Smithsonian source now yields, and what a two-publisher
corroboration would produce.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

trace = json.loads((ROOT / "scratch" / "apollo11_trace.json").read_text(encoding="utf-8"))
before = json.loads((ROOT / "scratch" / "apollo11_api_response.json").read_text(encoding="utf-8"))

print("=" * 108)
print("THE SMITHSONIAN CHUNK CAPTURED IN THE BEFORE RUN")
print("=" * 108)
si = [c for c in trace["chunks"] if "si.edu" in str(c.get("url"))]
if not si:
    print("no si.edu chunk in the captured trace")
    raise SystemExit(0)

for c in si:
    print(f"  url        : {c.get('url')}")
    print(f"  group      : {c.get('independence_group')}")
    print(f"  BEFORE     : class={c.get('evidence_class')} date={c.get('date_status')} "
          f"evt={c.get('event_match')} nli={c.get('raw_nli_label')}/{c.get('raw_nli_score')}")
    print(f"  text       : {str(c.get('chunk'))[:600]}")

print()
print("=" * 108)
print("THE SAME CHUNK, RE-CLASSIFIED WITH THE FIX (functions called directly)")
print("=" * 108)

from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
)
from source_reliability import get_source_profile

text = si[0].get("chunk")
attrs = extract_claim_attributes(CLAIM)
profile = build_claim_profile(CLAIM)
date_check = check_date_compatibility(CLAIM, text, attrs)
level, signals = evaluate_event_match(
    CLAIM, text, attrs, date_check["status"], 0.7, nli_entails=True
)
klass = classify_evidence(
    {
        "date_status": date_check["status"],
        "event_match": level,
        "similarity": 0.7,
        "raw_nli_label": si[0].get("raw_nli_label"),
        "raw_nli_score": si[0].get("raw_nli_score"),
        "nli_status": "ok",
        "location_match": signals["location_match"],
        "claim_has_location": signals.get("claim_has_location", True),
        "detail_coverage": signals["detail_coverage"],
        "debunking": False,
    },
    "scraped_article",
    get_source_profile(si[0]["url"], "scraped_article"),
)

print(f"  date_status        : {date_check['status']}")
print(f"                       {date_check['reason']}")
print(f"  event_match        : {level}")
print(f"  claim_has_location : {signals.get('claim_has_location')}")
print(f"  location_ok        : {signals.get('location_ok')}")
print(f"  detail_coverage    : {signals['detail_coverage']}")
print(f"  nli                : {si[0].get('raw_nli_label')} {si[0].get('raw_nli_score')}")
print(f"  CLASSIFICATION     : {klass}")

print()
print("=" * 108)
print("TWO INDEPENDENT PRIMARY PUBLISHERS (Smithsonian + RocketSTEM) -> GATE")
print("=" * 108)

def _chunk(source_id, url, text, nli_label, nli_score, score=0.9):
    return {
        "source_id": source_id, "url": url,
        "domain": url.split("/")[2], "title": "t",
        "evidence_type": "scraped_article", "chunk": text,
        "similarity": 0.7, "nli_label": nli_label, "nli_score": nli_score,
        "raw_nli_label": nli_label, "raw_nli_score": nli_score,
        "source_reliability": 0.95, "source_tier": "official",
        "independence_group": url.split("/")[2].replace("www.", ""),
        "is_primary_source": True, "evidence_score": score,
        "evidence_class": "SUPPORT", "date_status": "match",
        "event_match": "strong", "location_match": False,
        "detail_coverage": 0.9, "nli_status": "ok", "debunking": False,
    }

results = [
    _chunk(1, "https://www.si.edu/spotlight/apollo-11", text,
           si[0].get("raw_nli_label"), si[0].get("raw_nli_score")),
    _chunk(2, "https://www.rocketstem.org/apollo-50th-celebration",
           "In July of 1969, the Apollo 11 mission marked humanity's first steps "
           "on the Moon by Neil Armstrong and Buzz Aldrin.",
           "entailment", 0.79),
]

from evidence_fusion import calculate_source_agreement, calculate_fusion_score
from final_verdict import deterministic_gate, generate_final_verdict

agreement = calculate_source_agreement(results)
gate = deterministic_gate(agreement, results)
verdict = generate_final_verdict(
    results, agreement, calculate_fusion_score(results, agreement),
    "Verdict:\nREAL\n\nConfidence:\n95%\n\nReasoning:\nTwo independent primary publishers.",
    "Verdict:\nREAL\n\nConfidence:\n93%\n\nReasoning:\nAccept.",
)

print(f"  independence_groups        : {agreement['independence_groups']}")
print(f"  independent_supporting      : {agreement['independent_supporting']}")
print(f"  primary_supporting_groups   : {agreement['primary_supporting_groups']}")
print(f"  article_supporting_groups   : {agreement['article_supporting_groups']}")
print(f"  evidence_sufficiency        : {agreement['evidence_sufficiency']}")
print(f"  permitted verdicts          : {sorted(gate['permitted_verdicts'])}")
print(f"  gate reason                 : {gate['permitted_reason']}")
print(f"  FINAL VERDICT               : {verdict['verdict']}")
print(f"  confidence                  : {verdict['confidence']} ({verdict['confidence_basis']})")
print(f"  supporting sources          : {[e['domain'] for e in verdict['supporting_evidence']]}")
