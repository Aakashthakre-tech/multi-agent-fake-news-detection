"""scratch/debug_mars.py - why is the contradicted claim not FAKE?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_text_evaluation_set import (
    ARTICLES, CLAIM_APOLLO_MARS, _stub_nli,
)
from claim_profile import build_claim_profile
from evidence_analyzer import (
    check_date_compatibility, evaluate_event_match, extract_claim_attributes,
)
from evidence_matcher import calculate_similarity_batch as sim_batch

attrs = extract_claim_attributes(CLAIM_APOLLO_MARS)
profile = build_claim_profile(CLAIM_APOLLO_MARS)
print("claim_date   :", attrs.get("claim_date"))
print("locations    :", attrs.get("locations"))
print("location_terms:", attrs.get("location_terms"))
print("keywords     :", profile.get("keywords"))
print("phrases      :", profile.get("phrases"))
print()

for key in ("nasa-mars", "smithsonian-mars"):
    article = ARTICLES[key]
    body = "\n\n".join(article["paragraphs"])
    content = f"TITLE: {article['title']}\n\n{body}"
    print("=" * 92)
    print(key)
    print("=" * 92)
    for c in content.split("\n\n"):
        dc = check_date_compatibility(CLAIM_APOLLO_MARS, c, attrs)
        sim = sim_batch(CLAIM_APOLLO_MARS, [c])[0]
        nli = _stub_nli(CLAIM_APOLLO_MARS, c)
        lvl, sig = evaluate_event_match(
            CLAIM_APOLLO_MARS, c, attrs, dc["status"], sim, nli_entails=True
        )
        print(f"  chunk: {c[:90]}")
        print(f"    date={dc['status']:<12} sim={sim:<8} cov={sig.get('detail_coverage'):<8} "
              f"evt={lvl:<9} nli={nli['label']}")
        print(f"    location_match={sig.get('location_match')} location_ok={sig.get('location_ok')}")
