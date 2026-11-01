"""scratch/debug_fixtures.py - why are the evaluation-set fixtures not classified?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_text_evaluation_set import (
    ARTICLES, CLAIM_APOLLO, _stub_nli, _scrape_stub, _ToolProxy,
)
from claim_profile import build_claim_profile
from evidence_analyzer import (
    analyze_evidence, check_date_compatibility, evaluate_event_match,
    extract_claim_attributes, calculate_similarity_batch,
)
from evidence_matcher import calculate_similarity_batch as sim_batch

article = ARTICLES["smithsonian-apollo11"]
body = "\n\n".join(article["paragraphs"])
content = f"TITLE: {article['title']}\n\n{body}"

print("=" * 92)
print("FIXTURE:", article["key"])
print("=" * 92)
print("content chars:", len(content))

attrs = extract_claim_attributes(CLAIM_APOLLO)
print("claim_date:", attrs.get("claim_date"))
print("locations  :", attrs.get("locations"))

chunks = content.split("\n\n")
for c in chunks:
    print()
    print("  CHUNK:", c[:110].replace("\n", " "))
    dc = check_date_compatibility(CLAIM_APOLLO, c, attrs)
    print("    date_status :", dc["status"], "|", dc["reason"][:90])
    sim = sim_batch(CLAIM_APOLLO, [c])[0]
    print("    similarity  :", sim)
    lvl, sig = evaluate_event_match(
        CLAIM_APOLLO, c, attrs, dc["status"], sim, nli_entails=True
    )
    print("    event_match :", lvl)
    print("    coverage    :", sig.get("detail_coverage"))
    print("    location_ok :", sig.get("location_ok"), "claim_has_location:", sig.get("claim_has_location"))
    nli = _stub_nli(CLAIM_APOLLO, c)
    print("    stub nli    :", nli["label"], nli["score"])

print()
print("=" * 92)
print("FULL ANALYZER RUN ON THE FIXTURE")
print("=" * 92)
import pipeline as pipeline_mod
from unittest.mock import patch

with patch("evidence_analyzer.check_nli", _stub_nli):
    results = analyze_evidence(
        CLAIM_APOLLO,
        [{
            "url": article["url"], "domain": article["domain"],
            "title": article["title"], "content": content,
            "evidence_type": "scraped_article", "scrape_status": "SUCCESS",
        }],
        verbose=False,
    )
for r in results:
    print(f"  class={r['evidence_class']:<16} kind={r.get('passage_kind'):<9} "
          f"date={r['date_status']:<8} evt={r['event_match']:<8} "
          f"nli={r['nli_label']}/{r['nli_score']} sim={r['similarity']} "
          f"score={r['evidence_score']} det={r['detail_coverage']}")
    print(f"      {r['chunk'][:120]}")
