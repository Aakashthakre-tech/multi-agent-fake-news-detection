import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from evidence_analyzer import split_into_chunks, check_nli, check_date_compatibility, evaluate_event_match, classify_evidence, extract_claim_attributes
from test_evidence_retrieval_reliability import CLAIM, NASA_ARTICLE
from source_reliability import get_source_profile

chunks = split_into_chunks(NASA_ARTICLE)
attrs = extract_claim_attributes(CLAIM)
profile = get_source_profile("https://www.nasa.gov/apollo", "scraped_article")

for lo in range(len(chunks)):
    for hi in range(lo + 1, min(lo + 4, len(chunks) + 1)):
        window = "\n\n".join(chunks[lo:hi])
        dc = check_date_compatibility(CLAIM, window, attrs)
        nli = check_nli(CLAIM, window)
        lvl, sigs = evaluate_event_match(CLAIM, window, attrs, dc["status"], 0.6, nli["label"] == "entailment")
        klass = classify_evidence({
            "date_status": dc["status"],
            "event_match": lvl,
            "similarity": 0.6,
            "raw_nli_label": nli["label"],
            "raw_nli_score": nli["score"],
            "location_match": sigs["location_match"],
            "claim_has_location": False,
            "detail_coverage": sigs["detail_coverage"],
            "debunking": False,
            "passage_kind": "window" if hi > lo + 1 else "paragraph"
        }, "scraped_article", profile)
        if "Eagle has landed" in window or "first human foot" in window:
            print(f"lo={lo} hi={hi} klass={klass} date={dc['status']} nli={nli['label']} {nli['score']:.4f} evt={lvl} cov={sigs['detail_coverage']}")
