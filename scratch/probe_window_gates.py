"""
scratch/probe_window_gates.py
=============================
Go/no-go test for the candidate fix.

A window is only useful if it still clears every existing SUPPORT gate:
  - not a snippet
  - primary publisher
  - NLI entailment, score >= 0.70
  - similarity >= 0.40
  - event_match == 'strong'
  - location_ok
  - detail_coverage >= 0.30
  - date_status in ('match', 'same_day_published')

This measures every gate for the single-paragraph candidates and for the
adjacent-paragraph windows around them.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    MAX_NLI_CHUNKS_PER_SOURCE,
    MIN_SIMILARITY,
    _count_tokens,
    _dedupe_chunks,
    _resolve_chunk_budget,
    calculate_similarity_batch,
    check_date_compatibility,
    classify_evidence,
    evaluate_event_match,
    extract_claim_attributes,
    extract_years,
    split_into_chunks,
)
from nli_checker import check_nli
from source_reliability import get_source_profile
from tools import scrape_url

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
if text.startswith("SCRAPE_FAILED"):
    print(text)
    raise SystemExit(0)

budget = _resolve_chunk_budget(CLAIM)
chunks = _dedupe_chunks(split_into_chunks(text, max_tokens=budget))
attrs = extract_claim_attributes(CLAIM)
profile = build_claim_profile(CLAIM)

sims = calculate_similarity_batch(CLAIM, chunks)
prelim = [(i, c, s) for i, (c, s) in enumerate(zip(chunks, sims)) if s >= MIN_SIMILARITY]
prelim.sort(key=lambda item: item[2], reverse=True)

source_dates = []
source_years = set()


def gate_report(label, window, similarity):
    date_check = check_date_compatibility(CLAIM, window, attrs)
    level, signals = evaluate_event_match(
        CLAIM, window, attrs, date_check["status"], similarity, nli_entails=True
    )
    r = check_nli(CLAIM, window)
    klass = classify_evidence(
        {
            "date_status": date_check["status"],
            "event_match": level,
            "similarity": similarity,
            "raw_nli_label": r["label"],
            "raw_nli_score": r["score"],
            "nli_status": r.get("status", "ok"),
            "location_match": signals["location_match"],
            "claim_has_location": signals.get("claim_has_location", True),
            "detail_coverage": signals["detail_coverage"],
            "debunking": False,
        },
        "scraped_article",
        get_source_profile(URL, "scraped_article"),
    )
    gates = {
        "nli>=0.70": r["score"] >= 0.70 and r["label"] == "entailment",
        "sim>=0.40": similarity >= 0.40,
        "evt=strong": level == "strong",
        "loc_ok": signals.get("location_ok", False),
        "cov>=0.30": signals["detail_coverage"] >= 0.30,
        "date=match": date_check["status"] in ("match", "same_day_published"),
    }
    failed = [k for k, v in gates.items() if not v]
    print(f"  {label}")
    print(f"     sim={similarity:.4f} nli={r['label']}/{r['score']} "
          f"date={date_check['status']} evt={level} cov={signals['detail_coverage']:.3f}")
    print(f"     CLASSIFICATION: {klass}   failed gates: {failed or 'none'}")
    return klass, r


print("=" * 100)
print("A. SINGLE-PARAGRAPH CANDIDATES (today's behaviour, top 3 by similarity)")
print("=" * 100)
for i, (idx, c, s) in enumerate(prelim[:MAX_NLI_CHUNKS_PER_SOURCE], 1):
    gate_report(f"#{i} chunk[{idx}] sim={s:.4f}", c, s)

print()
print("=" * 100)
print("B. ADJACENT-PARAGRAPH WINDOWS around the same top candidates")
print("=" * 100)
best_window = None
for i, (idx, c, s) in enumerate(prelim[:MAX_NLI_CHUNKS_PER_SOURCE], 1):
    for lo, hi, tag in ((idx, idx + 2, "next"), (max(0, idx - 1), idx + 1, "prev+self"),
                        (idx, idx + 3, "next2")):
        if hi > len(chunks):
            continue
        window = "\n\n".join(chunks[lo:hi])
        if _count_tokens(window) > budget + 30:
            continue
        wsim = calculate_similarity_batch(CLAIM, [window])[0]
        klass, r = gate_report(
            f"#{i} window[{lo}:{hi}] ({tag}) sim={wsim:.4f}", window, wsim
        )
        if klass == "SUPPORT" and best_window is None:
            best_window = (window, wsim, lo, hi)

print()
print("=" * 100)
print("C. OUTCOME")
print("=" * 100)
if best_window:
    print(f"  A window reaches SUPPORT: chunk[{best_window[2]}:{best_window[3]}] "
          f"sim={best_window[1]:.4f}")
    print(f"      {best_window[0][:260]}")
else:
    print("  No window reaches SUPPORT with the current gates.")
