"""
scratch/probe_nasa_chunk_ranking.py
===================================
Shows, for NASA's Apollo 11 article, which passages the analyzer actually
sent to the NLI model and which it discarded - and what the NLI model says
about the discarded ones.

The analyzer selects NLI candidates by LEXICAL SIMILARITY only. This probe
checks whether the passage that actually asserts the claim is discarded
before the NLI model ever sees it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_profile import build_claim_profile
from evidence_analyzer import (
    MAX_NLI_CHUNKS_PER_SOURCE,
    MIN_SIMILARITY,
    _dedupe_chunks,
    _resolve_chunk_budget,
    calculate_similarity_batch,
    check_date_compatibility,
    extract_claim_attributes,
    split_into_chunks,
)
from nli_checker import check_nli

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

from tools import scrape_url

text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
if text.startswith("SCRAPE_FAILED"):
    print(text)
    raise SystemExit(0)

attrs = extract_claim_attributes(CLAIM)
budget = _resolve_chunk_budget(CLAIM)
chunks = split_into_chunks(text, max_tokens=budget)
chunks = _dedupe_chunks(chunks)

print("=" * 100)
print(f"article: {len(text)} chars -> {len(chunks)} chunks (budget {budget} tokens)")
print(f"MIN_SIMILARITY={MIN_SIMILARITY}  MAX_NLI_CHUNKS_PER_SOURCE={MAX_NLI_CHUNKS_PER_SOURCE}")
print("=" * 100)

similarities = calculate_similarity_batch(CLAIM, chunks)
ranked = sorted(zip(chunks, similarities), key=lambda item: item[1], reverse=True)

print()
print("=" * 100)
print("SIMILARITY RANKING (this is the order the analyzer uses to pick NLI candidates)")
print("=" * 100)
for i, (chunk, sim) in enumerate(ranked, 1):
    marker = "  <== SELECTED for NLI" if i <= MAX_NLI_CHUNKS_PER_SOURCE else ""
    print(f"{i:>3}. sim={sim:.4f} {marker}")
    print(f"     {chunk[:170]}")

print()
print("=" * 100)
print("WHAT THE NLI MODEL SAYS ABOUT EVERY CHUNK THAT PASSED THE SIMILARITY FILTER")
print("=" * 100)
for i, (chunk, sim) in enumerate(ranked, 1):
    if sim < MIN_SIMILARITY:
        continue
    date_check = check_date_compatibility(CLAIM, chunk, attrs)
    r = check_nli(CLAIM, chunk)
    selected = "SELECTED" if i <= MAX_NLI_CHUNKS_PER_SOURCE else "DISCARDED"
    print()
    print(f"  #{i} sim={sim:.4f} {selected}  date={date_check['status']} "
          f"nli={r['label']}/{r['score']}")
    print(f"     {chunk[:200]}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
selected = ranked[:MAX_NLI_CHUNKS_PER_SOURCE]
selected_entail = [c for c, s in selected if check_nli(CLAIM, c)["label"] == "entailment"]
discarded = ranked[MAX_NLI_CHUNKS_PER_SOURCE:]
discarded_entail = [
    (c, s) for c, s in discarded
    if s >= MIN_SIMILARITY and check_nli(CLAIM, c)["label"] == "entailment"
]
print(f"  chunks sent to NLI            : {len(selected)}")
print(f"  of those, NLI entails         : {len(selected_entail)}")
print(f"  chunks discarded before NLI   : {len(discarded)}")
print(f"  of those, NLI would entail    : {len(discarded_entail)}")
for c, s in discarded_entail:
    print(f"      sim={s:.4f} {c[:190]}")
