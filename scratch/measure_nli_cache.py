"""
scratch/measure_nli_cache.py
============================
Measures the NLI cache: the same claim analysed twice in ONE process.

The first pass populates the cache, the second reads it. The measurement
compares the NLI wall-clock time of the two passes and confirms that the
entailment DECISIONS are identical - the cache must not change a single
classification.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nli_checker  # noqa: E402
from evidence_analyzer import analyze_evidence  # noqa: E402

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."

ARTICLE = (
    "TITLE: July 20, 1969: One Giant Leap For Mankind - NASA\n\n"
    "They leave behind an American flag, a patch honoring the fallen Apollo 1 "
    "crew, and a plaque on one of Eagle's legs. It reads, “Here men from the "
    "planet Earth first set foot upon the moon. July 1969 A.D. We came in peace "
    "for all mankind.”\n\n"
    "Armstrong and Aldrin blast off and dock with Collins in Columbia. Collins "
    "later says that “for the first time,” he “really felt that we were going to "
    "carry this thing off.”\n\n"
    "When the lunar module lands at 4:17 p.m EDT, only 30 seconds of fuel "
    "remain. Armstrong radios “Houston, Tranquility Base here. The Eagle has "
    "landed.” Mission control erupts in celebration."
)

SOURCE = {
    "url": "https://www.nasa.gov/history/july-20-1969",
    "domain": "nasa.gov",
    "title": "July 20, 1969: One Giant Leap For Mankind - NASA",
    "content": ARTICLE,
    "evidence_type": "scraped_article",
    "scrape_status": "SUCCESS",
}


def one_pass(label):
    nli_checker._NLI_CACHE.clear() if label == "cold" else None
    t0 = time.time()
    results = analyze_evidence(CLAIM, [SOURCE], verbose=False)
    elapsed = time.time() - t0
    signature = [
        (r.get("evidence_class"), r.get("nli_label"), r.get("nli_score"),
         r.get("passage_kind"), r.get("evidence_score"))
        for r in results
    ]
    print(f"  {label:<6} {elapsed:6.2f}s  records={len(results)}  "
          f"cache_entries={len(nli_checker._NLI_CACHE)}")
    return elapsed, signature


print("=" * 92)
print("NLI CACHE MEASUREMENT")
print("=" * 92)

cold_time, cold_sig = one_pass("cold")
warm_time, warm_sig = one_pass("warm")

print()
print(f"  cold {cold_time:.2f}s -> warm {warm_time:.2f}s   "
      f"({cold_time - warm_time:.2f}s saved, "
      f"{(1 - warm_time / max(cold_time, 0.001)) * 100:.1f}% faster)")
print()
print("  identical classifications:", cold_sig == warm_sig)
if cold_sig != warm_sig:
    print("   COLD:", cold_sig)
    print("   WARM:", warm_sig)
    raise SystemExit(1)
print("  the cache changed no entailment decision.")
