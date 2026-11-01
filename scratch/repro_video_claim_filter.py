"""
scratch/repro_video_claim_filter.py
==================================
Replays the CURRENT video fusion + claim-extraction code against the saved
artifacts of the run that produced zero verifiable claims
(video_module/outputs/tmpx191xwhp: 117 frames, 36 Whisper segments,
11 OCR frames with text, 32 candidate claims, 0 accepted).

The point is to determine, with the real code, whether meaningful factual
claims were wrongly discarded or whether the rejection was correct.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN = ROOT / "video_module" / "outputs" / "tmpx191xwhp"


def load(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def main():
    from video_module.video_pipeline import fuse_whisper_and_ocr
    from video_module.video_pipeline import prepare_candidate_claims

    transcript = load("transcript.json")
    ocr = load("ocr.json")
    if isinstance(transcript, list):
        transcript = {"segments": transcript}
    if isinstance(ocr, list):
        ocr = {"observations": ocr}

    print("=" * 70)
    print("SAVED ARTIFACTS (old run)")
    print("=" * 70)
    print("whisper segments:", len(transcript.get("segments", [])))
    print("ocr observations:", len(ocr.get("observations", [])))
    for o in ocr.get("observations", []):
        print(f"  frame {o.get('frame_index')} ts={o.get('timestamp_sec')} "
              f"conf={o.get('confidence')} text={o.get('text')!r}")

    print()
    print("=" * 70)
    print("RE-RUNNING CURRENT fuse_whisper_and_ocr")
    print("=" * 70)
    fusion = fuse_whisper_and_ocr(transcript, ocr)
    print("status:", fusion.get("status"))
    print("counts:", {k: v for k, v in fusion.items() if k != "fused_entries"})
    sources = {}
    for e in fusion.get("fused_entries", []):
        sources[e.get("source")] = sources.get(e.get("source"), 0) + 1
    print("fused entry sources:", sources)
    for e in fusion.get("fused_entries", []):
        if e.get("source") in ("ocr", "both"):
            print(f"  [{e.get('source')}] t={e.get('start_sec')} "
                  f"text={e.get('text')!r} osc={e.get('on_screen_context')!r}")

    print()
    print("=" * 70)
    print("RE-RUNNING CURRENT claim ranking")
    print("=" * 70)
    video_metadata = {
        "file_name": "tmpx191xwhp.mp4",
        "duration_sec": 213.24,
    }
    ranked = prepare_candidate_claims(fusion, video_metadata=video_metadata)
    claims = ranked.get("candidate_claims", [])
    print("counts:", {k: v for k, v in ranked.items() if k != "candidate_claims"})
    prov = {}
    for c in claims:
        prov[c.get("source_provenance")] = prov.get(c.get("source_provenance"), 0) + 1
    print("provenance:", prov)
    for c in claims:
        print(f"  [{c.get('selection_status')}] factual={c.get('is_factual')} "
              f"prov={c.get('source_provenance')} text={c.get('claim_text')!r}")
        print(f"        reason: {c.get('selection_reason')}")


if __name__ == "__main__":
    main()
