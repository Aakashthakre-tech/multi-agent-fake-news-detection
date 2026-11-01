"""
three_sample_inference_test.py
==============================
Isolated three-sample inference test for TruthLensAI audio pipeline.
Runs verify_audio() with verify_claims=False on three labeled samples.

Selected samples:
  - human_04_anchor_bulletin.wav   → REAL (English, SAMPLE_04, correct in benchmark)
  - synth_01_sapi_david_intro.wav  → AI_GENERATED (English TTS, SAMPLE_05, correct in benchmark)
  - human_03_marathi_report.wav    → REAL (Marathi, SAMPLE_03; known FP from benchmark)

Does NOT modify thresholds, model weights, or any existing module.
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_module.audio_pipeline import verify_audio

SAMPLES_DIR = PROJECT_ROOT / "audio_module" / "validation_samples"
OUTPUT_FILE = PROJECT_ROOT / "audio_module" / "outputs" / "three_sample_test_results.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "id": "TEST_01",
        "filename": "human_04_anchor_bulletin.wav",
        "ground_truth": "REAL",
        "description": "Human broadcast — English studio anchor bulletin",
        "language": "English",
        "notes": "Correctly predicted REAL in full benchmark (SAMPLE_04).",
    },
    {
        "id": "TEST_02",
        "filename": "synth_01_sapi_david_intro.wav",
        "ground_truth": "AI_GENERATED",
        "description": "Synthetic TTS — Microsoft SAPI David Desktop (English)",
        "language": "English",
        "notes": "Correctly predicted AI_GENERATED in full benchmark (SAMPLE_05). P(synth)=1.000.",
    },
    {
        "id": "TEST_03",
        "filename": "human_03_marathi_report.wav",
        "ground_truth": "REAL",
        "description": "Human speech — Marathi field report",
        "language": "Marathi",
        "notes": (
            "Known false-positive case from full benchmark (SAMPLE_03): "
            "model predicted AI_GENERATED with P(synth)=1.000. "
            "Root cause: model is likely trained on English-only data; "
            "Marathi is out-of-distribution. Included here to verify reproducibility "
            "of that specific failure, NOT to reinterpret as authentic."
        ),
    },
]


def run_test(test_case: dict) -> dict:
    sample_path = SAMPLES_DIR / test_case["filename"]

    if not sample_path.exists():
        return {
            **test_case,
            "file_path": str(sample_path),
            "status": "SKIPPED",
            "error": f"File not found: {sample_path}",
            "prediction": "UNAVAILABLE",
            "probability_synthetic": None,
            "probability_real": None,
            "confidence": None,
            "inference_time_sec": None,
            "match": None,
        }

    t0 = time.perf_counter()
    result = verify_audio(sample_path, verify_claims=False, whisper_model="base")
    elapsed = round(time.perf_counter() - t0, 3)

    synth = result.get("synthetic_speech_detection", {})
    prediction = synth.get("label", "UNAVAILABLE")
    prob_synth = synth.get("probability_synthetic")
    prob_real = synth.get("probability_real")
    confidence = synth.get("confidence")
    available = synth.get("available", False)
    inference_error = synth.get("error")
    duration = result.get("audio_metadata", {}).get("duration_sec")

    gt = test_case["ground_truth"]
    if prediction in ("REAL", "AI_GENERATED"):
        match = prediction == gt
    else:
        match = None  # UNCERTAIN or UNAVAILABLE — not classifiable as correct/wrong

    return {
        "id": test_case["id"],
        "filename": test_case["filename"],
        "file_path": str(sample_path.resolve()),
        "ground_truth": gt,
        "description": test_case["description"],
        "language": test_case["language"],
        "notes": test_case["notes"],
        "audio_duration_sec": duration,
        "prediction": prediction,
        "probability_synthetic": prob_synth,
        "probability_real": prob_real,
        "confidence": confidence,
        "detector_available": available,
        "inference_error": inference_error,
        "total_pipeline_time_sec": elapsed,
        "match": match,
        "status": "OK" if result["status"] == "success" else result.get("status", "error"),
    }


def main():
    print("=" * 70)
    print("TRUTHLENSAI — THREE-SAMPLE AUDIO INFERENCE TEST")
    print("Command: .venv\\Scripts\\python.exe scratch/three_sample_inference_test.py")
    print("verify_claims=False | No model/threshold changes")
    print("=" * 70)

    all_results = []

    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['description']} ({case['language']})")
        print(f"  File      : {case['filename']}")
        print(f"  GT        : {case['ground_truth']}")
        r = run_test(case)
        all_results.append(r)

        if r["status"] == "SKIPPED":
            print(f"  *** SKIPPED — {r['error']}")
            continue

        match_str = "✓ CORRECT" if r["match"] is True else ("✗ WRONG" if r["match"] is False else "~ UNCERTAIN/UNAVAILABLE")
        print(f"  Prediction: {r['prediction']}")
        print(f"  P(synth)  : {r['probability_synthetic']}")
        print(f"  P(real)   : {r['probability_real']}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Duration  : {r['audio_duration_sec']} s")
        print(f"  Inference : {r['total_pipeline_time_sec']} s (full pipeline)")
        print(f"  Result    : {match_str}")
        if r["inference_error"]:
            print(f"  Error     : {r['inference_error']}")

    # Summary
    ran = [r for r in all_results if r["status"] != "SKIPPED"]
    correct = sum(1 for r in ran if r["match"] is True)
    wrong = sum(1 for r in ran if r["match"] is False)
    uncertain = sum(1 for r in ran if r["match"] is None)

    output = {
        "test_name": "TruthLensAI Three-Sample Inference Test",
        "command": ".venv\\Scripts\\python.exe scratch/three_sample_inference_test.py",
        "verify_claims": False,
        "total_samples": len(TEST_CASES),
        "ran": len(ran),
        "correct": correct,
        "wrong": wrong,
        "uncertain_or_unavailable": uncertain,
        "samples": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Ran     : {len(ran)}/{len(TEST_CASES)}")
    print(f"  Correct : {correct}")
    print(f"  Wrong   : {wrong}")
    print(f"  Uncertain/Unavailable: {uncertain}")
    print(f"  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
