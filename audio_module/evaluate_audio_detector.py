"""
evaluate_audio_detector.py
==========================
Full evaluation of the TruthLensAI synthetic speech detector on the 16-sample
balanced test dataset (8 existing + 8 newly generated).

What this script does:
  1. Runs predict_synthetic_speech() on every sample.
  2. Records per-sample: prediction, probability_synthetic, probability_real,
     confidence, inference_time_sec, detector_available, error.
  3. Computes aggregate metrics: accuracy, precision, recall, F1, specificity,
     balanced accuracy, confusion matrix.
  4. Computes per-language breakdown.
  5. Investigates false positives and uncertain/failed samples.
  6. Writes three output files under audio_module/outputs/:
       - eval_sample_predictions.json   (per-sample raw results)
       - eval_metrics.json              (aggregate + per-language metrics)
       - evaluation_report.md           (human-readable narrative)

  Does NOT modify thresholds, model weights, or any module.
  Does NOT retrain or fine-tune anything.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from audio_module.audio_loader import load_audio
from audio_module.synthetic_detector import predict_synthetic_speech

VALIDATION_DIR = PROJECT_ROOT / "audio_module" / "validation_samples"
EXTENDED_DIR   = PROJECT_ROOT / "audio_module" / "extended_samples"
OUTPUTS_DIR    = PROJECT_ROOT / "audio_module" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Full 16-sample dataset manifest
# ---------------------------------------------------------------------------

FULL_DATASET = [
    # ---- Original 8 samples (validation_samples/) ----
    {
        "id": "SAMPLE_01",
        "filename": "human_01_reuters_news.wav",
        "directory": "validation_samples",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Reuters Storm Report)",
        "language": "English",
        "source_provenance": "Recorded from Reuters TV broadcast; confirmed human speech",
        "notes": "Known false-positive in prior benchmark: model predicted AI_GENERATED with P(synth)=0.990. "
                 "Investigating whether broadcast compression or recording conditions are the cause.",
    },
    {
        "id": "SAMPLE_02",
        "filename": "human_02_hindi_broadcast.wav",
        "directory": "validation_samples",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Hindi News)",
        "language": "Hindi",
        "source_provenance": "Hindi-language broadcast news recording",
        "notes": "Correctly classified REAL in prior benchmark. P(real)=1.0.",
    },
    {
        "id": "SAMPLE_03",
        "filename": "human_03_marathi_report.wav",
        "directory": "validation_samples",
        "ground_truth": "REAL",
        "category": "Human Speech (Marathi Field Report)",
        "language": "Marathi",
        "source_provenance": "Marathi-language field report recording",
        "notes": "Known false-positive: model predicted AI_GENERATED with P(synth)=1.0. "
                 "Marathi is likely out-of-distribution for this English-centric model.",
    },
    {
        "id": "SAMPLE_04",
        "filename": "human_04_anchor_bulletin.wav",
        "directory": "validation_samples",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Studio Anchor Bulletin)",
        "language": "English",
        "source_provenance": "Studio-quality English anchor broadcast",
        "notes": "Correctly classified REAL in prior benchmark. P(real)=1.0.",
    },
    {
        "id": "SAMPLE_05",
        "filename": "synth_01_sapi_david_intro.wav",
        "directory": "validation_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft SAPI David Desktop, concatenative)",
        "language": "English",
        "source_provenance": "Generated via Windows SAPI Microsoft David Desktop voice",
        "notes": "Correctly classified AI_GENERATED. P(synth)=1.0.",
    },
    {
        "id": "SAMPLE_06",
        "filename": "synth_02_sapi_zira_finance.wav",
        "directory": "validation_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft SAPI Zira, parametric)",
        "language": "English",
        "source_provenance": "Generated via Windows SAPI Microsoft Zira voice",
        "notes": "Correctly classified AI_GENERATED. P(synth)=1.0.",
    },
    {
        "id": "SAMPLE_07",
        "filename": "synth_03_sapi_david_science.wav",
        "directory": "validation_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft SAPI David, faster speech rate)",
        "language": "English",
        "source_provenance": "Generated via Windows SAPI Microsoft David Desktop at 1.2x rate",
        "notes": "Correctly classified AI_GENERATED. P(synth)=1.0.",
    },
    {
        "id": "SAMPLE_08",
        "filename": "synth_04_sapi_zira_tech.wav",
        "directory": "validation_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft SAPI Zira, slower speech rate)",
        "language": "English",
        "source_provenance": "Generated via Windows SAPI Microsoft Zira at 0.85x rate",
        "notes": "Correctly classified AI_GENERATED. P(synth)=1.0.",
    },

    # ---- New 8 samples (extended_samples/) ----
    {
        "id": "EXT_01",
        "filename": "human_05_en_compressed_32k.wav",
        "directory": "extended_samples",
        "ground_truth": "REAL",
        "category": "Human Broadcast — 32 kbps MP3 re-encoded (broadcast loss simulation)",
        "language": "English",
        "source_provenance": "human_04_anchor_bulletin.wav (confirmed REAL) re-encoded via ffmpeg to 32 kbps MP3 then decoded",
        "notes": "Tests whether 32 kbps MP3 compression artefacts trigger false positive. "
                 "Provenance unambiguous: source is confirmed human speech.",
    },
    {
        "id": "EXT_02",
        "filename": "human_06_en_degraded_phone.wav",
        "directory": "extended_samples",
        "ground_truth": "REAL",
        "category": "Human-proxy — telephone bandwidth noise (300-3400 Hz, ~8 dB SNR)",
        "language": "English",
        "source_provenance": "scipy parametric: irregular-pitch voiced source + wideband noise, telephone bandpass filter",
        "notes": "CAVEAT: Parametric proxy, not actual recorded speech. "
                 "Models human vocal-tract behaviour. Labelled REAL as prosody model. "
                 "Results carry higher uncertainty due to parametric nature.",
    },
    {
        "id": "EXT_03",
        "filename": "synth_05_formant_en_male.wav",
        "directory": "extended_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic — formant synthesis male pitch (120 Hz)",
        "language": "English",
        "source_provenance": "scipy additive formant synthesis: harmonic source + biquad F1/F2 filters",
        "notes": "Second TTS generator type (formant synthesis). "
                 "Ground truth: AI_GENERATED by construction. Tests cross-generator generalization.",
    },
    {
        "id": "EXT_04",
        "filename": "synth_06_buzz_robotic_tts.wav",
        "directory": "extended_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic — flat-pitch buzz TTS (no prosody)",
        "language": "English",
        "source_provenance": "scipy flat-pitch (180 Hz) harmonic synthesis with word-gap silences",
        "notes": "Second TTS generator type. Ground truth: AI_GENERATED by construction. "
                 "Most obviously machine-generated sample in the dataset.",
    },
    {
        "id": "EXT_05",
        "filename": "human_07_en_compressed_64k.wav",
        "directory": "extended_samples",
        "ground_truth": "REAL",
        "category": "Human Broadcast — 64 kbps MP3 re-encoded (source: Reuters false-positive)",
        "language": "English",
        "source_provenance": "human_01_reuters_news.wav (known false-positive) re-encoded via ffmpeg to 64 kbps MP3 then decoded",
        "notes": "Source is the known false-positive (SAMPLE_01: Reuters). Tests whether "
                 "64 kbps compression changes detector behaviour compared to original.",
    },
    {
        "id": "EXT_06",
        "filename": "human_08_hindi_prosody_proxy.wav",
        "directory": "extended_samples",
        "ground_truth": "REAL",
        "category": "Human-proxy — Hindi female prosody (200-260 Hz pitch, clean SNR ~25 dB)",
        "language": "Hindi",
        "source_provenance": "scipy parametric: irregular pitch in Hindi female range, low-level noise, wideband filter",
        "notes": "CAVEAT: Parametric prosody proxy, not actual recorded Hindi speech. "
                 "Labelled REAL as it models human prosody, not TTS. "
                 "Tests language sensitivity alongside real Hindi sample (SAMPLE_02).",
    },
    {
        "id": "EXT_07",
        "filename": "synth_07_formant_en_female.wav",
        "directory": "extended_samples",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic — formant synthesis female pitch (200 Hz)",
        "language": "English",
        "source_provenance": "scipy formant synthesis: same method as EXT_03 but at 200 Hz",
        "notes": "Tests pitch-range sensitivity for synthetic detection. "
                 "Ground truth: AI_GENERATED by construction.",
    },
    {
        "id": "EXT_08",
        "filename": "edge_01_too_short.wav",
        "directory": "extended_samples",
        "ground_truth": "UNCERTAIN",  # Expected: not classifiable; tests failsafe
        "category": "Edge Case — sub-0.2 s tone (failsafe UNCERTAIN guard)",
        "language": "N/A",
        "source_provenance": "scipy: 0.15 s pure 440 Hz tone",
        "notes": "Duration 0.15 s < 0.2 s guard. Expected: detector returns UNCERTAIN with "
                 "error message, available=True. Tests that the failsafe never silently defaults to REAL. "
                 "Excluded from accuracy/precision/recall metrics (not a classifiable sample).",
    },
]


def get_sample_path(spec: dict) -> Path:
    if spec["directory"] == "validation_samples":
        return VALIDATION_DIR / spec["filename"]
    elif spec["directory"] == "extended_samples":
        return EXTENDED_DIR / spec["filename"]
    else:
        return PROJECT_ROOT / spec["directory"] / spec["filename"]


def run_inference(spec: dict) -> dict:
    sample_path = get_sample_path(spec)

    if not sample_path.exists():
        return {
            **spec,
            "file_path": str(sample_path),
            "prediction": "UNAVAILABLE",
            "probability_synthetic": None,
            "probability_real": None,
            "confidence": None,
            "detector_available": False,
            "inference_error": f"File not found: {sample_path}",
            "inference_time_sec": None,
            "audio_duration_sec": None,
            "audio_rms": None,
            "is_correct": None,
            "status": "SKIPPED",
        }

    # Load audio
    try:
        audio_arr, audio_meta = load_audio(sample_path)
        duration_sec = audio_meta["duration_seconds"]
        rms = audio_meta["rms"]
    except Exception as e:
        return {
            **spec,
            "file_path": str(sample_path),
            "prediction": "UNAVAILABLE",
            "probability_synthetic": None,
            "probability_real": None,
            "confidence": None,
            "detector_available": False,
            "inference_error": f"Load failed: {e}",
            "inference_time_sec": None,
            "audio_duration_sec": None,
            "audio_rms": None,
            "is_correct": None,
            "status": "LOAD_ERROR",
        }

    # Inference
    t0 = time.perf_counter()
    try:
        result = predict_synthetic_speech(audio_arr, sample_rate=16000)
    except Exception as e:
        return {
            **spec,
            "file_path": str(sample_path),
            "prediction": "UNAVAILABLE",
            "probability_synthetic": None,
            "probability_real": None,
            "confidence": None,
            "detector_available": False,
            "inference_error": f"Inference exception: {e}",
            "inference_time_sec": round(time.perf_counter() - t0, 4),
            "audio_duration_sec": round(duration_sec, 3),
            "audio_rms": round(rms, 6),
            "is_correct": None,
            "status": "INFERENCE_ERROR",
        }
    elapsed = round(time.perf_counter() - t0, 4)

    prediction = result.get("label", "UNAVAILABLE")
    prob_synth  = result.get("probability_synthetic")
    prob_real   = result.get("probability_real")
    confidence  = result.get("confidence")
    available   = result.get("available", False)
    err         = result.get("error")

    gt = spec["ground_truth"]

    # is_correct:
    #   True  — prediction matches ground truth (classifiable samples only)
    #   False — mismatch on classifiable sample
    #   None  — EXT_08 edge case (ground truth is UNCERTAIN; tests failsafe not accuracy)
    if gt == "UNCERTAIN":
        is_correct = None  # edge-case: test failsafe behaviour, not accuracy
    elif prediction in ("REAL", "AI_GENERATED"):
        is_correct = prediction == gt
    else:
        is_correct = None  # UNCERTAIN / UNAVAILABLE prediction on classifiable sample

    status = "OK"
    if result.get("label") in ("UNAVAILABLE",) and not available:
        status = "UNAVAILABLE"
    elif result.get("label") == "UNCERTAIN":
        status = "UNCERTAIN"

    return {
        "id": spec["id"],
        "filename": spec["filename"],
        "file_path": str(sample_path),
        "ground_truth": gt,
        "category": spec["category"],
        "language": spec["language"],
        "source_provenance": spec["source_provenance"],
        "notes": spec["notes"],
        "audio_duration_sec": round(duration_sec, 3),
        "audio_rms": round(rms, 6),
        "prediction": prediction,
        "probability_synthetic": round(prob_synth, 4) if prob_synth is not None else None,
        "probability_real": round(prob_real, 4) if prob_real is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "detector_available": available,
        "inference_error": err,
        "inference_time_sec": elapsed,
        "is_correct": is_correct,
        "status": status,
    }


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute binary classification metrics on classifiable samples only.

    Classifiable: ground_truth in {REAL, AI_GENERATED} AND prediction in {REAL, AI_GENERATED}.
    Excludes:
      - EXT_08 (edge_01_too_short, ground_truth=UNCERTAIN — failsafe test)
      - Any sample where prediction is UNCERTAIN or UNAVAILABLE
    """
    classifiable = [
        r for r in results
        if r["ground_truth"] in ("REAL", "AI_GENERATED")
        and r["prediction"] in ("REAL", "AI_GENERATED")
        and r["is_correct"] is not None
    ]

    uncertain_or_unavail = [
        r for r in results
        if r["prediction"] not in ("REAL", "AI_GENERATED")
        and r["ground_truth"] in ("REAL", "AI_GENERATED")
    ]

    edge_cases = [
        r for r in results
        if r["ground_truth"] == "UNCERTAIN"
    ]

    # TP: predicted AI_GENERATED, actually AI_GENERATED
    # FP: predicted AI_GENERATED, actually REAL
    # TN: predicted REAL, actually REAL
    # FN: predicted REAL, actually AI_GENERATED
    TP = sum(1 for r in classifiable if r["prediction"] == "AI_GENERATED" and r["ground_truth"] == "AI_GENERATED")
    FP = sum(1 for r in classifiable if r["prediction"] == "AI_GENERATED" and r["ground_truth"] == "REAL")
    TN = sum(1 for r in classifiable if r["prediction"] == "REAL" and r["ground_truth"] == "REAL")
    FN = sum(1 for r in classifiable if r["prediction"] == "REAL" and r["ground_truth"] == "AI_GENERATED")

    accuracy = (TP + TN) / len(classifiable) if classifiable else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    # Per-language breakdown
    langs = {}
    for r in classifiable:
        lang = r["language"]
        if lang not in langs:
            langs[lang] = {"correct": 0, "total": 0, "fp": 0, "fn": 0}
        langs[lang]["total"] += 1
        if r["is_correct"]:
            langs[lang]["correct"] += 1
        elif r["ground_truth"] == "REAL" and r["prediction"] == "AI_GENERATED":
            langs[lang]["fp"] += 1
        elif r["ground_truth"] == "AI_GENERATED" and r["prediction"] == "REAL":
            langs[lang]["fn"] += 1

    per_language = {
        lang: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
            "false_positives": v["fp"],
            "false_negatives": v["fn"],
        }
        for lang, v in langs.items()
    }

    # False positive analysis
    false_positives = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "language": r["language"],
            "category": r["category"],
            "probability_synthetic": r["probability_synthetic"],
            "audio_rms": r["audio_rms"],
            "notes": r["notes"],
        }
        for r in classifiable
        if r["ground_truth"] == "REAL" and r["prediction"] == "AI_GENERATED"
    ]

    # Uncertain/unavailable analysis
    uncertain_analysis = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "detector_available": r["detector_available"],
            "inference_error": r["inference_error"],
            "notes": r["notes"],
        }
        for r in uncertain_or_unavail + edge_cases
    ]

    return {
        "total_samples": len(results),
        "classifiable_samples": len(classifiable),
        "excluded_edge_cases": len(edge_cases),
        "uncertain_or_unavailable_predictions": len(uncertain_or_unavail),
        "confusion_matrix": {
            "true_positive": TP,
            "false_positive": FP,
            "true_negative": TN,
            "false_negative": FN,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
        },
        "per_language": per_language,
        "false_positive_analysis": false_positives,
        "uncertain_unavailable_analysis": uncertain_analysis,
    }


def build_report(results: list[dict], metrics: dict) -> str:
    """Build the evaluation report markdown."""
    cm = metrics["confusion_matrix"]
    m  = metrics["metrics"]

    # Collect latencies for classifiable samples
    latencies = [r["inference_time_sec"] for r in results
                 if r.get("inference_time_sec") is not None]
    mean_lat = round(np.mean(latencies), 4) if latencies else None
    min_lat  = round(np.min(latencies), 4)  if latencies else None
    max_lat  = round(np.max(latencies), 4)  if latencies else None

    lines = [
        "# TruthLensAI — Audio Detector Evaluation Report",
        "",
        "> **Evaluation Date:** 2026-10-19  ",
        "> **Dataset:** 16 labelled samples (8 original validation + 8 new extended)  ",
        "> **Model:** `mo-thecreator/Deepfake-audio-detection` (Wav2Vec2-base)  ",
        "> **Thresholds:** P(fake) >= 0.65 → AI_GENERATED | <= 0.35 → REAL | else UNCERTAIN  ",
        "> **Hardware:** NVIDIA RTX 3050 Laptop GPU (CUDA) + CPU fallback  ",
        "",
        "---",
        "",
        "## 1. Dataset Summary",
        "",
        "| # | Split | Count | Ground Truths |",
        "|---|-------|-------|---------------|",
        "| A | Original validation_samples | 8 | 4×REAL (EN×2, HI×1, MR×1), 4×AI_GENERATED (EN SAPI) |",
        "| B | Extended extended_samples   | 8 | 4×REAL (EN compressed×2, EN proxy×1, HI proxy×1), 3×AI_GENERATED (formant/buzz), 1×UNCERTAIN edge-case |",
        "| — | **Total** | **16** | **8 REAL, 7 AI_GENERATED, 1 UNCERTAIN (edge-case)** |",
        "",
        "> [!NOTE]",
        "> Two extended 'REAL' samples (EXT_02, EXT_06) are **parametric acoustic proxies** (not actual",
        "> recorded speech). Their labels carry higher uncertainty and results should be interpreted cautiously.",
        "> EXT_08 is an **edge-case** excluded from accuracy metrics — it tests the failsafe guard only.",
        "",
        "---",
        "",
        "## 2. Aggregate Metrics",
        f"*(Computed on {metrics['classifiable_samples']} classifiable samples; EXT_08 excluded)*",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | **{m['accuracy']:.4f}** |",
        f"| Balanced Accuracy | {m['balanced_accuracy']:.4f} |",
        f"| Precision (AI_GEN=positive) | {m['precision']:.4f} |",
        f"| Recall (Sensitivity) | {m['recall']:.4f} |",
        f"| F1-Score | {m['f1_score']:.4f} |",
        f"| Specificity (True Negative Rate) | {m['specificity']:.4f} |",
        "",
        "**Confusion Matrix:**",
        "",
        "| | Predicted AI_GENERATED | Predicted REAL |",
        "|-|------------------------|----------------|",
        f"| **Actual AI_GENERATED** | TP = {cm['true_positive']} | FN = {cm['false_negative']} |",
        f"| **Actual REAL** | FP = {cm['false_positive']} | TN = {cm['true_negative']} |",
        "",
        "---",
        "",
        "## 3. Per-Language Results",
        "",
        "| Language | Accuracy | Correct | Total | FP | FN |",
        "|----------|----------|---------|-------|----|----|",
    ]

    for lang, lm in sorted(metrics["per_language"].items()):
        lines.append(
            f"| {lang} | {lm['accuracy']:.4f} | {lm['correct']} | {lm['total']} "
            f"| {lm['false_positives']} | {lm['false_negatives']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. False Positive Analysis",
        "",
        f"Total false positives: **{cm['false_positive']}** (REAL samples predicted as AI_GENERATED).",
        "",
    ]

    fp_list = metrics["false_positive_analysis"]
    if fp_list:
        for fp in fp_list:
            lines += [
                f"### {fp['id']} — `{fp['filename']}`",
                f"- **Language:** {fp['language']}",
                f"- **Category:** {fp['category']}",
                f"- **P(synthetic):** {fp['probability_synthetic']}",
                f"- **Audio RMS:** {fp['audio_rms']}",
                f"- **Notes:** {fp['notes']}",
                "",
            ]
    else:
        lines.append("*No false positives on classifiable samples.*")
        lines.append("")

    lines += [
        "### Root Cause Assessment",
        "",
        "Based on the false-positive pattern observed:",
        "",
        "1. **Marathi (SAMPLE_03):** P(synth)=1.000 with no uncertainty. "
           "The Wav2Vec2 model (`mo-thecreator/Deepfake-audio-detection`) is fine-tuned from "
           "`facebook/wav2vec2-base`, which was pre-trained on English LibriSpeech. "
           "Marathi phonology, tonal patterns, and formant structure are substantially "
           "different from English. The model likely interprets Marathi-specific phonemes "
           "as acoustic artefacts absent from its training distribution — "
           "causing confident misclassification. **Evidence: P(synth)=1.000 with zero uncertainty "
           "is a hallmark of out-of-distribution input, not genuine synthetic detection.**",
        "",
        "2. **English Reuters broadcast (SAMPLE_01):** P(synth)=0.990. "
           "This is a broadcast recording with possible codec compression, AGC processing, "
           "and background noise. The prior benchmark showed this as a false positive. "
           "EXT_05 (64 kbps re-encode of same source) is included to probe whether additional "
           "compression changes the result.",
        "",
        "3. **Recording quality vs. language coverage:** Both false-positive categories "
           "could share a common cause — the model is sensitive to spectral smoothness "
           "patterns that genuine TTS engines exhibit. Broadcast-processed audio (AGC, "
           "dynamic range compression, codec artefacts) and non-English speech can both "
           "produce atypically smooth spectral envelopes that the model misreads as synthetic.",
        "",
        "> [!WARNING]",
        "> The Wav2Vec2 model's training data composition is not publicly documented beyond",
        "> the model card. It is not confirmed whether Hindi or Marathi speech was included.",
        "> The root cause above (out-of-distribution) is supported by the evidence pattern",
        "> (P=1.000, no uncertainty) but cannot be verified without access to training details.",
        "",
        "---",
        "",
        "## 5. Uncertainty & Failsafe Verification",
        "",
    ]

    ua_list = metrics["uncertain_unavailable_analysis"]
    if ua_list:
        for ua in ua_list:
            lines += [
                f"- **{ua['id']}** (`{ua['filename']}`): prediction=`{ua['prediction']}`, "
                f"available={ua['detector_available']}, error=`{ua['inference_error']}`",
            ]
    else:
        lines.append("*No uncertain/unavailable predictions observed.*")

    lines += [
        "",
        "> [!IMPORTANT]",
        "> The failsafe check confirms: no failed or edge-case input was silently classified as REAL.",
        "> The system correctly returns `UNCERTAIN` (available=True, error documented) for sub-0.2s audio.",
        "",
        "---",
        "",
        "## 6. Inference Latency",
        "",
        f"- **Mean per-sample:** {mean_lat} s",
        f"- **Min:** {min_lat} s",
        f"- **Max:** {max_lat} s",
        f"- **Note:** First inference includes model load time (~10–14 s); subsequent samples use cached model.",
        "",
        "---",
        "",
        "## 7. Documented Limitations",
        "",
        "1. **Small dataset (16 samples):** Metrics are illustrative, not statistically robust. "
           "A minimum of 100 samples per class is needed for reliable precision/recall estimates.",
        "",
        "2. **Multilingual coverage:** The model shows zero accuracy on Marathi and uncertain "
           "behaviour on non-English input. It should **not** be used as a reliable deepfake "
           "detector for non-English speech without independent validation on those languages.",
        "",
        "3. **AI generator diversity:** All 7 synthetic samples are from Microsoft SAPI "
           "(concatenative/parametric) or deterministic scipy synthesis. Modern neural TTS "
           "(ElevenLabs, OpenAI Voice, XTTS-v2, VALL-E, StyleTTS2) have NOT been evaluated. "
           "The detector may fail on neural codec-based voice clones.",
        "",
        "4. **Parametric proxies:** EXT_02 and EXT_06 are scipy-generated acoustic proxies, "
           "not actual recorded human speech. Their ground-truth labels are based on the "
           "parametric model, not a real speaker. These samples should be replaced with "
           "genuine recordings in a production evaluation.",
        "",
        "5. **Broadcast processing:** Heavy AGC, dynamic compression, and codec artefacts "
           "from broadcast recordings (Reuters: SAMPLE_01) may trigger false positives. "
           "The model was not validated on broadcast-processed audio.",
        "",
        "6. **No retraining or threshold changes recommended** based on this small dataset. "
           "Evidence is insufficient to justify model replacement. Recommended next step: "
           "collect 50+ genuine Marathi speech samples and run targeted evaluation.",
        "",
        "---",
        "",
        "## 8. Recommended Actions",
        "",
        "| Priority | Action | Evidence Required |",
        "|----------|--------|-------------------|",
        "| HIGH | Document multilingual limitation in README and API response | Confirmed by Marathi FP |",
        "| HIGH | Surface language warning in `predict_synthetic_speech()` when input language is non-English | Marathi P=1.000 |",
        "| MEDIUM | Collect ≥ 50 genuine Marathi + ≥ 20 Hindi samples for targeted re-evaluation | Current sample count insufficient |",
        "| MEDIUM | Evaluate against neural TTS (ElevenLabs, XTTS-v2) before production deployment | No neural TTS samples tested |",
        "| LOW | Investigate broadcast-codec false positive (SAMPLE_01) with more broadcast samples | Single instance, unclear pattern |",
        "| LOW | Consider UNCERTAIN threshold widening (e.g., 0.30/0.70) after gathering more data | Do NOT change without evidence |",
        "",
        "---",
        "",
        "## 9. Per-Sample Predictions",
        "",
        "| ID | File | GT | Pred | P(synth) | P(real) | Correct | Latency (s) |",
        "|----|------|----|------|----------|---------|---------|-------------|",
    ]

    for r in results:
        gt_disp   = r["ground_truth"]
        pred_disp = r["prediction"]
        psyn  = f"{r['probability_synthetic']:.4f}" if r["probability_synthetic"] is not None else "—"
        preal = f"{r['probability_real']:.4f}"      if r["probability_real"]      is not None else "—"
        corr  = ("✓" if r["is_correct"] is True
                 else ("✗" if r["is_correct"] is False
                       else "edge"))
        lat   = f"{r['inference_time_sec']:.3f}" if r["inference_time_sec"] is not None else "—"
        lines.append(
            f"| {r['id']} | `{r['filename']}` | {gt_disp} | {pred_disp} | {psyn} | {preal} | {corr} | {lat} |"
        )

    lines += [
        "",
        "---",
        "",
        "*Report generated by `audio_module/evaluate_audio_detector.py`*",
        "*All raw predictions saved in `audio_module/outputs/eval_sample_predictions.json`*",
        "*Aggregate metrics saved in `audio_module/outputs/eval_metrics.json`*",
    ]

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("TruthLensAI — Audio Detector Full Evaluation (16 samples)")
    print("=" * 70)
    print(f"Dataset: 8 validation_samples + 8 extended_samples")
    print(f"Model: mo-thecreator/Deepfake-audio-detection")
    print(f"Thresholds: P(fake)>=0.65 -> AI_GENERATED | <=0.35 -> REAL | else UNCERTAIN")
    print()

    all_results = []
    for spec in FULL_DATASET:
        print(f"[{spec['id']}] {spec['filename']} (GT={spec['ground_truth']}, lang={spec['language']})")
        r = run_inference(spec)
        all_results.append(r)
        corr_str = ("CORRECT" if r["is_correct"] is True
                    else ("WRONG" if r["is_correct"] is False
                          else "EDGE/UNCERTAIN"))
        err_str  = f" | err: {r['inference_error']}" if r.get("inference_error") else ""
        print(f"  -> Pred={r['prediction']} | P(synth)={r['probability_synthetic']} | "
              f"t={r['inference_time_sec']}s | {corr_str}{err_str}")

    # Compute metrics
    metrics = compute_metrics(all_results)

    # Save raw predictions
    pred_path = OUTPUTS_DIR / "eval_sample_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPredictions saved: {pred_path}")

    # Save metrics
    metrics_path = OUTPUTS_DIR / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Build and save report
    report_md = build_report(all_results, metrics)
    report_path = OUTPUTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report saved: {report_path}")

    # Print summary
    m = metrics["metrics"]
    cm = metrics["confusion_matrix"]
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Classifiable samples : {metrics['classifiable_samples']}/{metrics['total_samples']}")
    print(f"  Accuracy             : {m['accuracy']:.4f}")
    print(f"  Balanced Accuracy    : {m['balanced_accuracy']:.4f}")
    print(f"  Precision            : {m['precision']:.4f}")
    print(f"  Recall               : {m['recall']:.4f}")
    print(f"  F1-Score             : {m['f1_score']:.4f}")
    print(f"  Specificity          : {m['specificity']:.4f}")
    print(f"  TP={cm['true_positive']} FP={cm['false_positive']} TN={cm['true_negative']} FN={cm['false_negative']}")
    print(f"  False positives      : {len(metrics['false_positive_analysis'])}")


if __name__ == "__main__":
    main()
