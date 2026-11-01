import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from audio_module.audio_loader import load_audio
from audio_module.synthetic_detector import predict_synthetic_speech

SAMPLES_META = [
    {
        "id": "SAMPLE_01",
        "file": "human_01_reuters_news.wav",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Reuters Storm Report)",
        "language": "English",
    },
    {
        "id": "SAMPLE_02",
        "file": "human_02_hindi_broadcast.wav",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Hindi News)",
        "language": "Hindi",
    },
    {
        "id": "SAMPLE_03",
        "file": "human_03_marathi_report.wav",
        "ground_truth": "REAL",
        "category": "Human Speech (Marathi Field Report)",
        "language": "Marathi",
    },
    {
        "id": "SAMPLE_04",
        "file": "human_04_anchor_bulletin.wav",
        "ground_truth": "REAL",
        "category": "Human Broadcast (Studio Anchor Bulletin)",
        "language": "English",
    },
    {
        "id": "SAMPLE_05",
        "file": "synth_01_sapi_david_intro.wav",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft David Concatenative)",
        "language": "English",
    },
    {
        "id": "SAMPLE_06",
        "file": "synth_02_sapi_zira_finance.wav",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft Zira Parametric)",
        "language": "English",
    },
    {
        "id": "SAMPLE_07",
        "file": "synth_03_sapi_david_science.wav",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft David Fast Rate)",
        "language": "English",
    },
    {
        "id": "SAMPLE_08",
        "file": "synth_04_sapi_zira_tech.wav",
        "ground_truth": "AI_GENERATED",
        "category": "Synthetic TTS (Microsoft Zira Slow Rate)",
        "language": "English",
    },
]

def main():
    print("=" * 70)
    print("TRUTHLENS AI — SYNTHETIC SPEECH DETECTOR VALIDATION BENCHMARK")
    print("=" * 70)

    sample_dir = Path("audio_module/validation_samples")
    results = []

    # Check CUDA
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    print(f"CUDA Available: {has_cuda} ({device_name})")

    for meta in SAMPLES_META:
        audio_path = sample_dir / meta["file"]
        if not audio_path.exists():
            print(f"Skipping missing sample: {audio_path}")
            continue

        audio, audio_props = load_audio(audio_path)
        duration = audio_props["duration_seconds"]

        # 1. CUDA Inference
        t0 = time.perf_counter()
        res_cuda = predict_synthetic_speech(audio, device="cuda" if has_cuda else "cpu")
        lat_cuda = time.perf_counter() - t0

        # 2. CPU Inference (Fallback verification)
        t0 = time.perf_counter()
        res_cpu = predict_synthetic_speech(audio, device="cpu")
        lat_cpu = time.perf_counter() - t0

        pred_label = res_cuda["label"]
        prob_synth = res_cuda["probability_synthetic"]
        prob_real = res_cuda["probability_real"]
        gt = meta["ground_truth"]

        is_correct = (pred_label == gt)
        is_uncertain = (pred_label == "UNCERTAIN")

        sample_res = {
            "id": meta["id"],
            "filename": meta["file"],
            "ground_truth": gt,
            "category": meta["category"],
            "language": meta["language"],
            "duration_sec": duration,
            "prediction": pred_label,
            "probability_synthetic": prob_synth,
            "probability_real": prob_real,
            "confidence": res_cuda["confidence"],
            "is_correct": is_correct,
            "is_uncertain": is_uncertain,
            "latency_cuda_sec": round(lat_cuda, 3),
            "latency_cpu_sec": round(lat_cpu, 3),
        }
        results.append(sample_res)

        status_mark = "CORRECT" if is_correct else ("UNCERTAIN" if is_uncertain else "WRONG")
        print(f"[{meta['id']}] GT: {gt:<12} | Pred: {pred_label:<12} | P(synth): {prob_synth:.4f} | P(real): {prob_real:.4f} | CUDA: {lat_cuda:.2f}s | CPU: {lat_cpu:.2f}s | {status_mark}")

    # Metrics calculation
    tp = sum(1 for r in results if r["ground_truth"] == "AI_GENERATED" and r["prediction"] == "AI_GENERATED")
    fn = sum(1 for r in results if r["ground_truth"] == "AI_GENERATED" and r["prediction"] != "AI_GENERATED")
    tn = sum(1 for r in results if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
    fp = sum(1 for r in results if r["ground_truth"] == "REAL" and r["prediction"] != "REAL")
    uncertain_count = sum(1 for r in results if r["is_uncertain"])

    total = len(results)
    exact_accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    
    spec_denom = (tn + fp)
    specificity = tn / spec_denom if spec_denom else 0.0
    balanced_acc = (recall + specificity) / 2.0

    benchmark_summary = {
        "hardware": {
            "cuda_available": has_cuda,
            "gpu": device_name,
            "torch_version": torch.__version__,
        },
        "total_samples": total,
        "confusion_matrix": {
            "true_positive": tp,
            "false_negative": fn,
            "true_negative": tn,
            "false_positive": fp,
            "uncertain": uncertain_count,
        },
        "metrics": {
            "accuracy": round(exact_accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
        },
        "per_sample_results": results,
    }

    out_file = Path("audio_module/outputs/validation_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Samples    : {total}")
    print(f"Confusion Matrix : TP={tp}, FN={fn}, TN={tn}, FP={fp}, Uncertain={uncertain_count}")
    print(f"Accuracy         : {exact_accuracy * 100:.1f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.1f}%")
    print(f"Precision        : {precision * 100:.1f}%")
    print(f"Recall           : {recall * 100:.1f}%")
    print(f"F1 Score         : {f1 * 100:.1f}%")
    print(f"Specificity      : {specificity * 100:.1f}%")
    print(f"Results saved to : {out_file}")

if __name__ == "__main__":
    main()
