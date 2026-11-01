"""
evaluate.py
===========
Phase 4: Held-out test evaluation.

Run EXACTLY ONCE after training is complete and a single best model is selected.
Results are written to:
  - video_module/outputs/AI_GENERATED_VIDEO_EVALUATION_REPORT.md
  - video_module/outputs/ai_gen_evaluation_metrics.json
  - video_module/outputs/confusion_matrix_aigen.png

Usage
-----
    python -m video_module.ai_generated_detector.evaluate \
        [--split test] [--device auto]

    # Also evaluate OOD generator-disjoint split:
    python -m video_module.ai_generated_detector.evaluate --split ood
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from video_module.ai_generated_detector.config import (
    BEST_MODEL, CKPT_DIR, OUTPUTS_DIR, EXP_LOG,
    CACHE_TEST, CACHE_OOD, TEST_CSV, OOD_CSV,
    NUM_FRAMES, FEATURE_DIM, GRU_HIDDEN, DROPOUT,
    CLASS_NAMES, CLASS_MAP, INV_CLASS_MAP,
    MODEL_NAME, MODEL_VERSION, EXPERIMENTAL_WARNING,
    BATCH_SIZE, SEED,
)
from video_module.ai_generated_detector.dataset import FeatureDataset
from video_module.ai_generated_detector.model import build_model
from video_module.ai_generated_detector.train import compute_metrics, set_seed


SPLIT_TO_CSV_CACHE = {
    "test": (TEST_CSV, CACHE_TEST),
    "ood":  (OOD_CSV,  CACHE_OOD),
}


# ════════════════════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ════════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    out_path: Path,
    split_name: str = "test",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {split_name} split\n{MODEL_NAME}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color=color, fontsize=12)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[eval] Confusion matrix saved → {out_path}")
    except ImportError:
        print("[eval] matplotlib not installed — skipping confusion matrix plot")


# ════════════════════════════════════════════════════════════════════════════════
# Training curves plot
# ════════════════════════════════════════════════════════════════════════════════

def save_training_curves(log_path: Path, out_path: Path) -> None:
    if not log_path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(log_path) as f:
            log = json.load(f)
        hist = log.get("history", [])
        if not hist:
            return

        epochs      = [h["epoch"] for h in hist]
        train_loss  = [h["train"]["loss"] for h in hist]
        val_loss    = [h["val"]["loss"] for h in hist]
        val_f1      = [h["val"]["macro_f1"] for h in hist]
        val_bal_acc = [h["val"]["balanced_acc"] for h in hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, train_loss, label="train loss")
        ax1.plot(epochs, val_loss,   label="val loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
        ax1.set_title("Loss curves")

        ax2.plot(epochs, val_f1,      label="val macro-F1")
        ax2.plot(epochs, val_bal_acc, label="val balanced-acc")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.legend()
        ax2.set_title("Validation metrics")

        plt.suptitle(MODEL_NAME, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[eval] Training curves saved → {out_path}")
    except Exception as e:
        print(f"[eval] Could not save training curves: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Error analysis
# ════════════════════════════════════════════════════════════════════════════════

def collect_errors(
    ds:     FeatureDataset,
    preds:  List[int],
    probs:  List[float],
    max_n:  int = 10,
) -> Dict:
    ids  = ds.get_video_ids()
    gens = ds.get_generators()
    labels = ds.get_labels()

    fps, fns = [], []
    for i, (l, p, prob, vid, gen) in enumerate(zip(labels, preds, probs, ids, gens)):
        entry = {"video_id": vid, "generator": gen, "true_label": INV_CLASS_MAP.get(l,"?"),
                 "pred_label": INV_CLASS_MAP.get(p,"?"), "prob_ai_generated": round(prob, 4)}
        if l == 0 and p == 1:   # REAL predicted as AI_GENERATED
            fps.append(entry)
        elif l == 1 and p == 0: # AI_GENERATED predicted as REAL
            fns.append(entry)

    return {
        "false_positives": fps[:max_n],
        "false_negatives": fns[:max_n],
        "fp_count": len(fps),
        "fn_count": len(fns),
    }


# ════════════════════════════════════════════════════════════════════════════════
# Per-generator breakdown
# ════════════════════════════════════════════════════════════════════════════════

def per_generator_metrics(
    ds:     FeatureDataset,
    preds:  List[int],
    labels: List[int],
    probs:  List[float],
) -> Dict:
    gens = ds.get_generators()
    gen_set = sorted(set(gens))
    result = {}
    for gen in gen_set:
        idxs = [i for i, g in enumerate(gens) if g == gen]
        if len(idxs) < 2:
            continue
        sub_labels = [labels[i] for i in idxs]
        sub_preds  = [preds[i]  for i in idxs]
        sub_probs  = [probs[i]  for i in idxs]
        m = compute_metrics(sub_labels, sub_preds, sub_probs)
        result[gen] = {"n": len(idxs), **m}
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ════════════════════════════════════════════════════════════════════════════════

def evaluate(split: str = "test", device_str: str = "auto") -> Dict:
    set_seed(SEED)

    csv_path, cache_path = SPLIT_TO_CSV_CACHE[split]

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[eval] Device: {device}, split: {split}")

    # Load checkpoint
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"No checkpoint at {BEST_MODEL}. Run train.py first.")
    ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=False)
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch','?')}")
    print(f"[eval] Validation macro-F1 at save: {ckpt.get('val_metrics',{}).get('macro_f1','?')}")

    # Reconstruct model from checkpoint config
    ckpt_cfg = ckpt.get("config", {})
    model = build_model(
        feature_dim = ckpt_cfg.get("feature_dim", FEATURE_DIM),
        hidden_dim  = ckpt_cfg.get("hidden_dim",  GRU_HIDDEN),
        dropout     = ckpt_cfg.get("dropout",     DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    num_frames = ckpt_cfg.get("num_frames", NUM_FRAMES)

    # Dataset
    ds     = FeatureDataset(csv_path, cache_path, num_frames=num_frames, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Inference
    all_labels, all_preds, all_probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(features)
            probs_batch = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds_batch = logits.argmax(dim=1).cpu().tolist()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds_batch)
            all_probs.extend(probs_batch)
    infer_time_s = time.time() - t0
    per_video_ms = 1000 * infer_time_s / max(len(ds), 1)
    print(f"[eval] Inference: {infer_time_s:.2f}s total, {per_video_ms:.1f} ms/video")

    # Metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    errors  = collect_errors(ds, all_preds, all_probs)
    gen_breakdown = per_generator_metrics(ds, all_preds, all_labels, all_probs)

    # Peak VRAM
    peak_vram_mb = 0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) // 1024 // 1024

    # Full result dict
    result = {
        "split":              split,
        "model_name":         MODEL_NAME,
        "model_version":      MODEL_VERSION,
        "experimental_warning": EXPERIMENTAL_WARNING,
        "checkpoint_path":    str(BEST_MODEL),
        "val_macro_f1_at_save": ckpt.get("val_metrics", {}).get("macro_f1"),
        "dataset": {
            "csv_path":  str(csv_path),
            "cache_path":str(cache_path),
            "n_samples": len(ds),
        },
        "inference": {
            "total_time_s":   round(infer_time_s, 2),
            "per_video_ms":   round(per_video_ms, 2),
            "peak_vram_mb":   peak_vram_mb,
            "device":         str(device),
        },
        "metrics": metrics,
        "per_generator": gen_breakdown,
        "errors": errors,
    }

    return result


# ════════════════════════════════════════════════════════════════════════════════
# Report writers
# ════════════════════════════════════════════════════════════════════════════════

def write_json_report(result: Dict, ood_result: Optional[Dict], out_path: Path) -> None:
    payload = {"in_distribution": result}
    if ood_result:
        payload["out_of_distribution"] = ood_result
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] JSON metrics → {out_path}")


def write_markdown_report(
    result:     Dict,
    ood_result: Optional[Dict],
    train_log:  Optional[Dict],
    out_path:   Path,
) -> None:
    m = result["metrics"]
    cm = m["confusion_matrix"]
    pc = m["per_class"]
    inf = result["inference"]

    lines = [
        "# TruthLensAI — AI-Generated Video Detection",
        "## Pilot Evaluation Report",
        "",
        "> **PILOT EXPERIMENT** — trained on ~500 videos from 2 generators.",
        "> Results are preliminary. Do not claim production-grade performance.",
        "",
        f"**Model:** {result['model_name']} v{result['model_version']}  ",
        f"**Checkpoint:** `{result['checkpoint_path']}`  ",
        f"**Val macro-F1 at save:** {result['val_macro_f1_at_save']}  ",
        "",
        "---",
        "",
        "## Dataset",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Split | `{result['split']}` |",
        f"| Samples | {result['dataset']['n_samples']} |",
        f"| Cache | `{Path(result['dataset']['cache_path']).name}` |",
        f"| License | CC-BY-NC-4.0 (GenVideo synthetic) / public domain (real) |",
        "",
        "---",
        "",
        "## In-Distribution Test Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Balanced Accuracy | {m['balanced_acc']:.4f} |",
        f"| Macro-F1 | {m['macro_f1']:.4f} |",
        f"| Weighted-F1 | {m['weighted_f1']:.4f} |",
        f"| ROC-AUC | {m['roc_auc'] if m['roc_auc'] is not None else 'N/A'} |",
        "",
        "### Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for cls in CLASS_NAMES:
        p = pc[cls]
        lines.append(f"| {cls} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {p['support']} |")

    lines += [
        "",
        "### Confusion Matrix",
        "",
        "```",
        f"              Pred REAL   Pred AI_GEN",
        f"True REAL      {cm[0][0]:>8}    {cm[0][1]:>8}",
        f"True AI_GEN    {cm[1][0]:>8}    {cm[1][1]:>8}",
        "```",
        "",
        "---",
        "",
        "## Per-Generator Breakdown",
        "",
        "| Generator | N | Accuracy | Balanced Acc | Macro-F1 |",
        "|-----------|---|----------|--------------|----------|",
    ]
    for gen, gm in result["per_generator"].items():
        lines.append(
            f"| {gen} | {gm['n']} | {gm['accuracy']:.4f} | "
            f"{gm['balanced_acc']:.4f} | {gm['macro_f1']:.4f} |"
        )

    if ood_result:
        ood_m = ood_result["metrics"]
        lines += [
            "",
            "---",
            "",
            "## Out-of-Distribution (Generator-Disjoint) Test",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| N samples | {ood_result['dataset']['n_samples']} |",
            f"| Accuracy | {ood_m['accuracy']:.4f} |",
            f"| Balanced Accuracy | {ood_m['balanced_acc']:.4f} |",
            f"| Macro-F1 | {ood_m['macro_f1']:.4f} |",
            f"| ROC-AUC | {ood_m['roc_auc'] if ood_m['roc_auc'] is not None else 'N/A'} |",
        ]

    err = result["errors"]
    lines += [
        "",
        "---",
        "",
        "## Error Analysis",
        "",
        f"- **False Positives** (REAL → AI_GENERATED): {err['fp_count']}",
        f"- **False Negatives** (AI_GENERATED → REAL): {err['fn_count']}",
        "",
    ]
    if err["false_positives"]:
        lines.append("### Top False Positives")
        lines.append("| video_id | generator | conf |")
        lines.append("|----------|-----------|------|")
        for e in err["false_positives"][:5]:
            lines.append(f"| {e['video_id']} | {e['generator']} | {e['prob_ai_generated']:.3f} |")

    if err["false_negatives"]:
        lines.append("")
        lines.append("### Top False Negatives")
        lines.append("| video_id | generator | conf |")
        lines.append("|----------|-----------|------|")
        for e in err["false_negatives"][:5]:
            lines.append(f"| {e['video_id']} | {e['generator']} | {e['prob_ai_generated']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## Inference Performance",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Device | {inf['device']} |",
        f"| Total inference time | {inf['total_time_s']:.2f}s |",
        f"| Per-video latency | {inf['per_video_ms']:.1f} ms |",
        f"| Peak VRAM | {inf['peak_vram_mb']} MB |",
        "",
    ]

    if train_log:
        lines += [
            "---",
            "",
            "## Training Summary",
            "",
            f"| Item | Value |",
            f"|------|-------|",
            f"| Epochs run | {train_log.get('epochs_run', '?')} |",
            f"| Best val macro-F1 | {train_log.get('best_val_macro_f1', '?')} |",
            f"| Training time | {train_log.get('total_train_time_s', '?')}s |",
            f"| Device | {train_log.get('config', {}).get('device', '?')} |",
            "",
        ]

    lines += [
        "---",
        "",
        "## Limitations",
        "",
        "1. **Pilot scale**: Only ~500 videos from 2 generators. Results cannot be generalized.",
        "2. **Generator coverage**: ZeroScope and VideoCraft2 only. Modern generators (Sora, Pika, SVD) not represented.",
        "3. **Real video source**: Limited to one source; may reflect source-specific patterns.",
        "4. **No temporal forensics**: BiGRU over spatial features may not capture all motion artifacts.",
        "5. **Test set size**: Small test split (~75 videos) gives wide confidence intervals on all metrics.",
        "6. **Class balance**: If imbalanced, balanced accuracy and macro-F1 are primary metrics.",
        "7. **CPU training** (if GPU unavailable): training dynamics may differ from GPU run.",
        "",
        "---",
        "",
        f"*{EXPERIMENTAL_WARNING}*",
    ]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] Markdown report → {out_path}")


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained AI-gen detector on held-out test data.")
    ap.add_argument("--split",  choices=["test", "ood"], default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--also-ood", action="store_true", default=True,
                    help="Also run OOD evaluation after the primary split.")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Primary evaluation
    result = evaluate(args.split, args.device)

    # Optional OOD
    ood_result = None
    if args.also_ood and OOD_CSV.exists() and CACHE_OOD.exists():
        print("\n[eval] Running OOD (generator-disjoint) evaluation...")
        ood_result = evaluate("ood", args.device)
    elif args.also_ood:
        print("[eval] OOD split/cache not found — skipping OOD evaluation.")

    # Load training log
    train_log = None
    if EXP_LOG.exists():
        with open(EXP_LOG) as f:
            train_log = json.load(f)

    # Write reports
    json_out = OUTPUTS_DIR / "ai_gen_evaluation_metrics.json"
    md_out   = OUTPUTS_DIR / "AI_GENERATED_VIDEO_EVALUATION_REPORT.md"
    cm_out   = OUTPUTS_DIR / "confusion_matrix_aigen.png"
    curves_out = OUTPUTS_DIR / "training_curves_aigen.png"

    write_json_report(result, ood_result, json_out)
    write_markdown_report(result, ood_result, train_log, md_out)

    save_confusion_matrix(result["metrics"]["confusion_matrix"], CLASS_NAMES, cm_out, args.split)
    save_training_curves(EXP_LOG, curves_out)

    # Console summary
    m = result["metrics"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Split          : {args.split} (n={result['dataset']['n_samples']})")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  Balanced Acc   : {m['balanced_acc']:.4f}")
    print(f"  Macro-F1       : {m['macro_f1']:.4f}")
    print(f"  Weighted-F1    : {m['weighted_f1']:.4f}")
    print(f"  ROC-AUC        : {m['roc_auc']}")
    print(f"  Confusion mat  : {m['confusion_matrix']}")
    print(f"  Inference      : {result['inference']['per_video_ms']:.1f} ms/video")
    print("=" * 60)
    print(f"\nReports saved:")
    print(f"  {md_out}")
    print(f"  {json_out}")
    print(f"  {cm_out}")


if __name__ == "__main__":
    main()
