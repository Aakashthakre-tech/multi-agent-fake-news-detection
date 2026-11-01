"""
evaluate_videomae_test.py
==========================
One-time final evaluation of the selected VideoMAE candidate model on the held-out UNSEEN 42-video test split.
- Evaluates strictly after model selection on validation split is frozen.
- Computes Accuracy, Balanced Accuracy, Macro Precision/Recall/F1, Weighted F1, ROC-AUC, per-class metrics, and Confusion Matrix.
- Saves results to videomae_final_test_metrics.json.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
CHECKPOINTS_DIR = _CLASSIFIER_DIR / "checkpoints"

import sys
sys.path.insert(0, str(_CLASSIFIER_DIR))
from run_videomae_experiments import VideoMAEClassifierHead

def evaluate_test(candidate_id: str):
    ckpt_path = CHECKPOINTS_DIR / f"{candidate_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = VideoMAEClassifierHead(
        in_dim=768,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=2,
        dropout=cfg.get("dropout", 0.4),
        norm_type=cfg.get("norm_type", "layernorm"),
        architecture=cfg.get("architecture", "mlp"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load cache
    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    if "test" not in data:
        raise RuntimeError("Test split features not found in cache. Run extract_videomae_features.py --include_test first.")

    test_feat = data["test"]["features"]
    test_labels = data["test"]["labels"].numpy()
    test_cids = data["test"]["cids"]

    with torch.no_grad():
        logits = model(test_feat)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = torch.argmax(logits, dim=1).numpy()

    test_acc = accuracy_score(test_labels, preds)
    test_bal_acc = balanced_accuracy_score(test_labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, preds, average="weighted", zero_division=0)
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(test_labels, preds, average=None, zero_division=0)
    roc_auc = roc_auc_score(test_labels, probs[:, 1])
    cm = confusion_matrix(test_labels, preds)

    eval_report = {
        "candidate_id": candidate_id,
        "checkpoint_path": str(ckpt_path),
        "best_validation_epoch": ckpt.get("epoch"),
        "test_total_samples": len(test_labels),
        "test_accuracy": round(float(test_acc), 4),
        "test_balanced_accuracy": round(float(test_bal_acc), 4),
        "test_macro_precision": round(float(p_macro), 4),
        "test_macro_recall": round(float(r_macro), 4),
        "test_macro_f1": round(float(f1_macro), 4),
        "test_weighted_f1": round(float(f1_weighted), 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "per_class": {
            "real": {
                "precision": round(float(p_class[0]), 4),
                "recall": round(float(r_class[0]), 4),
                "f1_score": round(float(f1_class[0]), 4),
                "support": int(support_class[0]),
            },
            "fake": {
                "precision": round(float(p_class[1]), 4),
                "recall": round(float(r_class[1]), 4),
                "f1_score": round(float(f1_class[1]), 4),
                "support": int(support_class[1]),
            },
        },
        "confusion_matrix": {
            "true_real_pred_real (TN)": int(cm[0, 0]),
            "true_real_pred_fake (FP)": int(cm[0, 1]),
            "true_fake_pred_real (FN)": int(cm[1, 0]),
            "true_fake_pred_fake (TP)": int(cm[1, 1]),
        },
    }

    out_file = _CLASSIFIER_DIR / "videomae_final_test_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("=" * 70)
    print("VIDEOMAE FINAL HELD-OUT TEST EVALUATION REPORT")
    print("=" * 70)
    print(json.dumps(eval_report, indent=2))
    print("\nClassification Report:\n", classification_report(test_labels, preds, target_names=["Real", "Fake"], digits=4))
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")
    print(f"\nSaved final metrics to: {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, required=True, help="Candidate ID (e.g. vmae_exp1_mlp64)")
    args = parser.parse_args()
    evaluate_test(args.candidate)
