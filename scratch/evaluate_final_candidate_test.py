"""
evaluate_final_candidate_test.py
================================
Runs the FINAL one-time evaluation of the selected candidate model (exp6_compact_proj64)
on the held-out UNSEEN 42-video test split.
Calculates: Accuracy, Balanced Accuracy, Macro F1, Weighted F1, ROC-AUC, per-class metrics,
and the full confusion matrix.
"""

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
ckpt_path = _CLASSIFIER_DIR / "checkpoints" / "exp6_compact_proj64.pt"
cache_path = _CLASSIFIER_DIR / "features_cache.pt"

import sys
sys.path.insert(0, str(_CLASSIFIER_DIR))
from experiment_runner import ModularTemporalVideoClassifier

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = ModularTemporalVideoClassifier(
    embed_dim=cfg.get("embed_dim", 1280),
    proj_dim=cfg.get("proj_dim", 64),
    hidden_dim=cfg.get("hidden_dim", 128),
    mlp_dim=cfg.get("mlp_dim", 48),
    num_classes=2,
    dropout=cfg.get("dropout", 0.40),
    pooling_mode=cfg.get("pooling_mode", "projected_dynamics"),
    norm_type=cfg.get("norm_type", "layernorm"),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

data = torch.load(cache_path, map_location="cpu", weights_only=False)
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
    "candidate_id": "exp6_compact_proj64",
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

print("=== FINAL HELD-OUT TEST EVALUATION REPORT ===")
print(json.dumps(eval_report, indent=2))
print("\nClassification Report:\n", classification_report(test_labels, preds, target_names=["Real", "Fake"], digits=4))
print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")

with open(_CLASSIFIER_DIR / "final_candidate_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2)
print(f"\nSaved final metrics to {_CLASSIFIER_DIR / 'final_candidate_test_metrics.json'}")
