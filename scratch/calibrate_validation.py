"""
calibrate_validation.py
=======================
Examines predicted probabilities, calibration, and optimal threshold strictly on the VALIDATION split.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss, confusion_matrix

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
sys.path.insert(0, str(_CLASSIFIER_DIR))

from experiment_runner import ModularTemporalVideoClassifier

ckpt_path = _CLASSIFIER_DIR / "checkpoints" / "exp6_compact_proj64.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

cache = torch.load(_CLASSIFIER_DIR / "features_cache.pt", map_location="cpu", weights_only=False)
val_feat = cache["val"]["features"]
val_y = cache["val"]["labels"].numpy()

model = ModularTemporalVideoClassifier(
    embed_dim=1280,
    proj_dim=64,
    hidden_dim=128,
    mlp_dim=48,
    num_classes=2,
    dropout=0.40,
    pooling_mode="projected_dynamics",
    norm_type="layernorm",
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(val_feat)
    probs = torch.softmax(logits, dim=1).numpy()

fake_probs = probs[:, 1]
brier = brier_score_loss(val_y, fake_probs)
auc = roc_auc_score(val_y, fake_probs)

print(f"=== EXP6 CALIBRATION ON VALIDATION SPLIT ===")
print(f"Val Samples: {len(val_y)} (Real={sum(val_y==0)}, Fake={sum(val_y==1)})")
print(f"Brier Score: {brier:.4f}")
print(f"ROC-AUC:     {auc:.4f}")

# Threshold scan strictly on validation to see behavior
print("\n--- Validation Threshold Behavior ---")
best_th = 0.50
best_bal = 0.0
for th in np.arange(0.35, 0.65, 0.05):
    preds = (fake_probs >= th).astype(int)
    acc = accuracy_score(val_y, preds)
    bal = balanced_accuracy_score(val_y, preds)
    _, _, f1, _ = precision_recall_fscore_support(val_y, preds, average="macro", zero_division=0)
    cm = confusion_matrix(val_y, preds)
    print(f"Th: {th:.2f} | Acc: {acc:.4f} | Bal Acc: {bal:.4f} | Macro F1: {f1:.4f} | TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    if bal > best_bal:
        best_bal = bal
        best_th = th

print(f"\nDefault th=0.50 yields balanced accuracy: {balanced_accuracy_score(val_y, (fake_probs>=0.5).astype(int)):.4f}")
