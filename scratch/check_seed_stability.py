"""
check_seed_stability.py
=======================
Evaluate compact projected dynamics architecture across different random seeds on validation split.
"""

import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
sys.path.insert(0, str(_CLASSIFIER_DIR))

from experiment_runner import train_single_experiment

seeds = [42, 101, 2024]
results = []

for s in seeds:
    cfg = {
        "id": f"seed_eval_{s}",
        "description": f"Compact Proj64 Dynamics with seed={s}",
        "pooling_mode": "projected_dynamics",
        "proj_dim": 64,
        "norm_type": "layernorm",
        "mlp_dim": 48,
        "dropout": 0.40,
        "lr": 4e-4,
        "weight_decay": 1.5e-2,
        "epochs": 40,
        "patience": 10,
        "seed": s,
    }
    r = train_single_experiment(cfg["id"], cfg)
    results.append(r)
    cm = r.get("confusion_matrix", {})
    print(f"Seed {s:4d} -> Val Acc: {r['val_accuracy']:.4f} | Macro F1: {r['val_macro_f1']:.4f} | Bal Acc: {r['val_balanced_accuracy']:.4f} | AUC: {r['val_roc_auc']:.4f} | TN={cm.get('TN')}, FP={cm.get('FP')}")

accs = [r["val_accuracy"] for r in results]
f1s = [r["val_macro_f1"] for r in results]
print(f"\nMean Val Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Mean Val F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
