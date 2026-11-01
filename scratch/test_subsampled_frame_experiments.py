"""
test_subsampled_frame_experiments.py
====================================
Evaluates the impact of frame sampling strategies on validation performance
by evaluating interior vs boundary frames and frame counts (6, 8, 10, 12 frames).
"""

import sys
from pathlib import Path
import torch
import numpy as np

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
sys.path.insert(0, str(_CLASSIFIER_DIR))

from experiment_runner import (
    ModularTemporalVideoClassifier,
    train_single_experiment,
    update_results_table,
    CACHE_PATH,
)

# Load existing cache
raw_cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

sampling_strategies = {
    "12_frames_all": list(range(12)),
    "10_frames_interior": list(range(1, 11)), # Drops frame 0 (fade-in/logo) and frame 11 (outro)
    "8_frames_interior": [1, 2, 4, 5, 7, 8, 9, 10], # 8 frames excluding boundaries
    "8_frames_uniform": [0, 1, 3, 4, 6, 7, 9, 11], # 8 frames uniform across full sequence
    "6_frames_sparse": [0, 2, 4, 6, 8, 10], # 6 frames sparsely sampled
}

print("=== FRAME SAMPLING COMPARISON ON VALIDATION SET ===")

frame_exp_results = []

for s_name, indices in sampling_strategies.items():
    print(f"\nEvaluating frame subset: {s_name} (Indices: {indices}, Count: {len(indices)})")
    
    # Create sub-sampled cache in-memory
    sub_cache = {
        "train": {
            "features": raw_cache["train"]["features"][:, indices, :],
            "labels": raw_cache["train"]["labels"],
            "cids": raw_cache["train"]["cids"],
        },
        "val": {
            "features": raw_cache["val"]["features"][:, indices, :],
            "labels": raw_cache["val"]["labels"],
            "cids": raw_cache["val"]["cids"],
        },
        "test": {
            "features": raw_cache["test"]["features"][:, indices, :],
            "labels": raw_cache["test"]["labels"],
            "cids": raw_cache["test"]["cids"],
        },
        "config": raw_cache.get("config", {}),
        "class_map": raw_cache.get("class_map", {}),
    }
    
    tmp_cache_path = _CLASSIFIER_DIR / f"temp_cache_{s_name}.pt"
    torch.save(sub_cache, tmp_cache_path)
    
    exp_config = {
        "id": f"frames_{s_name}",
        "description": f"Frame sampling strategy: {s_name} ({len(indices)} frames) with Compact Proj64 + Dynamics",
        "pooling_mode": "projected_dynamics",
        "proj_dim": 64,
        "norm_type": "layernorm",
        "mlp_dim": 48,
        "dropout": 0.40,
        "lr": 4e-4,
        "weight_decay": 1.5e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    }
    
    res = train_single_experiment(exp_config["id"], exp_config, cache_path=tmp_cache_path)
    frame_exp_results.append(res)
    
    cm = res.get("confusion_matrix", {})
    print(f"[{s_name}] -> Val Acc: {res['val_accuracy']:.4f} | Macro F1: {res['val_macro_f1']:.4f} | Bal Acc: {res['val_balanced_accuracy']:.4f} | AUC: {res['val_roc_auc']:.4f}")
    print(f"             Confusion: TN={cm.get('TN')}, FP={cm.get('FP')}, FN={cm.get('FN')}, TP={cm.get('TP')} (Best Epoch: {res['best_epoch']})")
    
    # Clean up temp cache
    if tmp_cache_path.exists():
        tmp_cache_path.unlink()
