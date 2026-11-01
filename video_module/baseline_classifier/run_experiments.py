"""
run_experiments.py
==================
Executes a structured suite of controlled experiments on the training and validation splits.
"""

import sys
from pathlib import Path
import torch

_CLASSIFIER_DIR = Path(__file__).resolve().parent
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))

from experiment_runner import train_single_experiment, update_results_table

EXPERIMENTS = [
    {
        "id": "exp1_norm_hybrid_reg",
        "description": "Hybrid pooling (attn+mean+max) with LayerNorm, strong weight decay (1e-2), and early stopping",
        "pooling_mode": "hybrid",
        "proj_dim": None,
        "norm_type": "layernorm",
        "hidden_dim": 128,
        "mlp_dim": 128,
        "dropout": 0.45,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
    {
        "id": "exp2_projected_bottleneck",
        "description": "Linear projection (1280 -> 128) + Mean+Max pooling + LayerNorm, regularized head",
        "pooling_mode": "mean_max",
        "proj_dim": 128,
        "norm_type": "layernorm",
        "mlp_dim": 64,
        "dropout": 0.40,
        "lr": 4e-4,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
    {
        "id": "exp3_projected_dynamics",
        "description": "Linear projection (1280 -> 128) + Mean+Max+Temporal Difference Dynamics + LayerNorm",
        "pooling_mode": "projected_dynamics",
        "proj_dim": 128,
        "norm_type": "layernorm",
        "mlp_dim": 64,
        "dropout": 0.40,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
    {
        "id": "exp4_l2_projected_dynamics",
        "description": "L2 feature normalization + Projection (1280 -> 128) + Temporal Dynamics",
        "pooling_mode": "projected_dynamics",
        "proj_dim": 128,
        "norm_type": "l2",
        "mlp_dim": 64,
        "dropout": 0.40,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
    {
        "id": "exp5_dynamics_labelsmooth",
        "description": "Projected Dynamics (1280 -> 128) + Label Smoothing (0.08) + LayerNorm",
        "pooling_mode": "projected_dynamics",
        "proj_dim": 128,
        "norm_type": "layernorm",
        "mlp_dim": 64,
        "dropout": 0.40,
        "label_smoothing": 0.08,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
    {
        "id": "exp6_compact_proj64",
        "description": "Ultra-compact bottleneck (1280 -> 64) + Temporal Dynamics + LayerNorm (~80k params)",
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
    },
    {
        "id": "exp7_pure_mean_max",
        "description": "Pure Mean+Max pooling without projection (2560-d) + LayerNorm + High Weight Decay",
        "pooling_mode": "mean_max",
        "proj_dim": None,
        "norm_type": "layernorm",
        "mlp_dim": 64,
        "dropout": 0.45,
        "lr": 2e-4,
        "weight_decay": 1.5e-2,
        "epochs": 40,
        "patience": 10,
        "seed": 42,
    },
]

def main():
    print("=== TRUTHLENSAI CONTROLLED VIDEO CLASSIFIER EXPERIMENT SUITE ===")
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Add baseline exp0 record
    results = [
        {
            "experiment_id": "exp0_original_baseline",
            "config": {
                "description": "Original EfficientNet-B0 + Attention + Mean + Max (3840-d), no input norm, lr=1e-3, wd=1e-4",
                "pooling_mode": "hybrid",
                "proj_dim": None,
                "norm_type": "none",
                "lr": 0.001,
                "weight_decay": 0.0001,
                "dropout": 0.35,
                "label_smoothing": 0.0,
            },
            "best_epoch": 4,
            "epochs_trained": 35,
            "duration_sec": 4.5,
            "val_accuracy": 0.7674,
            "val_macro_f1": 0.7352,
            "val_balanced_accuracy": 0.7300,
            "val_roc_auc": 0.7822,
            "val_loss": 0.6185,
            "confusion_matrix": {"TN": 9, "FP": 9, "FN": 1, "TP": 24},
        }
    ]

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        print(f"\n---> Running [{exp_id}]: {exp['description']}")
        res = train_single_experiment(exp_id, exp, device=device)
        results.append(res)
        cm = res.get("confusion_matrix", {})
        print(f"     Finished in {res['duration_sec']}s (Best Epoch: {res['best_epoch']}/{res['epochs_trained']})")
        print(f"     Val Acc: {res['val_accuracy']:.4f} | Macro F1: {res['val_macro_f1']:.4f} | Bal Acc: {res['val_balanced_accuracy']:.4f} | AUC: {res['val_roc_auc']:.4f} | Val Loss: {res['val_loss']:.4f}")
        print(f"     Confusion: TN={cm.get('TN')}, FP={cm.get('FP')}, FN={cm.get('FN')}, TP={cm.get('TP')}")

    update_results_table(results)
    print("\n[Done] All experiments completed. Results saved to experiments_results.csv and experiments_log.json")

if __name__ == "__main__":
    main()
