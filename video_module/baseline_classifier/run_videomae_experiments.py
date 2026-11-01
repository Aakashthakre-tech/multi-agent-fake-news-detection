"""
run_videomae_experiments.py
===========================
Executes controlled experiments on frozen VideoMAE representations (MCG-NJU/videomae-base).
- Consumes strictly train and val from videomae_features_cache.pt.
- Compares 5 controlled regularized classification head configurations.
- Selects the best candidate based strictly on validation Macro F1, balanced accuracy, and class recall.
- Saves all configs, history, and metrics to videomae_experiments_results.csv and videomae_experiments_log.json.
- Prepares comparison table with exp6_compact_proj64 baseline.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
CHECKPOINTS_DIR = _CLASSIFIER_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

EXP_RESULTS_CSV = _CLASSIFIER_DIR / "videomae_experiments_results.csv"
EXP_LOG_JSON = _CLASSIFIER_DIR / "videomae_experiments_log.json"

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class VideoMAEClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.4,
        norm_type: str = "layernorm",
        architecture: str = "mlp",
    ):
        super().__init__()
        if norm_type == "layernorm":
            self.input_norm = nn.LayerNorm(in_dim)
        else:
            self.input_norm = nn.Identity()

        if architecture == "linear":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_classes)
            )
        elif architecture == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif architecture == "deep_mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.input_norm(x)
        return self.head(normed)

def train_and_eval_head(
    exp_id: str,
    cfg: Dict[str, Any],
    train_feat: torch.Tensor,
    train_labels: torch.Tensor,
    val_feat: torch.Tensor,
    val_labels: torch.Tensor,
    patience: int = 15,
) -> Dict[str, Any]:
    set_seed(cfg.get("seed", 42))

    train_ds = TensorDataset(train_feat, train_labels)
    val_ds = TensorDataset(val_feat, val_labels)

    batch_size = cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VideoMAEClassifierHead(
        in_dim=768,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=2,
        dropout=cfg.get("dropout", 0.4),
        norm_type=cfg.get("norm_type", "layernorm"),
        architecture=cfg.get("architecture", "mlp"),
    )

    weight_decay = cfg.get("weight_decay", 1e-2)
    lr = cfg.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    if cfg.get("use_class_weights", False):
        # Inverse class frequency
        n_real = (train_labels == 0).sum().item()
        n_fake = (train_labels == 1).sum().item()
        total = n_real + n_fake
        weights = torch.tensor([total / (2.0 * n_real), total / (2.0 * n_fake)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get("label_smoothing", 0.0))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    scheduler = None
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 40), eta_min=1e-5)

    epochs = cfg.get("epochs", 40)
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_val_metrics = {}
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                val_logits_list.append(model(x_b))

        val_logits = torch.cat(val_logits_list, dim=0)
        val_probs = torch.softmax(val_logits, dim=1).numpy()
        val_preds = torch.argmax(val_logits, dim=1).numpy()
        y_val_np = val_labels.numpy()

        val_loss = criterion(val_logits, val_labels).item()
        val_acc = accuracy_score(y_val_np, val_preds)
        val_bal_acc = balanced_accuracy_score(y_val_np, val_preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val_np, val_preds, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_val_np, val_preds, average=None, zero_division=0)
        val_auc = roc_auc_score(y_val_np, val_probs[:, 1])
        cm = confusion_matrix(y_val_np, val_preds)

        if f1_macro > best_val_macro_f1:
            best_val_macro_f1 = f1_macro
            best_epoch = epoch
            patience_counter = 0
            best_val_metrics = {
                "val_accuracy": round(float(val_acc), 4),
                "val_balanced_accuracy": round(float(val_bal_acc), 4),
                "val_macro_precision": round(float(p_macro), 4),
                "val_macro_recall": round(float(r_macro), 4),
                "val_macro_f1": round(float(f1_macro), 4),
                "val_roc_auc": round(float(val_auc), 4),
                "val_loss": round(float(val_loss), 4),
                "real_precision": round(float(p_c[0]), 4),
                "real_recall": round(float(r_c[0]), 4),
                "real_f1": round(float(f1_c[0]), 4),
                "fake_precision": round(float(p_c[1]), 4),
                "fake_recall": round(float(r_c[1]), 4),
                "fake_f1": round(float(f1_c[1]), 4),
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            }
            # Save checkpoint
            ckpt_path = CHECKPOINTS_DIR / f"{exp_id}.pt"
            torch.save({
                "exp_id": exp_id,
                "epoch": best_epoch,
                "config": cfg,
                "model_state_dict": model.state_dict(),
                "val_metrics": best_val_metrics,
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    duration = round(time.time() - t0, 2)
    return {
        "exp_id": exp_id,
        "description": cfg.get("description", ""),
        "architecture": cfg.get("architecture", ""),
        "hidden_dim": cfg.get("hidden_dim", 0),
        "dropout": cfg.get("dropout", 0.0),
        "lr": cfg.get("lr", 0.0),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "use_class_weights": cfg.get("use_class_weights", False),
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "best_epoch": best_epoch,
        "duration_sec": duration,
        "checkpoint": str(CHECKPOINTS_DIR / f"{exp_id}.pt"),
        **best_val_metrics,
    }

def main():
    if not CACHE_PATH.exists():
        print(f"Error: Cache path {CACHE_PATH} not found. Please run extract_videomae_features.py first.")
        sys.exit(1)

    print("Loading cached VideoMAE representations...")
    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    train_feat = data["train"]["features"]
    train_labels = data["train"]["labels"]
    val_feat = data["val"]["features"]
    val_labels = data["val"]["labels"]

    print(f"Train features: {train_feat.shape}, Val features: {val_feat.shape}")

    configs = [
        {
            "id": "vmae_exp0_linear",
            "description": "Linear probing (LayerNorm + Linear 768->2)",
            "architecture": "linear",
            "hidden_dim": 0,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 40,
            "seed": 42,
        },
        {
            "id": "vmae_exp1_mlp64",
            "description": "LayerNorm + 64-d projection + Dropout(0.4) + Linear(64->2)",
            "architecture": "mlp",
            "hidden_dim": 64,
            "dropout": 0.4,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 40,
            "seed": 42,
        },
        {
            "id": "vmae_exp2_mlp128",
            "description": "LayerNorm + 128-d projection + Dropout(0.5) + Linear(128->2)",
            "architecture": "mlp",
            "hidden_dim": 128,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 40,
            "seed": 42,
        },
        {
            "id": "vmae_exp3_mlp64_weighted",
            "description": "LayerNorm + 64-d projection with inverse class weights",
            "architecture": "mlp",
            "hidden_dim": 64,
            "dropout": 0.4,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "use_class_weights": True,
            "epochs": 40,
            "seed": 42,
        },
        {
            "id": "vmae_exp4_mlp64_label_smooth",
            "description": "LayerNorm + 64-d projection with label smoothing (0.1) & cosine decay",
            "architecture": "mlp",
            "hidden_dim": 64,
            "dropout": 0.4,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,
            "scheduler": "cosine",
            "epochs": 40,
            "seed": 42,
        },
    ]

    print(f"\nRunning {len(configs)} VideoMAE head experiments...")
    all_results = []
    for c in configs:
        print(f"\n--- Running [{c['id']}]: {c['description']} ---")
        res = train_and_eval_head(c["id"], c, train_feat, train_labels, val_feat, val_labels)
        all_results.append(res)
        print(f"    Val Acc: {res['val_accuracy']:.4f} | Val Macro F1: {res['val_macro_f1']:.4f} | Val Bal Acc: {res['val_balanced_accuracy']:.4f} | AUC: {res['val_roc_auc']:.4f}")
        print(f"    Real: Rec={res['real_recall']:.4f}, F1={res['real_f1']:.4f} | Fake: Rec={res['fake_recall']:.4f}, F1={res['fake_f1']:.4f}")

    # Save to JSON and CSV
    with open(EXP_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = list(all_results[0].keys())
    with open(EXP_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\nAll results saved to:\n  {EXP_LOG_JSON}\n  {EXP_RESULTS_CSV}")

    # Select best candidate
    best_candidate = max(all_results, key=lambda x: (x["val_macro_f1"], x["val_balanced_accuracy"]))
    print("\n" + "=" * 70)
    print("BEST VIDEOMAE CANDIDATE SELECTED ON VALIDATION:")
    print("=" * 70)
    print(f"Candidate ID: {best_candidate['exp_id']}")
    print(f"Validation Macro F1: {best_candidate['val_macro_f1']:.4f}")
    print(f"Validation Accuracy: {best_candidate['val_accuracy']:.4f}")
    print(f"Validation Balanced Accuracy: {best_candidate['val_balanced_accuracy']:.4f}")
    print(f"Validation ROC-AUC: {best_candidate['val_roc_auc']:.4f}")
    print(f"Confusion Matrix: TN={best_candidate['TN']}, FP={best_candidate['FP']}, FN={best_candidate['FN']}, TP={best_candidate['TP']}")

if __name__ == "__main__":
    main()
