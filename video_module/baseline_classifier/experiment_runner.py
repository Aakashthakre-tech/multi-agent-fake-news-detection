"""
experiment_runner.py
====================
Reproducible experiment runner for TruthLensAI video classifier optimization.

Features:
1. Strict protocol: Train ONLY on training split, validate ONLY on validation split.
2. Protects held-out test split: Never used for model selection, thresholding, or hyperparameter search.
3. Modular architecture and pooling options:
   - pooling_mode: 'hybrid' (original baseline), 'mean_max', 'mean', 'attention', 'projected_dynamics'
   - feature normalization: None, 'layernorm', 'l2'
   - regularized MLP with tunable bottleneck and dropout
4. Early stopping with validation macro F1 and validation loss.
5. Logs all metrics to experiments_log.json and experiments_results.csv.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Ensure path
_CLASSIFIER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CLASSIFIER_DIR.parent.parent
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))

CHECKPOINTS_DIR = _CLASSIFIER_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = _CLASSIFIER_DIR / "features_cache.pt"
EXPERIMENTS_CSV = _CLASSIFIER_DIR / "experiments_results.csv"
EXPERIMENTS_JSON = _CLASSIFIER_DIR / "experiments_log.json"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModularTemporalVideoClassifier(nn.Module):
    """
    Modular, regularized video classifier supporting:
    - Input normalization (LayerNorm, L2 norm)
    - Projection head to reduce dimensionality before pooling
    - Multiple temporal pooling modes ('hybrid', 'mean_max', 'mean', 'attention', 'projected_dynamics')
    - Regularized classification head
    """
    def __init__(
        self,
        embed_dim: int = 1280,
        proj_dim: Optional[int] = 128,
        hidden_dim: int = 128,
        mlp_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.4,
        pooling_mode: str = "projected_dynamics",
        norm_type: str = "layernorm",  # "none", "layernorm", "l2"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.pooling_mode = pooling_mode
        self.norm_type = norm_type

        # Input Normalization
        if norm_type == "layernorm":
            self.input_norm = nn.LayerNorm(embed_dim)
        else:
            self.input_norm = nn.Identity()

        # Optional linear projection to reduce embedding dimensions
        if proj_dim is not None and proj_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            working_dim = proj_dim
        else:
            self.projection = nn.Identity()
            working_dim = embed_dim

        # Temporal Aggregation
        if pooling_mode == "hybrid":
            # Original: attention + mean + max -> 3 * working_dim
            self.attn = nn.Sequential(
                nn.Linear(working_dim, min(hidden_dim, 128)),
                nn.Tanh(),
                nn.Linear(min(hidden_dim, 128), 1),
                nn.Softmax(dim=1),
            )
            head_in_dim = working_dim * 3

        elif pooling_mode == "mean_max":
            head_in_dim = working_dim * 2

        elif pooling_mode == "mean":
            head_in_dim = working_dim

        elif pooling_mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(working_dim, min(hidden_dim, 64)),
                nn.Tanh(),
                nn.Linear(min(hidden_dim, 64), 1),
                nn.Softmax(dim=1),
            )
            head_in_dim = working_dim

        elif pooling_mode == "projected_dynamics":
            # mean + max + temporal difference dynamics
            # 1. mean (working_dim)
            # 2. max (working_dim)
            # 3. mean absolute temporal diff (working_dim)
            head_in_dim = working_dim * 3

        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(head_in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.norm_type == "l2":
            x = nn.functional.normalize(x, p=2, dim=-1)
        else:
            x = self.input_norm(x)

        x = self.projection(x)  # (B, T, working_dim)

        if self.pooling_mode == "hybrid":
            attn_weights = self.attn(x)
            attn_rep = torch.sum(x * attn_weights, dim=1)
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            feat = torch.cat([attn_rep, mean_rep, max_rep], dim=1)

        elif self.pooling_mode == "mean_max":
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            feat = torch.cat([mean_rep, max_rep], dim=1)

        elif self.pooling_mode == "mean":
            feat = torch.mean(x, dim=1)

        elif self.pooling_mode == "attention":
            attn_weights = self.attn(x)
            feat = torch.sum(x * attn_weights, dim=1)

        elif self.pooling_mode == "projected_dynamics":
            mean_rep = torch.mean(x, dim=1)
            max_rep, _ = torch.max(x, dim=1)
            # Temporal difference between consecutive frames
            if x.size(1) > 1:
                diff = torch.abs(x[:, 1:, :] - x[:, :-1, :])
                diff_rep = torch.mean(diff, dim=1)
            else:
                diff_rep = torch.zeros_like(mean_rep)
            feat = torch.cat([mean_rep, max_rep, diff_rep], dim=1)

        logits = self.classifier(feat)
        return logits


def train_single_experiment(
    exp_id: str,
    config: Dict[str, Any],
    cache_path: Path = CACHE_PATH,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run one controlled experiment and return evaluation metrics."""
    if device is None:
        device = torch.device("cpu")

    seed = config.get("seed", 42)
    set_seed(seed)

    # 1. Load cached features
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_feat = data["train"]["features"]
    train_labels = data["train"]["labels"]
    val_feat = data["val"]["features"]
    val_labels = data["val"]["labels"]

    # Class weights
    counts = torch.bincount(train_labels)
    total = float(len(train_labels))
    use_class_weights = config.get("use_class_weights", True)
    if use_class_weights:
        class_weights = (total / (2.0 * counts.float())).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], device=device)

    train_ds = TensorDataset(train_feat, train_labels)
    val_ds = TensorDataset(val_feat, val_labels)

    batch_size = config.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Build model
    model = ModularTemporalVideoClassifier(
        embed_dim=config.get("embed_dim", 1280),
        proj_dim=config.get("proj_dim", 128),
        hidden_dim=config.get("hidden_dim", 128),
        mlp_dim=config.get("mlp_dim", 64),
        num_classes=2,
        dropout=config.get("dropout", 0.4),
        pooling_mode=config.get("pooling_mode", "projected_dynamics"),
        norm_type=config.get("norm_type", "layernorm"),
    ).to(device)

    # Loss with optional label smoothing
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config.get("epochs", 40)
    scheduler_type = config.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None

    # Tracking & Early stopping
    patience = config.get("patience", 10)
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0

    checkpoint_save_path = CHECKPOINTS_DIR / f"{exp_id}.pt"
    start_time = time.time()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_trues = [], []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_trues.extend(by.cpu().numpy())

        if scheduler:
            scheduler.step()

        epoch_tr_loss = running_loss / len(train_labels)
        epoch_tr_acc = accuracy_score(train_trues, train_preds)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds, val_trues, val_probs = [], [], []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_running_loss += loss.item() * len(by)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(by.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_labels)
        epoch_val_acc = accuracy_score(val_trues, val_preds)
        val_bal_acc = balanced_accuracy_score(val_trues, val_preds)
        val_prec, val_rec, val_macro_f1, _ = precision_recall_fscore_support(
            val_trues, val_preds, average="macro", zero_division=0
        )
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(
            val_trues, val_preds, average=None, zero_division=0
        )
        try:
            val_auc = roc_auc_score(val_trues, val_probs)
        except Exception:
            val_auc = 0.5
        cm = confusion_matrix(val_trues, val_preds)

        history["train_loss"].append(round(epoch_tr_loss, 4))
        history["train_acc"].append(round(epoch_tr_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))

        # Selection criterion: Validation Macro F1, tie-breaker: validation loss
        is_best = False
        if val_macro_f1 > best_val_macro_f1 or (np.isclose(val_macro_f1, best_val_macro_f1, atol=1e-4) and epoch_val_loss < best_val_loss):
            best_val_macro_f1 = val_macro_f1
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            is_best = True
            patience_counter = 0

            best_val_metrics = {
                "val_accuracy": round(float(epoch_val_acc), 4),
                "val_macro_f1": round(float(val_macro_f1), 4),
                "val_balanced_accuracy": round(float(val_bal_acc), 4),
                "val_roc_auc": round(float(val_auc), 4),
                "val_loss": round(float(epoch_val_loss), 4),
                "val_precision_real": round(float(p_class[0]), 4),
                "val_recall_real": round(float(r_class[0]), 4),
                "val_f1_real": round(float(f1_class[0]), 4),
                "val_precision_fake": round(float(p_class[1]), 4),
                "val_recall_fake": round(float(r_class[1]), 4),
                "val_f1_fake": round(float(f1_class[1]), 4),
                "confusion_matrix": {
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1]),
                },
            }

            # Save checkpoint
            torch.save({
                "experiment_id": exp_id,
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": best_val_metrics,
                "config": config,
                "class_map": {"real": 0, "fake": 1},
            }, checkpoint_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    duration_sec = round(time.time() - start_time, 2)

    result_record = {
        "experiment_id": exp_id,
        "config": config,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "duration_sec": duration_sec,
        "checkpoint_path": str(checkpoint_save_path),
        "history": history,
        **best_val_metrics,
    }

    return result_record


def update_results_table(all_results: List[Dict[str, Any]]):
    """Save summary CSV and JSON table of all experiments."""
    with open(EXPERIMENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = [
        "experiment_id",
        "description",
        "pooling_mode",
        "proj_dim",
        "norm_type",
        "lr",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "duration_sec",
        "best_epoch",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_roc_auc",
        "val_loss",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    with open(EXPERIMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            cfg = r.get("config", {})
            cm = r.get("confusion_matrix", {})
            row = {
                "experiment_id": r.get("experiment_id"),
                "description": cfg.get("description", ""),
                "pooling_mode": cfg.get("pooling_mode", ""),
                "proj_dim": cfg.get("proj_dim", ""),
                "norm_type": cfg.get("norm_type", ""),
                "lr": cfg.get("lr", ""),
                "weight_decay": cfg.get("weight_decay", ""),
                "dropout": cfg.get("dropout", ""),
                "label_smoothing": cfg.get("label_smoothing", 0.0),
                "duration_sec": r.get("duration_sec", 0.0),
                "best_epoch": r.get("best_epoch", 0),
                "val_accuracy": r.get("val_accuracy", 0.0),
                "val_macro_f1": r.get("val_macro_f1", 0.0),
                "val_balanced_accuracy": r.get("val_balanced_accuracy", 0.0),
                "val_roc_auc": r.get("val_roc_auc", 0.0),
                "val_loss": r.get("val_loss", 0.0),
                "TN": cm.get("TN", 0),
                "FP": cm.get("FP", 0),
                "FN": cm.get("FN", 0),
                "TP": cm.get("TP", 0),
            }
            writer.writerow(row)
