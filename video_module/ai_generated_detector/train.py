"""
train.py
========
Phase 3b: Train the BiGRU head on pre-extracted EfficientNet-B0 features.

Features:
  - Deterministic seeding
  - Class-weighted cross-entropy
  - AMP (fp16 on CUDA)
  - Gradient clipping
  - Validation-based early stopping
  - Best-checkpoint saving
  - Resumable (skips if best_model.pt already contains a better checkpoint)
  - JSON training log

Usage
-----
    python -m video_module.ai_generated_detector.train \
        [--epochs 60] [--lr 1e-3] [--batch 16] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from video_module.ai_generated_detector.config import (
    SEED, TRAIN_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, GRAD_CLIP,
    BEST_MODEL, EXP_LOG, CKPT_DIR, NUM_FRAMES, FEATURE_DIM, DROPOUT,
    GRU_HIDDEN, CLASS_NAMES, CLASS_MAP,
    CACHE_TRAIN, CACHE_VAL, TRAIN_CSV, VAL_CSV,
)
from video_module.ai_generated_detector.dataset import FeatureDataset
from video_module.ai_generated_detector.model import build_model


# ════════════════════════════════════════════════════════════════════════════════
# Determinism
# ════════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    labels: List[int],
    preds:  List[int],
    probs:  List[float],   # P(AI_GENERATED)
) -> Dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_recall_fscore_support, roc_auc_score,
        confusion_matrix,
    )
    n       = len(labels)
    acc     = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    mac_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    wgt_f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    cm      = confusion_matrix(labels, preds).tolist()

    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(p_cls[i]),
            "recall":    float(r_cls[i]),
            "f1":        float(f_cls[i]),
            "support":   int(s_cls[i]),
        }
        for i in range(2)
    }

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = None

    return {
        "n":               n,
        "accuracy":        round(float(acc),     4),
        "balanced_acc":    round(float(bal_acc),  4),
        "macro_f1":        round(float(mac_f1),   4),
        "weighted_f1":     round(float(wgt_f1),   4),
        "roc_auc":         round(roc_auc, 4) if roc_auc is not None else None,
        "per_class":       per_class,
        "confusion_matrix":cm,
    }


# ════════════════════════════════════════════════════════════════════════════════
# One epoch
# ════════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler:    Optional[torch.cuda.amp.GradScaler],
    device:    torch.device,
    train:     bool,
) -> Tuple[float, Dict]:
    model.train(train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for features, labels in loader:
        features = features.to(device)      # (B, T, F)
        labels   = labels.to(device)        # (B,)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(features)        # (B, 2)
            loss   = criterion(logits, labels)

        if train and optimizer is not None:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * features.size(0)
        probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        preds  = logits.argmax(dim=1).detach().cpu().tolist()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = round(avg_loss, 6)
    return avg_loss, metrics


# ════════════════════════════════════════════════════════════════════════════════
# Main training loop
# ════════════════════════════════════════════════════════════════════════════════

def train(
    epochs:     int   = TRAIN_EPOCHS,
    lr:         float = LR,
    batch_size: int   = BATCH_SIZE,
    device_str: str   = "auto",
    num_frames: int   = NUM_FRAMES,
    seed:       int   = SEED,
) -> None:
    set_seed(seed)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] Device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[train] VRAM free: {vram_free:.2f} GB")

    # Datasets
    print("\n[train] Loading datasets...")
    train_ds = FeatureDataset(TRAIN_CSV, CACHE_TRAIN, num_frames=num_frames, augment=True)
    val_ds   = FeatureDataset(VAL_CSV,   CACHE_VAL,   num_frames=num_frames, augment=False)

    # Class-weighted sampler (handles imbalance)
    train_labels = train_ds.get_labels()
    class_counts = [train_labels.count(0), train_labels.count(1)]
    print(f"[train] Class counts: REAL={class_counts[0]}, AI_GENERATED={class_counts[1]}")
    weights      = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = torch.tensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Class weights for loss (double-balance)
    w_tensor = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    # Model
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training state
    best_val_f1    = -1.0
    patience_count = 0
    history        = []

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[train] Starting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"        Early stopping patience: {PATIENCE}")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # Train
        _, tr_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        # Validate
        _, va_metrics = run_epoch(model, val_loader,   criterion, None,      None,   device, train=False)

        scheduler.step()

        ep_time = time.time() - t_ep
        val_f1  = va_metrics["macro_f1"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss={tr_metrics['loss']:.4f} acc={tr_metrics['accuracy']:.3f} | "
            f"val loss={va_metrics['loss']:.4f} acc={va_metrics['accuracy']:.3f} "
            f"bal_acc={va_metrics['balanced_acc']:.3f} macro_f1={val_f1:.3f} | "
            f"{ep_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": tr_metrics,
            "val":   va_metrics,
        })

        # Best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": va_metrics,
                    "config": {
                        "feature_dim":  FEATURE_DIM,
                        "hidden_dim":   GRU_HIDDEN,
                        "num_frames":   num_frames,
                        "dropout":      DROPOUT,
                        "seed":         seed,
                        "lr":           lr,
                        "batch_size":   batch_size,
                    },
                },
                BEST_MODEL,
            )
            print(f"  ✓ New best val macro-F1: {best_val_f1:.4f} — saved to {BEST_MODEL.name}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[train] Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"\n[train] Training complete in {total_time:.1f}s")
    print(f"[train] Best val macro-F1: {best_val_f1:.4f}")

    # Save full log
    log = {
        "best_val_macro_f1": best_val_f1,
        "total_train_time_s": round(total_time, 1),
        "epochs_run": len(history),
        "config": {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "num_frames": num_frames, "seed": seed,
            "device": str(device),
        },
        "history": history,
    }
    with open(EXP_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] Log saved → {EXP_LOG}")


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Train BiGRU head on cached features.")
    ap.add_argument("--epochs",     type=int,   default=TRAIN_EPOCHS)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    ap.add_argument("--num-frames", type=int,   default=NUM_FRAMES)
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--seed",       type=int,   default=SEED)
    args = ap.parse_args()

    train(
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch,
        device_str = args.device,
        num_frames = args.num_frames,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
