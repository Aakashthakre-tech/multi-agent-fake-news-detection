"""
run_videomae_head_experiments.py
=================================
Runs 5 controlled VideoMAE classifier-head experiments.
- Reads train+val from videomae_features_cache.pt
- Trains lightweight heads on frozen 768-d VideoMAE embeddings
- Selects best candidate on validation Macro-F1 (+ balanced accuracy)
- Saves per-experiment checkpoints, JSON log, and CSV results
- Skips experiments that already have a valid checkpoint+log entry (crash-safe)

Experiments:
  vmae_exp0_linear          : LayerNorm + Dropout + Linear(768->2)
  vmae_exp1_mlp64           : LayerNorm + Linear(768->64) + LN + ReLU + Dropout + Linear(64->2)
  vmae_exp2_mlp128          : LayerNorm + Linear(768->128) + LN + ReLU + Dropout + Linear(128->2)
  vmae_exp3_mlp64_weighted  : vmae_exp1 + inverse class-frequency weights
  vmae_exp4_mlp64_smooth    : vmae_exp1 + label smoothing(0.1) + cosine-annealing LR
"""

import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
CHECKPOINTS_DIR = _CLASSIFIER_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_LOG_JSON = _CLASSIFIER_DIR / "videomae_experiments_log.json"
EXP_RESULTS_CSV = _CLASSIFIER_DIR / "videomae_experiments_results.csv"


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class VideoMAEHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=64, num_classes=2,
                 dropout=0.4, norm_type="layernorm", architecture="mlp"):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim) if norm_type == "layernorm" else nn.Identity()
        if architecture == "linear":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))
        elif architecture == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
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
                nn.Linear(hidden_dim // 2, num_classes),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        return self.head(self.input_norm(x))


def evaluate(model, feat, labels, criterion):
    model.eval()
    ds = TensorDataset(feat, labels)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    logits_all = []
    with torch.inference_mode():
        for xb, _ in loader:
            logits_all.append(model(xb))
    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = torch.argmax(logits, dim=1).numpy()
    y = labels.numpy()
    loss = criterion(logits, labels).item()
    acc = float(accuracy_score(y, preds))
    bal_acc = float(balanced_accuracy_score(y, preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(y, preds, average=None, zero_division=0)
    try:
        auc = float(roc_auc_score(y, probs[:, 1]))
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y, preds)
    return {
        "val_loss": round(loss, 4),
        "val_accuracy": round(acc, 4),
        "val_balanced_accuracy": round(bal_acc, 4),
        "val_macro_f1": round(float(f1_macro), 4),
        "val_macro_precision": round(float(p_macro), 4),
        "val_macro_recall": round(float(r_macro), 4),
        "val_roc_auc": round(auc, 4),
        "real_precision": round(float(p_c[0]), 4),
        "real_recall": round(float(r_c[0]), 4),
        "real_f1": round(float(f1_c[0]), 4),
        "fake_precision": round(float(p_c[1]), 4),
        "fake_recall": round(float(r_c[1]), 4),
        "fake_f1": round(float(f1_c[1]), 4),
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }


def train_experiment(exp_id, cfg, train_feat, train_labels, val_feat, val_labels):
    set_seed(cfg.get("seed", 42))
    ckpt_path = CHECKPOINTS_DIR / f"{exp_id}.pt"

    train_ds = TensorDataset(train_feat, train_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16), shuffle=True)

    model = VideoMAEHead(
        in_dim=768,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=2,
        dropout=cfg.get("dropout", 0.4),
        norm_type=cfg.get("norm_type", "layernorm"),
        architecture=cfg.get("architecture", "mlp"),
    )

    # Loss
    if cfg.get("use_class_weights", False):
        n_real = (train_labels == 0).sum().item()
        n_fake = (train_labels == 1).sum().item()
        total = n_real + n_fake
        w = torch.tensor([total / (2.0 * n_real), total / (2.0 * n_fake)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-2)
    )

    scheduler = None
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("epochs", 40), eta_min=1e-5
        )

    epochs = cfg.get("epochs", 40)
    patience = cfg.get("patience", 15)
    best_f1 = -1.0
    best_metrics = {}
    best_epoch = 0
    patience_cnt = 0
    history = {"train_loss": [], "train_acc": [], "val_macro_f1": [], "val_accuracy": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        if scheduler:
            scheduler.step()

        t_loss = total_loss / total
        t_acc = correct / total
        vm = evaluate(model, val_feat, val_labels, criterion)
        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_macro_f1"].append(vm["val_macro_f1"])
        history["val_accuracy"].append(vm["val_accuracy"])

        if vm["val_macro_f1"] > best_f1:
            best_f1 = vm["val_macro_f1"]
            best_epoch = epoch
            best_metrics = vm
            patience_cnt = 0
            torch.save({
                "exp_id": exp_id, "epoch": epoch, "config": cfg,
                "model_state_dict": model.state_dict(),
                "val_metrics": best_metrics,
            }, ckpt_path)
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    duration = round(time.time() - t0, 2)
    result = {
        "exp_id": exp_id,
        "description": cfg.get("description", ""),
        "architecture": cfg.get("architecture", ""),
        "hidden_dim": cfg.get("hidden_dim", 0),
        "dropout": cfg.get("dropout", 0.0),
        "lr": cfg.get("lr", 0.0),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "use_class_weights": cfg.get("use_class_weights", False),
        "scheduler": cfg.get("scheduler", "none"),
        "epochs_trained": best_epoch + patience,
        "best_epoch": best_epoch,
        "duration_sec": duration,
        "checkpoint": str(ckpt_path),
        "history": history,
        **best_metrics,
    }
    return result


EXPERIMENT_CONFIGS = [
    {
        "id": "vmae_exp0_linear",
        "description": "Linear probing: LayerNorm + Dropout(0.3) + Linear(768->2)",
        "architecture": "linear",
        "hidden_dim": 0,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 15,
        "seed": 42,
    },
    {
        "id": "vmae_exp1_mlp64",
        "description": "MLP-64: LayerNorm + Linear(768->64) + LN + ReLU + Dropout(0.4) + Linear(64->2)",
        "architecture": "mlp",
        "hidden_dim": 64,
        "dropout": 0.4,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 15,
        "seed": 42,
    },
    {
        "id": "vmae_exp2_mlp128",
        "description": "MLP-128: LayerNorm + Linear(768->128) + LN + ReLU + Dropout(0.5) + Linear(128->2)",
        "architecture": "mlp",
        "hidden_dim": 128,
        "dropout": 0.5,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 40,
        "patience": 15,
        "seed": 42,
    },
    {
        "id": "vmae_exp3_mlp64_weighted",
        "description": "MLP-64 + inverse class-frequency weighting",
        "architecture": "mlp",
        "hidden_dim": 64,
        "dropout": 0.4,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "use_class_weights": True,
        "epochs": 40,
        "patience": 15,
        "seed": 42,
    },
    {
        "id": "vmae_exp4_mlp64_smooth",
        "description": "MLP-64 + label smoothing(0.1) + cosine-annealing LR",
        "architecture": "mlp",
        "hidden_dim": 64,
        "dropout": 0.4,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "label_smoothing": 0.1,
        "scheduler": "cosine",
        "epochs": 40,
        "patience": 15,
        "seed": 42,
    },
]


def main():
    print("=" * 70)
    print("VIDEOMAE HEAD EXPERIMENTS")
    print("=" * 70)

    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}. Run feature extraction first.")
        sys.exit(1)

    print(f"Loading features from: {CACHE_PATH}")
    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

    if "train" not in data or "val" not in data:
        print(f"ERROR: Cache missing 'train' or 'val'. Keys: {list(data.keys())}")
        sys.exit(1)

    train_feat = data["train"]["features"]
    train_labels = data["train"]["labels"]
    val_feat = data["val"]["features"]
    val_labels = data["val"]["labels"]

    print(f"Train: {train_feat.shape}, Val: {val_feat.shape}")
    print(f"Train labels: real={(train_labels==0).sum().item()}, fake={(train_labels==1).sum().item()}")
    print(f"Val labels:   real={(val_labels==0).sum().item()},  fake={(val_labels==1).sum().item()}")

    # Load existing log (for crash-resumption)
    existing_results = []
    completed_ids = set()
    if EXP_LOG_JSON.exists():
        with open(EXP_LOG_JSON, encoding="utf-8") as f:
            existing_results = json.load(f)
        completed_ids = {r["exp_id"] for r in existing_results}
        print(f"\nFound {len(completed_ids)} already-completed experiments: {completed_ids}")

    all_results = list(existing_results)

    for cfg in EXPERIMENT_CONFIGS:
        exp_id = cfg["id"]
        if exp_id in completed_ids:
            print(f"\n[SKIP] {exp_id} — already completed.")
            continue

        print(f"\n{'='*60}")
        print(f"Running [{exp_id}]: {cfg['description']}")
        print(f"{'='*60}")
        result = train_experiment(exp_id, cfg, train_feat, train_labels, val_feat, val_labels)
        all_results.append(result)

        print(f"  Val Acc={result['val_accuracy']:.4f} | Macro-F1={result['val_macro_f1']:.4f} | "
              f"Bal-Acc={result['val_balanced_accuracy']:.4f} | AUC={result['val_roc_auc']:.4f}")
        print(f"  Real: prec={result['real_precision']:.3f} rec={result['real_recall']:.3f} f1={result['real_f1']:.3f}")
        print(f"  Fake: prec={result['fake_precision']:.3f} rec={result['fake_recall']:.3f} f1={result['fake_f1']:.3f}")
        print(f"  CM: TN={result['TN']} FP={result['FP']} FN={result['FN']} TP={result['TP']}")
        print(f"  Best epoch={result['best_epoch']}, Duration={result['duration_sec']}s")

        # Save incrementally after each experiment
        with open(EXP_LOG_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        fieldnames = [k for k in all_results[0].keys() if k != "history"]
        with open(EXP_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: v for k, v in r.items() if k != "history"})

        print(f"  Saved to {EXP_LOG_JSON}")

    # --- Select best candidate ---
    print("\n" + "=" * 70)
    print("VALIDATION-BASED CANDIDATE SELECTION")
    print("=" * 70)
    valid_results = [r for r in all_results if r.get("val_macro_f1", 0) > 0]
    best = max(valid_results, key=lambda x: (x["val_macro_f1"], x["val_balanced_accuracy"]))

    print(f"\nBest VideoMAE candidate: {best['exp_id']}")
    print(f"  Val Macro-F1:        {best['val_macro_f1']:.4f}")
    print(f"  Val Accuracy:        {best['val_accuracy']:.4f}")
    print(f"  Val Balanced Acc:    {best['val_balanced_accuracy']:.4f}")
    print(f"  Val ROC-AUC:         {best['val_roc_auc']:.4f}")
    print(f"  Real recall:         {best['real_recall']:.4f}")
    print(f"  Fake recall:         {best['fake_recall']:.4f}")
    print(f"  Checkpoint:          {best['checkpoint']}")

    # --- Compare against exp6_compact_proj64 ---
    # exp6 baseline: val_macro_f1=0.7568, val_balanced_accuracy=0.7533 (from experiments_log.json)
    EXP6_VAL_F1 = 0.7568
    EXP6_VAL_BAL_ACC = 0.7533
    EXP6_VAL_ACC = 0.7674

    print(f"\nComparison vs exp6_compact_proj64 (EfficientNet baseline):")
    print(f"  {'Metric':<25} {'exp6 (EfficientNet)':<22} {'VideoMAE best':<22}")
    print(f"  {'-'*70}")
    print(f"  {'Val Macro-F1':<25} {EXP6_VAL_F1:<22.4f} {best['val_macro_f1']:<22.4f}")
    print(f"  {'Val Balanced Acc':<25} {EXP6_VAL_BAL_ACC:<22.4f} {best['val_balanced_accuracy']:<22.4f}")
    print(f"  {'Val Accuracy':<25} {EXP6_VAL_ACC:<22.4f} {best['val_accuracy']:<22.4f}")

    videomae_wins = best["val_macro_f1"] > EXP6_VAL_F1

    if videomae_wins:
        print(f"\n[OK] VideoMAE ({best['exp_id']}) outperforms exp6_compact_proj64 on val Macro-F1.")
        print(f"  Recommendation: Proceed to test evaluation with {best['exp_id']}.")
    else:
        print(f"\n[INFO] VideoMAE best val Macro-F1 ({best['val_macro_f1']:.4f}) does NOT exceed exp6 ({EXP6_VAL_F1:.4f}).")
        print(f"  Retaining exp6_compact_proj64 as the leading candidate.")

    # Save selection summary
    selection = {
        "best_videomae_candidate": best["exp_id"],
        "best_videomae_checkpoint": best["checkpoint"],
        "videomae_val_macro_f1": best["val_macro_f1"],
        "videomae_val_balanced_accuracy": best["val_balanced_accuracy"],
        "videomae_val_accuracy": best["val_accuracy"],
        "videomae_val_roc_auc": best["val_roc_auc"],
        "efficientnet_baseline_id": "exp6_compact_proj64",
        "efficientnet_val_macro_f1": EXP6_VAL_F1,
        "efficientnet_val_balanced_accuracy": EXP6_VAL_BAL_ACC,
        "videomae_outperforms": bool(videomae_wins),
        "recommended_for_test": best["exp_id"] if videomae_wins else "exp6_compact_proj64",
    }

    sel_path = _CLASSIFIER_DIR / "videomae_candidate_selection.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSelection saved to: {sel_path}")
    print(f"Experiments log: {EXP_LOG_JSON}")
    print(f"Results CSV: {EXP_RESULTS_CSV}")


if __name__ == "__main__":
    main()
