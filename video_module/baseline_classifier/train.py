import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# Reconfigure stdout for utf-8 and unbuffered line-level output
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Import local modules
from model import TemporalVideoClassifier, get_feature_extractor
from dataset import sample_video_frames

PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
AUDIT_DIR = PROJECT_ROOT / "video_module" / "dataset_audit"
OUTPUT_DIR = PROJECT_ROOT / "video_module" / "baseline_classifier"
CACHE_PATH = OUTPUT_DIR / "features_cache.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"real": 0, "fake": 1}
INV_CLASS_MAP = {0: "real", 1: "fake"}

CONFIG = {
    "num_frames": 12,
    "embed_dim": 1280,
    "hidden_dim": 256,
    "mlp_dim": 256,
    "dropout": 0.35,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 35,
    "seed": 42,
}

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def check_environment():
    print("=== ENVIRONMENT & HARDWARE INSPECTION ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available to PyTorch: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"GPU Device: {device_name} ({vram_gb} GB VRAM)")
        device = torch.device("cuda:0")
    else:
        print("Note: PyTorch is currently running on CPU (build has +cpu). Training will utilize optimized CPU multithreading.")
        device = torch.device("cpu")
    print(f"Selected Compute Device: {device}\n")
    return device

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def extract_or_load_features(device: torch.device) -> Dict[str, Any]:
    if CACHE_PATH.exists():
        print(f"[Cache] Found cached features at {CACHE_PATH}. Loading directly...")
        cached_data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        print(f"        Train: {cached_data['train']['features'].shape}")
        print(f"        Val:   {cached_data['val']['features'].shape}")
        print(f"        Test:  {cached_data['test']['features'].shape}")
        return cached_data

    print("[Features] Cache not found. Beginning EfficientNet-B0 frame feature extraction...")
    extractor = get_feature_extractor(device)

    train_rows = load_manifest(AUDIT_DIR / "train_manifest.csv")
    val_rows = load_manifest(AUDIT_DIR / "val_manifest.csv")
    test_rows = load_manifest(AUDIT_DIR / "test_manifest.csv")

    def process_split(rows: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        features_list = []
        labels_list = []
        cids_list = []

        start_time = time.time()
        print(f"\n[Extract] Processing {split_name} split ({len(rows)} videos)...")
        for i, row in enumerate(rows, 1):
            cid = row["cascade_id"]
            lbl_str = row["label"]
            lbl_int = CLASS_MAP[lbl_str]
            vid_path = Path(row["file_path"])

            # Sample frames (num_frames, 3, 224, 224)
            frames_tensor = sample_video_frames(vid_path, num_frames=CONFIG["num_frames"])
            frames_tensor = frames_tensor.to(device)

            with torch.no_grad():
                # Extractor outputs (num_frames, 1280)
                emb = extractor(frames_tensor).cpu()  # (T, 1280)

            features_list.append(emb)
            labels_list.append(lbl_int)
            cids_list.append(cid)

            if i % 10 == 0 or i == len(rows):
                elapsed = time.time() - start_time
                print(f"   [{i}/{len(rows)}] {cid} ({lbl_str}) - {elapsed:.1f}s elapsed", flush=True)

        # Stack into (N, T, 1280)
        all_features = torch.stack(features_list, dim=0)
        all_labels = torch.tensor(labels_list, dtype=torch.long)
        return {
            "features": all_features,
            "labels": all_labels,
            "cids": cids_list,
        }

    train_data = process_split(train_rows, "train")
    val_data = process_split(val_rows, "val")
    test_data = process_split(test_rows, "test")

    cached_payload = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "config": CONFIG,
        "class_map": CLASS_MAP,
    }
    torch.save(cached_payload, CACHE_PATH)
    print(f"\n[Cache] Successfully cached all extracted features to {CACHE_PATH}")
    return cached_payload

def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Real (0)", "Fake (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold"
            )

    ax.set_ylabel("True Label", fontsize=12, labelpad=8)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_title("Test Set Confusion Matrix\n(TruthLensAI Baseline Video Classifier)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(CONFIG["seed"])
    device = check_environment()

    # 1. Prepare Features & DataLoaders
    data = extract_or_load_features(device)

    train_feat = data["train"]["features"]
    train_labels = data["train"]["labels"]
    val_feat = data["val"]["features"]
    val_labels = data["val"]["labels"]
    test_feat = data["test"]["features"]
    test_labels = data["test"]["labels"]

    # Compute balanced class weights for CrossEntropyLoss
    # weight = N / (n_classes * class_count)
    counts = torch.bincount(train_labels)
    total_samples = float(len(train_labels))
    class_weights = total_samples / (2.0 * counts.float())
    class_weights = class_weights.to(device)
    print(f"[Class Imbalance] Train counts: real={counts[0].item()}, fake={counts[1].item()}")
    print(f"                 Class weights: real={class_weights[0]:.3f}, fake={class_weights[1]:.3f}")

    train_ds = TensorDataset(train_feat, train_labels)
    val_ds = TensorDataset(val_feat, val_labels)
    test_ds = TensorDataset(test_feat, test_labels)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # 2. Build Model
    model = TemporalVideoClassifier(
        num_classes=2,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        mlp_dim=CONFIG["mlp_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)

    # 3. Training Loop with Validation Selection
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_model_path = OUTPUT_DIR / "best_model.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    print("\n=== STARTING BASELINE TRAINING ===")
    print(f"Total Epochs: {CONFIG['epochs']} | Batch Size: {CONFIG['batch_size']}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_trues = [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_trues.extend(batch_y.cpu().numpy())

        scheduler.step()

        epoch_train_loss = running_loss / len(train_labels)
        epoch_train_acc = accuracy_score(train_trues, train_preds)

        # Validation Step
        model.eval()
        val_running_loss = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_running_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(batch_y.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_labels)
        epoch_val_acc = accuracy_score(val_trues, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_trues, val_preds, average="macro", zero_division=0
        )

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        # Model Checkpointing by Validation Macro F1 and Loss
        is_best = False
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and epoch_val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = epoch_val_loss
            is_best = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_loss": best_val_loss,
                "config": CONFIG,
                "class_map": CLASS_MAP,
            }, best_model_path)

        if epoch % 5 == 0 or is_best or epoch == CONFIG["epochs"]:
            mark = " [BEST CHECKPOINT SAVED]" if is_best else ""
            print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.3f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.3f} Macro-F1: {val_f1:.3f}{mark}")

    # 4. Final Evaluation on UNSEEN Test Split
    print("\n=== FINAL TEST EVALUATION (HELD-OUT UNSEEN SPLIT) ===")
    print(f"Loading best checkpoint from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_trues, test_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class (1)
            test_preds.extend(preds.cpu().numpy())
            test_trues.extend(batch_y.cpu().numpy())

    # Calculate Test Metrics
    acc = accuracy_score(test_trues, test_preds)
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        test_trues, test_preds, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        test_trues, test_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        test_trues, test_preds, average="weighted", zero_division=0
    )
    roc_auc = roc_auc_score(test_trues, test_probs)
    cm = confusion_matrix(test_trues, test_preds)

    print("\n" + classification_report(test_trues, test_preds, target_names=["Real", "Fake"], digits=4))
    print(f"Test Accuracy:  {acc * 100:.2f}%")
    print(f"Test Macro F1:  {f1_macro:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")

    # Save Confusion Matrix Plot
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)
    print(f"Saved confusion matrix plot: {cm_path}")

    # Save Metrics JSON
    test_metrics = {
        "model_architecture": "EfficientNet-B0 + Temporal Attention Pooling + MLP",
        "checkpoint_epoch": checkpoint["epoch"],
        "test_total_samples": len(test_trues),
        "test_accuracy": round(float(acc), 4),
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
            }
        },
        "confusion_matrix": {
            "true_real_pred_real (TN)": int(cm[0, 0]),
            "true_real_pred_fake (FP)": int(cm[0, 1]),
            "true_fake_pred_real (FN)": int(cm[1, 0]),
            "true_fake_pred_fake (TP)": int(cm[1, 1]),
        }
    }

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)

    print(f"\nAll artifacts successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
