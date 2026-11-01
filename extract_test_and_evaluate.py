"""
extract_test_and_evaluate.py
============================
1. Extracts VideoMAE features for the 42 held-out test videos from test_manifest.csv
   using the exact same 16-frame OpenCV uniform sampling and mean pooling.
2. Updates videomae_features_cache.pt with key 'test'.
3. Evaluates the selected best candidate (vmae_exp0_linear) on the held-out test set.
4. Generates comprehensive test metrics, confusion matrix, and detailed sample-level error analysis.
5. Saves results to videomae_final_candidate_test_metrics.json and videomae_test_error_analysis.json.
"""

import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoImageProcessor, VideoMAEModel

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
_AUDIT_DIR = _PROJECT_ROOT / "video_module" / "dataset_audit"

VIDEOMAE_CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
TEST_MANIFEST_PATH = _AUDIT_DIR / "test_manifest.csv"
MODEL_ID = "MCG-NJU/videomae-base"
CLASS_MAP = {"real": 0, "fake": 1}
INV_CLASS_MAP = {0: "real", 1: "fake"}


class VideoMAEHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=64, num_classes=2,
                 dropout=0.3, norm_type="layernorm", architecture="linear"):
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
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        return self.head(self.input_norm(x))


def get_ram_gb() -> float:
    return round(psutil.virtual_memory().available / (1024 ** 3), 2)


def sample_frames_cv2(video_path: Path, num_frames: int = 16) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if not frames:
                raise ValueError(f"No frames in {video_path}")
            while len(frames) < num_frames:
                frames.append(frames[-1])
            return np.stack(frames[:num_frames])

        idxs = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
        sampled, last = [], None
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled.append(last)
            elif last is not None:
                sampled.append(last)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, f2 = cap.read()
                if ret2 and f2 is not None:
                    last = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    sampled.append(last)
                else:
                    raise ValueError(f"Cannot decode any frame from {video_path}")
        return np.stack(sampled)
    finally:
        cap.release()


def extract_test_features():
    cache = torch.load(VIDEOMAE_CACHE_PATH, map_location="cpu", weights_only=False)
    if "test" in cache:
        feat = cache["test"]["features"]
        print(f"Test features already in cache: shape={feat.shape}")
        if feat.shape == (42, 768) and not torch.isnan(feat).any():
            print("Cached test features are complete and valid.")
            return cache["test"]

    print(f"Extracting test features from {TEST_MANIFEST_PATH}...")
    test_rows = list(csv.DictReader(open(TEST_MANIFEST_PATH, encoding="utf-8")))
    print(f"Total test rows: {len(test_rows)}")

    # Load local VideoMAE model
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / "hub" / "models--MCG-NJU--videomae-base"
    st_path = list(cache_dir.rglob("model.safetensors"))[0]

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = VideoMAEModel.from_pretrained(MODEL_ID, local_files_only=True)

    raw_sd = load_file(str(st_path))
    sd = model.state_dict()
    for i in range(model.config.num_hidden_layers):
        q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
        v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
        if q_key in raw_sd:
            sd[f"encoder.layer.{i}.attention.attention.query.bias"] = raw_sd[q_key]
        if v_key in raw_sd:
            sd[f"encoder.layer.{i}.attention.attention.value.bias"] = raw_sd[v_key]
        sd[f"encoder.layer.{i}.attention.attention.key.bias"] = torch.zeros(model.config.hidden_size)

    model.load_state_dict(sd, strict=False)
    model.eval()
    del raw_sd, sd
    gc.collect()

    features, labels, cids, metadata_list = [], [], [], []
    t0 = time.time()
    for idx, row in enumerate(test_rows):
        cid = row["cascade_id"]
        lbl_str = row["label"]
        lbl_int = CLASS_MAP[lbl_str]
        vid_path = Path(row["file_path"])

        frames = sample_frames_cv2(vid_path, num_frames=16)
        inputs = processor(list(frames), return_tensors="pt")
        with torch.inference_mode():
            out = model(inputs["pixel_values"])
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()

        assert not torch.isnan(emb).any(), f"NaN in {cid}"
        features.append(emb)
        labels.append(lbl_int)
        cids.append(cid)
        metadata_list.append({
            "cascade_id": cid,
            "label": lbl_str,
            "file_path": str(vid_path),
            "source": row.get("source", ""),
            "manipulation_type": row.get("manipulation_type", ""),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_rows):
            print(f"  [{idx+1}/{len(test_rows)}] {cid} ({lbl_str}) in {time.time()-t0:.1f}s | RAM={get_ram_gb()}GB")

    del model, processor
    gc.collect()

    test_data = {
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "cids": cids,
        "metadata": metadata_list,
    }

    cache["test"] = test_data
    torch.save(cache, VIDEOMAE_CACHE_PATH)
    print(f"Updated cache with test features: shape={test_data['features'].shape}")
    return test_data


def evaluate_final_candidate():
    selection_path = _CLASSIFIER_DIR / "videomae_candidate_selection.json"
    assert selection_path.exists(), f"Missing {selection_path}"
    with open(selection_path, encoding="utf-8") as f:
        selection = json.load(f)

    best_candidate_id = selection["best_videomae_candidate"]
    ckpt_path = Path(selection["best_videomae_checkpoint"])
    print(f"\nEvaluating Best Candidate: {best_candidate_id}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = VideoMAEHead(
        in_dim=768,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=2,
        dropout=cfg.get("dropout", 0.3),
        norm_type=cfg.get("norm_type", "layernorm"),
        architecture=cfg.get("architecture", "linear"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cache = torch.load(VIDEOMAE_CACHE_PATH, map_location="cpu", weights_only=False)
    test_feat = cache["test"]["features"]
    test_labels = cache["test"]["labels"]
    test_cids = cache["test"]["cids"]
    test_meta = cache["test"].get("metadata", [{} for _ in range(len(test_cids))])

    with torch.inference_mode():
        logits = model(test_feat)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = torch.argmax(logits, dim=1).numpy()

    y = test_labels.numpy()
    acc = float(accuracy_score(y, preds))
    bal_acc = float(balanced_accuracy_score(y, preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y, preds, average="weighted", zero_division=0)
    p_c, r_c, f1_c, support = precision_recall_fscore_support(y, preds, average=None, zero_division=0)
    auc = float(roc_auc_score(y, probs[:, 1]))
    cm = confusion_matrix(y, preds)

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print("\n" + "=" * 60)
    print(f"HELD-OUT TEST SET EVALUATION: {best_candidate_id}")
    print("=" * 60)
    print(f"Total Test Samples:      {len(y)}")
    print(f"Test Accuracy:           {acc:.4f}")
    print(f"Test Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Test Macro Precision:    {p_macro:.4f}")
    print(f"Test Macro Recall:       {r_macro:.4f}")
    print(f"Test Macro-F1:           {f1_macro:.4f}")
    print(f"Test Weighted-F1:        {f1_weighted:.4f}")
    print(f"Test ROC-AUC:            {auc:.4f}")
    print("\nPer-class Performance:")
    print(f"  Real (0): Prec={p_c[0]:.4f}, Rec={r_c[0]:.4f}, F1={f1_c[0]:.4f}, Support={support[0]}")
    print(f"  Fake (1): Prec={p_c[1]:.4f}, Rec={r_c[1]:.4f}, F1={f1_c[1]:.4f}, Support={support[1]}")
    print(f"\nConfusion Matrix:")
    print(f"  True Real  (TN={tn}, FP={fp}) -> Real Accuracy: {tn/(tn+fp):.4f}")
    print(f"  True Fake  (FN={fn}, TP={tp}) -> Fake Accuracy: {tp/(fn+tp):.4f}")

    # Baseline comparison
    exp6_test_path = _CLASSIFIER_DIR / "final_candidate_test_metrics.json"
    exp6_metrics = {}
    if exp6_test_path.exists():
        with open(exp6_test_path, encoding="utf-8") as f:
            exp6_metrics = json.load(f)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON ON HELD-OUT TEST SET:")
    print(f"{'Metric':<25} {'EfficientNet (exp6)':<22} {'VideoMAE (linear)':<22} {'Delta':<10}")
    print("-" * 75)
    e_f1 = exp6_metrics.get("test_macro_f1", 0.5411)
    e_bal = exp6_metrics.get("test_balanced_accuracy", 0.5417)
    e_acc = exp6_metrics.get("test_accuracy", 0.5476)
    e_auc = exp6_metrics.get("test_roc_auc", 0.6042)

    print(f"{'Test Macro-F1':<25} {e_f1:<22.4f} {f1_macro:<22.4f} {f1_macro - e_f1:+.4f}")
    print(f"{'Test Balanced Accuracy':<25} {e_bal:<22.4f} {bal_acc:<22.4f} {bal_acc - e_bal:+.4f}")
    print(f"{'Test Accuracy':<25} {e_acc:<22.4f} {acc:<22.4f} {acc - e_acc:+.4f}")
    print(f"{'Test ROC-AUC':<25} {e_auc:<22.4f} {auc:<22.4f} {auc - e_auc:+.4f}")

    # Detailed Error Analysis
    sample_analysis = []
    false_positives = []
    false_negatives = []
    correct_predictions = []

    for i in range(len(y)):
        cid = test_cids[i]
        true_l = int(y[i])
        pred_l = int(preds[i])
        prob_real = float(probs[i, 0])
        prob_fake = float(probs[i, 1])
        meta = test_meta[i] if i < len(test_meta) else {}

        item = {
            "cascade_id": cid,
            "true_label": INV_CLASS_MAP[true_l],
            "pred_label": INV_CLASS_MAP[pred_l],
            "prob_real": round(prob_real, 4),
            "prob_fake": round(prob_fake, 4),
            "confidence": round(max(prob_real, prob_fake), 4),
            "correct": bool(true_l == pred_l),
            "error_type": "FP" if true_l == 0 and pred_l == 1 else ("FN" if true_l == 1 and pred_l == 0 else "None"),
            "file_path": meta.get("file_path", ""),
            "source": meta.get("source", ""),
            "manipulation_type": meta.get("manipulation_type", ""),
        }
        sample_analysis.append(item)
        if item["error_type"] == "FP":
            false_positives.append(item)
        elif item["error_type"] == "FN":
            false_negatives.append(item)
        else:
            correct_predictions.append(item)

    print(f"\nError Analysis Summary:")
    print(f"  Total Correct:        {len(correct_predictions)} / {len(y)}")
    print(f"  False Positives (FP): {len(false_positives)} (Real classified as Fake)")
    print(f"  False Negatives (FN): {len(false_negatives)} (Fake classified as Real)")

    if false_positives:
        print("\n  False Positive Samples (Real called Fake):")
        for fp_item in false_positives:
            print(f"    - {fp_item['cascade_id']}: p(fake)={fp_item['prob_fake']:.3f}, source={fp_item['source']}")

    if false_negatives:
        print("\n  False Negative Samples (Fake called Real):")
        for fn_item in false_negatives:
            print(f"    - {fn_item['cascade_id']}: p(fake)={fn_item['prob_fake']:.3f}, manip={fn_item['manipulation_type']}")

    # Save output artifacts
    test_metrics_data = {
        "candidate_id": best_candidate_id,
        "checkpoint_path": str(ckpt_path),
        "validation_macro_f1": selection["videomae_val_macro_f1"],
        "validation_balanced_accuracy": selection["videomae_val_balanced_accuracy"],
        "test_total_samples": len(y),
        "test_accuracy": round(acc, 4),
        "test_balanced_accuracy": round(bal_acc, 4),
        "test_macro_precision": round(float(p_macro), 4),
        "test_macro_recall": round(float(r_macro), 4),
        "test_macro_f1": round(float(f1_macro), 4),
        "test_weighted_f1": round(float(f1_weighted), 4),
        "test_roc_auc": round(auc, 4),
        "per_class": {
            "real": {
                "precision": round(float(p_c[0]), 4),
                "recall": round(float(r_c[0]), 4),
                "f1_score": round(float(f1_c[0]), 4),
                "support": int(support[0]),
            },
            "fake": {
                "precision": round(float(p_c[1]), 4),
                "recall": round(float(r_c[1]), 4),
                "f1_score": round(float(f1_c[1]), 4),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "true_real_pred_real (TN)": tn,
            "true_real_pred_fake (FP)": fp,
            "true_fake_pred_real (FN)": fn,
            "true_fake_pred_fake (TP)": tp,
        },
        "comparison_vs_baseline_exp6": {
            "baseline_model": "exp6_compact_proj64 (EfficientNet)",
            "baseline_test_macro_f1": e_f1,
            "videomae_test_macro_f1": round(float(f1_macro), 4),
            "macro_f1_delta": round(float(f1_macro - e_f1), 4),
            "baseline_test_balanced_accuracy": e_bal,
            "videomae_test_balanced_accuracy": round(bal_acc, 4),
            "balanced_accuracy_delta": round(float(bal_acc - e_bal), 4),
            "videomae_outperforms_on_test": bool(f1_macro > e_f1),
        },
    }

    metrics_out = _CLASSIFIER_DIR / "videomae_final_candidate_test_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"\nSaved test metrics to: {metrics_out}")

    error_analysis_out = _CLASSIFIER_DIR / "videomae_test_error_analysis.json"
    with open(error_analysis_out, "w", encoding="utf-8") as f:
        json.dump({
            "candidate_id": best_candidate_id,
            "total_samples": len(y),
            "num_correct": len(correct_predictions),
            "num_fp": len(false_positives),
            "num_fn": len(false_negatives),
            "samples": sample_analysis,
        }, f, indent=2)
    print(f"Saved error analysis to: {error_analysis_out}")


def main():
    extract_test_features()
    evaluate_final_candidate()


if __name__ == "__main__":
    main()
