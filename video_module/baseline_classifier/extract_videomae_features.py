"""
extract_videomae_features.py
=============================
Extracts frozen spatio-temporal embeddings from MCG-NJU/videomae-base for FVC splits.
Features:
- Extracts strictly train and validation first. Protects the test split.
- Uses inference mode (torch.no_grad()) and model in eval mode.
- 100% exact weight mapping with mapped attention biases.
- Fast uniform 16-frame seek sampling with OpenCV (matching dataset.py).
- Incremental checkpointing per 10 videos for crash-proof resumption.
- Caches embeddings with comprehensive reproducibility and provenance metadata.
- Memory safe: garbage collected per video to ensure RAM stability on CPU.
"""

import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import psutil
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, VideoMAEModel

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
_AUDIT_DIR = _PROJECT_ROOT / "video_module" / "dataset_audit"
VIDEOMAE_CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
CHECKPOINT_PARTIAL_PATH = _CLASSIFIER_DIR / "videomae_train_partial.pt"
MODEL_ID = "MCG-NJU/videomae-base"

CLASS_MAP = {"real": 0, "fake": 1}

def get_ram_available_gb() -> float:
    return round(psutil.virtual_memory().available / (1024**3), 2)

def load_videomae_clean():
    """Loads MCG-NJU/videomae-base with 100% exact attention bias mapping."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(MODEL_ID)
    st_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")
    raw_sd = load_file(st_path)
    sd = model.state_dict()
    for i in range(model.config.num_hidden_layers):
        q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
        v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
        sd[f"encoder.layer.{i}.attention.attention.query.bias"] = raw_sd[q_key]
        sd[f"encoder.layer.{i}.attention.attention.value.bias"] = raw_sd[v_key]
        sd[f"encoder.layer.{i}.attention.attention.key.bias"] = torch.zeros(model.config.hidden_size)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return processor, model

def sample_video_frames_cv2(video_path: Path, num_frames: int = 16) -> np.ndarray:
    """Fast seek sampling of 16 frames uniformly across video (matches baseline dataset.py)."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            frames_list = []
            while len(frames_list) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames_list:
                raise ValueError(f"No decodable frames found in {video_path}")
            while len(frames_list) < num_frames:
                frames_list.append(frames_list[-1])
            return np.stack(frames_list[:num_frames])

        frame_indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        sampled_frames = []
        last_valid_frame = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_valid_frame = rgb_frame
                sampled_frames.append(rgb_frame)
            else:
                if last_valid_frame is not None:
                    sampled_frames.append(last_valid_frame)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_alt, frame_alt = cap.read()
                    if ret_alt and frame_alt is not None:
                        last_valid_frame = cv2.cvtColor(frame_alt, cv2.COLOR_BGR2RGB)
                        sampled_frames.append(last_valid_frame)
                    else:
                        raise ValueError(f"Could not decode any valid frames from {video_path}")

        return np.stack(sampled_frames)
    finally:
        cap.release()

def extract_split(
    manifest_path: Path,
    split_name: str,
    processor: AutoImageProcessor,
    model: VideoMAEModel,
    device: torch.device
) -> Dict[str, Any]:
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    print(f"\n[{split_name.upper()}] Starting fast extraction for {len(rows)} samples...")

    # Check for existing partial cache to allow seamless resume
    features_list = []
    labels_list = []
    cids_list = []
    start_idx = 0

    partial_file = _CLASSIFIER_DIR / f"videomae_{split_name}_partial.pt"
    if partial_file.exists():
        part_data = torch.load(partial_file, map_location="cpu", weights_only=False)
        features_list = list(part_data["features"])
        labels_list = list(part_data["labels"].tolist())
        cids_list = list(part_data["cids"])
        start_idx = len(cids_list)
        print(f"  Resuming from partial checkpoint: {start_idx}/{len(rows)} already completed.")

    t_start = time.time()
    for idx, row in enumerate(rows[start_idx:], start_idx + 1):
        cid = row["cascade_id"]
        lbl_str = row["label"]
        lbl_int = CLASS_MAP[lbl_str]
        vid_path = Path(row["file_path"])

        t_v0 = time.time()
        frames_np = sample_video_frames_cv2(vid_path, num_frames=16)
        inputs = processor(list(frames_np), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values)
            pooled_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # [768]

        features_list.append(pooled_emb)
        labels_list.append(lbl_int)
        cids_list.append(cid)

        # Cleanup
        del outputs, inputs, pixel_values, frames_np

        if idx % 10 == 0 or idx == len(rows):
            gc.collect()
            elapsed = time.time() - t_start
            processed_here = idx - start_idx
            rate = elapsed / max(1, processed_here)
            remain = rate * (len(rows) - idx)
            print(f"  [{idx}/{len(rows)}] {cid} ({lbl_str}) in {time.time()-t_v0:.2f}s | Elapsed: {elapsed/60:.1f}m | ETA: {remain/60:.1f}m | Free RAM: {get_ram_available_gb()}GB")
            # Save partial checkpoint
            torch.save({
                "features": torch.stack(features_list),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "cids": cids_list,
            }, partial_file)

    features_tensor = torch.stack(features_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Clean up partial file upon successful full split completion
    if partial_file.exists():
        partial_file.unlink()

    duration = round(time.time() - t_start, 2)
    print(f"[{split_name.upper()}] Extraction complete! Tensor shape: {features_tensor.shape}, Duration: {duration}s")
    return {
        "features": features_tensor,
        "labels": labels_tensor,
        "cids": cids_list,
        "duration_sec": duration,
    }

def main(include_test: bool = False):
    device = torch.device("cpu")
    print("=" * 70)
    print("VIDEOMAE FAST FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Initial Available RAM: {get_ram_available_gb()} GB")

    processor, model = load_videomae_clean()
    model = model.to(device)

    cache = {}
    if VIDEOMAE_CACHE_PATH.exists():
        print(f"Loading existing cache from: {VIDEOMAE_CACHE_PATH}")
        cache = torch.load(VIDEOMAE_CACHE_PATH, map_location="cpu", weights_only=False)

    # 1. Val split (already cached if previous run saved it)
    if "val" not in cache:
        cache["val"] = extract_split(_AUDIT_DIR / "val_manifest.csv", "val", processor, model, device)
        torch.save(cache, VIDEOMAE_CACHE_PATH)
        print(f"Saved val cache to {VIDEOMAE_CACHE_PATH}")
    else:
        print(f"[CACHE HIT] Val split already cached: {cache['val']['features'].shape}")

    # 2. Train split
    if "train" not in cache:
        cache["train"] = extract_split(_AUDIT_DIR / "train_manifest.csv", "train", processor, model, device)
        torch.save(cache, VIDEOMAE_CACHE_PATH)
        print(f"Saved train cache to {VIDEOMAE_CACHE_PATH}")
    else:
        print(f"[CACHE HIT] Train split already cached: {cache['train']['features'].shape}")

    # 3. Test split (only if explicitly requested)
    if include_test and "test" not in cache:
        cache["test"] = extract_split(_AUDIT_DIR / "test_manifest.csv", "test", processor, model, device)
        torch.save(cache, VIDEOMAE_CACHE_PATH)
        print(f"Saved test cache to {VIDEOMAE_CACHE_PATH}")

    # Metadata
    cache["metadata"] = {
        "model_id": MODEL_ID,
        "hidden_size": 768,
        "num_frames": 16,
        "resolution": [224, 224],
        "sampling_method": "uniform_fast_seek_opencv",
        "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "pooling": "mean_across_tubelet_tokens",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(cache["train"]["labels"]) if "train" in cache else 0,
        "val_samples": len(cache["val"]["labels"]) if "val" in cache else 0,
        "test_samples": len(cache["test"]["labels"]) if "test" in cache else 0,
    }
    torch.save(cache, VIDEOMAE_CACHE_PATH)
    print("\nExtraction finished and metadata written to cache.")
    print(f"Cache File: {VIDEOMAE_CACHE_PATH} (Size: {VIDEOMAE_CACHE_PATH.stat().st_size / (1024**2):.2f} MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_test", action="store_true", help="Extract test split")
    args = parser.parse_args()
    main(include_test=args.include_test)
