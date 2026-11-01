"""
verify_and_sample_videomae.py
==============================
Benchmarks VideoMAE (MCG-NJU/videomae-base) on a 3-video sample (1 train, 1 val, 1 test).
Measures:
1. Exact model configuration and preprocessing requirements.
2. System RAM before loading, after loading, and peak during forward pass.
3. Frame sampling time, preprocessing time, and forward pass latency on CPU.
4. Output tensor dimensions and numerical validity (NaN/Inf, min/max/norm).
5. Garbage collection and memory stability.
"""

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
from transformers import AutoImageProcessor, VideoMAEModel

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_AUDIT_DIR = _PROJECT_ROOT / "video_module" / "dataset_audit"
MODEL_ID = "MCG-NJU/videomae-base"

def get_ram_info():
    v = psutil.virtual_memory()
    return {
        "total_gb": round(v.total / (1024**3), 2),
        "available_gb": round(v.available / (1024**3), 2),
        "used_percent": v.percent
    }

def sample_video_frames_cv2(video_path: Path, num_frames: int = 16) -> np.ndarray:
    """Sample num_frames uniformly from video_path as uint8 numpy array [T, H, W, C] in RGB."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        total_frames = len(frames)
        if total_frames == 0:
            raise RuntimeError(f"Zero frames decoded from {video_path}")
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return np.stack([frames[idx] for idx in indices])

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    current_idx = 0
    target_idx_set = set(indices)
    max_idx = indices[-1]

    while cap.isOpened() and current_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_idx_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((current_idx, rgb))
        current_idx += 1
    cap.release()

    frame_dict = dict(frames)
    sampled = []
    last_frame = None
    for idx in indices:
        if idx in frame_dict:
            last_frame = frame_dict[idx]
            sampled.append(last_frame)
        elif last_frame is not None:
            sampled.append(last_frame)
        else:
            raise RuntimeError(f"Failed to sample initial frame from {video_path}")
    return np.stack(sampled)

def run_benchmark():
    print("=" * 70)
    print("STEP 2 & 3: VIDEOMAE BENCHMARK & NUMERICAL VALIDITY AUDIT")
    print("=" * 70)

    ram_start = get_ram_info()
    print(f"System RAM (Initial): Total={ram_start['total_gb']}GB, Available={ram_start['available_gb']}GB ({ram_start['used_percent']}% used)")

    # 1. Load processor and model
    t0 = time.time()
    print(f"\nLoading processor and model from {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(MODEL_ID)
    model.eval()
    load_time = time.time() - t0

    ram_after_model = get_ram_info()
    model_ram_mb = round((ram_start['available_gb'] - ram_after_model['available_gb']) * 1024, 1)
    print(f"Model loaded in {load_time:.2f}s.")
    print(f"RAM after model load: Available={ram_after_model['available_gb']}GB (Model footprint: ~{model_ram_mb} MB)")
    print(f"Model config: hidden_size={model.config.hidden_size}, num_hidden_layers={model.config.num_hidden_layers}, num_attention_heads={model.config.num_attention_heads}")

    # 2. Pick 1 sample from each split
    import csv
    train_row = list(csv.DictReader(open(_AUDIT_DIR / "train_manifest.csv", encoding="utf-8")))[0]
    val_row = list(csv.DictReader(open(_AUDIT_DIR / "val_manifest.csv", encoding="utf-8")))[0]
    test_row = list(csv.DictReader(open(_AUDIT_DIR / "test_manifest.csv", encoding="utf-8")))[0]

    samples = [
        ("train", train_row),
        ("val", val_row),
        ("test", test_row),
    ]

    benchmark_records = []
    print("\nProcessing small 3-sample benchmark (1 train, 1 val, 1 test)...")

    for split, row in samples:
        cid = row["cascade_id"]
        vid_path = Path(row["file_path"])
        label = row["label"]
        print(f"\n--- Testing [{split.upper()}] video {cid} ({label}) ---")
        print(f"Path: {vid_path}")

        # Measure frame decoding
        t_decode_start = time.time()
        frames_np = sample_video_frames_cv2(vid_path, num_frames=16)  # [16, H, W, 3]
        decode_time = time.time() - t_decode_start
        print(f"1. Decoded 16 frames in {decode_time:.3f}s. Raw shape: {frames_np.shape}, dtype: {frames_np.dtype}")

        # Measure processor transformation
        t_proc_start = time.time()
        # VideoMAE processor accepts list of 16 numpy frames
        inputs = processor(list(frames_np), return_tensors="pt")
        proc_time = time.time() - t_proc_start
        pixel_values = inputs["pixel_values"]  # [1, 16, 3, 224, 224]
        print(f"2. Preprocessed in {proc_time:.3f}s. Input tensor shape: {pixel_values.shape}, min: {pixel_values.min().item():.3f}, max: {pixel_values.max().item():.3f}")

        # Measure forward pass
        ram_before_fwd = get_ram_info()
        t_fwd_start = time.time()
        with torch.no_grad():
            outputs = model(pixel_values)
            # last_hidden_state: [1, 1568, 768]
            last_hidden = outputs.last_hidden_state
            # Mean pool across tokens -> [1, 768]
            pooled_emb = last_hidden.mean(dim=1).squeeze(0)  # [768]
        fwd_time = time.time() - t_fwd_start
        ram_after_fwd = get_ram_info()

        # Numerical validity
        has_nan = torch.isnan(pooled_emb).any().item()
        has_inf = torch.isinf(pooled_emb).any().item()
        norm_val = torch.norm(pooled_emb, p=2).item()
        min_val = pooled_emb.min().item()
        max_val = pooled_emb.max().item()
        mean_val = pooled_emb.mean().item()

        print(f"3. Forward pass completed in {fwd_time:.3f}s.")
        print(f"   last_hidden_state shape: {last_hidden.shape}")
        print(f"   pooled_emb shape: {pooled_emb.shape}")
        print(f"   Numerical Validity: NaN={has_nan}, Inf={has_inf}")
        print(f"   Stats: norm={norm_val:.4f}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
        print(f"   RAM during fwd: Available={ram_after_fwd['available_gb']}GB")

        record = {
            "split": split,
            "cascade_id": cid,
            "label": label,
            "decode_sec": round(decode_time, 3),
            "proc_sec": round(proc_time, 3),
            "fwd_sec": round(fwd_time, 3),
            "total_sec": round(decode_time + proc_time + fwd_time, 3),
            "embedding_dim": list(pooled_emb.shape),
            "is_valid": (not has_nan and not has_inf and norm_val > 0),
            "norm": round(norm_val, 4),
        }
        benchmark_records.append(record)

        # Cleanup
        del outputs, last_hidden, pooled_emb, inputs, pixel_values, frames_np
        gc.collect()

    ram_final = get_ram_info()
    avg_total_sec = np.mean([r["total_sec"] for r in benchmark_records])
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY & RESOURCE ESTIMATION")
    print("=" * 70)
    for r in benchmark_records:
        print(f"[{r['split'].upper()}] {r['cascade_id']}: Decode={r['decode_sec']}s, Proc={r['proc_sec']}s, Fwd={r['fwd_sec']}s -> Total={r['total_sec']}s (Valid: {r['is_valid']}, Norm: {r['norm']})")

    print(f"\nAverage time per video: {avg_total_sec:.2f}s")
    print(f"Estimated extraction time for 203 train videos: {203 * avg_total_sec / 60:.1f} minutes")
    print(f"Estimated extraction time for 43 val videos:    {43 * avg_total_sec / 60:.1f} minutes")
    print(f"RAM Stability: Started with {ram_start['available_gb']}GB, currently {ram_final['available_gb']}GB available.")

    summary = {
        "model_id": MODEL_ID,
        "load_time_sec": round(load_time, 2),
        "model_ram_mb": model_ram_mb,
        "avg_total_sec_per_video": round(float(avg_total_sec), 2),
        "est_train_extract_min": round(203 * avg_total_sec / 60, 1),
        "est_val_extract_min": round(43 * avg_total_sec / 60, 1),
        "samples": benchmark_records,
        "ram_start_gb": ram_start,
        "ram_final_gb": ram_final,
        "system_stable": ram_final["available_gb"] >= 1.0,
    }

    out_json = _AUDIT_DIR / "videomae_benchmark_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark audit to: {out_json}")

if __name__ == "__main__":
    run_benchmark()
