"""
resume_extract_videomae.py
===========================
Crash-proof resumption of VideoMAE train extraction.

Key fixes vs original extract_videomae_features.py:
1. Loads safetensors DIRECTLY from local HF cache — no network call.
2. Pre-checks available RAM and aborts if below 900 MB.
3. GC + RAM check every 5 videos (not 10) to stay safe on tight systems.
4. Does NOT call hf_hub_download (was the stall cause).
5. On completion, merges train into videomae_features_cache.pt alongside val.
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
from safetensors.torch import load_file
from transformers import AutoImageProcessor, VideoMAEModel

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
_AUDIT_DIR = _PROJECT_ROOT / "video_module" / "dataset_audit"

VIDEOMAE_CACHE_PATH = _CLASSIFIER_DIR / "videomae_features_cache.pt"
PARTIAL_PATH = _CLASSIFIER_DIR / "videomae_train_partial.pt"
MODEL_ID = "MCG-NJU/videomae-base"
CLASS_MAP = {"real": 0, "fake": 1}
MIN_RAM_GB = 0.9  # abort extraction if free RAM drops below this


def get_ram_gb() -> float:
    return round(psutil.virtual_memory().available / (1024 ** 3), 2)


def find_local_safetensors(model_id: str) -> Path:
    """Locate model.safetensors in the HF local cache without a network call."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_slug = model_id.replace("/", "--")
    cache_dir = hf_home / "hub" / f"models--{model_slug}"
    candidates = list(cache_dir.rglob("model.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"model.safetensors not found under {cache_dir}. "
            f"Please run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_id}')\" first."
        )
    # Pick the largest (most complete)
    return max(candidates, key=lambda p: p.stat().st_size)


def load_videomae_local() -> tuple:
    """Load VideoMAE from local HF cache ONLY — no network calls."""
    st_path = find_local_safetensors(MODEL_ID)
    print(f"Loading safetensors from local cache: {st_path} ({st_path.stat().st_size / 1e6:.1f} MB)")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = VideoMAEModel.from_pretrained(MODEL_ID, local_files_only=True)

    # Apply the exact q/v bias mapping from the original script
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

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only encoder keys should be loaded; decoder keys are fine to be missing
    encoder_missing = [k for k in missing if "encoder" in k and "key.bias" not in k
                       and "query.bias" not in k and "value.bias" not in k]
    if encoder_missing:
        print(f"WARNING: Unexpected missing encoder keys: {encoder_missing[:5]}")
    model.eval()
    del raw_sd, sd
    gc.collect()
    print(f"VideoMAE loaded. Free RAM: {get_ram_gb()} GB")
    return processor, model


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


def main():
    print("=" * 70)
    print("VideoMAE TRAIN EXTRACTION — CRASH-PROOF RESUMPTION")
    print("=" * 70)
    print(f"Free RAM at start: {get_ram_gb()} GB")

    if get_ram_gb() < MIN_RAM_GB:
        print(f"ABORT: Only {get_ram_gb()} GB available — need at least {MIN_RAM_GB} GB.")
        sys.exit(1)

    # --- Load existing cache ---
    cache = {}
    if VIDEOMAE_CACHE_PATH.exists():
        print(f"Loading existing cache: {VIDEOMAE_CACHE_PATH}")
        cache = torch.load(VIDEOMAE_CACHE_PATH, map_location="cpu", weights_only=False)
        print(f"  Cache keys: {list(cache.keys())}")

    if "train" in cache:
        n = len(cache["train"]["cids"])
        print(f"Train already fully extracted ({n} samples). Nothing to do.")
        print("Verifying integrity...")
        f = cache["train"]["features"]
        assert f.shape == (203, 768), f"Expected [203,768], got {f.shape}"
        assert not torch.isnan(f).any(), "NaN in train features!"
        print("  OK — train features valid.")
        return

    # --- Load train manifest ---
    train_rows = list(csv.DictReader(open(_AUDIT_DIR / "train_manifest.csv", encoding="utf-8")))
    print(f"Train manifest: {len(train_rows)} rows")

    # --- Resume from partial ---
    features_list, labels_list, cids_list = [], [], []
    start_idx = 0
    if PARTIAL_PATH.exists():
        part = torch.load(PARTIAL_PATH, map_location="cpu", weights_only=False)
        features_list = list(part["features"])
        labels_list = list(part["labels"].tolist())
        cids_list = list(part["cids"])
        start_idx = len(cids_list)
        n_real = sum(1 for l in labels_list if l == 0)
        n_fake = sum(1 for l in labels_list if l == 1)
        has_nan = torch.isnan(part["features"]).any().item()
        has_inf = torch.isinf(part["features"]).any().item()
        if has_nan or has_inf:
            print(f"WARNING: Partial checkpoint has NaN={has_nan}/Inf={has_inf} — resetting to 0!")
            features_list, labels_list, cids_list = [], [], []
            start_idx = 0
        else:
            print(f"Resuming from partial: {start_idx}/{len(train_rows)} done (real={n_real}, fake={n_fake})")
        del part
        gc.collect()

    remaining_rows = train_rows[start_idx:]
    print(f"Remaining to extract: {len(remaining_rows)} videos")

    if not remaining_rows:
        print("Nothing remaining — merging partial into cache.")
    else:
        # --- Load model (local only, no network) ---
        print(f"\nFree RAM before model load: {get_ram_gb()} GB")
        processor, model = load_videomae_local()
        print(f"Free RAM after model load: {get_ram_gb()} GB")

        if get_ram_gb() < 0.5:
            print("ABORT: Critically low RAM after model load.")
            sys.exit(1)

        t_start = time.time()
        failed = []

        for i, row in enumerate(remaining_rows):
            global_idx = start_idx + i + 1  # 1-based
            cid = row["cascade_id"]
            lbl_str = row["label"]
            lbl_int = CLASS_MAP[lbl_str]
            vid_path = Path(row["file_path"])

            # RAM check before each video
            free_gb = get_ram_gb()
            if free_gb < MIN_RAM_GB:
                print(f"\nWARNING: Only {free_gb} GB free — saving partial and stopping.")
                torch.save({
                    "features": torch.stack(features_list),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "cids": cids_list,
                }, PARTIAL_PATH)
                print(f"Partial saved: {len(cids_list)} samples.")
                sys.exit(2)  # Exit code 2 = memory stop (not an error, resume later)

            t0 = time.time()
            try:
                frames = sample_frames_cv2(vid_path, num_frames=16)
                inputs = processor(list(frames), return_tensors="pt")
                pv = inputs["pixel_values"]

                with torch.inference_mode():
                    out = model(pv)
                    emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()

                # Validate embedding
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    raise ValueError(f"NaN/Inf in embedding for {cid}")
                if emb.shape != (768,):
                    raise ValueError(f"Wrong shape {emb.shape} for {cid}")

                features_list.append(emb)
                labels_list.append(lbl_int)
                cids_list.append(cid)

                elapsed = time.time() - t_start
                processed = i + 1
                rate = elapsed / max(1, processed)
                remain_sec = rate * (len(remaining_rows) - processed)
                elapsed_min = elapsed / 60
                remain_min = remain_sec / 60

                del out, inputs, pv, frames, emb
            except Exception as e:
                print(f"  [FAILED] {cid}: {e}")
                failed.append({"cid": cid, "label": lbl_str, "error": str(e)})
                # Put a zero embedding as placeholder so alignment is preserved
                features_list.append(torch.zeros(768))
                labels_list.append(lbl_int)
                cids_list.append(cid)
                del e
            
            # Checkpoint every 5 videos
            if global_idx % 5 == 0 or global_idx == len(train_rows):
                gc.collect()
                free_now = get_ram_gb()
                speed_str = f"{time.time()-t0:.1f}s" if i == 0 else ""
                print(f"  [{global_idx}/{len(train_rows)}] {cid} ({lbl_str}) | RAM={free_now}GB | Elapsed={elapsed_min:.1f}m | ETA={remain_min:.1f}m")
                torch.save({
                    "features": torch.stack(features_list),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "cids": cids_list,
                }, PARTIAL_PATH)

        # Unload model to free RAM
        del model, processor
        gc.collect()
        print(f"\nFree RAM after model unload: {get_ram_gb()} GB")

        if failed:
            print(f"\nWARNING: {len(failed)} videos failed:")
            for f_item in failed:
                print(f"  {f_item['cid']} ({f_item['label']}): {f_item['error']}")

    # --- Merge into main cache ---
    print("\nMerging train embeddings into main cache...")
    part = torch.load(PARTIAL_PATH, map_location="cpu", weights_only=False)
    
    assert len(part["cids"]) == len(train_rows), \
        f"Expected {len(train_rows)} samples, got {len(part['cids'])}"

    # Cross-validate labels vs manifest
    manifest_labels = {r["cascade_id"]: CLASS_MAP[r["label"]] for r in train_rows}
    mismatches = 0
    for cid, lbl in zip(part["cids"], part["labels"].tolist()):
        if cid in manifest_labels and manifest_labels[cid] != lbl:
            print(f"  LABEL MISMATCH: {cid} partial={lbl} manifest={manifest_labels[cid]}")
            mismatches += 1
    if mismatches > 0:
        print(f"ERROR: {mismatches} label mismatches — aborting merge.")
        sys.exit(3)

    feat = part["features"]
    n_nan = torch.isnan(feat).sum().item()
    n_inf = torch.isinf(feat).sum().item()
    print(f"Train features: shape={feat.shape}, NaN count={n_nan}, Inf count={n_inf}")

    cache["train"] = {
        "features": feat,
        "labels": part["labels"],
        "cids": part["cids"],
    }

    # Also store metadata
    cache["metadata"] = {
        "model_id": MODEL_ID,
        "hidden_size": 768,
        "num_frames": 16,
        "pooling": "mean_across_all_tubelet_tokens",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(part["cids"]),
        "val_samples": len(cache["val"]["cids"]) if "val" in cache else 0,
        "failed_samples": failed if "failed" in dir() else [],
    }

    torch.save(cache, VIDEOMAE_CACHE_PATH)
    print(f"Saved merged cache to {VIDEOMAE_CACHE_PATH} ({VIDEOMAE_CACHE_PATH.stat().st_size / 1e6:.1f} MB)")

    # Remove partial now that it's merged
    PARTIAL_PATH.unlink()
    print(f"Deleted partial file: {PARTIAL_PATH}")

    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Train: {cache['train']['features'].shape}")
    print(f"Val:   {cache['val']['features'].shape}")
    n_real_t = (cache['train']['labels'] == 0).sum().item()
    n_fake_t = (cache['train']['labels'] == 1).sum().item()
    print(f"Train label distribution: real={n_real_t}, fake={n_fake_t}")


if __name__ == "__main__":
    main()
