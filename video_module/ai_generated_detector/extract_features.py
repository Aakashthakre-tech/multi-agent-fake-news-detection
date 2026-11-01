"""
extract_features.py
===================
Phase 3a: One-time EfficientNet-B0 frame feature extraction.

For each video in a CSV split:
  1. Sample NUM_FRAMES uniformly spaced frames with OpenCV.
  2. Pass frames through frozen EfficientNet-B0 in mini-batches.
  3. Save per-video feature tensors to a .pt cache file.

Supports resuming: already-extracted video IDs are skipped.

Usage
-----
    python -m video_module.ai_generated_detector.extract_features \
        --split train   [--batch 16] [--num-frames 8] [--device cuda]

    # Extract all splits:
    for split in train val test ood; do
        python -m video_module.ai_generated_detector.extract_features --split $split
    done
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from video_module.ai_generated_detector.config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV, OOD_CSV,
    CACHE_TRAIN, CACHE_VAL, CACHE_TEST, CACHE_OOD,
    NUM_FRAMES, FRAME_SIZE, BACKBONE_NAME, FEATURE_DIM, EXTRACT_BATCH,
    CKPT_DIR,
)

SPLIT_MAP = {
    "train": (TRAIN_CSV, CACHE_TRAIN),
    "val":   (VAL_CSV,   CACHE_VAL),
    "test":  (TEST_CSV,  CACHE_TEST),
    "ood":   (OOD_CSV,   CACHE_OOD),
}

# ImageNet normalisation (EfficientNet-B0 expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════════
# Frame sampling
# ════════════════════════════════════════════════════════════════════════════════

def sample_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> Optional[np.ndarray]:
    """
    Return (num_frames, H, W, 3) uint8 RGB array or None on failure.
    Mirrors video_module/videomae_adapter.sample_frames_cv2 logic.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            frames_list: List[np.ndarray] = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if not frames_list:
                return None
            while len(frames_list) < num_frames:
                frames_list.append(frames_list[-1])
            return np.stack(frames_list[:num_frames])

        idxs = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
        sampled: List[np.ndarray] = []
        last: Optional[np.ndarray] = None
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled.append(last)
            elif last is not None:
                sampled.append(last)
            else:
                return None
        return np.stack(sampled)   # (T, H, W, 3)
    finally:
        cap.release()


# ════════════════════════════════════════════════════════════════════════════════
# Backbone loader
# ════════════════════════════════════════════════════════════════════════════════

def load_backbone(device: torch.device) -> torch.nn.Module:
    """Load frozen EfficientNet-B0 via timm."""
    try:
        import timm
    except ImportError:
        raise RuntimeError("timm is not installed. Run: pip install timm")

    model = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[backbone] {BACKBONE_NAME} loaded on {device} ({sum(p.numel() for p in model.parameters()):,} params frozen)")
    return model


def build_transform(img_size: int = FRAME_SIZE) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ════════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_video_features(
    video_path: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
    num_frames: int = NUM_FRAMES,
) -> Optional[torch.Tensor]:
    """
    Extract features for a single video.
    Returns: (num_frames, FEATURE_DIM) float32 or None on failure.
    """
    frames_np = sample_frames(video_path, num_frames)
    if frames_np is None:
        return None

    # (T, H, W, 3) → (T, 3, H, W) after transform
    frame_tensors = torch.stack([transform(f) for f in frames_np])  # (T,3,H,W)
    frame_tensors = frame_tensors.to(device)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        features = model(frame_tensors)   # (T, FEATURE_DIM)

    return features.cpu().float()


def load_csv(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_extraction(
    csv_path:   Path,
    cache_path: Path,
    device:     torch.device,
    num_frames: int = NUM_FRAMES,
    batch_size: int = EXTRACT_BATCH,
) -> None:
    """Extract features for all videos in a CSV split."""
    rows = load_csv(csv_path)
    if not rows:
        print(f"[extract] {csv_path.name}: no rows — skipping.")
        return

    # Load existing cache (for resume)
    cache: Dict[str, torch.Tensor] = {}
    if cache_path.exists():
        print(f"[extract] Resuming from existing cache: {cache_path}")
        existing = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(existing, dict):
            cache = existing
        print(f"  Already extracted: {len(cache)} videos")

    model     = load_backbone(device)
    transform = build_transform()

    pending = [r for r in rows if r["video_id"] not in cache]
    print(f"[extract] {csv_path.name}: {len(rows)} total, {len(pending)} pending")

    failed = []
    t0     = time.time()

    for i, row in enumerate(pending):
        vid_id = row["video_id"]
        vpath  = Path(row["abs_path"])

        if not vpath.exists():
            print(f"  MISS  {vid_id}: {vpath.name} not found")
            failed.append(vid_id)
            continue

        feat = extract_video_features(vpath, model, transform, device, num_frames)
        if feat is None:
            print(f"  FAIL  {vid_id}: cannot decode frames")
            failed.append(vid_id)
            continue

        cache[vid_id] = feat  # (T, F)

        if (i + 1) % 20 == 0 or (i + 1) == len(pending):
            # Periodic save (resume support)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_path)
            elapsed = time.time() - t0
            speed   = (i + 1) / elapsed
            eta     = (len(pending) - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{len(pending)}] saved — {speed:.1f} vid/s — ETA {eta:.0f}s")

    # Final save
    torch.save(cache, cache_path)
    print(f"\n[extract] Done: {len(cache)} features saved to {cache_path}")
    print(f"          Failed: {len(failed)} videos")
    if failed:
        print(f"          Failed IDs: {failed[:10]}")


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract EfficientNet-B0 features for a dataset split.")
    ap.add_argument("--split",      choices=["train","val","test","ood","all"], default="all")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--batch",      type=int, default=EXTRACT_BATCH)
    ap.add_argument("--device",     default="auto",
                    help="cuda | cpu | auto (default: auto — uses CUDA if available)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[extract] Device: {device}")

    if device.type == "cuda":
        print(f"[extract] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[extract] VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    splits = list(SPLIT_MAP.items()) if args.split == "all" else [(args.split, SPLIT_MAP[args.split])]

    for split_name, (csv_path, cache_path) in splits:
        print(f"\n{'='*55}")
        print(f"  Extracting split: {split_name}")
        print(f"{'='*55}")
        run_extraction(csv_path, cache_path, device, args.num_frames, args.batch)


if __name__ == "__main__":
    main()
