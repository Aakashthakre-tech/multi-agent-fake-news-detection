"""
extract_frame_sampling_caches.py
================================
Extract features for frame sampling experiments:
1. 8 frames (segment center)
2. 16 frames (segment center)
3. 12 frames (segment center)
"""

import csv
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

_PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
_CLASSIFIER_DIR = _PROJECT_ROOT / "video_module" / "baseline_classifier"
_AUDIT_DIR = _PROJECT_ROOT / "video_module" / "dataset_audit"

sys.path.insert(0, str(_CLASSIFIER_DIR))
from model import get_feature_extractor
from dataset import DEFAULT_TRANSFORM

CLASS_MAP = {"real": 0, "fake": 1}

def sample_video_frames_segment_center(
    video_path: Path,
    num_frames: int = 12,
    transform: transforms.Compose = DEFAULT_TRANSFORM,
) -> torch.Tensor:
    """Sample frames at the midpoint of K uniform segments across the video duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = num_frames

        # Center of each of the num_frames segments
        # Avoids boundary black frames at t=0 and t=end
        segment_size = total_frames / num_frames
        frame_indices = [int((i + 0.5) * segment_size) for i in range(num_frames)]
        frame_indices = [min(max(0, idx), total_frames - 1) for idx in frame_indices]

        sampled_tensors = []
        last_valid = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor_img = transform(pil_img)
                last_valid = tensor_img
                sampled_tensors.append(tensor_img)
            else:
                if last_valid is not None:
                    sampled_tensors.append(last_valid)
                else:
                    black = Image.new("RGB", (224, 224), (0, 0, 0))
                    last_valid = transform(black)
                    sampled_tensors.append(last_valid)

        while len(sampled_tensors) < num_frames:
            sampled_tensors.append(last_valid)

        return torch.stack(sampled_tensors[:num_frames], dim=0)
    finally:
        cap.release()


def load_manifest(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_and_save_cache(num_frames: int, out_cache_path: Path, device: torch.device):
    if out_cache_path.exists():
        print(f"Cache already exists at {out_cache_path}. Skipping extraction.")
        return

    print(f"\n========================================================")
    print(f"EXTRACTING FEATURES: {num_frames} FRAMES (SEGMENT-CENTER)")
    print(f"Target file: {out_cache_path}")
    print(f"========================================================")

    extractor = get_feature_extractor(device)
    train_rows = load_manifest(_AUDIT_DIR / "train_manifest.csv")
    val_rows = load_manifest(_AUDIT_DIR / "val_manifest.csv")
    test_rows = load_manifest(_AUDIT_DIR / "test_manifest.csv")

    def process(rows, name):
        t0 = time.time()
        feat_list, lbl_list, cid_list = [], [], []
        for i, r in enumerate(rows, 1):
            cid = r["cascade_id"]
            lbl = CLASS_MAP[r["label"]]
            vpath = Path(r["file_path"])
            frames = sample_video_frames_segment_center(vpath, num_frames=num_frames)
            with torch.no_grad():
                emb = extractor(frames.to(device)).cpu()
            feat_list.append(emb)
            lbl_list.append(lbl)
            cid_list.append(cid)
            if i % 30 == 0 or i == len(rows):
                print(f"   [{name}] {i}/{len(rows)} processed ({time.time() - t0:.1f}s)")
        return {
            "features": torch.stack(feat_list, dim=0),
            "labels": torch.tensor(lbl_list, dtype=torch.long),
            "cids": cid_list,
        }

    train_data = process(train_rows, "train")
    val_data = process(val_rows, "val")
    test_data = process(test_rows, "test")

    payload = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "num_frames": num_frames,
        "sampling_strategy": "segment_center",
        "class_map": CLASS_MAP,
    }
    torch.save(payload, out_cache_path)
    print(f"[Success] Saved {out_cache_path} ({out_cache_path.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    device = torch.device("cpu")
    # 1. 12 frames with segment-center
    build_and_save_cache(12, _CLASSIFIER_DIR / "features_cache_12frames_segcenter.pt", device)
    # 2. 16 frames with segment-center
    build_and_save_cache(16, _CLASSIFIER_DIR / "features_cache_16frames.pt", device)
    # 3. 8 frames with segment-center
    build_and_save_cache(8, _CLASSIFIER_DIR / "features_cache_8frames.pt", device)
