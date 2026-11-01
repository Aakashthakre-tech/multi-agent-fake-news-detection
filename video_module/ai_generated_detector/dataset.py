"""
dataset.py
==========
PyTorch Dataset that loads pre-extracted feature caches for training.

Never loads raw video during training — operates purely on cached .pt files.
Applies optional per-frame jitter augmentation (only during training).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .config import CLASS_MAP, NUM_FRAMES, FEATURE_DIM


class FeatureDataset(Dataset):
    """
    Loads pre-extracted EfficientNet-B0 features from a .pt cache file.

    Each item is (features, label):
        features : (T, F) float32  — T=num_frames, F=feature_dim
        label    : int             — 0=REAL, 1=AI_GENERATED
    """

    def __init__(
        self,
        csv_path:   Path,
        cache_path: Path,
        num_frames: int  = NUM_FRAMES,
        augment:    bool = False,
        seed:       int  = 42,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.augment    = augment
        self.rng        = random.Random(seed)

        # Load manifest
        self.rows = self._load_csv(csv_path)

        # Load feature cache
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {cache_path}\n"
                "Run extract_features.py first."
            )
        raw = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict in cache, got {type(raw)}")
        self.cache: Dict[str, torch.Tensor] = raw

        # Filter rows to only those present in cache
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["video_id"] in self.cache]
        after  = len(self.rows)
        if before != after:
            print(f"[dataset] {csv_path.name}: {before - after} rows skipped (not in cache)")

        if len(self.rows) == 0:
            raise ValueError(
                f"No valid rows found for {csv_path.name}. "
                "Check that feature extraction succeeded."
            )

        print(f"[dataset] {csv_path.name}: {len(self.rows)} samples "
              f"(REAL={sum(r['label']=='REAL' for r in self.rows)}, "
              f"AI_GEN={sum(r['label']=='AI_GENERATED' for r in self.rows)})")

    @staticmethod
    def _load_csv(path: Path) -> List[Dict]:
        if not path.exists():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row    = self.rows[idx]
        vid_id = row["video_id"]
        label  = CLASS_MAP.get(row["label"], -1)

        feat = self.cache[vid_id].float()   # (T, F)

        # Adjust frame count: truncate or repeat-pad
        T = feat.shape[0]
        if T > self.num_frames:
            # Uniform sub-sample
            idxs = torch.linspace(0, T - 1, self.num_frames).long()
            feat = feat[idxs]
        elif T < self.num_frames:
            # Repeat-pad
            pad  = feat[-1:].repeat(self.num_frames - T, 1)
            feat = torch.cat([feat, pad], dim=0)

        # Augmentation: random Gaussian noise on features (mild, training only)
        if self.augment:
            noise = torch.randn_like(feat) * 0.01
            feat  = feat + noise

        return feat, label

    def get_labels(self) -> List[int]:
        """Return all labels (for computing class weights)."""
        return [CLASS_MAP.get(r["label"], -1) for r in self.rows]

    def get_generators(self) -> List[str]:
        return [r.get("generator", "unknown") for r in self.rows]

    def get_video_ids(self) -> List[str]:
        return [r["video_id"] for r in self.rows]
