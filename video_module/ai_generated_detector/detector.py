"""
detector.py
===========
Inference interface for the AI-generated video detection module.

Loads the frozen EfficientNet-B0 backbone and trained BiGRU head to classify
whether an input video is authentic (REAL) or AI-generated (AI_GENERATED).

Separate and independent from the misinformation VideoMAE classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import timm

from .config import (
    BEST_MODEL,
    NUM_FRAMES,
    FRAME_SIZE,
    BACKBONE_NAME,
    CLASS_NAMES,
    EXPERIMENTAL_WARNING,
)
from .model import BiGRUHead
from .extract_features import sample_frames, IMAGENET_MEAN, IMAGENET_STD

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def _get_models(ckpt_path: Path, device: str = _DEVICE):
    global _MODEL_CACHE
    cache_key = f"{ckpt_path}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 1. Backbone
    backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)

    # 2. Classifier head
    classifier = BiGRUHead()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        classifier.load_state_dict(state.get("model_state", state))
    classifier.eval()
    classifier.to(device)

    bundle = {"backbone": backbone, "classifier": classifier}
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def predict_ai_generated(
    video_path: Union[str, Path],
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = _DEVICE,
) -> Dict:
    """
    Classify a video as REAL (0) or AI_GENERATED (1).

    Returns:
        dict with keys:
          - label: "REAL" | "AI_GENERATED"
          - label_idx: int (0 or 1)
          - probability: float (prob of AI_GENERATED)
          - confidence: float (prob of predicted class)
          - is_ai_generated: bool
          - warning: str
    """
    video_path = Path(video_path)
    ckpt_path = Path(ckpt_path) if ckpt_path else BEST_MODEL

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = sample_frames(video_path, num_frames=NUM_FRAMES)
    if frames is None:
        raise ValueError(f"Could not decode frames from: {video_path}")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((FRAME_SIZE, FRAME_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    frame_tensors = torch.stack([transform(f) for f in frames]).to(device)  # (T, 3, H, W)

    bundle = _get_models(ckpt_path, device=device)
    backbone = bundle["backbone"]
    classifier = bundle["classifier"]

    with torch.no_grad():
        features = backbone(frame_tensors)              # (T, FEATURE_DIM)
        feat_seq = features.unsqueeze(0)                # (1, T, FEATURE_DIM)
        logits = classifier(feat_seq)                   # (1, 2)
        probs = torch.softmax(logits, dim=-1).squeeze(0)# (2,)

    prob_fake = float(probs[1].item())
    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())

    return {
        "video_path": str(video_path),
        "label": pred_label,
        "label_idx": pred_idx,
        "probability_ai_generated": round(prob_fake, 4),
        "confidence": round(confidence, 4),
        "is_ai_generated": (pred_idx == 1),
        "warning": EXPERIMENTAL_WARNING,
    }
