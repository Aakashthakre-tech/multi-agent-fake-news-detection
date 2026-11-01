"""
videomae_adapter.py
===================
TruthLensAI VideoMAE visual classifier adapter.

Integrates the pretrained VideoMAE-base foundation model with the trained
vmae_exp0_linear classifier head (LayerNorm + Dropout(0.3) + Linear(768, 2)).

Design Principles:
1. Resilient & Standalone: Never raises uncaught exceptions to caller.
2. Exact Reproduction: Exactly matches the 16-frame uniform OpenCV seek sampling,
   AutoImageProcessor preprocessing, VideoMAE encoder feature extraction,
   tubelet mean pooling (768-d), and linear classification used during training/testing.
3. Offline & Safe: Loads local weights directly without network access.
4. Transparent Provenance: Reports model name, version, warning disclaimer,
   class labels, confidence, and per-class probabilities.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoImageProcessor, VideoMAEModel

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLASSIFIER_DIR = Path(__file__).resolve().parent / "baseline_classifier"

# Checkpoint paths
DEFAULT_VIDEOMAE_CHECKPOINT = _CLASSIFIER_DIR / "checkpoints" / "vmae_exp0_linear.pt"
MODEL_ID = "MCG-NJU/videomae-base"

MODEL_NAME = "TruthLensAI Video Classifier (VideoMAE-base + Linear Probe)"
MODEL_VERSION = "1.0-videomae"
EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL SIGNAL: This visual classifier achieved 61.90% test accuracy (0.6047 macro F1, 0.6759 ROC-AUC, "
    "70.83% fake recall) on the Fake Video Corpus benchmark. Visual classification must be treated as a "
    "supporting heuristic only and never as a definitive verdict."
)

CLASS_MAP = {"real": 0, "fake": 1}
INV_CLASS_MAP = {0: "real", 1: "fake"}

# Global caches
_CACHED_BACKBONE = None
_CACHED_PROCESSOR = None
_CACHED_HEAD = None
_CACHED_CONFIG = None
_DEVICE = None


def get_compute_device() -> torch.device:
    """Detect and return compute device (CUDA if available, else CPU)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


class VideoMAEHead(nn.Module):
    """
    VideoMAE classifier head matching vmae_exp0_linear:
    LayerNorm(768) + Dropout(dropout) + Linear(768, num_classes)
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        norm_type: str = "layernorm",
        architecture: str = "linear",
    ):
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
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.input_norm(x))


def sample_frames_cv2(video_path: Path, num_frames: int = 16) -> np.ndarray:
    """
    Sample exactly `num_frames` evenly spaced RGB frames using OpenCV seek,
    matching extract_videomae_features.py and extract_test_and_evaluate.py.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path.name}")
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
                raise ValueError(f"No frames could be decoded from {video_path.name}")
            while len(frames) < num_frames:
                frames.append(frames[-1])
            return np.stack(frames[:num_frames])

        idxs = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
        sampled = []
        last = None
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
                    raise ValueError(f"Cannot decode any frame from {video_path.name}")
        return np.stack(sampled)
    finally:
        cap.release()


def load_videomae_classifier(
    checkpoint_path: Optional[Path | str] = None,
) -> Tuple[AutoImageProcessor, VideoMAEModel, VideoMAEHead, Dict[str, Any]]:
    """
    Safely load the VideoMAE processor, backbone model, and trained classifier head.
    Caches loaded components in memory for subsequent inference calls.
    """
    global _CACHED_PROCESSOR, _CACHED_BACKBONE, _CACHED_HEAD, _CACHED_CONFIG

    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_VIDEOMAE_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VideoMAE checkpoint not found at: {ckpt_path}")

    # Return cached instances if default requested and already loaded
    if (
        checkpoint_path is None
        and _CACHED_PROCESSOR is not None
        and _CACHED_BACKBONE is not None
        and _CACHED_HEAD is not None
    ):
        return _CACHED_PROCESSOR, _CACHED_BACKBONE, _CACHED_HEAD, _CACHED_CONFIG

    device = get_compute_device()

    # 1. Locate local safetensors checkpoint for VideoMAE-base to prevent network stalls
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / "hub" / f"models--{MODEL_ID.replace('/', '--')}"
    st_candidates = list(cache_dir.rglob("model.safetensors")) if cache_dir.exists() else []

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    backbone = VideoMAEModel.from_pretrained(MODEL_ID, local_files_only=True)

    if st_candidates:
        st_path = max(st_candidates, key=lambda p: p.stat().st_size)
        raw_sd = load_file(str(st_path))
        sd = backbone.state_dict()
        for i in range(backbone.config.num_hidden_layers):
            q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
            v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
            if q_key in raw_sd:
                sd[f"encoder.layer.{i}.attention.attention.query.bias"] = raw_sd[q_key]
            if v_key in raw_sd:
                sd[f"encoder.layer.{i}.attention.attention.value.bias"] = raw_sd[v_key]
            sd[f"encoder.layer.{i}.attention.attention.key.bias"] = torch.zeros(backbone.config.hidden_size)

        backbone.load_state_dict(sd, strict=False)
        del raw_sd, sd

    backbone.to(device)
    backbone.eval()

    # 2. Load trained head checkpoint
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt_data.get("config", {})

    head = VideoMAEHead(
        in_dim=768,
        hidden_dim=config.get("hidden_dim", 64),
        num_classes=2,
        dropout=config.get("dropout", 0.3),
        norm_type=config.get("norm_type", "layernorm"),
        architecture=config.get("architecture", "linear"),
    )
    head.load_state_dict(ckpt_data["model_state_dict"])
    head.to(device)
    head.eval()

    _CACHED_PROCESSOR = processor
    _CACHED_BACKBONE = backbone
    _CACHED_HEAD = head
    _CACHED_CONFIG = config

    return _CACHED_PROCESSOR, _CACHED_BACKBONE, _CACHED_HEAD, _CACHED_CONFIG


def predict_videomae(
    video_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Run the VideoMAE visual classifier on a single video file.

    Returns a standardized dictionary matching classifier_adapter format:
    - available (bool): Whether prediction succeeded.
    - model_name (str): Identifier.
    - model_version (str): Model version.
    - predicted_label (str | None): 'fake' or 'real'.
    - confidence (float | None): Confidence score (0.0 to 1.0).
    - probabilities (dict | None): Raw class probabilities {'real': p0, 'fake': p1}.
    - experimental_warning (str): Disclaimer regarding benchmark accuracy.
    - error (str | None): Error message if unavailable.
    """
    v_path = Path(video_path)

    # 1. Input Path Validation
    if not v_path.exists():
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Video file not found: {v_path.name}",
        }

    if not v_path.is_file():
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Expected a file, but got directory: {v_path.name}",
        }

    # 2. Model Loading
    try:
        processor, backbone, head, config = load_videomae_classifier(checkpoint_path)
    except Exception as exc:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Failed to load VideoMAE classifier: {str(exc)}",
        }

    # 3. Frame Sampling & Preprocessing
    try:
        frames_np = sample_frames_cv2(v_path, num_frames=16)
        inputs = processor(list(frames_np), return_tensors="pt")
    except Exception as exc:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Frame extraction/preprocessing failed: {str(exc)}",
        }

    # 4. Feature Extraction & Classifier Forward Pass
    device = get_compute_device()
    try:
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            out = backbone(pixel_values)
            # Tubelet token mean pooling: (1, T*H*W, 768) -> (1, 768)
            emb = out.last_hidden_state.mean(dim=1)
            logits = head(emb)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1).item())

        pred_label = INV_CLASS_MAP.get(pred_idx, "unknown")
        confidence = float(probs[pred_idx])
        prob_dict = {
            INV_CLASS_MAP.get(idx, str(idx)): round(float(p), 4)
            for idx, p in enumerate(probs)
        }

        return {
            "available": True,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": None,
        }

    except Exception as exc:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Inference execution failed: {str(exc)}",
        }
