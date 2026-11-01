"""
classifier_adapter.py
=====================
Additive adapter providing safe, structured access to the trained TruthLensAI
fake-vs-real video classifier (EfficientNet-B0 + Temporal Attention Pooling).

Design Principles & Safety Rules:
1. Standalone & Resilient: Never raises uncaught exceptions to caller.
   Returns a structured dictionary with availability status and error messages.
2. Provenance & Transparency: Explicitly includes model metadata and an
   experimental warning acknowledging the 52.38% test accuracy benchmark.
3. Strict Local Model Loading: Loads model weights solely from verified local checkpoints.
4. Privacy: Does not expose internal file system details, credentials, or sensitive paths.
5. Reusable: Reuses model.py and dataset.py from video_module/baseline_classifier/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Module paths
_CLASSIFIER_DIR = Path(__file__).resolve().parent / "baseline_classifier"
OPTIMIZED_CHECKPOINT_PATH = _CLASSIFIER_DIR / "checkpoints" / "exp6_compact_proj64.pt"
BASELINE_CHECKPOINT_PATH = _CLASSIFIER_DIR / "best_model.pt"
VIDEOMAE_CHECKPOINT_PATH = _CLASSIFIER_DIR / "checkpoints" / "vmae_exp0_linear.pt"
DEFAULT_CHECKPOINT_PATH = OPTIMIZED_CHECKPOINT_PATH if OPTIMIZED_CHECKPOINT_PATH.exists() else BASELINE_CHECKPOINT_PATH

MODEL_NAME = "TruthLensAI Video Classifier (EfficientNet-B0 + Compact Projected Dynamics)"
MODEL_VERSION = "2.0-optimized"
EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL SIGNAL: This visual classifier achieved 54.76% test accuracy (0.5411 macro F1, 0.6042 ROC-AUC) "
    "on the Fake Video Corpus benchmark. Many real-world fake videos use authentic footage in "
    "a false context (misattribution). Visual classification must be treated as a "
    "supporting heuristic only and never as a definitive verdict."
)

VIDEOMAE_MODEL_NAME = "TruthLensAI Video Classifier (VideoMAE-base + Linear Probe)"
VIDEOMAE_MODEL_VERSION = "1.0-videomae"
VIDEOMAE_EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL SIGNAL: This visual classifier achieved 61.90% test accuracy (0.6047 macro F1, 0.6759 ROC-AUC, "
    "70.83% fake recall) on the Fake Video Corpus benchmark. Visual classification must be treated as a "
    "supporting heuristic only and never as a definitive verdict."
)

# Global cached model instances to prevent reloading overhead
_CACHED_MODEL = None
_CACHED_EXTRACTOR = None
_CACHED_CONFIG = None
_CACHED_CLASS_MAP = None
_DEVICE = None


def get_compute_device() -> torch.device:
    """Detect and return compute device (CUDA if available, else CPU)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def load_classifier(checkpoint_path: Optional[Path] = None) -> Tuple[Any, Any, Dict[str, Any], Dict[str, int]]:
    """
    Safely load the feature extractor and classifier model.
    Cached in-memory after first successful load.
    """
    global _CACHED_MODEL, _CACHED_EXTRACTOR, _CACHED_CONFIG, _CACHED_CLASS_MAP

    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found at: {ckpt_path.name}")

    # Return cached default model if not requesting a different checkpoint
    if checkpoint_path is None and _CACHED_MODEL is not None and _CACHED_EXTRACTOR is not None:
        return _CACHED_MODEL, _CACHED_EXTRACTOR, _CACHED_CONFIG, _CACHED_CLASS_MAP

    # Add baseline_classifier directory to sys.path temporarily for imports
    if str(_CLASSIFIER_DIR) not in sys.path:
        sys.path.insert(0, str(_CLASSIFIER_DIR))

    from model import TemporalVideoClassifier, get_feature_extractor

    device = get_compute_device()

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {
        "num_frames": 12,
        "embed_dim": 1280,
        "hidden_dim": 256,
        "mlp_dim": 256,
        "dropout": 0.35,
    })
    class_map = checkpoint.get("class_map", {"real": 0, "fake": 1})

    # Initialize models
    extractor = get_feature_extractor(device)
    model = TemporalVideoClassifier(
        num_classes=len(class_map),
        embed_dim=config.get("embed_dim", 1280),
        hidden_dim=config.get("hidden_dim", 256),
        mlp_dim=config.get("mlp_dim", 256),
        dropout=config.get("dropout", 0.35),
        pooling_mode=config.get("pooling_mode", "hybrid"),
        proj_dim=config.get("proj_dim", None),
        norm_type=config.get("norm_type", "none"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _CACHED_MODEL = model
    _CACHED_EXTRACTOR = extractor
    _CACHED_CONFIG = config
    _CACHED_CLASS_MAP = class_map

    return _CACHED_MODEL, _CACHED_EXTRACTOR, _CACHED_CONFIG, _CACHED_CLASS_MAP


def predict_video(
    video_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    model_family: str = "efficientnet",
) -> Dict[str, Any]:
    """
    Run the visual classifier on a single video file.

    Args:
        video_path: Path to the target video file.
        checkpoint_path: Optional custom path to PyTorch checkpoint.
        model_family: Architecture family: 'videomae' (default, state-of-the-art)
                      or 'efficientnet' (baseline).

    Returns:
        Structured dictionary containing:
        - available (bool): Whether prediction succeeded.
        - model_name (str): Model identifier.
        - model_version (str): Model version.
        - predicted_label (str | None): 'fake' or 'real' if available.
        - confidence (float | None): Confidence score (0.0 to 1.0).
        - probabilities (dict | None): Raw class probabilities.
        - experimental_warning (str): Disclaimer regarding test accuracy.
        - error (str | None): Description of failure if unavailable.
    """
    # Detect VideoMAE by parameter or checkpoint filename
    is_videomae = (
        model_family.lower() == "videomae"
        or (checkpoint_path and "vmae" in Path(checkpoint_path).name.lower())
    )

    if is_videomae and (checkpoint_path is None or "vmae" in Path(checkpoint_path).name.lower()):
        from video_module.videomae_adapter import predict_videomae
        return predict_videomae(video_path, checkpoint_path=checkpoint_path)
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
        model, extractor, config, class_map = load_classifier(
            Path(checkpoint_path) if checkpoint_path else None
        )
    except Exception as exc:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Failed to load classifier: {str(exc)}",
        }

    # 3. Frame Sampling & Preprocessing
    if str(_CLASSIFIER_DIR) not in sys.path:
        sys.path.insert(0, str(_CLASSIFIER_DIR))
    from dataset import sample_video_frames

    device = get_compute_device()
    try:
        frames_tensor = sample_video_frames(
            v_path,
            num_frames=config.get("num_frames", 12)
        )
        frames_tensor = frames_tensor.to(device)
    except Exception as exc:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": None,
            "confidence": None,
            "probabilities": None,
            "experimental_warning": EXPERIMENTAL_WARNING,
            "error": f"Frame extraction failed: {str(exc)}",
        }

    # 4. Feature Extraction & Classifier Forward Pass
    try:
        with torch.no_grad():
            frame_features = extractor(frames_tensor).unsqueeze(0)  # (1, T, 1280)
            logits = model(frame_features)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1).item())

        inv_class_map = {v: k for k, v in class_map.items()}
        pred_label = inv_class_map.get(pred_idx, "unknown")
        confidence = float(probs[pred_idx])

        prob_dict = {
            inv_class_map.get(idx, str(idx)): round(float(prob), 4)
            for idx, prob in enumerate(probs)
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
