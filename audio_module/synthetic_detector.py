"""
synthetic_detector.py
=====================
Pretrained Wav2Vec2-based synthetic speech (voice deepfake) detector.

Model Card & Provenance:
- Model ID: mo-thecreator/Deepfake-audio-detection
- Base: facebook/wav2vec2-base
- License: Apache-2.0
- Labels: 0 = fake (AI_GENERATED), 1 = real (REAL)
- Validation: 98.82% hold-out accuracy reported on creator training corpus.
  NOTE: In-the-wild generalization across novel vocoders (ElevenLabs, XTTS, etc.)
  remains unverified by independent benchmarks.

Safety & Calibration:
- Applies three-tier thresholding (0.35 / 0.65) to produce REAL, AI_GENERATED,
  or UNCERTAIN.
- In case of failure, error, or unprocessable audio, returns status UNAVAILABLE
  or UNCERTAIN with available=False. It NEVER silently defaults to REAL or HUMAN.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from audio_module.config import (
    PROB_FAKE_THRESHOLD_HIGH,
    PROB_FAKE_THRESHOLD_LOW,
    SYNTHETIC_SPEECH_LICENSE,
    SYNTHETIC_SPEECH_MODEL_ID,
    SYNTHETIC_SPEECH_MODEL_NAME,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger("TruthLensAI.AudioModule.SyntheticDetector")

# Global singleton cache for model and feature extractor
_GLOBAL_MODEL: Optional[Any] = None
_GLOBAL_EXTRACTOR: Optional[Any] = None
_GLOBAL_DEVICE: Optional[str] = None


def get_synthetic_detector(
    device: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """
    Retrieve or initialize the cached Wav2Vec2 synthetic speech detector and extractor.

    Returns:
        Tuple of (model, feature_extractor, device_str).
    """
    global _GLOBAL_MODEL, _GLOBAL_EXTRACTOR, _GLOBAL_DEVICE

    if _GLOBAL_MODEL is not None and _GLOBAL_EXTRACTOR is not None:
        return _GLOBAL_MODEL, _GLOBAL_EXTRACTOR, _GLOBAL_DEVICE or "cpu"

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading synthetic speech detector '{SYNTHETIC_SPEECH_MODEL_ID}' onto {device}...")
    try:
        extractor = AutoFeatureExtractor.from_pretrained(SYNTHETIC_SPEECH_MODEL_ID)
        model = AutoModelForAudioClassification.from_pretrained(SYNTHETIC_SPEECH_MODEL_ID)
        model.to(device)
        model.eval()

        _GLOBAL_MODEL = model
        _GLOBAL_EXTRACTOR = extractor
        _GLOBAL_DEVICE = device
        logger.info("Synthetic speech detector loaded successfully.")
        return _GLOBAL_MODEL, _GLOBAL_EXTRACTOR, _GLOBAL_DEVICE
    except Exception as e:
        logger.error(f"Failed to load synthetic speech detector: {e}")
        raise


def predict_synthetic_speech(
    audio_data: Union[np.ndarray, torch.Tensor],
    sample_rate: int = TARGET_SAMPLE_RATE,
    device: Optional[str] = None,
    chunk_duration_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Analyze speech audio to detect whether it was AI-synthesized or spoken by a human.

    Args:
        audio_data: 1D numpy array or torch Tensor of audio samples at 16 kHz.
        sample_rate: Audio sampling rate (expected 16000 Hz).
        device: 'cuda', 'cpu', or None for auto-detection.
        chunk_duration_sec: Window length in seconds for chunked inference on long audio
                            to avoid VRAM exhaustion on 4 GB GPUs.

    Returns:
        Structured dictionary conforming to TruthLensAI provenance specification.
    """
    unverified_claims_note = (
        "Hold-out validation 98.82% reported on creator dataset; cross-generator "
        "and in-the-wild accuracy across novel vocoders remains unverified by third-party benchmark."
    )

    # Base response for failure / unavailable cases
    unavailable_result = {
        "is_ai_generated": None,
        "label": "UNAVAILABLE",
        "confidence": 0.0,
        "probability_synthetic": 0.0,
        "probability_real": 0.0,
        "thresholds_applied": {
            "fake_threshold_high": PROB_FAKE_THRESHOLD_HIGH,
            "fake_threshold_low": PROB_FAKE_THRESHOLD_LOW,
        },
        "model_name": SYNTHETIC_SPEECH_MODEL_NAME,
        "model_id": SYNTHETIC_SPEECH_MODEL_ID,
        "model_license": SYNTHETIC_SPEECH_LICENSE,
        "available": False,
        "unverified_claims_note": unverified_claims_note,
        "error": None,
    }

    # Normalize audio array
    if isinstance(audio_data, torch.Tensor):
        audio_arr = audio_data.detach().cpu().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_arr = audio_data
    else:
        unavailable_result["error"] = f"Unsupported audio_data type: {type(audio_data)}"
        unavailable_result["label"] = "UNAVAILABLE"
        return unavailable_result

    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=0)  # mixdown to mono
    audio_arr = audio_arr.astype(np.float32)

    total_samples = len(audio_arr)
    if total_samples < int(sample_rate * 0.2):  # less than 0.2s
        result = dict(unavailable_result)
        result["label"] = "UNCERTAIN"
        result["available"] = True
        result["error"] = "Audio duration too short (< 0.2s) for reliable acoustic spoofing detection."
        return result

    # Load model
    try:
        model, extractor, actual_device = get_synthetic_detector(device=device)
    except Exception as e:
        logger.warning(f"Synthetic detector unavailable: {e}")
        unavailable_result["error"] = f"Failed to initialize model: {str(e)}"
        unavailable_result["label"] = "UNAVAILABLE"
        return unavailable_result

    # Chunking strategy for long audio to avoid quadratic self-attention OOM on 4 GB VRAM
    chunk_samples = int(chunk_duration_sec * sample_rate)
    
    if total_samples <= chunk_samples:
        chunks = [audio_arr]
    else:
        # Take overlapping or sliding chunks across the audio
        stride_samples = int(chunk_samples * 0.75)
        chunks = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio_arr[start:end]
            if len(chunk) >= int(sample_rate * 0.5):
                chunks.append(chunk)
            start += stride_samples
            if len(chunks) >= 8:  # cap max chunks at 8 (approx 60-80s) for responsiveness
                break

    all_prob_fake = []
    all_prob_real = []

    try:
        with torch.no_grad():
            for chunk in chunks:
                inputs = extractor(
                    chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(actual_device)
                
                try:
                    logits = model(input_values).logits
                except torch.cuda.OutOfMemoryError:
                    logger.warning("CUDA OOM during synthetic speech inference; falling back to CPU.")
                    torch.cuda.empty_cache()
                    model_cpu = model.cpu()
                    logits = model_cpu(input_values.cpu()).logits
                
                # Softmax probabilities
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Model labels: 0 -> fake (synthetic), 1 -> real (bonafide)
                prob_fake = float(probs[0])
                prob_real = float(probs[1])
                
                all_prob_fake.append(prob_fake)
                all_prob_real.append(prob_real)

        # Average probabilities across chunks
        mean_prob_fake = float(np.mean(all_prob_fake))
        mean_prob_real = float(np.mean(all_prob_real))

        # Three-tier decision rule
        if mean_prob_fake >= PROB_FAKE_THRESHOLD_HIGH:
            decision_label = "AI_GENERATED"
            is_ai = True
            confidence = mean_prob_fake
        elif mean_prob_fake <= PROB_FAKE_THRESHOLD_LOW:
            decision_label = "REAL"
            is_ai = False
            confidence = mean_prob_real
        else:
            decision_label = "UNCERTAIN"
            is_ai = None
            # In uncertain band, confidence reflects distance from midpoint 0.5
            confidence = round(1.0 - (abs(mean_prob_fake - 0.5) * 2), 4)

        return {
            "is_ai_generated": is_ai,
            "label": decision_label,
            "confidence": round(confidence, 4),
            "probability_synthetic": round(mean_prob_fake, 4),
            "probability_real": round(mean_prob_real, 4),
            "thresholds_applied": {
                "fake_threshold_high": PROB_FAKE_THRESHOLD_HIGH,
                "fake_threshold_low": PROB_FAKE_THRESHOLD_LOW,
            },
            "model_name": SYNTHETIC_SPEECH_MODEL_NAME,
            "model_id": SYNTHETIC_SPEECH_MODEL_ID,
            "model_license": SYNTHETIC_SPEECH_LICENSE,
            "available": True,
            "unverified_claims_note": unverified_claims_note,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Inference error in synthetic speech detector: {e}")
        err_res = dict(unavailable_result)
        err_res["label"] = "UNCERTAIN"
        err_res["available"] = False
        err_res["error"] = str(e)
        return err_res
