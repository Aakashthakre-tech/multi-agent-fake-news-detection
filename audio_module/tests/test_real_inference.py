"""
test_real_inference.py
======================
Integration tests running real pretrained model inference on actual audio samples.
Verifies real Wav2Vec2 weights, CUDA execution, CPU fallback, and structured outputs.
"""

from pathlib import Path
import pytest
import torch

from audio_module.audio_loader import load_audio
from audio_module.synthetic_detector import get_synthetic_detector, predict_synthetic_speech

VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation_samples"


@pytest.mark.integration
def test_real_model_loading_and_architecture():
    """Verify that real pretrained weights load and exhibit expected configuration."""
    model, extractor, device = get_synthetic_detector()
    assert model is not None
    assert extractor is not None
    assert model.config.num_labels == 2
    assert model.config.id2label[0] == "fake"
    assert model.config.id2label[1] == "real"
    assert device in ("cuda", "cpu")


@pytest.mark.integration
def test_real_inference_on_human_audio():
    """Verify inference pipeline on genuine human speech sample."""
    sample_path = VALIDATION_DIR / "human_01_reuters_news.wav"
    if not sample_path.exists():
        pytest.skip(f"Sample not found: {sample_path}")

    audio, meta = load_audio(sample_path)
    result = predict_synthetic_speech(audio, sample_rate=meta["sample_rate"])

    assert result["available"] is True
    assert result["error"] is None
    assert result["label"] in ("REAL", "AI_GENERATED", "UNCERTAIN")
    assert 0.0 <= result["probability_synthetic"] <= 1.0
    assert 0.0 <= result["probability_real"] <= 1.0
    assert abs((result["probability_synthetic"] + result["probability_real"]) - 1.0) < 1e-4


@pytest.mark.integration
def test_real_inference_on_synthetic_audio():
    """Verify inference pipeline on synthetic TTS speech sample."""
    sample_path = VALIDATION_DIR / "synth_01_sapi_david_intro.wav"
    if not sample_path.exists():
        pytest.skip(f"Sample not found: {sample_path}")

    audio, meta = load_audio(sample_path)
    result = predict_synthetic_speech(audio, sample_rate=meta["sample_rate"])

    assert result["available"] is True
    assert result["error"] is None
    assert result["label"] in ("REAL", "AI_GENERATED", "UNCERTAIN")
    assert 0.0 <= result["probability_synthetic"] <= 1.0
    assert 0.0 <= result["probability_real"] <= 1.0


@pytest.mark.integration
def test_real_inference_explicit_cpu_fallback():
    """Verify that forcing device='cpu' executes successfully without error."""
    sample_path = VALIDATION_DIR / "synth_01_sapi_david_intro.wav"
    if not sample_path.exists():
        pytest.skip(f"Sample not found: {sample_path}")

    audio, meta = load_audio(sample_path)
    result = predict_synthetic_speech(audio, sample_rate=meta["sample_rate"], device="cpu")

    assert result["available"] is True
    assert result["error"] is None
    assert result["label"] in ("REAL", "AI_GENERATED", "UNCERTAIN")
