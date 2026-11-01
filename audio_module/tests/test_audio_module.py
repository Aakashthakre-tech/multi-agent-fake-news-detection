"""
test_audio_module.py
====================
Unit test suite for TruthLensAI Audio Module.

Tests cover:
- Audio format validation (.wav, .mp3, .m4a, .flac, .aac, .ogg)
- Error handling for missing files, unsupported formats, corrupt audio, and silent audio
- AI-generated speech detection logic and 3-tier threshold calibration
- Decoupled behavior: failure never silently defaults to HUMAN/REAL
- Claim extraction and filler filtering
- Multi-signal consistency synthesis
- End-to-end pipeline execution with mocked heavyweight inference
"""

import math
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from audio_module.audio_loader import (
    AudioFileNotFoundError,
    CorruptAudioError,
    SilentAudioError,
    UnsupportedAudioFormatError,
    load_audio,
    validate_audio_path,
)
from audio_module.audio_pipeline import verify_audio
from audio_module.claim_extractor import extract_claims_from_transcript
from audio_module.consistency import analyze_audio_consistency
from audio_module.synthetic_detector import predict_synthetic_speech


# ====================================================================
# FIXTURES & HELPERS
# ====================================================================

@pytest.fixture
def temp_wav_file(tmp_path: Path) -> Path:
    """Generate a valid, non-silent 1-second 16kHz sine wave WAV file."""
    wav_path = tmp_path / "valid_sample.wav"
    sr = 16000
    duration = 1.0
    freq = 440.0
    num_samples = int(sr * duration)
    
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000.0 * math.sin(2.0 * math.pi * freq * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
        
    return wav_path


@pytest.fixture
def silent_wav_file(tmp_path: Path) -> Path:
    """Generate a 1-second 16kHz silent WAV file."""
    wav_path = tmp_path / "silent_sample.wav"
    sr = 16000
    duration = 1.0
    num_samples = int(sr * duration)
    
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray(num_samples * 2)  # All zeros
        wf.writeframes(frames)
        
    return wav_path


# ====================================================================
# 1. AUDIO LOADER & VALIDATION TESTS
# ====================================================================

def test_validate_audio_path_valid(temp_wav_file: Path):
    meta = validate_audio_path(temp_wav_file)
    assert meta["format"] == ".wav"
    assert meta["size_bytes"] > 0
    assert meta["exists"] is True


def test_validate_audio_path_nonexistent(tmp_path: Path):
    nonexistent = tmp_path / "ghost_audio.mp3"
    with pytest.raises(AudioFileNotFoundError):
        validate_audio_path(nonexistent)


def test_validate_audio_path_unsupported(tmp_path: Path):
    unsupported = tmp_path / "sample.txt"
    unsupported.write_text("not an audio file")
    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_path(unsupported)


def test_load_audio_valid(temp_wav_file: Path):
    samples, meta = load_audio(temp_wav_file)
    assert meta["sample_rate"] == 16000
    assert 0.95 <= meta["duration_seconds"] <= 1.05
    assert meta["rms"] > 0.05
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32


def test_load_audio_silent_fails(silent_wav_file: Path):
    with pytest.raises(SilentAudioError) as exc_info:
        load_audio(silent_wav_file)
    assert "silent" in str(exc_info.value).lower()


def test_load_audio_corrupt_fails(tmp_path: Path):
    corrupt = tmp_path / "broken.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbled_data_stream")
    with pytest.raises(CorruptAudioError):
        load_audio(corrupt)


# ====================================================================
# 2. SYNTHETIC SPEECH DETECTOR TESTS (CALIBRATION & FAILSAFE)
# ====================================================================

def test_synthetic_detector_ai_generated():
    """Mock model output predicting high fake probability (class 0)."""
    # Logits: class 0 (fake) = 3.0, class 1 (real) = -1.0 -> softmax ~0.98 fake
    mock_logits = torch.tensor([[3.0, -1.0]])
    mock_model = MagicMock()
    mock_model.return_value.logits = mock_logits
    
    mock_extractor = MagicMock()
    mock_extractor.return_value.input_values = torch.zeros(1, 16000)

    with patch("audio_module.synthetic_detector.get_synthetic_detector", return_value=(mock_model, mock_extractor, "cpu")):
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = predict_synthetic_speech(dummy_audio, sample_rate=16000)

        assert result["label"] == "AI_GENERATED"
        assert result["is_ai_generated"] is True
        assert result["probability_synthetic"] >= 0.65
        assert result["available"] is True


def test_synthetic_detector_real():
    """Mock model output predicting high real probability (class 1)."""
    # Logits: class 0 (fake) = -2.0, class 1 (real) = 2.5 -> softmax ~0.99 real
    mock_logits = torch.tensor([[-2.0, 2.5]])
    mock_model = MagicMock()
    mock_model.return_value.logits = mock_logits
    
    mock_extractor = MagicMock()
    mock_extractor.return_value.input_values = torch.zeros(1, 16000)

    with patch("audio_module.synthetic_detector.get_synthetic_detector", return_value=(mock_model, mock_extractor, "cpu")):
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = predict_synthetic_speech(dummy_audio, sample_rate=16000)

        assert result["label"] == "REAL"
        assert result["is_ai_generated"] is False
        assert result["probability_synthetic"] <= 0.35
        assert result["available"] is True


def test_synthetic_detector_uncertain_boundary():
    """Softmax between 0.35 and 0.65 must produce UNCERTAIN, never default to REAL."""
    # Logits: class 0 = 0.05, class 1 = -0.05 -> softmax ~0.52 fake
    mock_logits = torch.tensor([[0.05, -0.05]])
    mock_model = MagicMock()
    mock_model.return_value.logits = mock_logits
    
    mock_extractor = MagicMock()
    mock_extractor.return_value.input_values = torch.zeros(1, 16000)

    with patch("audio_module.synthetic_detector.get_synthetic_detector", return_value=(mock_model, mock_extractor, "cpu")):
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = predict_synthetic_speech(dummy_audio, sample_rate=16000)

        assert result["label"] == "UNCERTAIN"
        assert result["is_ai_generated"] is None
        assert 0.35 < result["probability_synthetic"] < 0.65


def test_synthetic_detector_failure_does_not_default_to_human():
    """Condition 6: Failure MUST return UNAVAILABLE or UNCERTAIN; never silently default to REAL/HUMAN."""
    with patch("audio_module.synthetic_detector.get_synthetic_detector", side_effect=RuntimeError("Weights missing")):
        dummy_audio = np.random.randn(16000).astype(np.float32)
        result = predict_synthetic_speech(dummy_audio, sample_rate=16000)

        assert result["available"] is False
        assert result["label"] in ("UNAVAILABLE", "UNCERTAIN")
        assert result["is_ai_generated"] is None
        assert result["label"] != "REAL"


# ====================================================================
# 3. CLAIM EXTRACTION TESTS
# ====================================================================

def test_claim_extractor_filters_filler_and_selects_fact():
    transcript = {
        "text": "Hello everyone, thank you for watching. The Eiffel Tower is located in Paris, France.",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello everyone, thank you for watching."},
            {"start": 2.1, "end": 5.0, "text": "The Eiffel Tower is located in Paris, France."},
        ],
    }
    extracted = extract_claims_from_transcript(transcript, max_claims=1)
    assert extracted["primary_claim"] is not None
    assert "Eiffel Tower" in extracted["primary_claim"] or "Paris" in extracted["primary_claim"]
    assert len(extracted["selected_claims"]) == 1


def test_claim_extractor_empty_transcript():
    transcript = {"text": "", "segments": []}
    extracted = extract_claims_from_transcript(transcript)
    assert extracted["primary_claim"] is None
    assert extracted["selected_claims"] == []


# ====================================================================
# 4. MULTI-SIGNAL CONSISTENCY TESTS
# ====================================================================

def test_consistency_ai_narrated_fact():
    synthetic_result = {"label": "AI_GENERATED", "available": True, "confidence": 0.95}
    factual_result = {"verdict": "TRUE", "confidence": 0.92, "claim": "Fact statement"}
    consistency = analyze_audio_consistency(synthetic_result, factual_result, has_claims=True)

    assert consistency["category"] == "AI_NARRATED_VERIFIED_FACT"
    assert consistency["risk_level"] == "LOW"


def test_consistency_synthetic_disinformation():
    synthetic_result = {"label": "AI_GENERATED", "available": True, "confidence": 0.98}
    factual_result = {"verdict": "FALSE", "confidence": 0.99, "claim": "Fabricated rumor"}
    consistency = analyze_audio_consistency(synthetic_result, factual_result, has_claims=True)

    assert consistency["category"] == "SYNTHETIC_DISINFORMATION"
    assert consistency["risk_level"] == "CRITICAL"


def test_consistency_human_disinformation():
    synthetic_result = {"label": "REAL", "available": True, "confidence": 0.90}
    factual_result = {"verdict": "FALSE", "confidence": 0.95, "claim": "Fake speech"}
    consistency = analyze_audio_consistency(synthetic_result, factual_result, has_claims=True)

    assert consistency["category"] == "HUMAN_DISINFORMATION"
    assert consistency["risk_level"] == "HIGH"


def test_consistency_uncertain_provenance():
    synthetic_result = {"label": "UNCERTAIN", "available": True, "confidence": 0.5}
    factual_result = {"verdict": "TRUE", "confidence": 0.85, "claim": "Valid fact"}
    consistency = analyze_audio_consistency(synthetic_result, factual_result, has_claims=True)

    assert consistency["category"] == "UNCERTAIN_PROVENANCE_VERIFIED_FACT"
    assert consistency["risk_level"] == "LOW"


# ====================================================================
# 5. PIPELINE INTEGRATION TESTS
# ====================================================================

def test_verify_audio_corrupt_file_returns_error_schema(tmp_path: Path):
    bad_file = tmp_path / "bad.wav"
    bad_file.write_bytes(b"invalid_audio_bytes")
    
    result = verify_audio(bad_file, verify_claims=False)
    assert result["status"] == "error"
    assert result["synthetic_speech_detection"]["label"] == "UNAVAILABLE"
    assert result["synthetic_speech_detection"]["available"] is False
    assert result["error"] is not None


def test_verify_audio_end_to_end_mocked(temp_wav_file: Path):
    mock_synthetic = {
        "is_ai_generated": False,
        "label": "REAL",
        "confidence": 0.88,
        "probability_synthetic": 0.12,
        "probability_real": 0.88,
        "thresholds_applied": {"fake_threshold_high": 0.65, "fake_threshold_low": 0.35},
        "model_name": "Wav2Vec2 Deepfake Audio Detector",
        "model_id": "mo-thecreator/Deepfake-audio-detection",
        "model_license": "Apache-2.0",
        "available": True,
        "unverified_claims_note": "Note",
        "error": None,
    }
    mock_transcription = {
        "text": "The moon is Earth's only natural satellite.",
        "language": "en",
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "The moon is Earth's only natural satellite."}],
        "model": "base",
        "available": True,
    }

    with patch("audio_module.audio_pipeline.predict_synthetic_speech", return_value=mock_synthetic), \
         patch("audio_module.audio_pipeline.transcribe_audio", return_value=mock_transcription):
        
        result = verify_audio(temp_wav_file, verify_claims=False)

        assert result["status"] == "success"
        assert result["audio_metadata"]["format"] == ".wav"
        assert result["audio_metadata"]["duration_sec"] > 0
        assert result["synthetic_speech_detection"]["label"] == "REAL"
        assert result["transcription"]["text"] == "The moon is Earth's only natural satellite."
        assert result["claim_extraction"]["primary_claim"] is not None
        assert "consistency_analysis" in result
        assert result["consistency_analysis"]["category"] in (
            "HUMAN_AUTHENTIC_FACT",
            "HUMAN_SPEECH_UNVERIFIED_CONTENT",
        )
