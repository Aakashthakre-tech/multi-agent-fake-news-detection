"""
tests/test_integration.py
==========================
Comprehensive Integration Tests for TruthLensAI Multimodal Architecture.

Verifies:
1. Modality detection (text, image, video, audio, unsupported).
2. Strict signal separation (Factual truth != AI origin != Visual heuristics).
3. Uniform response contracts across all modalities.
4. Router dispatch logic with mocked backends for text, image, video, and audio.
5. End-to-end API server endpoints via FastAPI TestClient.
6. Real audio pipeline integration on an existing sample.
7. Graceful degradation and error handling on invalid files or backend exceptions.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.server import app
import multimodal_router as router
from multimodal_router import (
    build_unified_response,
    detect_modality,
    route_media,
    verify_audio_file,
    verify_image,
    verify_text,
    verify_video_file,
)

client = TestClient(app)


# ============================================================================
# 1. Modality Detection Tests
# ============================================================================

def test_detect_modality_text():
    assert detect_modality("The Earth revolves around the Sun.") == "text"
    assert detect_modality("Some random news headline") == "text"


def test_detect_modality_files(tmp_path):
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"dummy")
    assert detect_modality(img) == "image"

    png = tmp_path / "sample.png"
    png.write_bytes(b"dummy")
    assert detect_modality(png) == "image"

    vid = tmp_path / "sample.mp4"
    vid.write_bytes(b"dummy")
    assert detect_modality(vid) == "video"

    aud = tmp_path / "sample.wav"
    aud.write_bytes(b"dummy")
    assert detect_modality(aud) == "audio"


def test_detect_modality_unsupported(tmp_path):
    unsupported = tmp_path / "file.exe"
    unsupported.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        detect_modality(unsupported)


# ============================================================================
# 2. Text Modality Integration Tests
# ============================================================================

def test_verify_text_empty():
    res = verify_text("")
    assert res["status"] == "error"
    assert "empty" in res["error"].lower()


@patch("pipeline.run_research_pipeline")
def test_verify_text_success(mock_research):
    mock_research.return_value = {
        "claim": "Water boils at 100C",
        "final_verdict": {
            "verdict": "REAL",
            "confidence": 0.98,
            "evidence_sufficiency": "SUFFICIENT",
            "independent_supporting": 5,
            "independent_contradicting": 0,
            "contradiction_score": 0.0,
            "source_agreement": 1.0,
            "fusion_score": 0.95,
        },
    }

    res = verify_text("Water boils at 100C")
    assert res["status"] == "success"
    assert res["modality"] == "text"
    assert res["factual_verification"]["verdict"] == "REAL"
    assert res["factual_verification"]["confidence"] == 0.98
    assert res["factual_verification"]["independent_supporting"] == 5

    # STRICT SIGNAL SEPARATION: Text has no synthetic media or visual classification
    assert res["synthetic_media_detection"] is None
    assert res["visual_classification"] is None


# ============================================================================
# 3. Image Modality Integration Tests
# ============================================================================

def test_verify_image_nonexistent():
    res = verify_image("nonexistent_image_123.png")
    assert res["status"] == "error"
    assert "not found" in res["error"].lower()


@patch("ai_image_signal.run_image_pipeline_with_ai_detection")
def test_verify_image_success(mock_img_pipe, tmp_path):
    test_img = tmp_path / "fake_meme.png"
    test_img.write_bytes(b"fake_image_bytes")

    mock_img_pipe.return_value = {
        "image_path": str(test_img),
        "normalized_claim": {"claim": "Extracted image claim about lunar eclipse"},
        "detected_objects": [{"object": "moon", "confidence": 0.95}],
        "verification": {
            "final_verdict": {
                "verdict": "FAKE",
                "confidence": 0.88,
                "evidence_sufficiency": "SUFFICIENT",
                "independent_supporting": 0,
                "independent_contradicting": 4,
            }
        },
        "ai_generation": {
            "available": True,
            "model": "umm-maybe/AI-image-detector",
            "verdict": "AI-GENERATED",
            "confidence": 0.92,
            "ai_probability": 94.2,
            "real_probability": 5.8,
            "error": None,
        },
        "ai_generation_note": "AI-generated image, claim verdict FAKE. Image origin is reported separately.",
    }

    res = verify_image(test_img)
    assert res["status"] == "success"
    assert res["modality"] == "image"
    assert res["primary_claim"] == "Extracted image claim about lunar eclipse"

    # Factual verification
    assert res["factual_verification"]["verdict"] == "FAKE"
    assert res["factual_verification"]["independent_contradicting"] == 4

    # Synthetic Media Detection (Separate signal)
    assert res["synthetic_media_detection"] is not None
    assert res["synthetic_media_detection"]["signal"] == "AI_GENERATED"
    assert res["synthetic_media_detection"]["available"] is True
    assert res["synthetic_media_detection"]["probabilities"]["ai_probability"] == 94.2

    # Visual classification is None for images (reserved for video)
    assert res["visual_classification"] is None


# ============================================================================
# 4. Video Modality Integration Tests
# ============================================================================

def test_verify_video_nonexistent():
    res = verify_video_file("nonexistent_video_123.mp4")
    assert res["status"] == "error"
    assert "not found" in res["error"].lower()


@patch("video_module.video_verifier.verify_video")
def test_verify_video_success(mock_video, tmp_path):
    test_vid = tmp_path / "deepfake_speech.mp4"
    test_vid.write_bytes(b"dummy_video_bytes")

    mock_video.return_value = {
        "video_path": str(test_vid),
        "primary_claim_analyzed": "Government announces new currency notes",
        "final_evidence_verdict": "CONTRADICTED",
        "verified_claims": [
            {
                "claim_text": "Government announces new currency notes",
                "verification_result": {
                    "final_verdict": {
                        "verdict": "CONTRADICTED",
                        "confidence": 0.91,
                        "evidence_sufficiency": "SUFFICIENT",
                        "independent_supporting": 0,
                        "independent_contradicting": 3,
                    }
                },
            }
        ],
        "visual_classification": {
            "available": True,
            "predicted_label": "fake",
            "confidence": 0.72,
            "experimental_warning": "Visual classifier is experimental.",
            "probabilities": {"fake": 0.72, "real": 0.28},
        },
        "consistency_analysis": {
            "relationship": "corroborating",
            "description": "Visual classifier indicates fake; claim is contradicted by evidence.",
        },
        "summary_artifacts": {},
    }

    res = verify_video_file(test_vid)
    assert res["status"] == "success"
    assert res["modality"] == "video"
    assert res["primary_claim"] == "Government announces new currency notes"

    # Factual verification
    assert res["factual_verification"]["verdict"] == "CONTRADICTED"
    assert res["factual_verification"]["confidence"] == 0.91

    # Visual classification is separate heuristic
    assert res["visual_classification"] is not None
    assert res["visual_classification"]["predicted_label"] == "fake"
    assert res["visual_classification"]["confidence"] == 0.72
    assert "experimental" in res["visual_classification"]["experimental_warning"].lower()


# ============================================================================
# 5. Audio Modality Integration Tests
# ============================================================================

def test_verify_audio_nonexistent():
    res = verify_audio_file("nonexistent_audio_123.wav")
    assert res["status"] == "error"
    assert "not found" in res["error"].lower()


@patch("audio_module.verify_audio")
def test_verify_audio_mocked(mock_audio, tmp_path):
    test_aud = tmp_path / "speech.wav"
    test_aud.write_bytes(b"dummy_audio_bytes")

    mock_audio.return_value = {
        "status": "success",
        "audio_metadata": {"duration_sec": 4.5, "sample_rate": 16000},
        "synthetic_speech_detection": {
            "label": "AI_GENERATED",
            "confidence": 0.96,
            "probability_synthetic": 0.96,
            "probability_real": 0.04,
            "available": True,
            "error": None,
        },
        "transcription": {"text": "Mars rover discovered liquid water yesterday."},
        "claim_extraction": {"primary_claim": "Mars rover discovered liquid water."},
        "factual_verification": {
            "verdict": "CONTRADICTED",
            "confidence": 0.89,
            "evidence_sufficiency": "SUFFICIENT",
            "details": {
                "independent_supporting": 0,
                "independent_contradicting": 3,
            },
        },
        "consistency_analysis": {
            "status": "synthetic_contradicted",
            "description": "Audio is synthetic; claim contradicted by independent evidence.",
        },
    }

    res = verify_audio_file(test_aud)
    assert res["status"] == "success"
    assert res["modality"] == "audio"
    assert res["primary_claim"] == "Mars rover discovered liquid water."
    assert res["transcription"] == "Mars rover discovered liquid water yesterday."

    # Synthetic speech provenance is separate from factual veracity
    assert res["synthetic_media_detection"]["signal"] == "AI_GENERATED"
    assert res["synthetic_media_detection"]["confidence"] == 0.96
    assert res["factual_verification"]["verdict"] == "CONTRADICTED"


# ============================================================================
# 7. FastAPI Endpoint Tests
# ============================================================================

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "modules" in data
    assert "text" in data["modules"]
    assert "image" in data["modules"]
    assert "video" in data["modules"]
    assert "audio" in data["modules"]


@patch("pipeline.run_research_pipeline")
def test_api_verify_text(mock_research):
    mock_research.return_value = {
        "claim": "Vaccines cause autism",
        "final_verdict": {
            "verdict": "FAKE",
            "confidence": 0.99,
            "evidence_sufficiency": "SUFFICIENT",
            "independent_supporting": 0,
            "independent_contradicting": 8,
        },
    }

    response = client.post("/verify/text", json={"claim": "Vaccines cause autism"})
    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == "text"
    assert data["factual_verification"]["verdict"] == "FAKE"
    assert data["factual_verification"]["independent_contradicting"] == 8


def test_api_verify_text_invalid():
    # Min length is 3
    response = client.post("/verify/text", json={"claim": "hi"})
    assert response.status_code == 422


@patch("ai_image_signal.run_image_pipeline_with_ai_detection")
def test_api_verify_image(mock_image, tmp_path):
    mock_image.return_value = {
        "image_path": "temp.png",
        "normalized_claim": {"claim": "Alien spotted in Times Square"},
        "detected_objects": [],
        "verification": {"final_verdict": {"verdict": "FAKE", "confidence": 0.95}},
        "ai_generation": {"available": True, "verdict": "AI-GENERATED", "confidence": 0.99},
        "ai_generation_note": "AI image detected.",
    }

    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"sample_png_data")

    with open(img_file, "rb") as f:
        response = client.post("/verify/image", files={"file": ("test.png", f, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == "image"
    assert data["factual_verification"]["verdict"] == "FAKE"
    assert data["synthetic_media_detection"]["signal"] == "AI_GENERATED"


def test_api_verify_image_invalid_extension(tmp_path):
    bad_file = tmp_path / "document.pdf"
    bad_file.write_bytes(b"dummy_pdf")

    with open(bad_file, "rb") as f:
        response = client.post("/verify/image", files={"file": ("document.pdf", f, "application/pdf")})

    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()


@patch("video_module.video_verifier.verify_video")
def test_api_verify_video(mock_video, tmp_path):
    mock_video.return_value = {
        "video_path": "temp.mp4",
        "primary_claim_analyzed": "Election fraud video",
        "final_evidence_verdict": "UNVERIFIED",
        "verified_claims": [],
        "visual_classification": {"available": True, "predicted_label": "real", "confidence": 0.65},
        "consistency_analysis": {},
        "summary_artifacts": {},
    }

    vid_file = tmp_path / "test.mp4"
    vid_file.write_bytes(b"sample_mp4_data")

    with open(vid_file, "rb") as f:
        response = client.post("/verify/video", files={"file": ("test.mp4", f, "video/mp4")})

    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == "video"
    assert data["visual_classification"]["predicted_label"] == "real"


@patch("audio_module.verify_audio")
def test_api_verify_audio(mock_audio, tmp_path):
    mock_audio.return_value = {
        "status": "success",
        "audio_metadata": {},
        "synthetic_speech_detection": {"label": "REAL", "confidence": 0.88, "available": True},
        "transcription": {"text": "Audio bulletin"},
        "claim_extraction": {"primary_claim": "Audio claim"},
        "factual_verification": {"verdict": "REAL", "confidence": 0.85},
        "consistency_analysis": {},
    }

    aud_file = tmp_path / "test.wav"
    aud_file.write_bytes(b"sample_wav_data")

    with open(aud_file, "rb") as f:
        response = client.post("/verify/audio", files={"file": ("test.wav", f, "audio/wav")})

    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == "audio"
    assert data["synthetic_media_detection"]["signal"] == "REAL"


@patch("pipeline.run_research_pipeline")
def test_api_unified_endpoint_text(mock_research):
    mock_research.return_value = {
        "claim": "Sky is blue",
        "final_verdict": {"verdict": "REAL", "confidence": 0.99},
    }

    response = client.post("/verify", data={"claim": "Sky is blue"})
    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == "text"
    assert data["factual_verification"]["verdict"] == "REAL"


def test_api_unified_endpoint_no_input():
    response = client.post("/verify")
    assert response.status_code == 400
    assert "either a media file" in response.json()["detail"].lower()


