"""
tests/test_audio_real_sample.py
===============================
Dedicated integration test for real audio pipeline inference on an existing WAV sample.
Isolated into its own test module to avoid CUDA / VRAM memory contention with visual models.
"""

from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from multimodal_router import verify_audio_file


def test_verify_audio_real_sample():
    sample_path = _PROJECT_ROOT / "audio_module" / "validation_samples" / "human_01_reuters_news.wav"
    if not sample_path.exists():
        pytest.skip(f"Validation audio sample not found: {sample_path}")

    res = verify_audio_file(sample_path, verify_claims=False)
    assert res["status"] == "success"
    assert res["modality"] == "audio"
    assert res["synthetic_media_detection"] is not None
    assert res["synthetic_media_detection"]["available"] is True
    assert res["synthetic_media_detection"]["signal"] in ("REAL", "AI_GENERATED", "UNCERTAIN")
    assert isinstance(res["synthetic_media_detection"]["confidence"], (float, int))
