"""
test_ai_generated_detector.py
==============================
Unit tests for the dedicated AI-generated video detection module.
"""

from pathlib import Path
import pytest
import torch

from video_module.ai_generated_detector.config import (
    BEST_MODEL,
    NUM_CLASSES,
    FEATURE_DIM,
    NUM_FRAMES,
    CACHE_TEST,
    TEST_CSV,
)
from video_module.ai_generated_detector.model import BiGRUHead, build_model
from video_module.ai_generated_detector.dataset import FeatureDataset
from video_module.ai_generated_detector.detector import predict_ai_generated


def test_bigru_head_forward():
    model = build_model()
    x = torch.randn(4, NUM_FRAMES, FEATURE_DIM)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_feature_dataset_loading():
    if not TEST_CSV.exists() or not CACHE_TEST.exists():
        pytest.skip("Test CSV or feature cache not generated yet")

    ds = FeatureDataset(TEST_CSV, CACHE_TEST, num_frames=NUM_FRAMES, augment=False)
    assert len(ds) > 0
    feat, label = ds[0]
    assert feat.shape == (NUM_FRAMES, FEATURE_DIM)
    assert label in (0, 1)


def test_predict_ai_generated_interface():
    real_sample = next(Path("video_module/ai_generated_data/real").glob("*.avi"), None)
    if not real_sample or not BEST_MODEL.exists():
        pytest.skip("No real video or checkpoint available")

    res = predict_ai_generated(real_sample)
    assert isinstance(res, dict)
    assert "label" in res
    assert res["label"] in ("REAL", "AI_GENERATED")
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probability_ai_generated" in res
    assert "is_ai_generated" in res
    assert isinstance(res["is_ai_generated"], bool)
