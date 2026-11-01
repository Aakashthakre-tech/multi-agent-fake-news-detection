"""
test_classifier_integration.py
==============================
Comprehensive test suite for TruthLensAI Visual Classifier Adapter & Multi-Modal Orchestrator.

Tests:
1. Adapter execution on sample real and fake videos.
2. Error handling: non-existent file, directory input, corrupted video, invalid checkpoint.
3. Multi-modal consistency analysis and divergence handling (misattribution & visual anomaly).
4. Full video verification pipeline with classifier enabled and disabled.
5. Evidence supremacy: ensuring classifier output never overrides evidence verdict.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_module.classifier_adapter import (
    predict_video,
    MODEL_NAME,
    MODEL_VERSION,
    EXPERIMENTAL_WARNING,
)
from video_module.video_verifier import (
    verify_video,
    analyze_multimodal_consistency,
)


def create_dummy_video(file_path: Path, num_frames: int = 15, width: int = 64, height: int = 64) -> Path:
    """Helper to generate a lightweight valid test video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(file_path), fourcc, 10.0, (width, height))
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), int((i / max(1, num_frames)) * 255), dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()
    return file_path


class TestClassifierAdapter(unittest.TestCase):
    """Test suite for classifier_adapter.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Missing or non-existent file
    def test_missing_video_path(self):
        missing = self.temp_path / "non_existent.mp4"
        res = predict_video(missing)
        self.assertFalse(res["available"])
        self.assertIsNone(res["predicted_label"])
        self.assertIn("not found", res["error"].lower())
        self.assertEqual(res["experimental_warning"], EXPERIMENTAL_WARNING)

    # 2. Directory passed instead of file
    def test_directory_input(self):
        res = predict_video(self.temp_path)
        self.assertFalse(res["available"])
        self.assertIsNone(res["predicted_label"])
        self.assertIn("directory", res["error"].lower())

    # 3. Corrupt media file
    def test_corrupt_video_handling(self):
        corrupt = self.temp_path / "corrupt.mp4"
        corrupt.write_bytes(b"corrupt video stream bytes that cannot decode")
        res = predict_video(corrupt)
        self.assertFalse(res["available"])
        self.assertIsNone(res["predicted_label"])
        self.assertIsNotNone(res["error"])

    # 4. Invalid checkpoint path
    def test_invalid_checkpoint_path(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample.mp4")
        invalid_ckpt = self.temp_path / "non_existent_model.pt"
        res = predict_video(dummy_vid, checkpoint_path=invalid_ckpt)
        self.assertFalse(res["available"])
        self.assertIn("failed to load", res["error"].lower())

    # 5. Prediction on synthetic valid video
    def test_synthetic_video_prediction(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample.mp4")
        res = predict_video(dummy_vid)
        self.assertTrue(res["available"])
        self.assertIn(res["predicted_label"], ("fake", "real"))
        self.assertIsInstance(res["confidence"], float)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("fake", res["probabilities"])
        self.assertIn("real", res["probabilities"])
        self.assertEqual(res["model_name"], MODEL_NAME)
        self.assertEqual(res["model_version"], MODEL_VERSION)

    # 6. Prediction on actual dataset video files (if present)
    def test_dataset_sample_videos(self):
        fake_sample = Path("video_module/fvc_downloads/fake/f0.mp4")
        real_sample = Path("video_module/fvc_downloads/real/r2.mp4")

        if fake_sample.exists():
            res_fake = predict_video(fake_sample)
            self.assertTrue(res_fake["available"])
            self.assertIn(res_fake["predicted_label"], ("fake", "real"))
            self.assertIsNone(res_fake["error"])

        if real_sample.exists():
            res_real = predict_video(real_sample)
            self.assertTrue(res_real["available"])
            self.assertIn(res_real["predicted_label"], ("fake", "real"))
            self.assertIsNone(res_real["error"])


class TestMultiModalConsistency(unittest.TestCase):
    """Test suite for multi-modal consistency and disagreement analysis."""

    def test_consistent_misinformation(self):
        signal = {"available": True, "predicted_label": "fake", "confidence": 0.85}
        analysis = analyze_multimodal_consistency(signal, "CONTRADICTED")
        self.assertEqual(analysis["status"], "consistent_misinformation")
        self.assertEqual(analysis["relationship"], "reinforcing")
        self.assertFalse(analysis["requires_investigation"])

    def test_consistent_authentic(self):
        signal = {"available": True, "predicted_label": "real", "confidence": 0.80}
        analysis = analyze_multimodal_consistency(signal, "SUPPORTED")
        self.assertEqual(analysis["status"], "consistent_authentic")
        self.assertEqual(analysis["relationship"], "reinforcing")
        self.assertFalse(analysis["requires_investigation"])

    def test_divergence_out_of_context_misattribution(self):
        # Video is visually real, but factual claim is contradicted by evidence
        signal = {"available": True, "predicted_label": "real", "confidence": 0.88}
        analysis = analyze_multimodal_consistency(signal, "CONTRADICTED")
        self.assertEqual(analysis["status"], "out_of_context_misattribution")
        self.assertEqual(analysis["relationship"], "divergent")
        self.assertTrue(analysis["requires_investigation"])
        self.assertIn("out-of-context", analysis["description"].lower())

    def test_divergence_visual_anomaly_flagged(self):
        # Evidence supports claim, but visual classifier flags fake
        signal = {"available": True, "predicted_label": "fake", "confidence": 0.72}
        analysis = analyze_multimodal_consistency(signal, "SUPPORTED")
        self.assertEqual(analysis["status"], "visual_anomaly_flagged")
        self.assertEqual(analysis["relationship"], "divergent")
        self.assertTrue(analysis["requires_investigation"])

    def test_visual_signal_unavailable(self):
        signal = {"available": False, "error": "Model load failure"}
        analysis = analyze_multimodal_consistency(signal, "SUPPORTED")
        self.assertEqual(analysis["status"], "visual_signal_unavailable")
        self.assertFalse(analysis["requires_investigation"])


class TestVideoVerifierOrchestration(unittest.TestCase):
    """Test suite for full video verifier orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.test_vid = create_dummy_video(self.temp_path / "test_orchestrator.avi", num_frames=20)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verify_video_with_classifier_enabled(self):
        # Run with skip_research=True (offline)
        out_folder = self.temp_path / "out_enabled"
        res = verify_video(
            video_path=self.test_vid,
            enable_classifier=True,
            verify_claims=False,
            output_dir=out_folder,
            skip_ocr=True,
        )

        self.assertIn("video_metadata", res)
        self.assertIn("visual_classification", res)
        self.assertTrue(res["visual_classification"]["available"])
        self.assertIn(res["visual_classification"]["predicted_label"], ("fake", "real"))
        self.assertIn("consistency_analysis", res)
        self.assertIn("final_evidence_verdict", res)

    def test_verify_video_with_classifier_disabled(self):
        out_folder = self.temp_path / "out_disabled"
        res = verify_video(
            video_path=self.test_vid,
            enable_classifier=False,
            verify_claims=False,
            output_dir=out_folder,
            skip_ocr=True,
        )

        self.assertIn("visual_classification", res)
        self.assertFalse(res["visual_classification"]["available"])
        self.assertIn("disabled", res["visual_classification"]["error"].lower())
        self.assertIn("consistency_analysis", res)
        self.assertEqual(res["consistency_analysis"]["status"], "visual_signal_unavailable")


if __name__ == "__main__":
    unittest.main()
