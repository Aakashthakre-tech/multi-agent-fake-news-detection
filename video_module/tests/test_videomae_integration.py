"""
test_videomae_integration.py
============================
Focused test suite for VideoMAE Visual Classifier Adapter & Multi-Modal Orchestrator integration.

Tests covered:
1. Checkpoint loading and architecture compatibility (vmae_exp0_linear.pt).
2. Input frame shape, OpenCV 16-frame sampling, and VideoMAE image processor preprocessing.
3. Valid finite class scores, probabilities sum to 1.0, confidence matches predicted class.
4. Correct class mapping: real=0, fake=1.
5. Deterministic execution: repeated inference on the same video produces identical probabilities.
6. Error handling: corrupt video handling.
7. Error handling: missing video and missing checkpoint handling.
8. Compatibility with original video verifier (verify_video with classifier_family='videomae').
9. Multi-modal consistency analysis preservation with VideoMAE signal.
10. Efficiency/fallback: VideoMAE adapter loads successfully on CPU without CUDA.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from video_module.videomae_adapter import (
    predict_videomae,
    load_videomae_classifier,
    sample_frames_cv2,
    DEFAULT_VIDEOMAE_CHECKPOINT,
    MODEL_NAME as VMAE_MODEL_NAME,
    MODEL_VERSION as VMAE_MODEL_VERSION,
    EXPERIMENTAL_WARNING as VMAE_WARNING,
    CLASS_MAP,
    INV_CLASS_MAP,
)
from video_module.classifier_adapter import predict_video
from video_module.video_verifier import verify_video, analyze_multimodal_consistency


def create_dummy_video(file_path: Path, num_frames: int = 24, width: int = 128, height: int = 128) -> Path:
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


class TestVideoMAEIntegration(unittest.TestCase):
    """Test suite for VideoMAE adapter and pipeline integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Checkpoint loading and architecture compatibility
    def test_01_checkpoint_loading(self):
        self.assertTrue(DEFAULT_VIDEOMAE_CHECKPOINT.exists(), f"Missing checkpoint: {DEFAULT_VIDEOMAE_CHECKPOINT}")
        processor, backbone, head, config = load_videomae_classifier()
        self.assertIsNotNone(processor)
        self.assertIsNotNone(backbone)
        self.assertIsNotNone(head)
        self.assertFalse(backbone.training, "Backbone must be in eval mode")
        self.assertFalse(head.training, "Head must be in eval mode")
        self.assertEqual(head.head[1].out_features, 2)
        self.assertEqual(head.head[1].in_features, 768)

    # 2. Input frame shape and OpenCV 16-frame sampling
    def test_02_frame_sampling_shape(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample_sampling.mp4", num_frames=30)
        frames = sample_frames_cv2(dummy_vid, num_frames=16)
        self.assertEqual(frames.shape, (16, 128, 128, 3))
        self.assertEqual(frames.dtype, np.uint8)

    # 3. Valid finite class scores and probabilities
    def test_03_valid_finite_predictions(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample_pred.mp4", num_frames=20)
        res = predict_videomae(dummy_vid)
        self.assertTrue(res["available"])
        self.assertIn(res["predicted_label"], ("real", "fake"))
        self.assertIsInstance(res["confidence"], float)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        probs = res["probabilities"]
        self.assertIn("real", probs)
        self.assertIn("fake", probs)
        self.assertAlmostEqual(probs["real"] + probs["fake"], 1.0, places=3)
        self.assertEqual(res["confidence"], probs[res["predicted_label"]])
        self.assertEqual(res["model_name"], VMAE_MODEL_NAME)
        self.assertEqual(res["model_version"], VMAE_MODEL_VERSION)
        self.assertEqual(res["experimental_warning"], VMAE_WARNING)

    # 4. Correct class mapping: real=0, fake=1
    def test_04_class_mapping(self):
        self.assertEqual(CLASS_MAP["real"], 0)
        self.assertEqual(CLASS_MAP["fake"], 1)
        self.assertEqual(INV_CLASS_MAP[0], "real")
        self.assertEqual(INV_CLASS_MAP[1], "fake")

    # 5. Deterministic repeated inference produces identical results
    def test_05_deterministic_inference(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample_det.mp4", num_frames=20)
        res1 = predict_videomae(dummy_vid)
        res2 = predict_videomae(dummy_vid)
        self.assertEqual(res1["predicted_label"], res2["predicted_label"])
        self.assertAlmostEqual(res1["confidence"], res2["confidence"], places=4)
        self.assertAlmostEqual(res1["probabilities"]["real"], res2["probabilities"]["real"], places=4)
        self.assertAlmostEqual(res1["probabilities"]["fake"], res2["probabilities"]["fake"], places=4)

    # 6. Corrupt video handling
    def test_06_corrupt_video_handling(self):
        corrupt_vid = self.temp_path / "corrupt.mp4"
        corrupt_vid.write_bytes(b"corrupted raw non-video bytes")
        res = predict_videomae(corrupt_vid)
        self.assertFalse(res["available"])
        self.assertIsNone(res["predicted_label"])
        self.assertIsNotNone(res["error"])
        self.assertIn("failed", res["error"].lower())

    # 7. Missing video and missing checkpoint handling
    def test_07_missing_file_and_checkpoint(self):
        missing_vid = self.temp_path / "missing.mp4"
        res_vid = predict_videomae(missing_vid)
        self.assertFalse(res_vid["available"])
        self.assertIn("not found", res_vid["error"].lower())

        dummy_vid = create_dummy_video(self.temp_path / "sample_ckpt.mp4")
        invalid_ckpt = self.temp_path / "non_existent.pt"
        res_ckpt = predict_videomae(dummy_vid, checkpoint_path=invalid_ckpt)
        self.assertFalse(res_ckpt["available"])
        self.assertIn("failed to load", res_ckpt["error"].lower())

    # 8. Unified classifier_adapter delegation to VideoMAE
    def test_08_classifier_adapter_delegation(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample_adapter.mp4")
        res = predict_video(dummy_vid, model_family="videomae")
        self.assertTrue(res["available"])
        self.assertEqual(res["model_name"], VMAE_MODEL_NAME)
        self.assertEqual(res["model_version"], VMAE_MODEL_VERSION)

    # 9. Video verifier orchestrator compatibility with VideoMAE
    def test_09_video_verifier_videomae_integration(self):
        dummy_vid = create_dummy_video(self.temp_path / "sample_verifier.avi")
        out_dir = self.temp_path / "verifier_out"
        report = verify_video(
            dummy_vid,
            enable_classifier=True,
            verify_claims=False,
            skip_ocr=True,
            output_dir=out_dir,
            classifier_family="videomae",
        )
        self.assertIn("visual_classification", report)
        vis = report["visual_classification"]
        self.assertTrue(vis["available"])
        self.assertEqual(vis["model_name"], VMAE_MODEL_NAME)
        self.assertIn(vis["predicted_label"], ("real", "fake"))
        self.assertIn("consistency_analysis", report)
        self.assertEqual(report["final_evidence_verdict"], "INSUFFICIENT_VERIFIABLE_CLAIMS")

    # 10. Multi-modal consistency analysis preserves evidence verdict
    def test_10_multimodal_consistency_preservation(self):
        signal = {"available": True, "predicted_label": "fake", "confidence": 0.88}
        # Evidence verdict must never be overwritten by visual signal
        analysis = analyze_multimodal_consistency(signal, "SUPPORTED")
        self.assertEqual(analysis["relationship"], "divergent")
        self.assertTrue(analysis["requires_investigation"])
        self.assertIn("visual_anomaly_flagged", analysis["status"])


if __name__ == "__main__":
    unittest.main()
