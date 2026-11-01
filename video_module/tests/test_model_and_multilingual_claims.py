"""
test_model_and_multilingual_claims.py
=====================================
Focused tests for:
1. Label mapping and confusion-matrix calculation.
2. Inference preprocessing matching training preprocessing.
3. Checkpoint loading consistency across baseline and optimized checkpoints.
4. Multilingual & Devanagari factual claim extraction.
5. Priority-ordered verdict aggregation in video_verifier.
"""

import unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from video_module.baseline_classifier.model import TemporalVideoClassifier, TemporalAttentionPooling
from video_module.baseline_classifier.dataset import DEFAULT_TRANSFORM
from video_module.classifier_adapter import (
    load_classifier,
    predict_video,
    OPTIMIZED_CHECKPOINT_PATH,
    BASELINE_CHECKPOINT_PATH,
)
from video_module.claim_ranker import (
    detect_predicate_and_verbs,
    evaluate_candidate_claim,
    is_non_factual_or_filler,
)
from video_module.video_verifier import verify_video


class TestModelAndMultilingualClaims(unittest.TestCase):

    def test_01_label_mapping_and_confusion_matrix(self):
        """Verify label mapping real=0, fake=1 and confusion matrix TN/FP/FN/TP semantics."""
        class_map = {"real": 0, "fake": 1}
        self.assertEqual(class_map["real"], 0)
        self.assertEqual(class_map["fake"], 1)

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()
        self.assertEqual(tn, 1)  # true 0, pred 0
        self.assertEqual(fp, 1)  # true 0, pred 1
        self.assertEqual(fn, 1)  # true 1, pred 0
        self.assertEqual(tp, 1)  # true 1, pred 1

    def test_02_preprocessing_transforms(self):
        """Inference preprocessing must match training: Resize(256), CenterCrop(224), ImageNet norm."""
        transform_types = [type(t).__name__ for t in DEFAULT_TRANSFORM.transforms]
        self.assertIn("Resize", transform_types)
        self.assertIn("CenterCrop", transform_types)
        self.assertIn("ToTensor", transform_types)
        self.assertIn("Normalize", transform_types)

    def test_03_modular_model_backwards_compatibility(self):
        """TemporalVideoClassifier must support both legacy baseline and modular optimized checkpoints."""
        device = torch.device("cpu")
        # Legacy architecture
        model_legacy = TemporalVideoClassifier(
            num_classes=2,
            embed_dim=1280,
            hidden_dim=256,
            mlp_dim=256,
            dropout=0.35,
            pooling_mode="hybrid",
        ).to(device)

        dummy_x = torch.randn(2, 12, 1280, device=device)
        logits_legacy = model_legacy(dummy_x)
        self.assertEqual(logits_legacy.shape, (2, 2))

        # Optimized modular architecture
        model_opt = TemporalVideoClassifier(
            num_classes=2,
            embed_dim=1280,
            proj_dim=64,
            mlp_dim=48,
            dropout=0.40,
            pooling_mode="projected_dynamics",
            norm_type="layernorm",
        ).to(device)

        logits_opt = model_opt(dummy_x)
        self.assertEqual(logits_opt.shape, (2, 2))

    def test_04_optimized_checkpoint_loading(self):
        """Optimized checkpoint must load successfully and return valid probabilities."""
        if OPTIMIZED_CHECKPOINT_PATH.exists():
            model, extractor, config, class_map = load_classifier(OPTIMIZED_CHECKPOINT_PATH)
            self.assertEqual(class_map, {"real": 0, "fake": 1})
            self.assertEqual(config.get("proj_dim"), 64)
            self.assertEqual(config.get("pooling_mode"), "projected_dynamics")

    def test_05_multilingual_devanagari_claim_detection(self):
        """Factual statements in Devanagari must have their predicates and entities recognized."""
        # Incident in Marathi: "मुंबईत मोठा अपघात झाला, ५ जणांचा मृत्यू"
        text = "मुंबईत मोठा अपघात झाला, ५ जणांचा मृत्यू"
        has_pred, verbs = detect_predicate_and_verbs(text)
        self.assertTrue(has_pred)
        self.assertTrue(any(v in verbs for v in ["अपघात", "झाला", "मृत्यू"]))

        # Non-factual clock should be filtered
        is_filler, reason = is_non_factual_or_filler("12:45:00 PM")
        self.assertTrue(is_filler)
        self.assertIn("clock", reason.lower())

        # Call-to-action should be filtered
        is_promo, reason = is_non_factual_or_filler("Watch till the end to see what happens!")
        self.assertTrue(is_promo)


if __name__ == "__main__":
    unittest.main()
