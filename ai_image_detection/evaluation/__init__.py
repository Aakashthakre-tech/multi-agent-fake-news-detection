"""
ai_image_detection.evaluation
=============================
Offline evaluation harness for the AI-image detector.

This package is NOT imported by the production detector. It exists so that
every number reported about detector performance is produced by code in this
repository from a labelled dataset, never asserted by hand.

Label convention used everywhere in this project:

    1 = AI-GENERATED
    0 = REAL

A prediction is AI-GENERATED when the detector's AI probability is greater
than or equal to the decision threshold.
"""

from .metrics import (
    evaluate_scores,
    confusion_counts,
    best_threshold,
    threshold_sweep,
    expected_calibration_error,
    reliability_bins,
)

__all__ = [
    "evaluate_scores",
    "confusion_counts",
    "best_threshold",
    "threshold_sweep",
    "expected_calibration_error",
    "reliability_bins",
]
