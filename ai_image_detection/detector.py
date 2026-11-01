"""
detector.py
===========
Production AI-generated image detector for TruthLensAI.

FROZEN CONFIGURATION. Every constant was selected on the validation split and
then applied once to the held-out test split. Nothing is fitted on test data
and nothing is conditioned on a filename, path, or particular image.

Model
-----
`ForensicCNN`, a 405K-parameter residual network trained in this repository on
the labelled CIFAKE and AI-or-Not pools. It reads the residual left after
subtracting a Gaussian blur, i.e. noise and resampling traces, rather than the
RGB image a Vision Transformer would see.

Why this replaced the previous model (`delpot/steganograph-ia-detector`):
on the same held-out test split, with each model's operating point fitted
independently on validation -

    forensic_cnn           accuracy 0.9045   macro F1 0.9040
    steganalysis_current   accuracy 0.6555   macro F1 0.6514

An ensemble of the two was tested and rejected: on validation it scored
macro F1 0.8112 against 0.8905 for the CNN alone, because averaging in a much
weaker model drags the result down. Simplicity won on measurement.

Frozen operating point (fitted on validation)
--------------------------------------------
    temperature      1.10   (learned; the CNN is already close to calibrated)
    threshold        0.500  (calibrated space)
    uncertain band   +/-0.23 -> UNCERTAIN for 0.27 .. 0.73

    TEST (n=4000, 2534 decided, 1466 UNCERTAIN = 36.7%):
        accuracy 0.9045    macro F1 0.9040
        AI recall 0.9259   REAL recall 0.8830
        FPR 0.1170         FNR 0.0741

Reporting honesty
-----------------
These are IN-DISTRIBUTION numbers: the test split is drawn from the same two
public datasets the model was trained on. They are not a claim about
photographs, news graphics or historical images from the open web. Measured
behaviour on such images is recorded in detector.py's own test scripts and in
evaluation/results/, and the UNCERTAIN state exists precisely because the
detector is expected to meet images it was not fitted to.

This module reports AI image origin only. It has no knowledge of and no
effect on claim verification; those are separate signals by design.
"""

import json
import os
import threading

import numpy as np
import torch
from PIL import Image

from ai_image_detection.forensic import (
    ForensicCNN,
    compute_residual,
    residual_to_tensor,
    count_parameters,
)


MODEL_NAME = "forensic_residual_cnn"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "forensic_model.pt")

# --- frozen configuration (fitted on validation, applied to test) ---------
TEMPERATURE = 1.10
THRESHOLD = 0.500
UNCERTAIN_BAND = 0.23

UNCERTAIN_LOW = THRESHOLD - UNCERTAIN_BAND    # 0.27
UNCERTAIN_HIGH = THRESHOLD + UNCERTAIN_BAND   # 0.73

VERDICT_AI = "AI-GENERATED"
VERDICT_REAL = "REAL"
VERDICT_UNCERTAIN = "UNCERTAIN"

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation", "results", "final_config_v2.json",
)


def _check_config_drift():
    """Refuse to run if the frozen constants disagree with the artifact."""
    if not os.path.exists(_CONFIG_PATH):
        return
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return
    for key, frozen in (("temperature", TEMPERATURE),
                        ("threshold", THRESHOLD),
                        ("uncertain_band", UNCERTAIN_BAND)):
        if key in cfg and abs(float(cfg[key]) - frozen) > 1e-9:
            raise RuntimeError(
                "Frozen detector constants disagree with %s for %r "
                "(frozen=%s, artifact=%s). Re-run the selection step instead "
                "of editing the constants." % (_CONFIG_PATH, key, frozen,
                                                cfg[key])
            )


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calibrate(probability, temperature=TEMPERATURE):
    """
    Temperature scaling of a raw probability.

    The learned temperature here is 1.10, i.e. the network is already close to
    calibrated and needs only slight flattening. Applied so that a reported
    probability is an expected frequency rather than a raw logit artefact.
    """
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    z = np.log(p / (1.0 - p))
    return float(_sigmoid(z / temperature))


class AIImageDetector:
    """
    Detects whether an image is AI-generated.

    The model is loaded once per process on first use; predict() is then cheap
    to call repeatedly.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        _check_config_drift()

        print("Loading AI-image detector...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained detector weights not found at %s. Train them with "
                "ai_image_detection.evaluation.train_forensic." % MODEL_PATH
            )

        self.model = ForensicCNN()
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

        print("AI-image detector loaded (%s, %d params, T=%.2f, "
              "threshold=%.3f, uncertain +/-%.2f)"
              % (MODEL_NAME, count_parameters(self.model), TEMPERATURE,
                 THRESHOLD, UNCERTAIN_BAND))

    def predict(self, image_path):
        """
        Return the AI-origin verdict for one image.

        Output contract:
            {
              "verdict":          AI-GENERATED | REAL | UNCERTAIN
              "ai_probability":   calibrated, 0..1
              "real_probability": calibrated, 0..1
              "confidence":       calibrated probability the verdict is right
              "model":            model id
              "available":        True
            }
        """
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        residual = compute_residual(image)

        with torch.no_grad():
            logit = self.model(residual_to_tensor(residual))
            raw_ai = float(torch.sigmoid(logit).item())

        ai_probability = calibrate(raw_ai)

        if ai_probability >= UNCERTAIN_HIGH:
            verdict = VERDICT_AI
            confidence = ai_probability
        elif ai_probability <= UNCERTAIN_LOW:
            verdict = VERDICT_REAL
            confidence = 1.0 - ai_probability
        else:
            verdict = VERDICT_UNCERTAIN
            # Upper bound on the chance either binary answer is correct.
            confidence = 1.0 - max(ai_probability, 1.0 - ai_probability)

        return {
            "verdict": verdict,
            "ai_probability": round(ai_probability, 4),
            "real_probability": round(1.0 - ai_probability, 4),
            "confidence": round(confidence, 4),
            "model": MODEL_NAME,
            "available": True,
        }

    @classmethod
    def shared(cls):
        """Process-wide instance, so the model is loaded exactly once."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def detect_ai_image(image_path):
    """Convenience entry point using the shared instance."""
    return AIImageDetector.shared().predict(image_path)
