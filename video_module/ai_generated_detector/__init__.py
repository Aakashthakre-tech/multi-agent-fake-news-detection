"""
ai_generated_detector
=====================
TruthLensAI dedicated binary classifier for AI-generated video detection.

Task:
    Class 0 (REAL):          Authentic, non-AI-generated video.
    Class 1 (AI_GENERATED):  Video fully synthesized by an AI generation model.

This module is separate from the misinformation/VideoMAE classifier.
"""

from .detector import predict_ai_generated

__all__ = ["predict_ai_generated"]
