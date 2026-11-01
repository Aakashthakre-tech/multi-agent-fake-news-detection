"""
consistency.py
==============
Multi-signal consistency analysis separating Audio Provenance from Factual Truth.

Core Principle:
- Acoustic Provenance (Is the voice human or synthetic?) is orthogonal to
  Factual Veracity (Is the statement true or false?).
- An AI-generated voice may narrate genuine educational facts (e.g. AI textbook reader).
- A human speaker may spread malicious falsehoods (e.g. traditional disinformation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("TruthLensAI.AudioModule.Consistency")


def analyze_audio_consistency(
    synthetic_result: Dict[str, Any],
    factual_result: Optional[Dict[str, Any]],
    has_claims: bool = True,
) -> Dict[str, Any]:
    """
    Synthesize acoustic provenance and factual verification into a structured matrix.

    Args:
        synthetic_result: Output from predict_synthetic_speech().
        factual_result: Output from run_research_pipeline() or factual verification.
        has_claims: Whether verifiable factual claims were extracted.

    Returns:
        Structured consistency synthesis dictionary.
    """
    prov_label = synthetic_result.get("label", "UNCERTAIN").upper()
    is_prov_available = synthetic_result.get("available", False)

    if not is_prov_available and prov_label != "UNCERTAIN":
        prov_label = "UNAVAILABLE"

    # Normalize factual verdict
    factual_verdict = "NO_CLAIMS"
    factual_confidence = 0.0
    claim_text = None

    if factual_result and has_claims:
        # Check standard verdict keys from pipeline.py / run_research_pipeline
        raw_verdict = (
            factual_result.get("verdict")
            or factual_result.get("classification")
            or factual_result.get("status")
            or "UNVERIFIED"
        )
        factual_verdict = str(raw_verdict).upper()
        factual_confidence = float(
            factual_result.get("confidence")
            or factual_result.get("score")
            or 0.0
        )
        claim_text = factual_result.get("claim") or factual_result.get("claim_text")
    elif not has_claims:
        factual_verdict = "NO_VERIFIABLE_CLAIMS"

    # Category matrix mapping
    is_synthetic = prov_label == "AI_GENERATED"
    is_human = prov_label == "REAL"
    is_prov_ambiguous = prov_label in ("UNCERTAIN", "UNAVAILABLE")

    is_true = factual_verdict in ("TRUE", "VERIFIED", "ACCURATE", "SUPPORTED")
    is_false = factual_verdict in ("FALSE", "REFUTED", "MISLEADING", "FABRICATED", "FAKE")
    is_unverified = factual_verdict in ("UNVERIFIED", "INCONCLUSIVE", "NOT_ENOUGH_INFO", "NO_CLAIMS", "NO_VERIFIABLE_CLAIMS")

    if is_synthetic and is_false:
        category = "SYNTHETIC_DISINFORMATION"
        risk_level = "CRITICAL"
        description = (
            "High threat: Audio voice is synthetic (AI-generated) and spoken claims "
            "are factually false or misleading."
        )
    elif is_synthetic and is_true:
        category = "AI_NARRATED_VERIFIED_FACT"
        risk_level = "LOW"
        description = (
            "Benign AI synthesis: Voice is AI-generated, but the spoken factual "
            "content has been verified as accurate."
        )
    elif is_synthetic and is_unverified:
        category = "SYNTHETIC_SPEECH_UNVERIFIED_CONTENT"
        risk_level = "MEDIUM"
        description = (
            "Voice is AI-generated; however, the spoken claims could not be conclusively "
            "verified against factual sources."
        )
    elif is_human and is_false:
        category = "HUMAN_DISINFORMATION"
        risk_level = "HIGH"
        description = (
            "Traditional disinformation: Voice is human/authentic, but the spoken statements "
            "are factually false or misleading."
        )
    elif is_human and is_true:
        category = "HUMAN_AUTHENTIC_FACT"
        risk_level = "VERY_LOW"
        description = (
            "Authentic human audio delivering verified factual information."
        )
    elif is_human and is_unverified:
        category = "HUMAN_SPEECH_UNVERIFIED_CONTENT"
        risk_level = "LOW"
        description = (
            "Voice is bonafide human, but spoken content contains unverified or non-factual assertions."
        )
    elif is_prov_ambiguous and is_false:
        category = "UNCERTAIN_PROVENANCE_DISINFORMATION"
        risk_level = "HIGH"
        description = (
            "Factual verification confirms false/misleading content, while voice provenance "
            f"remains {prov_label}."
        )
    elif is_prov_ambiguous and is_true:
        category = "UNCERTAIN_PROVENANCE_VERIFIED_FACT"
        risk_level = "LOW"
        description = (
            f"Voice provenance is {prov_label}, but the spoken statements are verified as factually true."
        )
    else:
        category = "UNCERTAIN_PROVENANCE_UNVERIFIED_CONTENT"
        risk_level = "NEUTRAL"
        description = (
            f"Both speech provenance ({prov_label}) and factual verification ({factual_verdict}) "
            "are inconclusive or lack verifiable claims."
        )

    return {
        "category": category,
        "risk_level": risk_level,
        "description": description,
        "provenance_signal": {
            "label": prov_label,
            "confidence": synthetic_result.get("confidence", 0.0),
            "available": is_prov_available,
        },
        "veracity_signal": {
            "verdict": factual_verdict,
            "confidence": round(factual_confidence, 4),
            "claim": claim_text,
        },
    }
