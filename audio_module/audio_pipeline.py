"""
audio_pipeline.py
=================
Main orchestration pipeline for TruthLensAI Audio Module.

End-to-end workflow:
1. Audio validation, decoding (FFmpeg/wave), and silence check.
2. AI-generated speech detection (mo-thecreator/Deepfake-audio-detection).
3. Speech transcription (reusing Whisper from video_module).
4. Verifiable claim extraction and priority ranking (reusing claim_ranker).
5. Optional factual claim verification (reusing pipeline.run_research_pipeline).
6. Multi-signal consistency synthesis (separating Provenance from Factual Truth).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from audio_module.audio_loader import (
    AudioFileNotFoundError,
    CorruptAudioError,
    SilentAudioError,
    UnsupportedAudioFormatError,
    load_audio,
    validate_audio_path,
)
from audio_module.claim_extractor import extract_claims_from_transcript
from audio_module.config import (
    DEFAULT_MAX_CLAIMS_TO_VERIFY,
    DEFAULT_WHISPER_MODEL,
    TARGET_SAMPLE_RATE,
)
from audio_module.consistency import analyze_audio_consistency
from audio_module.synthetic_detector import predict_synthetic_speech
from audio_module.transcriber import transcribe_audio

logger = logging.getLogger("TruthLensAI.AudioModule.Pipeline")


def verify_audio(
    audio_path: Union[str, Path],
    verify_claims: bool = False,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    device: Optional[str] = None,
    max_claims_to_verify: int = DEFAULT_MAX_CLAIMS_TO_VERIFY,
) -> Dict[str, Any]:
    """
    Execute complete TruthLensAI verification on an audio file.

    Args:
        audio_path: Path to the target audio file (.wav, .mp3, .m4a, .flac, .aac, .ogg).
        verify_claims: Whether to perform live web research and factual claim verification.
        whisper_model: Whisper model size ("tiny", "base", "small", etc.).
        device: 'cuda', 'cpu', or None for auto-selection.
        max_claims_to_verify: Maximum number of claims to submit for factual verification.

    Returns:
        Structured dictionary with audio metadata, synthetic speech provenance,
        transcription, candidate claims, factual verification, and consistency analysis.
    """
    audio_path_obj = Path(audio_path)

    # 1. Validate Audio Path & Format
    try:
        format_meta = validate_audio_path(audio_path_obj)
    except (AudioFileNotFoundError, UnsupportedAudioFormatError, CorruptAudioError) as e:
        logger.error(f"Validation failed for {audio_path}: {e}")
        return _build_error_response(audio_path_obj, error_msg=str(e), error_type=type(e).__name__)

    # 2. Decode & Load Audio Samples
    try:
        samples, audio_props = load_audio(audio_path_obj, target_sr=TARGET_SAMPLE_RATE)
        sr = int(audio_props["sample_rate"])
        duration_sec = float(audio_props["duration_seconds"])
        rms = float(audio_props["rms"])
    except (CorruptAudioError, SilentAudioError, Exception) as e:
        logger.error(f"Audio loading/decoding failed for {audio_path}: {e}")
        return _build_error_response(audio_path_obj, error_msg=str(e), error_type=type(e).__name__)

    audio_metadata = {
        "path": str(audio_path_obj.resolve()),
        "filename": audio_path_obj.name,
        "format": format_meta["format"],
        "size_bytes": format_meta["size_bytes"],
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "rms_energy": round(rms, 6),
    }

    # 3. AI-Generated Speech Detection (Orthogonal to Factual Truth)
    try:
        synthetic_detection = predict_synthetic_speech(
            audio_data=samples,
            sample_rate=sr,
            device=device,
        )
    except Exception as e:
        logger.error(f"Unexpected error in synthetic speech detector: {e}")
        synthetic_detection = {
            "is_ai_generated": None,
            "label": "UNAVAILABLE",
            "confidence": 0.0,
            "probability_synthetic": 0.0,
            "probability_real": 0.0,
            "available": False,
            "error": str(e),
        }

    # 4. Transcription (Whisper)
    try:
        transcription = transcribe_audio(
            audio_path_obj,
            model_name=whisper_model,
        )
    except Exception as e:
        logger.warning(f"Whisper transcription failed for {audio_path}: {e}")
        transcription = {
            "text": "",
            "language": "unknown",
            "duration": duration_sec,
            "segments": [],
            "model": whisper_model,
            "available": False,
            "error": str(e),
        }

    # 5. Candidate Claim Extraction & Filtering
    claim_extraction = extract_claims_from_transcript(
        transcript=transcription,
        max_claims=max_claims_to_verify,
    )
    primary_claim = claim_extraction.get("primary_claim")

    # 6. Factual Claim Verification (Orthogonal to Speech Provenance)
    factual_verification = {
        "enabled": verify_claims,
        "claim": primary_claim,
        "verdict": None,
        "confidence": None,
        "evidence_sufficiency": None,
        "details": None,
        "error": None,
    }

    if verify_claims and primary_claim:
        try:
            logger.info(f"Running factual research pipeline for claim: '{primary_claim}'")
            # Lazy import to avoid circular dependency
            from pipeline import run_research_pipeline
            research_state = run_research_pipeline(
                primary_claim,
                extracted_text=_build_audio_extracted_text(transcription, primary_claim),
            )
            
            final_v = research_state.get("final_verdict", {})
            factual_verification["verdict"] = final_v.get("verdict", "UNVERIFIED")
            factual_verification["confidence"] = final_v.get("confidence")
            factual_verification["evidence_sufficiency"] = final_v.get("evidence_sufficiency")
            factual_verification["verification"] = research_state
            factual_verification["details"] = {
                "independent_supporting": final_v.get("independent_supporting", 0),
                "independent_contradicting": final_v.get("independent_contradicting", 0),
                "contradiction_score": final_v.get("contradiction_score", 0),
                "source_agreement": final_v.get("source_agreement", 0),
                "fusion_score": final_v.get("fusion_score", 0),
                "total_sources": final_v.get("total_sources", 0),
                "supporting_sources": final_v.get("supporting_sources", 0),
                "contradicting_sources": final_v.get("contradicting_sources", 0),
                "reasoning": final_v.get("reasoning") or final_v.get("permitted_reason"),
                "deterministic_gate_reason": final_v.get("deterministic_gate_reason"),
                "agent_verdict": final_v.get("verification_verdict"),
                "critic_verdict": final_v.get("critic_verdict"),
                "agents_agree": final_v.get("agents_agree"),
                "agent_confidence": final_v.get("agent_confidence"),
                "critic_confidence": final_v.get("critic_confidence"),
            }
        except Exception as e:
            logger.error(f"Factual research pipeline failed: {e}")
            factual_verification["error"] = str(e)
            factual_verification["verdict"] = "UNVERIFIED"
    elif verify_claims and not primary_claim:
        factual_verification["verdict"] = "INSUFFICIENT_VERIFIABLE_CLAIMS"
        factual_verification["evidence_sufficiency"] = "INSUFFICIENT"
        factual_verification["details"] = {
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "contradiction_score": 0.0,
            "source_agreement": None,
            "fusion_score": None,
            "reasoning": "No verifiable factual claims were extracted from audio transcription.",
        }

    # 7. Consistency Synthesis (Combining Provenance + Veracity)
    consistency = analyze_audio_consistency(
        synthetic_result=synthetic_detection,
        factual_result=factual_verification if verify_claims else None,
        has_claims=bool(primary_claim),
    )

    return {
        "status": "success",
        "audio_metadata": audio_metadata,
        "synthetic_speech_detection": synthetic_detection,
        "transcription": transcription,
        "claim_extraction": claim_extraction,
        "factual_verification": factual_verification,
        "consistency_analysis": consistency,
        "error": None,
    }


def _build_audio_extracted_text(
    transcription: Optional[dict],
    primary_claim: Optional[str],
) -> str:
    """
    Build a concise extracted-text representation for audio:
    the Whisper transcript text plus the primary extracted claim.
    No transcription is re-run here; only already-collected output is reused.
    """
    parts = []
    transcript_text = ""
    if isinstance(transcription, dict):
        transcript_text = str(transcription.get("text") or "").strip()
    if transcript_text:
        parts.append("Transcript: " + transcript_text)
    if primary_claim:
        parts.append("Extracted claim: " + str(primary_claim).strip())
    return " ".join(parts)


def _build_error_response(audio_path: Path, error_msg: str, error_type: str) -> Dict[str, Any]:
    """Build a standard structured failure response ensuring synthetic provenance is never silently REAL."""
    return {
        "status": "error",
        "audio_metadata": {
            "path": str(audio_path.resolve()) if audio_path else None,
            "filename": audio_path.name if audio_path else None,
            "format": audio_path.suffix.lower() if audio_path else None,
            "duration_sec": 0.0,
            "sample_rate": 0,
            "rms_energy": 0.0,
        },
        "synthetic_speech_detection": {
            "is_ai_generated": None,
            "label": "UNAVAILABLE",
            "confidence": 0.0,
            "probability_synthetic": 0.0,
            "probability_real": 0.0,
            "available": False,
            "error": f"{error_type}: {error_msg}",
        },
        "transcription": {
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "segments": [],
            "available": False,
            "error": error_msg,
        },
        "claim_extraction": {
            "primary_claim": None,
            "selected_claims": [],
            "all_candidates": [],
            "rejection_summary": {},
        },
        "factual_verification": {
            "enabled": False,
            "claim": None,
            "verdict": None,
            "confidence": None,
            "details": None,
            "error": None,
        },
        "consistency_analysis": {
            "category": "UNAVAILABLE",
            "risk_level": "UNKNOWN",
            "description": f"Processing halted due to audio error: {error_msg}",
            "provenance_signal": {"label": "UNAVAILABLE", "confidence": 0.0, "available": False},
            "veracity_signal": {"verdict": "UNVERIFIED", "confidence": 0.0, "claim": None},
        },
        "error": f"{error_type}: {error_msg}",
    }
