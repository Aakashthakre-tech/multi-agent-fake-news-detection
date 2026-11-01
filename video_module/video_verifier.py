"""
video_verifier.py
=================
TruthLensAI Multi-Modal Video Verification Orchestrator.

Integrates:
1. Video Preprocessing, Whisper Transcription, and OCR Fusion (video_module/video_pipeline.py).
2. Experimental Visual Classifier Adapter (video_module/classifier_adapter.py).
3. Multi-Agent Evidence Research & Verification Engine (pipeline.py).

Core Principles:
- Evidence-Driven: The final truth verdict is strictly determined by factual evidence,
  supporting SUPPORTED, CONTRADICTED, and INCONCLUSIVE outcomes.
- Multi-Signal Synthesis: Visual classification is treated as a supporting heuristic.
  Disagreements trigger descriptive investigation notes rather than automatic overrides.
- Resilient & Optional: The classifier can be toggled on or off (--disable-classifier).
  If the visual model fails or is unavailable, the fact-checking pipeline continues unaffected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Video pipeline components
from video_module.video_pipeline import (
    process_video_pipeline,
    prepare_truthlens_claim_payload,
)

# Baseline visual classifier adapter
from video_module.classifier_adapter import (
    predict_video,
    EXPERIMENTAL_WARNING,
    MODEL_NAME,
)

# Fact-checking evidence engine
try:
    from pipeline import run_research_pipeline
except Exception as _pipeline_import_err:
    run_research_pipeline = None


def analyze_multimodal_consistency(
    visual_signal: Dict[str, Any],
    claim_verdict: Optional[str],
) -> Dict[str, Any]:
    """
    Compare visual classifier prediction against evidence-based claim verdict.
    Disagreements trigger diagnostic investigation notes, never automatic overrides.
    """
    if not visual_signal.get("available"):
        return {
            "status": "visual_signal_unavailable",
            "relationship": "independent",
            "description": "Visual classification is unavailable; verdict rests entirely on factual evidence.",
            "requires_investigation": False,
        }

    pred_visual = visual_signal.get("predicted_label")  # 'fake' or 'real'
    evidence_verdict = (claim_verdict or "INCONCLUSIVE").upper()

    # Map evidence verdict to polarity
    is_claim_false = evidence_verdict in ("CONTRADICTED", "FAKE", "FALSE")
    is_claim_true = evidence_verdict in ("SUPPORTED", "REAL", "TRUE")
    is_claim_unclear = not (is_claim_false or is_claim_true)

    if "INSUFFICIENT" in evidence_verdict:
        return {
            "status": "insufficient_verifiable_claims",
            "relationship": "neutral",
            "description": (
                "No verifiable factual claims were found in the speech or on-screen text. "
                f"Visual classifier indicated '{pred_visual}' as a supporting signal, "
                "but without factual claims to research, video veracity cannot be established."
            ),
            "requires_investigation": False,
        }

    if is_claim_unclear:
        return {
            "status": "inconclusive_evidence",
            "relationship": "neutral",
            "description": f"Evidence-based verdict is {evidence_verdict}. Visual classifier indicated '{pred_visual}' as a supporting signal.",
            "requires_investigation": False,
        }

    # Case 1: Consistent Fake
    if is_claim_false and pred_visual == "fake":
        return {
            "status": "consistent_misinformation",
            "relationship": "reinforcing",
            "description": (
                "Consistent: Factual evidence contradicts the claim, and the visual classifier "
                "identifies visual patterns typical of synthetic or fabricated videos."
            ),
            "requires_investigation": False,
        }

    # Case 2: Consistent Real
    if is_claim_true and pred_visual == "real":
        return {
            "status": "consistent_authentic",
            "relationship": "reinforcing",
            "description": (
                "Consistent: Factual evidence supports the claim, and the visual classifier "
                "identifies authentic visual video characteristics."
            ),
            "requires_investigation": False,
        }

    # Case 3: Out-of-Context Video Misattribution (Evidence False, Visual Real)
    if is_claim_false and pred_visual == "real":
        return {
            "status": "out_of_context_misattribution",
            "relationship": "divergent",
            "description": (
                "Divergence (Out-of-Context Misattribution): The video footage appears visually authentic, "
                "but the verbal/on-screen claims are contradicted by factual evidence. This pattern is "
                "characteristic of real footage recycled with false narratives or incorrect dates. "
                "The evidence-based verdict stands."
            ),
            "requires_investigation": True,
        }

    # Case 4: Visual Discrepancy (Evidence True, Visual Fake)
    if is_claim_true and pred_visual == "fake":
        return {
            "status": "visual_anomaly_flagged",
            "relationship": "divergent",
            "description": (
                "Divergence (Visual Anomaly Flagged): Factual evidence supports the spoken/on-screen claims, "
                "but the visual classifier flagged potential visual anomalies. Because the visual model has "
                "an experimental 52.38% benchmark baseline, the evidence-based verdict is maintained. "
                "Manual visual review recommended."
            ),
            "requires_investigation": True,
        }

    return {
        "status": "unknown_alignment",
        "relationship": "unclassified",
        "description": f"Visual signal: {pred_visual}, Claim verdict: {evidence_verdict}.",
        "requires_investigation": False,
    }


def verify_video(
    video_path: str | Path,
    enable_classifier: bool = True,
    verify_claims: bool = True,
    output_dir: Optional[str | Path] = None,
    whisper_model: str = "base",
    ocr_stride: int = 1,
    skip_ocr: bool = False,
    max_claims_to_verify: int = 1,
    max_ocr_frames: Optional[int] = 15,
    classifier_family: str = "videomae",
    classifier_checkpoint: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Execute full multi-modal video verification pipeline.

    Args:
        video_path: Path to target video file.
        enable_classifier: If True, evaluates visual fake/real model. If False, skips classifier.
        verify_claims: If True, runs research evidence pipeline on extracted claims.
        output_dir: Optional directory for intermediate frame/audio artifacts.
        whisper_model: Whisper model candidate size ('base', 'small', etc.).
        ocr_stride: Sample stride for OCR processing over extracted frames.
        skip_ocr: If True, skips on-screen OCR.
        max_claims_to_verify: Maximum candidate claims to verify via live research.
        max_ocr_frames: Upper limit on frames processed by OCR (default: 15).
        classifier_family: Visual classifier family ('videomae' or 'efficientnet').
        classifier_checkpoint: Optional custom checkpoint path.

    Returns:
        Structured dictionary with summary, visual classification, research results,
        and evidence-driven final verdict.
    """
    v_path = Path(video_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    t_verify_start = time.time()
    step_timings: Dict[str, float] = {}

    print("\n" + "=" * 65)
    print(f"TRUTHLENSAI MULTI-MODAL VIDEO VERIFICATION [{time.strftime('%H:%M:%S')}]")
    print("=" * 65)
    print(f"Target Video: {v_path.name}")
    print(f"Visual Classifier Enabled: {enable_classifier}")
    print(f"Live Claim Research Enabled: {verify_claims}")
    print(f"OCR Max Frame Budget: {max_ocr_frames}")

    # ========================================================
    # STEP 1: VIDEO PREPROCESSING & CLAIM EXTRACTION
    # ========================================================
    t_step = time.time()
    print(f"\n[Step 1/3] [{time.strftime('%H:%M:%S')}] Preprocessing video (Frames, Audio, Whisper, OCR)...")
    out_p = Path(output_dir) if output_dir else None
    pipeline_summary = process_video_pipeline(
        video_path=v_path,
        output_dir=out_p,
        whisper_model=whisper_model,
        ocr_stride=ocr_stride,
        max_ocr_frames=max_ocr_frames,
        skip_ocr=skip_ocr,
    )
    step_timings["preprocessing_and_extraction"] = round(time.time() - t_step, 2)
    print(f"[Step 1/3] Finished in {step_timings['preprocessing_and_extraction']}s")

    candidate_claims_payload = prepare_truthlens_claim_payload(
        pipeline_summary.get("claims_summary", {})
        if "candidate_claims" in pipeline_summary.get("claims_summary", {})
        else {"candidate_claims": []}
    )

    # Fallback to load candidate claims directly from JSON artifact if empty
    candidate_claims_json_path = pipeline_summary.get("candidate_claims_path")
    if not candidate_claims_payload and candidate_claims_json_path and Path(candidate_claims_json_path).exists():
        try:
            with open(candidate_claims_json_path, "r", encoding="utf-8") as f:
                claims_data = json.load(f)
                candidate_claims_payload = prepare_truthlens_claim_payload(claims_data)
        except Exception:
            pass

    # Structured artifacts exposed to the UI/report layer
    transcript = pipeline_summary.get("transcript") or {}
    warnings = pipeline_summary.get("warnings") or []
    errors = pipeline_summary.get("errors") or []
    research_skipped_reason = None

    # ========================================================
    # STEP 2: VISUAL CLASSIFIER SIGNAL (SEPARATE SUPPORTING SIGNAL)
    # ========================================================
    t_step2 = time.time()
    visual_signal: Dict[str, Any] = {
        "available": False,
        "model_name": MODEL_NAME,
        "error": "Visual classifier disabled by configuration.",
        "experimental_warning": EXPERIMENTAL_WARNING,
    }

    if enable_classifier:
        print(f"\n[Step 2/3] [{time.strftime('%H:%M:%S')}] Running visual video classifier ({classifier_family})...")
        try:
            visual_signal = predict_video(
                v_path,
                checkpoint_path=classifier_checkpoint,
                model_family=classifier_family,
            )
            step_timings["visual_classification"] = round(time.time() - t_step2, 2)
            if visual_signal.get("available"):
                print(f"   Visual Model: {visual_signal.get('model_name')}")
                print(f"   Visual Class: {visual_signal.get('predicted_label')} "
                      f"(Confidence: {visual_signal.get('confidence')})")
                print(f"   Probabilities: {visual_signal.get('probabilities')}")
                print(f"   Visual classification completed in {step_timings['visual_classification']}s")
            else:
                print(f"   Visual Classifier Unavailable: {visual_signal.get('error')}")
        except Exception as exc:
            step_timings["visual_classification"] = round(time.time() - t_step2, 2)
            visual_signal = {
                "available": False,
                "model_name": MODEL_NAME,
                "error": f"Visual classifier execution failed: {str(exc)}",
                "experimental_warning": EXPERIMENTAL_WARNING,
            }
            print(f"   Visual Classifier Error: {exc}")
    else:
        step_timings["visual_classification"] = 0.0
        print("\n[Step 2/3] Visual classifier skipped (--disable-classifier).")

    # ========================================================
    # STEP 3: EVIDENCE-BASED CLAIM FACT-CHECKING
    # ========================================================
    t_step3 = time.time()
    verification_results: List[Dict[str, Any]] = []
    primary_verdict = "INCONCLUSIVE"
    primary_claim_text = None

    # Filter candidate claims: only research eligible, selected factual claims!
    eligible_claims = [
        c for c in candidate_claims_payload
        if c.get("selection_status") == "selected"
    ]

    if verify_claims and eligible_claims:
        print(f"\n[Step 3/3] [{time.strftime('%H:%M:%S')}] Sending top {min(len(eligible_claims), max_claims_to_verify)} selected factual claim(s) to Evidence Engine...")
        if run_research_pipeline is not None:
            for c_idx, claim_obj in enumerate(eligible_claims[:max_claims_to_verify], 1):
                c_text = claim_obj["claim"]
                p_score = claim_obj.get("priority_score", 0.0)
                s_mod = claim_obj.get("source_modality", "unknown")
                print(f"   Researching Claim #{c_idx}: \"{c_text[:80]}...\" (Priority: {p_score}, Modality: {s_mod})")
                try:
                    res = run_research_pipeline(
                        c_text,
                        extracted_text=_build_video_extracted_text(
                            transcript, eligible_claims[:max_claims_to_verify]
                        ),
                    )
                    claim_v_obj = res.get("final_verdict") or {}
                    claim_v = claim_v_obj.get("verdict")
                    confidence = claim_v_obj.get("confidence")
                    verification_results.append({
                        "claim_id": claim_obj["claim_id"],
                        "claim_text": c_text,
                        "priority_score": p_score,
                        "source_modality": s_mod,
                        "selection_reason": claim_obj.get("selection_reason", ""),
                        "claim_verdict": claim_v,
                        "confidence": confidence,
                        "evidence_sufficiency": res.get("fusion_report", {}).get("evidence_sufficiency", "UNKNOWN"),
                        "verification": res,
                    })
                    # Extract verdict with strict priority:
                    # CONTRADICTED (highest severity) > SUPPORTED > INCONCLUSIVE
                    norm_v = (claim_v or "INCONCLUSIVE").upper()
                    if norm_v in ("CONTRADICTED", "FAKE", "FALSE"):
                        primary_verdict = claim_v
                        primary_claim_text = c_text
                    elif norm_v in ("SUPPORTED", "REAL", "TRUE") and primary_verdict not in ("CONTRADICTED", "FAKE", "FALSE"):
                        primary_verdict = claim_v
                        primary_claim_text = c_text
                    elif primary_verdict == "INCONCLUSIVE":
                        primary_verdict = claim_v or "INCONCLUSIVE"
                        primary_claim_text = c_text
                except Exception as exc:
                    verification_results.append({
                        "claim_id": claim_obj["claim_id"],
                        "claim_text": c_text,
                        "priority_score": p_score,
                        "source_modality": s_mod,
                        "error": str(exc),
                    })
                    print(f"   Evidence research failed for claim: {exc}")
        else:
            print("   Note: run_research_pipeline could not be imported; skipping live web research.")
            research_skipped_reason = "Evidence engine (run_research_pipeline) could not be imported."
    elif verify_claims and not eligible_claims and candidate_claims_payload:
        print("\n[Step 3/3] No verifiable factual claims passed filtering (all candidates were conversational filler, greetings, apologies, or noise). Web research skipped.")
        primary_verdict = "INSUFFICIENT_VERIFIABLE_CLAIMS"
        research_skipped_reason = (
            "No verifiable factual claims passed filtering. All candidates were conversational "
            "filler, greetings, apologies, or noise. Web research skipped."
        )
    elif not candidate_claims_payload:
        print("\n[Step 3/3] No candidate claims extracted from audio/OCR; claim verification skipped.")
        primary_verdict = "INSUFFICIENT_VERIFIABLE_CLAIMS"
        research_skipped_reason = "No candidate claims were extracted from the audio or OCR."
    else:
        print("\n[Step 3/3] Live claim verification skipped (--skip-research).")
        research_skipped_reason = "Live claim verification was skipped by configuration."

    step_timings["claim_research"] = round(time.time() - t_step3, 2)
    step_timings["total_verification_sec"] = round(time.time() - t_verify_start, 2)

    # ========================================================
    # STEP 4: MULTI-MODAL SYNTHESIS & CONSISTENCY ANALYSIS
    # ========================================================
    consistency_analysis = analyze_multimodal_consistency(visual_signal, primary_verdict)

    final_report = {
        "video_path": str(v_path),
        "video_metadata": pipeline_summary.get("video_metadata", {}),
        "visual_classification": visual_signal,
        "candidate_claims_count": len(candidate_claims_payload),
        "selected_claims_count": len(eligible_claims),
        "rejected_claims_count": len(candidate_claims_payload) - len(eligible_claims),
        "candidate_claims": candidate_claims_payload,
        "verified_claims": verification_results,
        "final_evidence_verdict": primary_verdict,
        "primary_claim_analyzed": primary_claim_text,
        "consistency_analysis": consistency_analysis,
        "step_timings": step_timings,
        "pipeline_stage_timings": pipeline_summary.get("stage_timings", {}),
        "summary_artifacts": {
            "transcript_json": pipeline_summary.get("transcript_json_path"),
            "ocr_json": pipeline_summary.get("ocr_json_path"),
            "fused_transcript": pipeline_summary.get("fused_transcript_path"),
            "candidate_claims_json": pipeline_summary.get("candidate_claims_path"),
        },
        # Structured forensic artifacts exposed for the UI/report layer
        "research_skipped_reason": research_skipped_reason,
        "transcription_segments": transcript.get("segments") or [],
        "transcription_language": transcript.get("language"),
        "warnings": warnings,
        "errors": errors,
    }

    print("\n" + "=" * 65)
    print(f"VERIFICATION COMPLETE in {step_timings['total_verification_sec']}s")
    print("=" * 65)
    print(f"Final Evidence-Based Verdict : {primary_verdict}")
    print(f"Visual Classifier Signal     : {visual_signal.get('predicted_label', 'UNAVAILABLE')} "
          f"({visual_signal.get('confidence', 'N/A')})")
    print(f"Multi-Modal Consistency      : {consistency_analysis['status']}")
    print(f"Note                         : {consistency_analysis['description']}")
    if consistency_analysis.get("requires_investigation"):
        print("ALERT: Divergence detected between visual signal and factual evidence. Review recommended.")
    print("=" * 65 + "\n")

    return final_report


def _build_video_extracted_text(
    transcript: Optional[dict],
    claims: Optional[List[dict]],
) -> str:
    """
    Build a concise extracted-text representation for video:
    Whisper transcript text, candidate claim statements, and any on-screen
    OCR context already collected. No transcription or OCR is re-run here.
    """
    parts = []

    transcript_text = ""
    if isinstance(transcript, dict):
        transcript_text = str(transcript.get("text") or "").strip()
        if not transcript_text:
            segments = transcript.get("segments") or []
            transcript_text = " ".join(
                str(seg.get("text", "")).strip() for seg in segments if seg
            ).strip()
    if transcript_text:
        parts.append("Transcript: " + transcript_text)

    if claims:
        claim_texts = []
        ocr_contexts = []
        for claim in (claims or []):
            claim_text = claim.get("claim") or claim.get("original_text")
            if claim_text:
                claim_texts.append(str(claim_text).strip())
            on_screen = claim.get("on_screen_context")
            if on_screen:
                ocr_contexts.append(str(on_screen).strip())
        if claim_texts:
            parts.append("Claims: " + " | ".join(claim_texts))
        if ocr_contexts:
            parts.append("On-screen text: " + " | ".join(ocr_contexts))

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="TruthLensAI Video Verification Orchestrator")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--disable-classifier", action="store_true", help="Disable experimental visual classifier")
    parser.add_argument("--skip-research", action="store_true", help="Skip live web search and research on candidate claims")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip on-screen frame OCR")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--classifier-family", type=str, default="videomae", choices=["videomae", "efficientnet"],
                        help="Visual classifier architecture ('videomae' or 'efficientnet')")
    parser.add_argument("--classifier-checkpoint", type=str, default=None,
                        help="Custom path to visual classifier checkpoint (.pt)")
    parser.add_argument("--max-ocr-frames", type=int, default=15,
                        help="Upper limit on video frames processed by OCR via uniform sampling (default: 15)")
    args = parser.parse_args()

    res = verify_video(
        video_path=args.video_path,
        enable_classifier=not args.disable_classifier,
        verify_claims=not args.skip_research,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        skip_ocr=args.skip_ocr,
        classifier_family=args.classifier_family,
        classifier_checkpoint=args.classifier_checkpoint,
        max_ocr_frames=args.max_ocr_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
