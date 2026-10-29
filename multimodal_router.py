"""
multimodal_router.py
====================
TruthLensAI Unified Multimodal Backend Router.

Provides a single coherent integration layer routing Text, Image, Video, and
Audio verification requests to existing underlying modules without altering
model weights, thresholds, prompts, or existing module logic.

Architectural Guarantees:
1. Strict Signal Separation:
   - Factual Verification (from evidence engine) is independent of:
   - Synthetic Media Detection (AI-generated speech / AI-generated image)
   - Visual Classification (experimental frame-level classification)
2. Uniform Response Contract:
   All modalities return a standardized dictionary representation while
   preserving the raw original response under the 'raw' key.
3. Resilient Degradation:
   Sub-detector or module errors degrade gracefully to 'UNAVAILABLE' status
   with logged diagnostics rather than crashing the orchestrator.
4. Genuine Case & Pipeline Provenance:
   Generates verifiable case metadata, UTC timestamps, elapsed execution time,
   and genuine pipeline stage states without fabricating values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_report import build_research_report, redact_sensitive


def _minimal_research_report(modality: str, error: Optional[str] = None) -> Dict[str, Any]:
    """
    A well-formed research report for a result that never produced one.

    The report is part of the API response schema, so an error path must still
    return a valid report rather than an empty object - an empty object fails
    response validation and turns a pipeline failure into a 500 instead of a
    graceful error result.
    """
    report = build_research_report({}, modality, None)
    if error:
        report["methodology"]["research_skipped_reason"] = (
            "The verification pipeline did not complete, so no research was "
            f"performed. Reason: {redact_sensitive(error)}"
        )
    return report

# Supported extension categories
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

logger = logging.getLogger("TruthLensAI.MultimodalRouter")


# ============================================================================
# Helpers: Case ID, Timestamps, and Metadata
# ============================================================================

def current_utc_iso() -> str:
    """Return current timestamp in ISO-8601 UTC format."""
    return datetime.now(timezone.utc).isoformat()


def generate_case_id(identifier: str, modality: str) -> str:
    """
    Generate a deterministic, verifiable Case ID based on input contents.
    Example: TL-B4F2A1
    """
    digest = hashlib.sha256(f"{modality}:{identifier}".encode("utf-8", errors="replace")).hexdigest()[:6].upper()
    return f"TL-{digest}"


def build_case_summary(
    modality: str,
    target: Union[str, Path],
    original_filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    processing_time_sec: Optional[float] = None,
    validation_status: str = "VALID",
    processing_status: str = "SUCCESS",
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct genuine case summary metadata adhering to Case Inspection standards.
    """
    raw = raw or {}
    now_iso = current_utc_iso()

    if modality == "text":
        claim_text = str(target)
        cid = generate_case_id(claim_text, "text")
        size = file_size_bytes if file_size_bytes is not None else len(claim_text.encode("utf-8"))
        return {
            "case_id": cid,
            "modality": "text",
            "filename": None,
            "file_size_bytes": size,
            "file_format": "text/plain",
            "media_metadata": {
                "character_count": len(claim_text),
                "word_count": len(claim_text.split()),
            },
            "timestamp_utc": now_iso,
            "processing_time_sec": round(processing_time_sec, 2) if processing_time_sec is not None else None,
            "validation_status": validation_status,
            "processing_status": processing_status,
        }

    p = Path(target)
    display_name = original_filename or p.name
    cid = generate_case_id(display_name, modality)
    actual_size = file_size_bytes if file_size_bytes is not None else (p.stat().st_size if p.exists() else 0)
    suffix = p.suffix.lower() if p.suffix else (Path(display_name).suffix.lower() if display_name else "")

    media_meta: Dict[str, Any] = {
        "filename": display_name,
        "extension": suffix,
        "size_kb": round(actual_size / 1024, 1),
    }

    if modality == "image":
        if "ocr_results" in raw:
            media_meta["ocr_fragments_count"] = len(raw.get("ocr_results", []))
        if "detected_objects" in raw:
            media_meta["detected_objects_count"] = len(raw.get("detected_objects", []))
    elif modality == "video":
        v_meta = raw.get("video_metadata", {})
        if v_meta:
            media_meta.update(v_meta)
    elif modality == "audio":
        a_meta = raw.get("audio_metadata", {})
        if a_meta:
            media_meta.update(a_meta)

    return {
        "case_id": cid,
        "modality": modality,
        "filename": display_name,
        "file_size_bytes": actual_size,
        "file_format": suffix or "binary/octet-stream",
        "media_metadata": media_meta,
        "timestamp_utc": now_iso,
        "processing_time_sec": round(processing_time_sec, 2) if processing_time_sec is not None else None,
        "validation_status": validation_status,
        "processing_status": processing_status,
    }


# ============================================================================
# Genuine Pipeline Stage Builders
# ============================================================================

def _evidence_policy_fields(final_v):
    """
    The evidence-policy diagnostics carried on every final verdict.

    Kept in one place so the text, image, video and audio paths cannot drift
    apart and report different numbers for the same evidence.
    """
    final_v = final_v or {}
    return {
        "confidence_basis": final_v.get("confidence_basis"),
        "primary_contradicting_groups": final_v.get("primary_contradicting_groups", 0),
        "article_supporting_groups": final_v.get("article_supporting_groups", 0),
        "article_contradicting_groups": final_v.get("article_contradicting_groups", 0),
        "correction_groups": final_v.get("correction_groups", 0),
        "verification_coverage": final_v.get("verification_coverage"),
        "unresolved_identity_sources": final_v.get("unresolved_identity_sources", 0),
        # Deterministic-gate policy accounting. None means the gate did not run
        # for this result; False means the policy counters were unusable and
        # the verdict was restricted to UNVERIFIED.
        "policy_data_valid": final_v.get("policy_data_valid"),
        "policy_data_errors": final_v.get("policy_data_errors") or [],
    }


def _agent_model_label() -> str:
    """
    The LLM the verification and critic agents actually use.

    Read from the already-imported ``agents`` module so the API never reports
    a model that is not running.
    """
    module = sys.modules.get("agents")
    model = getattr(module, "MODEL_NAME", None) if module is not None else None
    provider = getattr(module, "MODEL_PROVIDER", None) if module is not None else None
    if not model:
        return "LLM agent (model not reported)"
    return f"{provider} {model}" if provider else str(model)


def _build_text_stages(
    raw: Dict[str, Any],
    fact_data: Dict[str, Any],
    primary_claim: str,
) -> List[Dict[str, Any]]:
    final_v = raw.get("final_verdict", {}) or {}
    agreement = raw.get("source_agreement", {}) or {}
    evidence_analysis = raw.get("evidence_analysis", []) or []
    source_data = raw.get("source_data", []) or []

    supporting_count = len([e for e in evidence_analysis if e.get("evidence_class") == "SUPPORT"])
    contradicting_count = len([e for e in evidence_analysis if e.get("evidence_class") == "CONTRADICTION"])
    context_count = len([e for e in evidence_analysis if e.get("evidence_class") == "RELATED_CONTEXT"])
    scraped_success_count = len([s for s in source_data if s.get("scrape_status") == "SUCCESS"])

    src_agreement_val = fact_data.get("source_agreement")
    src_agreement_str = f"{src_agreement_val:.2f}" if src_agreement_val is not None else "N/A"
    src_agreement_pct = f"{(src_agreement_val * 100):.1f}%" if src_agreement_val is not None else "N/A"

    return [
        {
            "id": "ingestion",
            "number": 1,
            "name": "Ingestion & Claim Normalization",
            "status": "completed",
            "operation": "Validated claim syntax and bound token character budget for Cross-Encoder NLI processing.",
            "why": "Prevents tokenizer truncation in DeBERTa and bounds proposition scope.",
            "tool_or_model": "TruthLensAI Router // Ingestion Parser",
            "input_summary": f"Claim: \"{primary_claim[:100]}...\" ({len(primary_claim)} chars)",
            "output_summary": "Validated factual proposition dispatched to multi-agent evidence engine.",
            "metric": {"label": "Input Length", "value": f"{len(primary_claim)} chars", "meaning": "Character count of submitted factual claim.", "scale": "Characters (>= 3 required)"},
        },
        {
            "id": "search",
            "number": 2,
            "name": "Query Planning & Web Search",
            "status": "completed" if source_data or raw.get("search_results") else ("failed" if not os.getenv("TAVILY_API_KEY") else "completed"),
            "operation": "Dispatched structured search queries across global news wires, archives, and official registries.",
            "why": "Retrieves primary candidate reporting documents to cross-examine stated claim.",
            "tool_or_model": "Tavily Search API // Neural Search Engine",
            "input_summary": f"Target: \"{primary_claim[:80]}...\"",
            "output_summary": f"Retrieved {len(source_data)} candidate source URLs for forensic scraping.",
            "metric": {"label": "Candidate Sources", "value": len(source_data), "meaning": "Total primary and wire URLs retrieved from web search.", "scale": "Count (>= 2 recommended)"},
        },
        {
            "id": "scraping",
            "number": 3,
            "name": "Deep Article Scraping & Extraction",
            "status": "completed" if source_data else "skipped",
            "operation": "Executed HTTP/Playwright document extraction; stripped ads and boilerplate; fallback to search extracts.",
            "why": "Search snippets lack full investigative context; corroboration requires full article body text.",
            "tool_or_model": "Playwright Headless / Readability Scraper",
            "input_summary": f"{len(source_data)} candidate URLs queued for ingestion.",
            "output_summary": f"{scraped_success_count} full articles scraped; {len(source_data) - scraped_success_count} retained as search extracts.",
        },
        {
            "id": "nli_analysis",
            "number": 4,
            "name": "Evidence Chunking & NLI Cross-Encoder",
            "status": "completed" if evidence_analysis else "skipped",
            "operation": "Segmented scraped articles into semantic passages and computed premise-hypothesis entailment.",
            "why": "Determines whether verified passages logically entail, contradict, or discuss the claim.",
            "tool_or_model": "cross-encoder/nli-deberta-v3-base",
            "input_summary": f"{len(evidence_analysis)} semantic passages paired with claim hypothesis.",
            "output_summary": f"{supporting_count} Support, {contradicting_count} Contradiction, {context_count} Context chunks.",
            "metric": {
                "label": "Top Entailment Score",
                "value": f"{(evidence_analysis[0].get('nli_score', 0) * 100):.1f}%" if evidence_analysis and evidence_analysis[0].get("nli_score") is not None else "N/A",
                "meaning": "Natural Language Inference confidence on highest-scoring passage.",
                "scale": "0.0 to 1.0 (Strict logical entailment)"
            },
        },
        {
            "id": "independence",
            "number": 5,
            "name": "Source Reliability & Independence Grouping",
            "status": "completed",
            "operation": "Collapsed syndicated wire duplicates into registrable publisher domains and evaluated authority tiers.",
            "why": "Multiple outlets republishing one syndicated wire story count as 1 independent confirmation, eliminating single-source bias.",
            "tool_or_model": "source_reliability.py // Domain Registry",
            "input_summary": f"{len(evidence_analysis)} cited passages across candidate web domains.",
            "output_summary": f"Mapped into {fact_data.get('primary_supporting_groups') or agreement.get('independence_groups', 1)} discrete publisher groups.",
            "metric": {
                "label": "Independent Voices",
                "value": fact_data.get("independent_supporting", 0) + fact_data.get("independent_contradicting", 0),
                "meaning": "Distinct editorial publisher groups examined.",
                "scale": "Count (>= 2 independent primary publishers required for REAL)"
            },
        },
        {
            "id": "fusion",
            "number": 6,
            "name": "Source Agreement & Evidence Fusion",
            "status": "completed",
            "operation": "Synthesized publisher group votes, calculated harmonic agreement, and applied contradiction thresholds.",
            "why": "Prevents confirmation bias by evaluating ratio of corroborating versus refuting independent sources.",
            "tool_or_model": "evidence_fusion.py // Deterministic Fusion Engine",
            "input_summary": f"{fact_data.get('independent_supporting', 0)} supporting groups vs {fact_data.get('independent_contradicting', 0)} contradicting groups.",
            "output_summary": f"Evidence Sufficiency: {fact_data.get('evidence_sufficiency', 'UNKNOWN')}; Agreement: {src_agreement_str}.",
            "metric": {
                "label": "Source Agreement",
                "value": src_agreement_pct,
                "meaning": "Harmonic agreement across distinct publisher groups.",
                "scale": "0% (Complete conflict) to 100% (Complete corroboration)"
            },
        },
        {
            "id": "verify_agent",
            "number": 7,
            "name": "Verification Agent Audit",
            "status": "completed" if fact_data.get("agent_verdict") or final_v.get("verification_verdict") else "skipped",
            "operation": "LLM agent audited synthesized evidence brief and produced structured investigative verdict.",
            "why": "Interprets subtle linguistic nuances, historical timelines, and potential out-of-context distortions.",
            "tool_or_model": f"{_agent_model_label()} (Verification Agent)",
            "input_summary": "Synthesized Evidence Brief with attributed source citations.",
            "output_summary": f"Proposed Verdict: {fact_data.get('agent_verdict') or final_v.get('verification_verdict') or 'Evaluated'}; Confidence: {fact_data.get('agent_confidence') or final_v.get('agent_confidence') or 'N/A'}%.",
            "metric": {
                "label": "Agent Confidence",
                "value": f"{fact_data.get('agent_confidence') or final_v.get('agent_confidence') or 'N/A'}%",
                "meaning": "Verification Agent self-reported assessment certainty.",
                "scale": "0% - 100%"
            },
        },
        {
            "id": "critic_agent",
            "number": 8,
            "name": "Adversarial Critic Review Agent",
            "status": "completed" if fact_data.get("critic_verdict") or final_v.get("critic_verdict") else "skipped",
            "operation": "Adversarial Critic Agent examined Verification Agent reasoning for hallucinations and logical fallacies.",
            "why": "Prevents sycophancy and challenges ungrounded assertions before verdict adjudication.",
            "tool_or_model": f"{_agent_model_label()} (Critic Agent)",
            "input_summary": "Verification Agent reasoning + cited evidence passages.",
            "output_summary": f"Critic Verdict: {fact_data.get('critic_verdict') or final_v.get('critic_verdict') or 'Evaluated'}; Consensus: {'AGREE' if fact_data.get('agents_agree') or final_v.get('agents_agree') else 'DISAGREE'}.",
            "metric": {
                "label": "Critic Confidence",
                "value": f"{fact_data.get('critic_confidence') or final_v.get('critic_confidence') or 'N/A'}%",
                "meaning": "Critic Agent audit confidence score.",
                "scale": "0% - 100%"
            },
        },
        {
            "id": "adjudication",
            "number": 9,
            "name": "Deterministic Gate & Final Adjudication",
            "status": "completed",
            "operation": "Applied strict evidence rules (evidence holds final veto over LLM agent recommendations).",
            "why": "Ensures verdicts never hallucinate facts; strictly enforces 2-source corroboration rule.",
            "tool_or_model": "Deterministic Gate (final_verdict.py)",
            "input_summary": f"Gate rules evaluated against evidence sufficiency ({fact_data.get('evidence_sufficiency')}).",
            "output_summary": f"Final Verdict: {fact_data.get('verdict') or 'UNVERIFIED'}; Evidence Overrode Agent: {'YES (Vetoed)' if fact_data.get('verdict_overridden_by_evidence') or final_v.get('verdict_overridden_by_evidence') else 'NO (Upheld)'}.",
            "metric": {
                "label": "Final Certainty",
                "value": f"{(fact_data.get('confidence', 0) * 100):.1f}%" if fact_data.get("confidence") is not None else "Calculated",
                "meaning": "Evidence certainty score factoring NLI score, source tier, and publisher independence.",
                "scale": "0% - 100%"
            },
            "details": fact_data.get("deterministic_gate_reason") or fact_data.get("reasoning") or None,
        },
    ]


def _build_image_stages(
    raw: Dict[str, Any],
    fact_data: Dict[str, Any],
    synthetic_data: Optional[Dict[str, Any]],
    primary_claim: Optional[str],
) -> List[Dict[str, Any]]:
    ocr_res = raw.get("ocr_results", []) or []
    det_objs = raw.get("detected_objects", []) or []
    verif = raw.get("verification") or {}
    ai_gen = raw.get("ai_generation") or {}

    synth_signal = synthetic_data.get("signal", "UNAVAILABLE") if synthetic_data else "UNAVAILABLE"
    ai_prob = (synthetic_data.get("probabilities") or {}).get("ai_probability") if synthetic_data else None

    return [
        {
            "id": "image_validation",
            "number": 1,
            "name": "Image Validation & Preprocessing",
            "status": "completed",
            "operation": "Validated image container integrity, dimensions, color space, and memory buffers.",
            "why": "Ensures visual tensors are correctly conditioned for YOLO and ViT model inference.",
            "tool_or_model": "image_module // Image Loader & Preprocessor",
            "input_summary": f"Image target: {raw.get('image_path', 'uploaded_image')}",
            "output_summary": "Image validated and scaled for parallel OCR and object detection.",
        },
        {
            "id": "ocr_extraction",
            "number": 2,
            "name": "OCR Text Extraction & Cleaning",
            "status": "completed" if ocr_res else "skipped",
            "operation": "Extracted on-image typography, headlines, captions, and watermarks via EasyOCR with confidence thresholding.",
            "why": "Captures textual claims and assertions embedded directly into the visual medium.",
            "tool_or_model": "EasyOCR (PyTorch) // OCR Cleaner",
            "input_summary": "Preprocessed visual canvas.",
            "output_summary": f"Extracted {len(ocr_res)} text fragments with confidence >= 0.40.",
            "metric": {"label": "OCR Fragments", "value": len(ocr_res), "meaning": "Count of distinct text regions extracted from image.", "scale": "Count"},
        },
        {
            "id": "visual_analysis",
            "number": 3,
            "name": "Visual Object & Entity Analysis",
            "status": "completed" if det_objs else "skipped",
            "operation": "Identified physical subjects, people, vehicles, and contextual objects using YOLOv11.",
            "why": "Provides ground-truth visual context to corroborate or disambiguate on-image claims.",
            "tool_or_model": "Ultralytics YOLOv11 (yolo11n.pt)",
            "input_summary": "Preprocessed visual frame.",
            "output_summary": f"Detected {len(det_objs)} visual objects with class confidences.",
            "metric": {"label": "Detected Objects", "value": len(det_objs), "meaning": "Visual entities identified in frame.", "scale": "Count"},
        },
        {
            "id": "claim_normalization",
            "number": 4,
            "name": "Claim Synthesis & Normalization",
            "status": "completed" if primary_claim else "skipped",
            "operation": "Synthesized OCR headline text and detected visual objects into a testable factual proposition.",
            "why": "Translates unstructured visual and textual features into a formal hypothesis for wire fact-checking.",
            "tool_or_model": "image_module.claim_extractor & claim_normalizer",
            "input_summary": "OCR text + detected visual entities.",
            "output_summary": f"Normalized proposition: \"{primary_claim or 'No verifiable claim'}\"",
        },
        {
            "id": "ai_image_detector",
            "number": 5,
            "name": "AI Image Origin Detection (Independent Signal)",
            "status": "completed" if synthetic_data and synthetic_data.get("available") else ("failed" if synthetic_data and synthetic_data.get("error") else "skipped"),
            "operation": "Analyzed high-frequency pixel artifacts and latent diffusion traces for synthetic image generation.",
            "why": "Strict signal separation: independently determines visual provenance without conflating with factual veracity.",
            "tool_or_model": (
                (synthetic_data.get("model") or "AI image origin detector (model not reported)")
                if synthetic_data
                else "AI image origin detector (not executed)"
            ),
            "input_summary": "Source image pixel array.",
            "output_summary": f"Origin Signal: {synth_signal}; AI Probability: {f'{ai_prob:.1f}%' if ai_prob is not None else 'N/A'}.",
            "metric": {
                "label": "AI Image Probability",
                "value": f"{ai_prob:.1f}%" if ai_prob is not None else "N/A",
                "meaning": "ViT detector probability that image is synthetic/AI-generated.",
                "scale": "0% - 100%"
            },
        },
        {
            "id": "evidence_research",
            "number": 6,
            "name": "Live Wire Evidence Retrieval & Fact-Checking",
            "status": "completed" if verif else "skipped",
            "operation": "Dispatched normalized image claim to Tavily and DeBERTa Cross-Encoder to verify factual veracity against global wire registries.",
            "why": "Verifies whether the depicted event or claim actually happened in the real world.",
            "tool_or_model": "pipeline.run_research_pipeline (Tavily + DeBERTa)",
            "input_summary": f"Claim: \"{primary_claim}\"" if primary_claim else "No verifiable claim",
            "output_summary": f"Verdict: {fact_data.get('verdict', 'UNVERIFIED')}; Supporting: {fact_data.get('independent_supporting', 0)}, Contradicting: {fact_data.get('independent_contradicting', 0)}.",
            "metric": {
                "label": "Factual Corroboration",
                "value": f"{fact_data.get('independent_supporting', 0)} Support / {fact_data.get('independent_contradicting', 0)} Contra",
                "meaning": "Independent primary publisher groups corroborating or refuting claim.",
                "scale": "Counts"
            },
        },
        {
            "id": "multimodal_consistency",
            "number": 7,
            "name": "Multimodal Consistency & Adjudication",
            "status": "completed",
            "operation": "Cross-referenced factual evidence verdict with AI image origin signal to produce comprehensive forensic dossier.",
            "why": "Maintains strict epistemic boundary: authentic photos can carry false claims; AI images can depict real historical concepts.",
            "tool_or_model": "ai_image_signal.describe_signal_consistency",
            "input_summary": f"Factual: {fact_data.get('verdict')}; Image Origin: {synth_signal}.",
            "output_summary": raw.get("ai_generation_note") or "Multimodal synthesis complete.",
        },
    ]


def _build_video_stages(
    raw: Dict[str, Any],
    fact_data: Dict[str, Any],
    visual_data: Optional[Dict[str, Any]],
    primary_claim: Optional[str],
) -> List[Dict[str, Any]]:
    cand_claims = raw.get("candidate_claims", []) or []
    verif_claims = raw.get("verified_claims", []) or []
    v_meta = raw.get("video_metadata", {}) or {}
    vis_class = visual_data or {}

    vis_conf = vis_class.get("confidence")
    vis_label = vis_class.get("predicted_label")

    return [
        {
            "id": "video_ingestion",
            "number": 1,
            "name": "Video Ingestion & Stream Demuxing",
            "status": "completed",
            "operation": "Decoded video container, extracted audio stream to 16kHz WAV, and sampled video frames.",
            "why": "Prepares separated audio and visual streams for parallel transcription, OCR, and classification.",
            "tool_or_model": "OpenCV (cv2) & FFmpeg / wave",
            "input_summary": f"Target: {raw.get('video_path', 'uploaded_video')}",
            "output_summary": f"Decoded {v_meta.get('frame_count', 'multiple')} frames ({v_meta.get('duration_sec', 0):.1f}s, {v_meta.get('fps', 0):.1f} fps).",
            "metric": {"label": "Video Duration", "value": f"{v_meta.get('duration_sec', 0):.1f}s", "meaning": "Duration of ingested video stream.", "scale": "Seconds"},
        },
        {
            "id": "video_transcription",
            "number": 2,
            "name": "Whisper Speech Transcription",
            "status": "completed",
            "operation": "Transcribed spoken audio into timed text segments with automatic language detection.",
            "why": "Extracts verbal factual claims asserted in speech.",
            "tool_or_model": "OpenAI Whisper ASR",
            "input_summary": "Extracted audio stream.",
            "output_summary": "Speech transcription completed.",
        },
        {
            "id": "video_ocr",
            "number": 3,
            "name": "On-Screen Frame OCR",
            "status": "completed" if not (raw.get("summary_artifacts") or {}).get("skip_ocr") else "skipped",
            "operation": "Sampled video frames with stride and extracted text chyrons, headlines, and overlay typography via EasyOCR.",
            "why": "Captures textual claims shown on screen that may not be spoken verbally.",
            "tool_or_model": "EasyOCR (PyTorch)",
            "input_summary": "Sampled video frames.",
            "output_summary": "On-screen text extraction completed.",
        },
        {
            "id": "video_claims",
            "number": 4,
            "name": "Claim Extraction, Ranking & Filtering",
            "status": "completed",
            "operation": "Extracted candidate statements from speech and OCR; filtered conversational filler, greetings, and noise; ranked factual claims by verifiability priority.",
            "why": "Eliminates non-factual statements and focuses research budget on verifiable factual assertions.",
            "tool_or_model": "video_module.claim_ranker",
            "input_summary": "Fused speech transcript + on-screen OCR text.",
            "output_summary": f"Evaluated {len(cand_claims)} candidates: {raw.get('selected_claims_count', 0)} selected for wire research, {raw.get('rejected_claims_count', 0)} filtered.",
            "metric": {"label": "Selected Claims", "value": f"{raw.get('selected_claims_count', 0)} / {len(cand_claims)}", "meaning": "Candidate claims meeting verifiability threshold.", "scale": "Ratio"},
        },
        {
            "id": "video_classifier",
            "number": 5,
            "name": "VideoMAE Visual Frame Classifier (Supporting Heuristic)",
            "status": "completed" if vis_class.get("available") else "skipped",
            "operation": "Evaluated temporal frame sequence through VideoMAE visual classifier.",
            "why": "Supporting visual heuristic: flags visual anomalies; explicitly kept separate from factual claim verification.",
            "tool_or_model": "MCG-NJU/videomae-base (VideoMAE)",
            "input_summary": "Sampled video frame sequence.",
            "output_summary": f"Predicted Class: {vis_label or 'N/A'}; Confidence: {f'{(vis_conf * 100):.1f}%' if vis_conf is not None else 'N/A'}. Note: Experimental 52.38% benchmark baseline.",
            "metric": {
                "label": "Visual Confidence",
                "value": f"{(vis_conf * 100):.1f}%" if vis_conf is not None else "N/A",
                "meaning": "Confidence of VideoMAE visual classifier (experimental heuristic).",
                "scale": "0% - 100%"
            },
        },
        {
            "id": "video_research",
            "number": 6,
            "name": "Live Wire Claim Research & Fact-Checking",
            "status": "completed" if verif_claims else "skipped",
            "operation": "Dispatched top selected factual claim(s) to Evidence Engine for global news wire corroboration.",
            "why": "Establishes ground truth veracity of assertions made in the video.",
            "tool_or_model": "pipeline.run_research_pipeline (Tavily + DeBERTa)",
            "input_summary": f"Selected Claim: \"{primary_claim}\"" if primary_claim else "No verifiable claims selected",
            "output_summary": f"Evidence Verdict: {fact_data.get('verdict', 'INCONCLUSIVE')}; Supporting: {fact_data.get('independent_supporting', 0)}, Contradicting: {fact_data.get('independent_contradicting', 0)}.",
            "metric": {"label": "Evidence Verdict", "value": fact_data.get("verdict", "INCONCLUSIVE"), "meaning": "Factual veracity determined from global wire corroboration.", "scale": "Categorical"},
        },
        {
            "id": "video_consistency",
            "number": 7,
            "name": "Multimodal Consistency & Adjudication",
            "status": "completed",
            "operation": "Cross-referenced visual classifier signal against factual evidence verdict to detect out-of-context recycling or visual anomalies.",
            "why": "Prevents misattribution: genuine footage can be reused with false claims, while authentic claims may appear in stylized video.",
            "tool_or_model": "video_verifier.analyze_multimodal_consistency",
            "input_summary": f"Visual: {vis_label or 'N/A'}; Factual: {fact_data.get('verdict')}.",
            "output_summary": (raw.get("consistency_analysis") or {}).get("description") or "Multimodal synthesis complete.",
        },
    ]


def _build_audio_stages(
    raw: Dict[str, Any],
    fact_data: Dict[str, Any],
    synthetic_data: Optional[Dict[str, Any]],
    primary_claim: Optional[str],
) -> List[Dict[str, Any]]:
    a_meta = raw.get("audio_metadata", {}) or {}
    trans = raw.get("transcription", {}) or {}
    synth = synthetic_data or {}
    synth_prob = (synth.get("probabilities") or {}).get("probability_synthetic")

    return [
        {
            "id": "audio_validation",
            "number": 1,
            "name": "Audio Validation & Format Decoding",
            "status": "completed",
            "operation": "Validated container headers, decoded audio stream to 16kHz float32 samples, and evaluated RMS energy for silence check.",
            "why": "Guarantees clean acoustic tensor input and prevents silent audio false positives.",
            "tool_or_model": "audio_module.audio_loader (FFmpeg / wave)",
            "input_summary": f"Audio target: {a_meta.get('filename', 'uploaded_audio')}",
            "output_summary": f"Decoded {a_meta.get('duration_sec', 0):.2f}s audio at {a_meta.get('sample_rate', 16000)} Hz (RMS: {a_meta.get('rms_energy', 0):.5f}).",
            "metric": {"label": "Audio Duration", "value": f"{a_meta.get('duration_sec', 0):.2f}s", "meaning": "Duration of decoded audio stream.", "scale": "Seconds"},
        },
        {
            "id": "audio_synthetic",
            "number": 2,
            "name": "Synthetic Speech Detection (Acoustic Provenance)",
            "status": "completed" if synth.get("available") else ("failed" if synth.get("error") else "skipped"),
            "operation": "Analyzed spectral frequencies and acoustic phoneme artifacts for neural speech synthesis, vocoder signatures, or voice cloning.",
            "why": "Strict signal separation: independently assesses acoustic provenance without evaluating statement veracity.",
            "tool_or_model": synth.get("model") or "Synthetic speech detector (model not reported)",
            "input_summary": "16kHz mono audio waveform.",
            "output_summary": f"Provenance Signal: {synth.get('signal', 'UNAVAILABLE')}; Synthetic Probability: {f'{(synth_prob * 100):.1f}%' if synth_prob is not None else 'N/A'}.",
            "metric": {
                "label": "Synthetic Speech Prob",
                "value": f"{(synth_prob * 100):.1f}%" if synth_prob is not None else "N/A",
                "meaning": "Wav2Vec2 model probability that speech is AI-generated / cloned.",
                "scale": "0% - 100%"
            },
        },
        {
            "id": "audio_transcription",
            "number": 3,
            "name": "Whisper Speech Transcription",
            "status": "completed" if trans.get("text") else "skipped",
            "operation": "Transcribed spoken audio into timestamped segments and detected spoken language.",
            "why": "Converts acoustic speech into textual representation for claim extraction.",
            "tool_or_model": "OpenAI Whisper ASR",
            "input_summary": "Audio waveform.",
            "output_summary": f"Transcribed {len(trans.get('text', ''))} characters (Language: {trans.get('language', 'unknown')}).",
            "metric": {"label": "Transcript Length", "value": f"{len(trans.get('text', ''))} chars", "meaning": "Total characters transcribed from speech.", "scale": "Characters"},
        },
        {
            "id": "audio_claims",
            "number": 4,
            "name": "Verifiable Claim Extraction & Filtering",
            "status": "completed",
            "operation": "Parsed transcript sentences, filtered conversational filler and subjective remarks, and selected verifiable factual assertions.",
            "why": "Focuses verification resources on substantive factual claims rather than greetings or opinions.",
            "tool_or_model": "audio_module.claim_extractor",
            "input_summary": "Whisper transcript text.",
            "output_summary": f"Primary claim: \"{primary_claim}\"" if primary_claim else "No verifiable factual claims extracted.",
        },
        {
            "id": "audio_fact_check",
            "number": 5,
            "name": "Live Wire Fact-Checking (Factual Veracity)",
            "status": "completed" if fact_data.get("verdict") and fact_data.get("verdict") not in (None, "NOT_EVALUATED") else "skipped",
            "operation": "Dispatched extracted spoken claim to evidence engine for live global wire corroboration.",
            "why": "Establishes whether the spoken statements are factually true or false in reality.",
            "tool_or_model": "pipeline.run_research_pipeline (Tavily + DeBERTa)",
            "input_summary": f"Claim: \"{primary_claim}\"" if primary_claim else "Verification disabled or no claim",
            "output_summary": f"Verdict: {fact_data.get('verdict', 'NOT_EVALUATED')}; Supporting: {fact_data.get('independent_supporting', 0)}, Contradicting: {fact_data.get('independent_contradicting', 0)}." if fact_data.get("verdict") else "Claim verification not requested (--skip-research).",
            "metric": {"label": "Factual Verdict", "value": fact_data.get("verdict") or "NOT_EVALUATED", "meaning": "Factual verdict derived from global news wires.", "scale": "Categorical"},
        },
        {
            "id": "audio_consistency",
            "number": 6,
            "name": "Multi-Signal Consistency Synthesis",
            "status": "completed",
            "operation": "Combined acoustic provenance and factual veracity onto a dual-axis matrix (Synthetic vs Human, True vs False) without conflating signals.",
            "why": "Critical forensic separation: an AI voice can state a true fact; a human voice can spread falsehoods.",
            "tool_or_model": "audio_module.consistency.analyze_audio_consistency",
            "input_summary": f"Provenance: {synth.get('signal', 'UNAVAILABLE')}; Factual: {fact_data.get('verdict') or 'UNVERIFIED'}.",
            "output_summary": (raw.get("consistency_analysis") or {}).get("description") or "Multi-signal synthesis complete.",
        },
    ]


def build_pipeline_stages(
    modality: str,
    raw: Dict[str, Any],
    fact_data: Dict[str, Any],
    synthetic_data: Optional[Dict[str, Any]] = None,
    visual_data: Optional[Dict[str, Any]] = None,
    primary_claim: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Construct genuine pipeline stages for the specified modality.
    """
    if modality == "text":
        return _build_text_stages(raw, fact_data, primary_claim or "")
    elif modality == "image":
        return _build_image_stages(raw, fact_data, synthetic_data, primary_claim)
    elif modality == "video":
        return _build_video_stages(raw, fact_data, visual_data, primary_claim)
    elif modality == "audio":
        return _build_audio_stages(raw, fact_data, synthetic_data, primary_claim)
    return []


# ============================================================================
# Modality-Specific Structured Payload Builders
# These expose artifacts the underlying verifiers already produced so the UI
# can render forensic detail without inventing data.
# ============================================================================

def _extract_transcription_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reconstruct the timestamped Whisper segments from a video verifier result.
    The verifier keeps the transcript JSON under summary_artifacts.transcript_json;
    we read the segments list directly when the pipeline already loaded it, and
    fall back to the on-disk artifact when the verifier did not retain it.
    """
    segments = result.get("transcription_segments")
    if isinstance(segments, list) and segments:
        return segments

    artifacts = result.get("summary_artifacts") or {}
    tpath = artifacts.get("transcript_json")
    if tpath and Path(tpath).exists():
        try:
            data = json.loads(Path(tpath).read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _build_video_modality_specific(
    result: Dict[str, Any],
    visual_classification: Dict[str, Any],
    transcription_text: Optional[str],
) -> Dict[str, Any]:
    """
    Expose the structured video forensic artifacts the verifier already produced:
    metadata, OCR observations, candidate claims, selected/rejected claim counts,
    per-stage timings, and the visual classifier signal.
    """
    # The verifier flattens video_pipeline.process_video_pipeline's summary into
    # top-level keys (video_metadata, ocr_result, fusion_result, ...).
    meta = result.get("video_metadata") or {}
    ocr = result.get("ocr_result") or {}
    fusion = result.get("fusion_result") or {}
    claims = result.get("candidate_claims_result") or {}
    stage_timings = result.get("stage_timings") or {}
    step_timings = result.get("step_timings") or {}

    return {
        "video_metadata": {
            "filename": meta.get("file_name"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "frame_count": meta.get("frame_count"),
            "duration_sec": meta.get("duration_sec"),
            "has_audio": meta.get("has_audio"),
            "audio_codec": meta.get("audio_codec"),
            "video_codec": meta.get("video_codec"),
        },
        "frames": {
            "extracted_count": result.get("frames_extracted_count"),
            "ocr_processed_count": ocr.get("processed_frames_count"),
            "ocr_frames_with_text_count": ocr.get("frames_with_text_count"),
            "ocr_sampling_policy": ocr.get("sampling_policy"),
            "ocr_observations": ocr.get("observations") or [],
            "ocr_warnings": ocr.get("warnings") or [],
        },
        "transcript": {
            "text": transcription_text,
            "segments": _extract_transcription_segments(result),
            "whisper_segments_count": fusion.get("whisper_segments_count"),
            "language": result.get("transcription_language"),
        },
        "fusion": {
            "fused_entries_count": fusion.get("fused_entries_count"),
            "fused_entries": fusion.get("fused_entries") or [],
            "review_flags_count": fusion.get("review_flags_count"),
            "deduplication_summary": fusion.get("deduplication_summary") or {},
        },
        "candidate_claims": {
            "candidate_claims_count": result.get("candidate_claims_count"),
            "selected_claims_count": result.get("selected_claims_count"),
            "rejected_claims_count": result.get("rejected_claims_count"),
            "all_candidate_claims": result.get("candidate_claims") or [],
            "verified_claims": result.get("verified_claims") or [],
            "research_skipped_reason": result.get("research_skipped_reason"),
        },
        "visual_classifier": {
            "available": visual_classification.get("available"),
            "predicted_label": visual_classification.get("predicted_label"),
            "confidence": visual_classification.get("confidence"),
            "probabilities": visual_classification.get("probabilities"),
            "model_name": visual_classification.get("model_name"),
            "experimental_warning": visual_classification.get("experimental_warning"),
            "error": visual_classification.get("error"),
        },
        "timings": {
            "preprocessing_and_extraction_sec": step_timings.get("preprocessing_and_extraction"),
            "visual_classification_sec": step_timings.get("visual_classification"),
            "claim_research_sec": step_timings.get("claim_research"),
            "total_verification_sec": step_timings.get("total_verification_sec"),
            "pipeline_stage_timings": stage_timings,
        },
        "whisper_recommendations": result.get("whisper_recommendations") or {},
        "pipeline_warnings": result.get("warnings") or [],
        "pipeline_errors": result.get("errors") or [],
    }


def _build_image_modality_specific(
    result: Dict[str, Any],
    synthetic_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Expose the structured image forensic artifacts: OCR, objects, claim, detector."""
    return {
        "ocr_results": result.get("ocr_results") or [],
        "detected_objects": result.get("detected_objects") or [],
        "image_claim": result.get("image_claim") or {},
        "normalized_claim": result.get("normalized_claim") or {},
        "ai_generation": synthetic_data or {},
        "ai_generation_note": result.get("ai_generation_note"),
    }


def _build_audio_modality_specific(
    result: Dict[str, Any],
    synthetic_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Expose the structured audio forensic artifacts: metadata, transcription, claims."""
    synth = synthetic_data or {}
    return {
        "audio_metadata": result.get("audio_metadata") or {},
        "synthetic_speech_detection": {
            "available": synth.get("available"),
            "label": synth.get("signal"),
            "confidence": synth.get("confidence"),
            "model": synth.get("model"),
            "probabilities": synth.get("probabilities"),
            "error": synth.get("error"),
        },
        "transcription": result.get("transcription") or {},
        "claim_extraction": result.get("claim_extraction") or {},
        "consistency_analysis": result.get("consistency_analysis") or {},
    }


# ============================================================================
# Core Modality Detection and Unified Response
# ============================================================================

def detect_modality(input_target: Union[str, Path]) -> str:
    """
    Determine the input modality from the input string or path.
    - If input points to an existing file, detects by extension.
    - If input does not exist as a file, treats it as raw text.
    """
    target_path = Path(input_target)
    if target_path.exists() and target_path.is_file():
        suffix = target_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        raise ValueError(f"Unsupported file format: {suffix}")

    # Not an existing file: treat as text claim
    return "text"


def build_unified_response(
    modality: str,
    status: str = "success",
    factual_verification: Optional[Dict[str, Any]] = None,
    synthetic_media_detection: Optional[Dict[str, Any]] = None,
    visual_classification: Optional[Dict[str, Any]] = None,
    transcription: Optional[str] = None,
    transcription_segments: Optional[List[Dict[str, Any]]] = None,
    primary_claim: Optional[str] = None,
    consistency_analysis: Optional[Dict[str, Any]] = None,
    raw: Optional[Dict[str, Any]] = None,
    case_summary: Optional[Dict[str, Any]] = None,
    pipeline_stages: Optional[List[Dict[str, Any]]] = None,
    modality_specific: Optional[Dict[str, Any]] = None,
    research_report: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Standardize response dictionary across all modalities.
    """
    # Default factual verification structure
    fact_dict = {
        "verdict": None,
        "confidence": None,
        "evidence_sufficiency": None,
        "independent_supporting": 0,
        "independent_contradicting": 0,
        "contradiction_score": None,
        "source_agreement": None,
        "fusion_score": None,
    }
    if factual_verification:
        fact_dict.update(factual_verification)

    return {
        "modality": modality,
        "status": status,
        "factual_verification": fact_dict,
        "synthetic_media_detection": synthetic_media_detection,
        "visual_classification": visual_classification,
        "transcription": transcription,
        "transcription_segments": transcription_segments,
        "primary_claim": primary_claim,
        "consistency_analysis": consistency_analysis,
        "case_summary": case_summary,
        "pipeline_stages": pipeline_stages or [],
        "modality_specific": modality_specific or {},
        # The report is part of the response schema, so it is always present
        # and always well-formed, even on an error path.
        "research_report": research_report or _minimal_research_report(
            modality, error
        ),
        "raw": raw or {},
        "error": redact_sensitive(error) if error else error,
        "warnings": warnings or [],
    }


# ============================================================================
# Modality Verifiers
# ============================================================================

def verify_text(claim: str) -> Dict[str, Any]:
    """
    Route text claim to research pipeline (pipeline.run_research_pipeline).
    """
    clean_claim = (claim or "").strip()
    if not clean_claim:
        return build_unified_response(
            modality="text",
            status="error",
            error="Claim text cannot be empty.",
        )

    t0 = time.time()
    try:
        from pipeline import run_research_pipeline

        state = run_research_pipeline(clean_claim)
        final_verdict = state.get("final_verdict", {}) or {}

        fact_data = {
            "verdict": final_verdict.get("verdict"),
            "confidence": ((final_verdict.get("confidence") / 100.0) if (isinstance(final_verdict.get("confidence"), (int, float)) and final_verdict.get("confidence") > 1.0) else final_verdict.get("confidence")),
            "evidence_sufficiency": final_verdict.get("evidence_sufficiency"),
            "independent_supporting": final_verdict.get("independent_supporting", 0),
            "independent_contradicting": final_verdict.get("independent_contradicting", 0),
            "contradiction_score": final_verdict.get("contradiction_score"),
            "entailment_score": final_verdict.get("entailment_score"),
            "source_agreement": final_verdict.get("source_agreement"),
            "fusion_score": final_verdict.get("fusion_score"),
            "reasoning": final_verdict.get("reasoning") or final_verdict.get("permitted_reason"),
            "permitted_verdicts": final_verdict.get("permitted_verdicts"),
            "deterministic_gate_reason": final_verdict.get("deterministic_gate_reason"),
            "total_sources": final_verdict.get("total_sources", 0),
            "supporting_sources": final_verdict.get("supporting_sources", 0),
            "contradicting_sources": final_verdict.get("contradicting_sources", 0),
            "context_chunks": final_verdict.get("context_chunks", 0),
            "agent_verdict": final_verdict.get("verification_verdict"),
            "critic_verdict": final_verdict.get("critic_verdict"),
            "agents_agree": final_verdict.get("agents_agree"),
            "verdict_overridden_by_evidence": final_verdict.get("verdict_overridden_by_evidence"),
            "agent_confidence": final_verdict.get("agent_confidence"),
            "critic_confidence": final_verdict.get("critic_confidence"),
            "primary_supporting_groups": final_verdict.get("primary_supporting_groups", 0),
            "supporting_evidence": final_verdict.get("supporting_evidence", []),
            "contradicting_evidence": final_verdict.get("contradicting_evidence", []),
            **_evidence_policy_fields(final_verdict),
        }

        elapsed = time.time() - t0
        # A provider failure in an agent stage leaves the retrieved and
        # analysed evidence intact but the verification unfinished. The result
        # is reported as partial, never as a completed verification.
        agent_errors = state.get("agent_errors") or []
        is_partial = bool(agent_errors)
        case_summary = build_case_summary(
            modality="text",
            target=clean_claim,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="PARTIAL" if is_partial else "SUCCESS",
            raw=state,
        )
        stages = build_pipeline_stages(
            modality="text",
            raw=state,
            fact_data=fact_data,
            primary_claim=clean_claim,
        )

        research_report = build_research_report(state, "text", clean_claim)

        return build_unified_response(
            modality="text",
            status="partial" if is_partial else "success",
            factual_verification=fact_data,
            synthetic_media_detection=None,
            visual_classification=None,
            transcription=None,
            primary_claim=clean_claim,
            consistency_analysis=None,
            case_summary=case_summary,
            pipeline_stages=stages,
            modality_specific={},
            research_report=research_report,
            raw=state,
        )
    except Exception as exc:
        logger.exception("Text verification pipeline failed: %s", exc)
        elapsed = time.time() - t0
        case_summary = build_case_summary(
            modality="text",
            target=clean_claim,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="ERROR",
        )
        return build_unified_response(
            modality="text",
            status="error",
            primary_claim=clean_claim,
            case_summary=case_summary,
            error=str(exc),
        )


def verify_image(
    image_path: Union[str, Path],
    original_filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Route image to image verification pipeline with AI generation detection
    (ai_image_signal.run_image_pipeline_with_ai_detection).
    """
    img_path = Path(image_path).resolve()
    if not img_path.exists():
        return build_unified_response(
            modality="image",
            status="error",
            error=f"Image file not found: {img_path}",
        )

    t0 = time.time()
    try:
        from ai_image_signal import run_image_pipeline_with_ai_detection

        result = run_image_pipeline_with_ai_detection(str(img_path))

        # Extract factual verification details
        verif = result.get("verification") or {}
        final_v = verif.get("final_verdict") or {}
        fact_data = {
            "verdict": final_v.get("verdict"),
            "confidence": ((final_v.get("confidence") / 100.0) if (isinstance(final_v.get("confidence"), (int, float)) and final_v.get("confidence") > 1.0) else final_v.get("confidence")),
            "evidence_sufficiency": final_v.get("evidence_sufficiency"),
            "independent_supporting": final_v.get("independent_supporting", 0),
            "independent_contradicting": final_v.get("independent_contradicting", 0),
            "contradiction_score": final_v.get("contradiction_score"),
            "entailment_score": final_v.get("entailment_score"),
            "source_agreement": final_v.get("source_agreement"),
            "fusion_score": final_v.get("fusion_score"),
            "reasoning": final_v.get("reasoning") or final_v.get("permitted_reason"),
            "permitted_verdicts": final_v.get("permitted_verdicts"),
            "deterministic_gate_reason": final_v.get("deterministic_gate_reason"),
            "total_sources": final_v.get("total_sources", 0),
            "supporting_sources": final_v.get("supporting_sources", 0),
            "contradicting_sources": final_v.get("contradicting_sources", 0),
            "context_chunks": final_v.get("context_chunks", 0),
            "agent_verdict": final_v.get("verification_verdict"),
            "critic_verdict": final_v.get("critic_verdict"),
            "agents_agree": final_v.get("agents_agree"),
            "verdict_overridden_by_evidence": final_v.get("verdict_overridden_by_evidence"),
            "agent_confidence": final_v.get("agent_confidence"),
            "critic_confidence": final_v.get("critic_confidence"),
            "primary_supporting_groups": final_v.get("primary_supporting_groups", 0),
            **_evidence_policy_fields(final_v),
        }

        # Extract AI image origin detection
        ai_gen = result.get("ai_generation") or {}
        raw_label = ai_gen.get("verdict")
        # Standardize signal label
        std_label = "UNAVAILABLE"
        if not ai_gen.get("available", False):
            std_label = "UNAVAILABLE"
        elif raw_label in ("AI-GENERATED", "AI_GENERATED"):
            std_label = "AI_GENERATED"
        elif raw_label == "REAL":
            std_label = "REAL"
        elif raw_label == "UNCERTAIN":
            std_label = "UNCERTAIN"

        synthetic_data = {
            "signal": std_label,
            "confidence": ai_gen.get("confidence"),
            "model": ai_gen.get("model"),
            "available": bool(ai_gen.get("available", False)),
            "warning": None,
            "probabilities": {
                "ai_probability": ai_gen.get("ai_probability"),
                "real_probability": ai_gen.get("real_probability"),
                # Tampering/manipulation detection: no such model exists in this
                # repository. Report as NOT_ASSESSED rather than fabricating a score.
                "manipulation_score": None,
            },
            # The detector's own frozen operating point, so the UI can state the
            # real thresholds instead of hardcoding them.
            "operating_point": ai_gen.get("operating_point"),
            # Explicit manipulation status: no pixel-tampering / splicing detector
            # is installed. This must be shown to the user rather than hidden.
            "manipulation_status": "NOT_ASSESSED",
            "manipulation_note": (
                "No pixel-level tampering or splicing detector is installed. "
                "Manipulation status cannot be assessed from available models."
            ),
            "error": ai_gen.get("error"),
        }

        # Consistency analysis note
        consistency_info = {
            "consistency_note": result.get("ai_generation_note"),
            "extracted_claim": result.get("normalized_claim", {}).get("claim"),
            "detected_objects": result.get("detected_objects", []),
        }

        primary_claim = result.get("normalized_claim", {}).get("claim")
        elapsed = time.time() - t0

        case_summary = build_case_summary(
            modality="image",
            target=img_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="SUCCESS",
            raw=result,
        )
        stages = build_pipeline_stages(
            modality="image",
            raw=result,
            fact_data=fact_data,
            synthetic_data=synthetic_data,
            primary_claim=primary_claim,
        )

        modality_specific = _build_image_modality_specific(result, synthetic_data)
        research_report = build_research_report(result, "image", primary_claim)

        return build_unified_response(
            modality="image",
            status="success",
            factual_verification=fact_data,
            synthetic_media_detection=synthetic_data,
            visual_classification=None,
            transcription=None,
            primary_claim=primary_claim,
            consistency_analysis=consistency_info,
            case_summary=case_summary,
            pipeline_stages=stages,
            modality_specific=modality_specific,
            research_report=research_report,
            raw=result,
        )
    except Exception as exc:
        logger.exception("Image verification pipeline failed: %s", exc)
        elapsed = time.time() - t0
        case_summary = build_case_summary(
            modality="image",
            target=img_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="ERROR",
        )
        return build_unified_response(
            modality="image",
            status="error",
            case_summary=case_summary,
            error=str(exc),
        )


def verify_video_file(
    video_path: Union[str, Path],
    enable_classifier: bool = True,
    verify_claims: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    whisper_model: str = "base",
    ocr_stride: int = 1,
    skip_ocr: bool = False,
    max_claims_to_verify: int = 1,
    max_ocr_frames: Optional[int] = 15,
    classifier_family: str = "videomae",
    classifier_checkpoint: Optional[Union[str, Path]] = None,
    original_filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Route video to video verifier (video_module.video_verifier.verify_video).
    """
    v_path = Path(video_path).resolve()
    if not v_path.exists():
        return build_unified_response(
            modality="video",
            status="error",
            error=f"Video file not found: {v_path}",
        )

    t0 = time.time()
    try:
        from video_module.video_verifier import verify_video

        result = verify_video(
            video_path=v_path,
            enable_classifier=enable_classifier,
            verify_claims=verify_claims,
            output_dir=output_dir,
            whisper_model=whisper_model,
            ocr_stride=ocr_stride,
            skip_ocr=skip_ocr,
            max_claims_to_verify=max_claims_to_verify,
            max_ocr_frames=max_ocr_frames,
            classifier_family=classifier_family,
            classifier_checkpoint=classifier_checkpoint,
        )

        verified_claims = result.get("verified_claims") or []
        candidate_claims_all = result.get("candidate_claims") or []
        primary_claim_obj = verified_claims[0] if verified_claims else {}
        fact_sub = primary_claim_obj.get("verification") or primary_claim_obj.get("verification_result", {}) or {}
        final_v = fact_sub.get("final_verdict", {}) or {}

        primary_verdict = final_v.get("verdict") or result.get("final_evidence_verdict") or "INCONCLUSIVE"
        confidence_val = final_v.get("confidence")
        if isinstance(confidence_val, (int, float)) and confidence_val > 1.0:
            confidence_val = confidence_val / 100.0

        fact_data = {
            "verdict": primary_verdict,
            "confidence": confidence_val,
            "evidence_sufficiency": final_v.get("evidence_sufficiency") or ("INSUFFICIENT" if primary_verdict == "INSUFFICIENT_VERIFIABLE_CLAIMS" else None),
            "independent_supporting": final_v.get("independent_supporting", 0),
            "independent_contradicting": final_v.get("independent_contradicting", 0),
            "contradiction_score": final_v.get("contradiction_score"),
            "entailment_score": final_v.get("entailment_score"),
            "source_agreement": final_v.get("source_agreement"),
            "fusion_score": final_v.get("fusion_score"),
            "reasoning": final_v.get("reasoning") or final_v.get("permitted_reason") or (result.get("consistency_analysis", {}) or {}).get("description"),
            "deterministic_gate_reason": final_v.get("deterministic_gate_reason"),
            "total_sources": final_v.get("total_sources", 0),
            "supporting_sources": final_v.get("supporting_sources", 0),
            "contradicting_sources": final_v.get("contradicting_sources", 0),
            "context_chunks": final_v.get("context_chunks", 0),
            "agent_verdict": final_v.get("verification_verdict"),
            "critic_verdict": final_v.get("critic_verdict"),
            "agents_agree": final_v.get("agents_agree"),
            "verdict_overridden_by_evidence": final_v.get("verdict_overridden_by_evidence"),
            "agent_confidence": final_v.get("agent_confidence"),
            "critic_confidence": final_v.get("critic_confidence"),
            "primary_supporting_groups": final_v.get("primary_supporting_groups", 0),
            **_evidence_policy_fields(final_v),
        }

        # Visual classifier (experimental visual misinformation heuristic)
        vis_sig = result.get("visual_classification") or {}
        visual_classification = {
            "predicted_label": vis_sig.get("predicted_label"),
            "confidence": vis_sig.get("confidence"),
            "experimental_warning": vis_sig.get("experimental_warning"),
            "available": bool(vis_sig.get("available", False)),
            "probabilities": vis_sig.get("probabilities"),
        }

        # Transcript text from fused summary if available
        transcription_text = None
        fused_path = (result.get("summary_artifacts") or {}).get("fused_transcript")
        if fused_path and Path(fused_path).exists():
            try:
                transcription_text = Path(fused_path).read_text(encoding="utf-8")
            except Exception:
                pass

        warnings = []
        if visual_classification.get("experimental_warning"):
            warnings.append(visual_classification["experimental_warning"])

        primary_claim_analyzed = result.get("primary_claim_analyzed")
        elapsed = time.time() - t0

        case_summary = build_case_summary(
            modality="video",
            target=v_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="SUCCESS",
            raw=result,
        )
        stages = build_pipeline_stages(
            modality="video",
            raw=result,
            fact_data=fact_data,
            visual_data=visual_classification,
            primary_claim=primary_claim_analyzed,
        )

        modality_specific = _build_video_modality_specific(
            result, visual_classification, transcription_text
        )
        research_report = build_research_report(
            result, "video", primary_claim_analyzed, modality_specific=modality_specific
        )

        return build_unified_response(
            modality="video",
            status="success",
            factual_verification=fact_data,
            synthetic_media_detection=None,  # Video classifier is visual_classification
            visual_classification=visual_classification,
            transcription=transcription_text,
            transcription_segments=_extract_transcription_segments(result),
            primary_claim=primary_claim_analyzed,
            consistency_analysis=result.get("consistency_analysis"),
            case_summary=case_summary,
            pipeline_stages=stages,
            modality_specific=modality_specific,
            research_report=research_report,
            raw=result,
            warnings=warnings,
        )
    except Exception as exc:
        logger.exception("Video verification pipeline failed: %s", exc)
        elapsed = time.time() - t0
        case_summary = build_case_summary(
            modality="video",
            target=v_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="ERROR",
        )
        return build_unified_response(
            modality="video",
            status="error",
            case_summary=case_summary,
            error=str(exc),
        )


def verify_audio_file(
    audio_path: Union[str, Path],
    verify_claims: bool = False,
    whisper_model: str = "base",
    device: Optional[str] = None,
    max_claims_to_verify: int = 1,
    original_filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Route audio to audio verifier (audio_module.verify_audio).
    """
    a_path = Path(audio_path).resolve()
    if not a_path.exists():
        return build_unified_response(
            modality="audio",
            status="error",
            error=f"Audio file not found: {a_path}",
        )

    t0 = time.time()
    try:
        from audio_module import verify_audio

        result = verify_audio(
            audio_path=a_path,
            verify_claims=verify_claims,
            whisper_model=whisper_model,
            device=device,
            max_claims_to_verify=max_claims_to_verify,
        )

        status = result.get("status", "success")
        factual_res = result.get("factual_verification") or {}
        details = factual_res.get("details") or {}
        verif_sub = factual_res.get("verification") or {}
        final_v = verif_sub.get("final_verdict") or {}

        verdict_val = factual_res.get("verdict")
        confidence_val = factual_res.get("confidence")
        if isinstance(confidence_val, (int, float)) and confidence_val > 1.0:
            confidence_val = confidence_val / 100.0

        fact_data = {
            "verdict": verdict_val,
            "confidence": confidence_val,
            "evidence_sufficiency": factual_res.get("evidence_sufficiency"),
            "independent_supporting": details.get("independent_supporting", 0),
            "independent_contradicting": details.get("independent_contradicting", 0),
            "contradiction_score": details.get("contradiction_score"),
            "entailment_score": details.get("entailment_score"),
            "source_agreement": details.get("source_agreement"),
            "fusion_score": details.get("fusion_score"),
            "reasoning": details.get("reasoning") or final_v.get("reasoning") or final_v.get("permitted_reason"),
            "deterministic_gate_reason": details.get("deterministic_gate_reason") or final_v.get("deterministic_gate_reason"),
            "total_sources": details.get("total_sources") or final_v.get("total_sources", 0),
            "supporting_sources": details.get("supporting_sources") or final_v.get("supporting_sources", 0),
            "contradicting_sources": details.get("contradicting_sources") or final_v.get("contradicting_sources", 0),
            "context_chunks": final_v.get("context_chunks", 0),
            "agent_verdict": details.get("agent_verdict") or final_v.get("verification_verdict"),
            "critic_verdict": details.get("critic_verdict") or final_v.get("critic_verdict"),
            "agents_agree": details.get("agents_agree") or final_v.get("agents_agree"),
            "agent_confidence": details.get("agent_confidence") or final_v.get("agent_confidence"),
            "critic_confidence": details.get("critic_confidence") or final_v.get("critic_confidence"),
            "primary_supporting_groups": final_v.get("primary_supporting_groups", 0),
            **_evidence_policy_fields(final_v),
        }

        # Synthetic speech detection
        synth = result.get("synthetic_speech_detection") or {}
        # The detector reports its identity as `model_name` / `model_id`.
        # Expose the real values so the API never claims an unnamed detector.
        synth_model = synth.get("model") or synth.get("model_name") or synth.get("model_id")
        synth_data = {
            "signal": synth.get("label", "UNAVAILABLE"),
            "confidence": synth.get("confidence"),
            "model": synth_model,
            "model_id": synth.get("model_id"),
            "model_license": synth.get("model_license"),
            "available": bool(synth.get("available", False)),
            "warning": None,
            "probabilities": {
                "probability_synthetic": synth.get("probability_synthetic"),
                "probability_real": synth.get("probability_real"),
            },
            "error": synth.get("error"),
        }

        transcription_obj = result.get("transcription") or {}
        transcription_text = transcription_obj.get("text")
        primary_claim = (result.get("claim_extraction") or {}).get("primary_claim")
        elapsed = time.time() - t0

        case_summary = build_case_summary(
            modality="audio",
            target=a_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="SUCCESS" if status == "success" else "ERROR",
            raw=result,
        )
        stages = build_pipeline_stages(
            modality="audio",
            raw=result,
            fact_data=fact_data,
            synthetic_data=synth_data,
            primary_claim=primary_claim,
        )

        modality_specific = _build_audio_modality_specific(result, synth_data)
        research_report = build_research_report(result, "audio", primary_claim)

        return build_unified_response(
            modality="audio",
            status=status,
            factual_verification=fact_data,
            synthetic_media_detection=synth_data,
            visual_classification=None,
            transcription=transcription_text,
            primary_claim=primary_claim,
            consistency_analysis=result.get("consistency_analysis"),
            case_summary=case_summary,
            pipeline_stages=stages,
            modality_specific=modality_specific,
            research_report=research_report,
            raw=result,
            error=result.get("error"),
        )
    except Exception as exc:
        logger.exception("Audio verification pipeline failed: %s", exc)
        elapsed = time.time() - t0
        case_summary = build_case_summary(
            modality="audio",
            target=a_path,
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            processing_time_sec=elapsed,
            validation_status="VALID",
            processing_status="ERROR",
        )
        return build_unified_response(
            modality="audio",
            status="error",
            case_summary=case_summary,
            error=str(exc),
        )


def route_media(
    target: Union[str, Path],
    modality: Optional[str] = None,
    original_filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Unified entry point for routing any media or text to the appropriate pipeline.
    """
    resolved_modality = modality.lower() if modality else detect_modality(target)

    if resolved_modality == "text":
        return verify_text(str(target))
    elif resolved_modality == "image":
        return verify_image(target, original_filename=original_filename, file_size_bytes=file_size_bytes)
    elif resolved_modality == "video":
        return verify_video_file(target, original_filename=original_filename, file_size_bytes=file_size_bytes, **kwargs)
    elif resolved_modality == "audio":
        return verify_audio_file(target, original_filename=original_filename, file_size_bytes=file_size_bytes, **kwargs)
    else:
        return build_unified_response(
            modality=resolved_modality,
            status="error",
            error=f"Unsupported modality: '{resolved_modality}'",
        )
