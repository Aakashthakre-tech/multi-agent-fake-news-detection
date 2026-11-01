"""
api/server.py
=============
TruthLensAI Multimodal Verification FastAPI Server.

Exposes RESTful endpoints for Text, Image, Video, and Audio fact-checking
and synthetic-media analysis using the unified multimodal_router.

Also exposes a Server-Sent Events (SSE) streaming endpoint for the text
pipeline so the frontend can display real-time pipeline progress.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The pipelines print progress markers that include non-ASCII glyphs. On a
# Windows console using a legacy code page this raises UnicodeEncodeError and
# aborts the run, so the whole server process is switched to UTF-8 output with
# unencodable characters replaced rather than raised.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from dotenv import load_dotenv
load_dotenv()

from research_report import redact_sensitive  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import multimodal_router as router
from multimodal_router import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    detect_modality,
    verify_audio_file,
    verify_image,
    verify_text,
    verify_video_file,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("TruthLensAI.API")

# FastAPI application instance
app = FastAPI(
    title="TruthLensAI Multimodal Verification API",
    description=(
        "Unified multimodal fact-checking and synthetic media detection API. "
        "Strictly separates factual claim verification from synthetic-media provenance."
    ),
    version="1.0.0",
)

# Enable CORS for future frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Request / Response Models
# ============================================================================

class TextVerificationRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="The news claim or factual statement to verify.")


class FactualVerificationDetails(BaseModel):
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    # What `confidence` actually measures.
    #   'evidence_strength'      - strength of the evidence behind a settled verdict
    #   'verification_coverage'   - share of retrieved material that could be read
    #                              and classified (used when the verdict is
    #                              UNVERIFIED; it is NOT a probability that the
    #                              claim is true or false)
    confidence_basis: Optional[str] = None
    evidence_sufficiency: Optional[str] = None
    independent_supporting: int = 0
    independent_contradicting: int = 0
    contradiction_score: Optional[float] = None
    entailment_score: Optional[float] = None
    source_agreement: Optional[float] = None
    fusion_score: Optional[float] = None
    reasoning: Optional[str] = None
    permitted_verdicts: Optional[List[str]] = None
    deterministic_gate_reason: Optional[str] = None
    total_sources: int = 0
    supporting_sources: int = 0
    contradicting_sources: int = 0
    context_chunks: int = 0
    agent_verdict: Optional[str] = None
    critic_verdict: Optional[str] = None
    agents_agree: Optional[bool] = None
    verdict_overridden_by_evidence: Optional[bool] = None
    agent_confidence: Optional[float] = None
    critic_confidence: Optional[float] = None
    primary_supporting_groups: int = 0
    # --- evidence-policy diagnostics -----------------------------------
    # Publishers whose article text we actually read (not a search snippet).
    primary_contradicting_groups: int = 0
    article_supporting_groups: int = 0
    article_contradicting_groups: int = 0
    # Publishers containing an explicit correction of a material claim detail.
    correction_groups: int = 0
    # Share of the retrieved pool that could be read and classified.
    verification_coverage: Optional[float] = None
    # Sources whose publisher identity could not be established; they are
    # reported for diagnostics but excluded from the independent counts.
    unresolved_identity_sources: int = 0
    # --- deterministic-gate policy accounting ---------------------------
    # False when the evidence policy counters were missing or malformed and
    # the gate therefore restricted the result to UNVERIFIED. This is an
    # internal evidence-accounting issue, not a finding about the claim.
    policy_data_valid: Optional[bool] = None
    policy_data_errors: List[str] = Field(default_factory=list)


class SyntheticMediaDetails(BaseModel):
    signal: Optional[str] = None
    confidence: Optional[float] = None
    model: Optional[str] = None
    model_id: Optional[str] = None
    model_license: Optional[str] = None
    available: bool = False
    warning: Optional[str] = None
    probabilities: Optional[Dict[str, Any]] = None
    # Manipulation / tampering detection: no pixel-level detector installed.
    # 'NOT_ASSESSED' is the only honest value for images; absent for audio/video.
    manipulation_status: Optional[str] = None
    manipulation_note: Optional[str] = None
    # The detector's own frozen operating point (threshold / uncertain band /
    # temperature), when the detector reports one.
    operating_point: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VisualClassificationDetails(BaseModel):
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    experimental_warning: Optional[str] = None
    available: bool = False
    probabilities: Optional[Dict[str, Any]] = None


class CaseSummary(BaseModel):
    case_id: str
    modality: str
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_format: Optional[str] = None
    media_metadata: Optional[Dict[str, Any]] = None
    timestamp_utc: str
    processing_time_sec: Optional[float] = None
    validation_status: str = "VALID"
    processing_status: str = "SUCCESS"


class PipelineStageDetails(BaseModel):
    id: str
    number: int
    name: str
    status: str
    operation: str
    why: str
    tool_or_model: str
    input_summary: str
    output_summary: str
    metric: Optional[Dict[str, Any]] = None
    details: Optional[str] = None
    duration_sec: Optional[float] = None
    warnings: Optional[List[str]] = None


class ResearchFinding(BaseModel):
    """One point-wise finding derived from a real classified evidence passage."""
    title: str
    explanation: str
    claim_text: Optional[str] = None
    status: str
    evidence_strength: Optional[float] = None
    strength_label: Optional[str] = None
    source_name: Optional[str] = None
    source_domain: Optional[str] = None
    source_url: Optional[str] = None
    source_tier: Optional[str] = None
    source_reliability: Optional[float] = None
    excerpt: Optional[str] = None
    excerpt_kind: Optional[str] = None
    date_status: Optional[str] = None
    event_match: Optional[str] = None
    nli_label: Optional[str] = None
    nli_score: Optional[float] = None
    independence_group: Optional[str] = None
    evidence_class: Optional[str] = None
    evidence_type: Optional[str] = None
    search_query: Optional[str] = None


class PolicyDiagnostics(BaseModel):
    """
    The deterministic gate's own policy-accounting diagnostics.

    ``gate_ran`` is False when the gate never evaluated this result (for
    example a modality with no verifiable claim to research).
    ``verdict_restricted`` is True when missing or malformed policy counters
    forced the gate to abstain with UNVERIFIED. That abstention is an internal
    evidence-accounting issue, not a finding about the claim.
    """
    gate_ran: bool = False
    policy_data_valid: Optional[bool] = None
    policy_data_errors: List[str] = Field(default_factory=list)
    verdict_restricted: bool = False
    explanation: Optional[str] = None


class AgentStatus(BaseModel):
    """Which agent stages completed, and which failed."""
    verification_agent: str = "unknown"
    critic_agent: str = "unknown"
    partial: bool = False
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchExecutiveSummary(BaseModel):
    input_claim: Optional[str] = None
    modality: Optional[str] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    # 'evidence_strength' or 'verification_coverage'
    confidence_basis: Optional[str] = None
    confidence_basis_note: Optional[str] = None
    confidence_basis_explanation: Optional[str] = None
    policy_diagnostics: Optional[PolicyDiagnostics] = None
    agent_status: Optional[AgentStatus] = None
    attempted: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    why_this_verdict: Optional[str] = None
    deterministic_gate_reason: Optional[str] = None
    agent_narrative: Optional[str] = None
    critic_review: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class ResearchSearchQuery(BaseModel):
    label: Optional[str] = None
    query: str
    intent: Optional[str] = None
    topic: Optional[str] = None


class ResearchScrapeFailure(BaseModel):
    url: Optional[str] = None
    domain: Optional[str] = None
    title: Optional[str] = None
    reason: Optional[str] = None
    fallback_used: Optional[Any] = None


class ResearchMethodologyStage(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None


class ResearchMethodology(BaseModel):
    research_skipped: bool = False
    research_skipped_reason: Optional[str] = None
    search_queries: List[ResearchSearchQuery] = Field(default_factory=list)
    search_errors: List[str] = Field(default_factory=list)
    sources_discovered: int = 0
    sources_retrieved: int = 0
    articles_scraped: int = 0
    scrapes_failed: List[ResearchScrapeFailure] = Field(default_factory=list)
    syndicated_sources_merged: int = 0
    evidence_passages: int = 0
    nli_evaluated: int = 0
    # --- evidence-policy counters (real, from the fusion layer) ----------
    independent_publishers: int = 0
    primary_supporting_groups: int = 0
    primary_contradicting_groups: int = 0
    article_supporting_groups: int = 0
    article_contradicting_groups: int = 0
    correction_groups: int = 0
    snippet_only_sources: int = 0
    article_sources: int = 0
    unresolved_identity_sources: int = 0
    verification_coverage: float = 0.0
    stages: List[ResearchMethodologyStage] = Field(default_factory=list)


class ResearchClaimResult(BaseModel):
    claim_text: Optional[str] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    evidence_sufficiency: Optional[str] = None
    reasoning: Optional[str] = None
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    timestamps: Optional[Any] = None
    frame_references: Optional[List[int]] = None
    source_modality: Optional[str] = None
    selection_reason: Optional[str] = None
    priority_score: Optional[float] = None
    limitations: List[str] = Field(default_factory=list)


class ResearchEvidenceGap(BaseModel):
    question: str
    status: str
    detail: Optional[str] = None
    what_would_help: Optional[str] = None


class ResearchReport(BaseModel):
    """Point-wise, evidence-grounded research report built from real pipeline state."""
    modality: str
    executive_summary: ResearchExecutiveSummary = Field(default_factory=ResearchExecutiveSummary)
    findings: Dict[str, List[ResearchFinding]] = Field(default_factory=dict)
    claims: List[ResearchClaimResult] = Field(default_factory=list)
    methodology: ResearchMethodology = Field(default_factory=ResearchMethodology)
    evidence_gaps: List[ResearchEvidenceGap] = Field(default_factory=list)


class UnifiedVerificationResponse(BaseModel):
    modality: str
    status: str
    factual_verification: FactualVerificationDetails
    synthetic_media_detection: Optional[SyntheticMediaDetails] = None
    visual_classification: Optional[VisualClassificationDetails] = None
    transcription: Optional[str] = None
    transcription_segments: Optional[List[Dict[str, Any]]] = None
    primary_claim: Optional[str] = None
    consistency_analysis: Optional[Dict[str, Any]] = None
    case_summary: Optional[CaseSummary] = None
    pipeline_stages: Optional[List[PipelineStageDetails]] = None
    modality_specific: Optional[Dict[str, Any]] = None
    research_report: Optional[ResearchReport] = None
    raw: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# SSE Pipeline Stage Classification
# ============================================================================

# Maps pipeline print() patterns to structured stage information
_STAGE_PATTERNS: List[tuple] = [
    # (regex pattern, stage_id, stage_label, icon)
    (r"STEP 1\s*[-–]\s*RESEARCH",                   "research",    "Web Research",            "travel_explore"),
    (r"Search Results",                              "research",    "Processing Search Results","travel_explore"),
    (r"STEP 2\s*[-–]\s*COLLECTING EVIDENCE",         "evidence",    "Collecting Evidence",     "collections_bookmark"),
    (r"Evaluating Candidate Source",                 "evidence",    "Scraping Source",         "link"),
    (r"Scrape Status:\s*SUCCESS",                    "evidence",    "Source Scraped ✓",        "check_circle"),
    (r"Scrape Status:\s*FAILED",                     "evidence",    "Scrape Failed (fallback)","warning"),
    (r"STEP 3\s*[-–]\s*ANALYZING EVIDENCE",          "analysis",    "Evidence Analysis",       "biotech"),
    (r"EVIDENCE LAYER SUMMARY",                      "analysis",    "Building Evidence Layer", "layers"),
    (r"STEP 4\s*[-–]\s*SOURCE AGREEMENT",            "fusion",      "Source Fusion",           "hub"),
    (r"Source Agreement:",                           "fusion",      "Computing Agreement",     "hub"),
    (r"STEP 5\s*[-–]\s*BUILDING EVIDENCE BRIEF",     "brief",       "Evidence Brief",          "summarize"),
    (r"Brief length:",                               "brief",       "Brief Ready",             "summarize"),
    (r"STEP 6\s*[-–]\s*VERIFICATION",                "verify_agent","Verification Agent",      "smart_toy"),
    (r"Verification Result",                         "verify_agent","Agent Analysis Complete", "smart_toy"),
    (r"STEP 7\s*[-–]\s*CRITIC REVIEW",               "critic",      "Critic Review",           "rate_review"),
    (r"Critic Review",                               "critic",      "Critic Analysis Complete","rate_review"),
    (r"STEP 8\s*[-–]\s*FINAL VERDICT",               "verdict",     "Final Verdict",           "gavel"),
    (r"TRUTHLENS AI\s*[-–]\s*FINAL ASSESSMENT",      "verdict",     "Verdict Ready",           "gavel"),
    (r"Merged \d+ duplicated",                       "evidence",    "Deduplicating Sources",   "merge"),
    (r"Evidence Collection Complete",                "evidence",    "Evidence Collection Done","collections_bookmark"),
    (r"Evidence Type:\s*(SCRAPED|SEARCH)",           "evidence",    "Evidence Classified",     "label"),
]

_STEP_ORDER = ["research", "evidence", "analysis", "fusion", "brief", "verify_agent", "critic", "verdict"]


def _classify_line(line: str) -> Optional[Dict[str, str]]:
    """Map a pipeline stdout line to a structured event dict, or None."""
    clean = line.strip()
    if not clean or set(clean) <= {"=", "-", " "}:
        return None  # Skip separator lines

    for pattern, stage_id, label, icon in _STAGE_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return {
                "stage": stage_id,
                "label": label,
                "icon": icon,
                "detail": clean[:200],
            }

    # Surface important data lines (URLs, domains, verdicts, scores)
    if re.search(r"URL:\s*https?://", clean):
        domain = re.search(r"https?://(?:www\.)?([^/\s]+)", clean)
        return {
            "stage": "evidence",
            "label": f"Source: {domain.group(1) if domain else 'web'}",
            "icon": "language",
            "detail": clean[:200],
        }
    if re.search(r"VERDICT:", clean, re.IGNORECASE):
        return {
            "stage": "verdict",
            "label": "Verdict Extracted",
            "icon": "gavel",
            "detail": clean[:200],
        }
    if re.search(r"CONFIDENCE:|NLI:|Evidence Score:", clean):
        return {
            "stage": "analysis",
            "label": "Score Computed",
            "icon": "analytics",
            "detail": clean[:200],
        }

    return None


# ============================================================================
# Capturing stdout Writer
# ============================================================================

class _QueueWriter(io.TextIOBase):
    """Intercepts sys.stdout write() calls and puts lines into a queue."""

    def __init__(self, q: queue.Queue, original: Any):
        self._buf = ""
        self._q = q
        self._original = original

    def write(self, s: str) -> int:
        # Also forward to original stdout so server logs still work
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._q.put(("line", line))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._q.put(("line", self._buf))
            self._buf = ""
        try:
            self._original.flush()
        except Exception:
            pass

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")


# ============================================================================
# SSE Generator
# ============================================================================

def _format_sse(event_type: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _stream_text_verification(claim: str) -> Generator[str, None, None]:
    """
    Generator that:
    1. Runs verify_text() in a background thread.
    2. Captures pipeline stdout in real time via _QueueWriter.
    3. Yields SSE events for each meaningful line.
    4. Emits 'result' event when done, or 'error' event on failure.
    """
    q: queue.Queue = queue.Queue()
    result_holder: Dict[str, Any] = {}
    current_stage_idx = [0]

    # --- Emit initial ready event ---
    yield _format_sse("status", {
        "stage": "ingestion",
        "label": "Ingestion & Validation",
        "icon": "input",
        "detail": f'Claim received ({len(claim)} chars). Dispatching to pipeline.',
        "step": 0,
        "total_steps": len(_STEP_ORDER),
    })

    def _worker():
        original_stdout = sys.stdout
        writer = _QueueWriter(q, original_stdout)
        sys.stdout = writer
        try:
            result_holder["data"] = verify_text(claim)
        except Exception as exc:
            result_holder["error"] = redact_sensitive(str(exc))
        finally:
            sys.stdout = original_stdout
            writer.flush()
            q.put(("done", None))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    emitted_stages = set()

    while True:
        try:
            kind, value = q.get(timeout=60)
        except queue.Empty:
            yield _format_sse("heartbeat", {"ts": "keep-alive"})
            continue

        if kind == "done":
            break

        if kind == "line":
            event_info = _classify_line(value)
            if event_info:
                stage = event_info["stage"]
                step_idx = _STEP_ORDER.index(stage) if stage in _STEP_ORDER else current_stage_idx[0]
                if step_idx >= current_stage_idx[0]:
                    current_stage_idx[0] = step_idx

                yield _format_sse("progress", {
                    **event_info,
                    "step": current_stage_idx[0] + 1,
                    "total_steps": len(_STEP_ORDER),
                    "raw": value[:300],
                })

    t.join(timeout=5)

    if "error" in result_holder:
        yield _format_sse("error", {"message": result_holder["error"]})
    elif "data" in result_holder:
        yield _format_sse("result", result_holder["data"])
    else:
        yield _format_sse("error", {"message": "Pipeline returned no result."})


# ============================================================================
# Video SSE Stage Patterns
# ============================================================================

_VIDEO_STAGE_PATTERNS: List[tuple] = [
    # (regex pattern, stage_id, stage_label, icon)
    (r"TRUTHLENSAI MULTI-MODAL VIDEO VERIFICATION",  "start",       "Video Intake",             "movie"),
    (r"\[1/7\].*Validating",                          "validation",  "Validating Video",         "verified"),
    (r"\[2/7\].*Extracting frames",                   "frames",      "Extracting Frames",        "photo_library"),
    (r"Extracted \d+ frames",                         "frames",      "Frames Extracted ✓",       "photo_library"),
    (r"\[3/7\].*Extracting audio",                    "audio",       "Extracting Audio",         "audio_file"),
    (r"Audio extracted",                              "audio",       "Audio Extracted ✓",        "audio_file"),
    (r"\[4/7\].*Transcrib",                           "whisper",     "Whisper Transcription",    "transcribe"),
    (r"Transcription completed",                      "whisper",     "Transcription Done ✓",     "transcribe"),
    (r"\[5/7\].*OCR",                                 "ocr",         "Frame OCR",                "document_scanner"),
    (r"OCR Policy:",                                  "ocr",         "OCR Policy Applied",       "document_scanner"),
    (r"OCR processed",                                "ocr",         "OCR Complete ✓",           "document_scanner"),
    (r"\[OCR \d+/\d+\]",                              "ocr",         "OCR Frame...",             "document_scanner"),
    (r"\[6/7\].*Fusing",                              "fusion",      "Speech + OCR Fusion",      "merge"),
    (r"Fusion completed",                             "fusion",      "Fusion Complete ✓",        "merge"),
    (r"\[7/7\].*Preparing candidate",                 "claims",      "Claim Extraction",         "fact_check"),
    (r"Step 1/3.*Preprocessing",                      "preprocessing", "Video Preprocessing",   "settings"),
    (r"Step 1/3.*Finished",                           "preprocessing", "Preprocessing Done ✓",  "settings"),
    (r"Step 2/3.*visual.*classifier",                 "classifier",  "VideoMAE Classifier",      "smart_toy"),
    (r"Visual Class:",                                "classifier",  "Classifier Result",        "smart_toy"),
    (r"Step 3/3.*Evidence Engine",                    "research",    "Evidence Research",        "travel_explore"),
    (r"Researching Claim",                            "research",    "Researching Claim...",     "travel_explore"),
    (r"STEP 1\s*[-–]\s*RESEARCH",                    "research",    "Web Research",             "travel_explore"),
    (r"STEP 2\s*[-–]\s*COLLECTING EVIDENCE",          "evidence",    "Collecting Evidence",      "collections_bookmark"),
    (r"STEP 3\s*[-–]\s*ANALYZING EVIDENCE",           "analysis",    "Evidence Analysis",        "biotech"),
    (r"STEP 4\s*[-–]\s*SOURCE AGREEMENT",             "fusion_ev",   "Source Fusion",            "hub"),
    (r"STEP 5\s*[-–]\s*BUILDING EVIDENCE BRIEF",      "brief",       "Evidence Brief",           "summarize"),
    (r"STEP 6\s*[-–]\s*VERIFICATION",                 "verify_agent","Verification Agent",       "smart_toy"),
    (r"STEP 7\s*[-–]\s*CRITIC REVIEW",                "critic",      "Critic Review",            "rate_review"),
    (r"STEP 8\s*[-–]\s*FINAL VERDICT",                "verdict",     "Final Verdict",            "gavel"),
    (r"VERIFICATION COMPLETE",                        "done",        "Verification Complete ✓",  "check_circle"),
]

_VIDEO_STEP_ORDER = [
    "start", "validation", "frames", "audio", "whisper",
    "ocr", "fusion", "claims", "preprocessing", "classifier",
    "research", "evidence", "analysis", "fusion_ev",
    "brief", "verify_agent", "critic", "verdict", "done",
]


def _classify_video_line(line: str) -> Optional[Dict[str, str]]:
    """Map a video pipeline stdout line to a structured event dict, or None."""
    clean = line.strip()
    if not clean or set(clean) <= {"=", "-", " "}:
        return None
    for pattern, stage_id, label, icon in _VIDEO_STAGE_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return {
                "stage": stage_id,
                "label": label,
                "icon": icon,
                "detail": clean[:250],
            }
    return None


def _stream_video_verification(
    temp_path: Path,
    original_filename: str,
    file_size_bytes: Optional[int],
    enable_classifier: bool,
    verify_claims: bool,
    whisper_model: str,
    skip_ocr: bool,
    max_claims_to_verify: int,
    max_ocr_frames: int,
    classifier_family: str,
) -> Generator[str, None, None]:
    """
    Generator that runs video verification in a background thread and emits
    SSE events for each meaningful pipeline stage line in real time.

    Event types:
    - status:    Initial ready event with file metadata
    - progress:  Per-stage update (stage id, label, icon, detail, step, total_steps)
    - heartbeat: Keep-alive sent every 30 s of idle queue
    - result:    Final UnifiedVerificationResponse on success
    - error:     Error detail string on failure
    """
    total_steps = len(_VIDEO_STEP_ORDER)
    q: queue.Queue = queue.Queue()
    result_holder: Dict[str, Any] = {}
    current_stage_idx = [0]

    yield _format_sse("status", {
        "stage": "start",
        "label": "Video Intake",
        "icon": "movie",
        "detail": (
            f"File: {original_filename} "
            f"({round(file_size_bytes / 1024 / 1024, 2)} MB) — dispatching pipeline."
            if file_size_bytes else f"File: {original_filename} — dispatching pipeline."
        ),
        "step": 1,
        "total_steps": total_steps,
    })

    def _worker():
        original_stdout = sys.stdout
        writer = _QueueWriter(q, original_stdout)
        sys.stdout = writer
        try:
            result_holder["data"] = verify_video_file(
                video_path=temp_path,
                enable_classifier=enable_classifier,
                verify_claims=verify_claims,
                whisper_model=whisper_model,
                skip_ocr=skip_ocr,
                max_claims_to_verify=max_claims_to_verify,
                max_ocr_frames=max_ocr_frames,
                classifier_family=classifier_family,
                original_filename=original_filename,
                file_size_bytes=file_size_bytes,
            )
        except Exception as exc:
            result_holder["error"] = redact_sensitive(str(exc))
        finally:
            sys.stdout = original_stdout
            writer.flush()
            q.put(("done", None))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        try:
            kind, value = q.get(timeout=30)
        except queue.Empty:
            yield _format_sse("heartbeat", {"ts": "keep-alive"})
            continue

        if kind == "done":
            break

        if kind == "line":
            event_info = _classify_video_line(value)
            if event_info:
                stage = event_info["stage"]
                step_idx = (
                    _VIDEO_STEP_ORDER.index(stage)
                    if stage in _VIDEO_STEP_ORDER
                    else current_stage_idx[0]
                )
                if step_idx >= current_stage_idx[0]:
                    current_stage_idx[0] = step_idx

                yield _format_sse("progress", {
                    **event_info,
                    "step": current_stage_idx[0] + 1,
                    "total_steps": total_steps,
                    "raw": value[:300],
                })

    t.join(timeout=5)

    # Ensure temp file is cleaned up after worker finishes
    _cleanup_temp_file(temp_path)

    if "error" in result_holder:
        yield _format_sse("error", {"message": result_holder["error"]})
    elif "data" in result_holder:
        yield _format_sse("result", result_holder["data"])
    else:
        yield _format_sse("error", {"message": "Video pipeline returned no result."})


# ============================================================================
# Helper Functions
# ============================================================================

def _save_upload_to_temp(upload_file: UploadFile, allowed_extensions: set[str]) -> Path:
    """Save an uploaded file to a temporary file with validated extension."""
    filename = upload_file.filename or "uploaded_media"
    suffix = Path(filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Allowed: {sorted(list(allowed_extensions))}",
        )

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(upload_file.file, tmp_file)
        tmp_file.close()
        return Path(tmp_file.name)
    except Exception as exc:
        logger.error(f"Failed to save uploaded file '{filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {str(exc)}",
        )


def _cleanup_temp_file(path: Optional[Path]) -> None:
    """Safely remove a temporary file if it exists."""
    if path and path.exists():
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Could not remove temp file {path}: {exc}")


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """
    Service health check and module capability matrix.
    Never exposes API key values.
    """
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))

    return {
        "status": "healthy",
        "service": "TruthLensAI Multimodal Verification Engine",
        "version": "1.0.0",
        "api_keys_configured": {
            "groq": has_groq,
            "tavily": has_tavily,
        },
        "modules": {
            "text": {
                "available": has_groq and has_tavily,
                "description": "Multi-agent research pipeline & evidence fusion engine",
            },
            "image": {
                "available": True,
                "description": "OCR, visual object analysis, claim normalization, & AI-image origin detection",
            },
            "video": {
                "available": True,
                "description": "Whisper transcription, OCR, visual frame classifier, & factual claim verification",
            },
            "audio": {
                "available": True,
                "description": "Whisper transcription, deepfake speech detection, & claim verification",
            },
        },
    }


# ============================================================================
# SSE Streaming Endpoint — Text Verification
# ============================================================================

@app.get("/verify/text/stream", tags=["Streaming"])
async def endpoint_verify_text_stream(claim: str) -> StreamingResponse:
    """
    Server-Sent Events streaming endpoint for text claim verification.
    Emits real-time pipeline progress events as the multi-agent pipeline runs.

    Event types:
    - status:   Initial pipeline ready event
    - progress: Incremental pipeline stage update
    - heartbeat: Keep-alive (every 60s of idle)
    - result:   Final UnifiedVerificationResponse payload (on success)
    - error:    Error detail string (on failure)
    """
    if not claim or not claim.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'claim' query parameter is required and cannot be empty.",
        )

    logger.info(f"SSE stream initiated for claim: {claim[:80]}...")

    return StreamingResponse(
        _stream_text_verification(claim.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# SSE Streaming Endpoint — Video Verification
# ============================================================================

@app.post("/verify/video/stream", tags=["Streaming"])
async def endpoint_verify_video_stream(
    file: UploadFile = File(..., description="Video file (.mp4, .mov, .avi, .mkv, .webm, etc.)"),
    enable_classifier: bool = Form(True, description="Enable visual fake/real classifier"),
    verify_claims: bool = Form(True, description="Enable web research on extracted claims"),
    whisper_model: str = Form("base", description="Whisper model size"),
    skip_ocr: bool = Form(False, description="Skip on-screen frame OCR"),
    max_claims_to_verify: int = Form(1, description="Max claims to fact-check"),
    max_ocr_frames: Optional[int] = Form(15, description="Max frame count to run OCR on (uniform sampling)"),
    classifier_family: str = Form("videomae", description="Visual classifier family"),
) -> StreamingResponse:
    """
    Server-Sent Events streaming endpoint for video verification.
    Emits real-time per-stage progress as the pipeline runs in a background thread.

    Event types:
    - status:    Initial intake event with file metadata
    - progress:  Per-stage pipeline update
    - heartbeat: Keep-alive every 30s of idle queue
    - result:    Final UnifiedVerificationResponse (on success)
    - error:     Error message (on failure)
    """
    temp_path = _save_upload_to_temp(file, VIDEO_EXTENSIONS)
    file_size = temp_path.stat().st_size if temp_path.exists() else None
    logger.info(
        f"Video SSE stream initiated: {file.filename!r} "
        f"({round(file_size / 1024 / 1024, 2) if file_size else '?'} MB)"
    )

    return StreamingResponse(
        _stream_video_verification(
            temp_path=temp_path,
            original_filename=file.filename or "video",
            file_size_bytes=file_size,
            enable_classifier=enable_classifier,
            verify_claims=verify_claims,
            whisper_model=whisper_model,
            skip_ocr=skip_ocr,
            max_claims_to_verify=max_claims_to_verify,
            max_ocr_frames=max_ocr_frames if max_ocr_frames is not None else 15,
            classifier_family=classifier_family,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# REST Verification Endpoints (unchanged)
# ============================================================================

@app.post("/verify/text", response_model=UnifiedVerificationResponse, tags=["Verification"])
def endpoint_verify_text(req: TextVerificationRequest) -> Dict[str, Any]:
    """
    Verify a textual claim through the multi-agent evidence engine.
    """
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Claim cannot be empty.",
        )
    result = verify_text(claim)
    return result


@app.post("/verify/image", response_model=UnifiedVerificationResponse, tags=["Verification"])
def endpoint_verify_image(
    file: UploadFile = File(..., description="Image file (.png, .jpg, .jpeg, .webp, .bmp, .tiff)")
) -> Dict[str, Any]:
    """
    Verify an image: extracts OCR text and objects, derives claim, conducts research,
    and independently evaluates AI image provenance.
    """
    temp_path = _save_upload_to_temp(file, IMAGE_EXTENSIONS)
    file_size = temp_path.stat().st_size if temp_path.exists() else None
    try:
        result = verify_image(
            temp_path,
            original_filename=file.filename,
            file_size_bytes=file_size,
        )
        return result
    finally:
        _cleanup_temp_file(temp_path)


@app.post("/verify/video", response_model=UnifiedVerificationResponse, tags=["Verification"])
def endpoint_verify_video(
    file: UploadFile = File(..., description="Video file (.mp4, .mov, .avi, .mkv, .webm, etc.)"),
    enable_classifier: bool = Form(True, description="Enable visual fake/real classifier"),
    verify_claims: bool = Form(True, description="Enable web research on extracted claims"),
    whisper_model: str = Form("base", description="Whisper model size"),
    skip_ocr: bool = Form(False, description="Skip on-screen frame OCR"),
    max_claims_to_verify: int = Form(1, description="Max claims to fact-check"),
    max_ocr_frames: Optional[int] = Form(15, description="Max frame count to run OCR on (uniform sampling)"),
    classifier_family: str = Form("videomae", description="Visual classifier family"),
) -> Dict[str, Any]:
    """
    Verify a video: pre-processes audio & frames, generates Whisper transcription & OCR,
    extracts factual claims, evaluates visual classifier, and synthesizes evidence.
    """
    temp_path = _save_upload_to_temp(file, VIDEO_EXTENSIONS)
    file_size = temp_path.stat().st_size if temp_path.exists() else None
    try:
        result = verify_video_file(
            video_path=temp_path,
            enable_classifier=enable_classifier,
            verify_claims=verify_claims,
            whisper_model=whisper_model,
            skip_ocr=skip_ocr,
            max_claims_to_verify=max_claims_to_verify,
            max_ocr_frames=max_ocr_frames,
            classifier_family=classifier_family,
            original_filename=file.filename,
            file_size_bytes=file_size,
        )
        return result
    finally:
        _cleanup_temp_file(temp_path)


@app.post("/verify/audio", response_model=UnifiedVerificationResponse, tags=["Verification"])
def endpoint_verify_audio(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a, .flac, .aac, .ogg)"),
    verify_claims: bool = Form(True, description="Enable web research on extracted claims"),
    whisper_model: str = Form("base", description="Whisper model size"),
    max_claims_to_verify: int = Form(1, description="Max claims to fact-check"),
) -> Dict[str, Any]:
    """
    Verify an audio recording: analyzes synthetic speech provenance (Deepfake detection),
    transcribes speech via Whisper, extracts claims, and optionally verifies veracity.
    """
    temp_path = _save_upload_to_temp(file, AUDIO_EXTENSIONS)
    file_size = temp_path.stat().st_size if temp_path.exists() else None
    try:
        result = verify_audio_file(
            audio_path=temp_path,
            verify_claims=verify_claims,
            whisper_model=whisper_model,
            max_claims_to_verify=max_claims_to_verify,
            original_filename=file.filename,
            file_size_bytes=file_size,
        )
        return result
    finally:
        _cleanup_temp_file(temp_path)


@app.post("/verify", response_model=UnifiedVerificationResponse, tags=["Verification"])
def endpoint_unified_verify(
    file: Optional[UploadFile] = File(None, description="Optional media file upload (image, video, audio)"),
    claim: Optional[str] = Form(None, description="Optional text claim to verify"),
    verify_claims: bool = Form(True, description="Whether to verify extracted claims"),
) -> Dict[str, Any]:
    """
    Omni-modal entry point: accepts either a media file or a raw text claim,
    automatically detects the modality, and dispatches to the correct pipeline.
    """
    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return endpoint_verify_image(file=file)
        elif suffix in VIDEO_EXTENSIONS:
            return endpoint_verify_video(file=file, verify_claims=verify_claims)
        elif suffix in AUDIO_EXTENSIONS:
            return endpoint_verify_audio(file=file, verify_claims=verify_claims)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{suffix}'. Supported: images, videos, audio.",
            )
    elif claim:
        return endpoint_verify_text(TextVerificationRequest(claim=claim))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a media file upload or a 'claim' text parameter must be provided.",
        )
