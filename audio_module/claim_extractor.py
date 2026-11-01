"""
claim_extractor.py
==================
Claim extraction, filtering, and priority scoring for audio transcripts in TruthLensAI.

Reuses the factual ranking heuristics from video_module.claim_ranker to:
1. Filter out greetings, apologies, conversational filler, and non-factual statements.
2. Identify verifiable factual assertions with entity, numerical, or temporal references.
3. Rank claims by verifiability priority score.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TruthLensAI.AudioModule.ClaimExtractor")

try:
    from video_module.claim_ranker import (
        process_and_rank_candidate_claims,
        evaluate_candidate_claim,
        clean_claim_text,
    )
    _CLAIM_RANKER_AVAILABLE = True
except ImportError:
    _CLAIM_RANKER_AVAILABLE = False
    logger.warning("video_module.claim_ranker could not be imported. Using fallback extraction.")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries."""
    if not text or not text.strip():
        return []
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = [s.strip() for s in raw_sentences if s and len(s.strip()) > 3]
    return cleaned


def extract_claims_from_transcript(
    transcript: Dict[str, Any],
    max_claims: int = 1,
) -> Dict[str, Any]:
    """
    Extract, filter, and rank candidate claims from an audio transcript.

    Args:
        transcript: Dictionary containing 'text' and optional 'segments'.
        max_claims: Maximum number of top verifiable claims to select for verification.

    Returns:
        Dictionary containing:
            - 'selected_claims': List of claims selected for fact-checking.
            - 'all_candidates': List of all candidate statements evaluated.
            - 'rejection_summary': Count and reasons for rejected candidates.
            - 'primary_claim': The single highest-priority claim string (or None).
    """
    full_text = transcript.get("text", "").strip()
    segments = transcript.get("segments", [])

    candidates: List[Dict[str, Any]] = []

    if segments:
        for idx, seg in enumerate(segments):
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            # Some segments might contain multiple sentences
            sub_sentences = _split_into_sentences(seg_text)
            if not sub_sentences:
                sub_sentences = [seg_text]

            for s_idx, sent in enumerate(sub_sentences):
                candidates.append({
                    "text": sent,
                    "source": "speech_transcript",
                    "start_sec": float(seg.get("start", 0.0)),
                    "end_sec": float(seg.get("end", 0.0)),
                    "context": {"segment_index": idx, "sub_sentence_index": s_idx},
                })
    elif full_text:
        sentences = _split_into_sentences(full_text)
        if not sentences:
            sentences = [full_text]
        for idx, sent in enumerate(sentences):
            candidates.append({
                "text": sent,
                "source": "speech_transcript",
                "start_sec": 0.0,
                "end_sec": 0.0,
                "context": {"sentence_index": idx},
            })

    if not candidates:
        return {
            "selected_claims": [],
            "all_candidates": [],
            "rejection_summary": {},
            "primary_claim": None,
        }

    if _CLAIM_RANKER_AVAILABLE:
        ranked_result = process_and_rank_candidate_claims(candidates)
        all_evaluated = ranked_result.get("candidate_claims", [])
        selected = [c for c in all_evaluated if c.get("selection_status") == "selected"]
        
        # If no claim was explicitly "selected" (e.g. short factual snippet),
        # take the highest priority candidate with score > 0.0 that has verbs
        if not selected and all_evaluated:
            viable = [c for c in all_evaluated if c.get("priority_score", 0.0) > 0.1]
            if viable:
                selected = [viable[0]]

        top_claims = selected[:max_claims]
        primary = top_claims[0].get("claim_text") if top_claims else None

        return {
            "selected_claims": top_claims,
            "all_candidates": all_evaluated,
            "rejection_summary": ranked_result.get("rejection_summary", {}),
            "primary_claim": primary,
        }
    else:
        # Minimal fallback if claim_ranker is absent
        ranked = []
        for idx, c in enumerate(candidates):
            txt = c["text"]
            score = 0.5 if len(txt.split()) >= 4 else 0.1
            obj = {
                "claim_id": f"claim_{idx+1:03d}",
                "claim_text": txt,
                "priority_score": score,
                "selection_status": "selected" if score >= 0.5 else "rejected",
                "selection_reason": "fallback_length_heuristic",
            }
            ranked.append(obj)
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)
        top = [c for c in ranked if c["selection_status"] == "selected"][:max_claims]
        return {
            "selected_claims": top,
            "all_candidates": ranked,
            "rejection_summary": {},
            "primary_claim": top[0]["claim_text"] if top else None,
        }
