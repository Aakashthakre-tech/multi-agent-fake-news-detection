"""
research_report.py
=================
Builds a structured, point-wise research report from the real state produced by
the existing TruthLensAI verification engines.

This module never invents content. Every value it emits is copied or derived
from values the backend already computed:

* the evidence layer (``evidence_analysis``) - NLI-scored, date-gated,
  reliability-scored passages that the evidence analyzer classified;
* the source ledger (``source_data``) - what was actually retrieved and
  whether the page body was scraped or fell back to a snippet;
* the retrieval diagnostics (``search_plans``, ``search_errors``,
  ``scrape_failures``) - the real queries issued and the real failures;
* the deterministic evidence gate (``final_verdict``) - the permitted verdicts
  and the reason a verdict was chosen;
* the verification agent and the critic agent - their own prose, quoted.

If the research engine produced nothing (for example a video with no
verifiable claims), the report says so explicitly and explains why, rather
than inventing a summary.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["build_research_report", "redact_sensitive"]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Values that must never reach the API response or the UI. A third-party
# provider's error message can carry the account's organisation id or a key.
_SENSITIVE_PATTERNS = (
    # Provider account identifiers, e.g. org_01kwyr..., proj_01kwyr...
    (re.compile(r"\b(org|proj)_[A-Za-z0-9]{6,}\b"), r"\1_[redacted]"),
    # Provider API keys, e.g. gsk_..., tvly-...
    (
        re.compile(
            r"\b(?:gsk|sk|xai|groq|tvly|tavily)[-_][A-Za-z0-9_\-]{12,}\b",
            re.IGNORECASE,
        ),
        "[redacted-key]",
    ),
    # Authorization headers.
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b", re.IGNORECASE),
        "Bearer [redacted]",
    ),
    # key=value pairs for well-known secret names.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|secret|token|password|authorization)\b"
            r"\s*[:=]\s*\S+"
        ),
        r"\1=[redacted]",
    ),
)


def redact_sensitive(text):
    """
    Remove provider identifiers and credentials from a message.

    Returns the text unchanged when it is not a string, so it can be applied
    unconditionally to an error field.
    """
    if not isinstance(text, str) or not text:
        return text
    redacted = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_MAX_EXCERPT = 420
_MAX_TITLE = 160


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _is_valid_url(url: Any) -> bool:
    """Only http(s) URLs are exposed as clickable links."""
    if not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate.lower().startswith(("http://", "https://")):
        return False
    return len(candidate) <= 2048 and not re.search(r"\s", candidate)


def _excerpt(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= _MAX_EXCERPT:
        return cleaned
    return cleaned[:_MAX_EXCERPT].rstrip() + "…"


def _strength_label(score: Optional[float]) -> str:
    """Interpret a real evidence_score. No percentage is invented."""
    if score is None:
        return "Not scored"
    if score >= 0.75:
        return "Strong"
    if score >= 0.5:
        return "Moderate"
    if score > 0:
        return "Weak"
    return "None"


def _extract_section(text: Any, heading: str) -> str:
    """
    Pull a named section out of an agent's free-text answer.

    The verification and critic agents answer with headed sections; this
    returns the section body verbatim so the report can quote the agent
    instead of paraphrasing it.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    pattern = re.compile(
        rf"^\s*{re.escape(heading)}\s*:\s*(.*?)(?=^\s*[A-Z][A-Za-z /-]{2,30}\s*:|\Z)",
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    body = match.group(1).strip()
    return _clean_text(body)


# ---------------------------------------------------------------------------
# Modality normalisation
# ---------------------------------------------------------------------------

def _text_research_state(state: Dict[str, Any], modality: str) -> Optional[Dict[str, Any]]:
    """
    Return the research-engine state for the modality, if research ran.

    * text  - the pipeline state is the research state;
    * image - ``verification`` holds the research state (or a no-claim stub);
    * audio - ``factual_verification.verification`` holds the research state;
    * video - the state carries claim-level results instead.
    """
    if not isinstance(state, dict):
        return None

    if modality == "image":
        candidate = state.get("verification")
        if isinstance(candidate, dict):
            return candidate
        return None

    if modality == "audio":
        fv = state.get("factual_verification")
        if isinstance(fv, dict):
            candidate = fv.get("verification")
            if isinstance(candidate, dict):
                return candidate
        return None

    if modality == "video":
        return None

    return state


def _video_claim_results(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verified-claim records produced by the video evidence engine."""
    results: List[Dict[str, Any]] = []
    verified = state.get("verified_claims")
    if not isinstance(verified, list):
        return results
    for entry in verified:
        if not isinstance(entry, dict):
            continue
        research = entry.get("verification") or entry.get("verification_result") or {}
        if not isinstance(research, dict):
            research = {}
        results.append({
            "claim": entry.get("claim") or entry.get("text") or "",
            "research_state": research,
            "selection_reason": entry.get("selection_reason"),
            "priority_score": entry.get("priority_score"),
            "timestamps": entry.get("timestamps"),
            "frame_references": entry.get("frame_references"),
            "source_modality": entry.get("source_modality") or "video",
        })
    return results


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _finding_from_chunk(
    chunk: Dict[str, Any],
    evidence_class: str,
    claim_text: Optional[str],
) -> Dict[str, Any]:
    """Turn one real classified evidence passage into a point-wise finding."""
    domain = _clean_text(chunk.get("domain")) or "unknown source"
    title = _clean_text(chunk.get("title"))
    url = chunk.get("url")
    score = _as_float(chunk.get("evidence_score"))
    reliability = _as_float(chunk.get("source_reliability"))
    nli_score = _as_float(chunk.get("nli_score"))
    tier = chunk.get("source_tier")
    tier_reason = _clean_text(chunk.get("source_tier_reason"))
    date_status = chunk.get("date_status")
    event_match = chunk.get("event_match")
    nli_label = chunk.get("nli_label")

    if evidence_class == "SUPPORT":
        status = "SUPPORTED"
        lead = "Supporting finding"
    elif evidence_class == "CONTRADICTION":
        status = "CONTRADICTED"
        lead = "Contradicting finding"
    elif evidence_class == "RELATED_CONTEXT":
        status = "CONTEXT"
        lead = "Contextual finding"
    else:
        status = "NON_EVIDENCE"
        lead = "Non-evidence passage"

    # Explanation is assembled only from recorded backend values.
    parts: List[str] = []
    parts.append(f"{lead} from {domain}")
    if title:
        parts.append(f"(\"{title}\")")
    parts.append(".")
    detail_bits: List[str] = []
    if tier:
        detail_bits.append(f"source tier {tier}")
    if reliability is not None:
        detail_bits.append(f"reliability {reliability:.2f}")
    if date_status:
        detail_bits.append(f"date check: {date_status}")
    if event_match:
        detail_bits.append(f"event match: {event_match}")
    if nli_label:
        nli_text = f"NLI {nli_label}"
        if nli_score is not None:
            nli_text += f" ({nli_score:.3f})"
        detail_bits.append(nli_text)
    if detail_bits:
        parts.append(" " + "; ".join(detail_bits) + ".")
    if tier_reason:
        parts.append(f" Tier basis: {tier_reason}.")

    return {
        "title": (title or domain)[:_MAX_TITLE],
        "explanation": "".join(parts),
        "claim_text": claim_text,
        "status": status,
        "evidence_strength": score,
        "strength_label": _strength_label(score),
        "source_name": title or domain,
        "source_domain": domain,
        "source_url": url if _is_valid_url(url) else None,
        "source_tier": tier,
        "source_reliability": reliability,
        "excerpt": _excerpt(chunk.get("chunk")),
        "excerpt_kind": "verbatim_passage",
        "date_status": date_status,
        "event_match": event_match,
        "nli_label": nli_label,
        "nli_score": nli_score,
        "independence_group": chunk.get("independence_group"),
        "evidence_class": evidence_class,
        "evidence_type": chunk.get("evidence_type"),
        "search_query": chunk.get("search_query") or None,
    }


def _build_findings(
    research_state: Dict[str, Any],
    claim_text: Optional[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """Group the real evidence layer into supporting/contradicting/contextual."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "supporting": [],
        "contradicting": [],
        "contextual": [],
        "unresolved": [],
    }

    evidence = research_state.get("evidence_analysis")
    if not isinstance(evidence, list):
        return buckets, 0

    counted = 0
    for chunk in evidence:
        if not isinstance(chunk, dict):
            continue
        counted += 1
        klass = chunk.get("evidence_class")
        finding = _finding_from_chunk(chunk, klass, claim_text)
        if klass == "SUPPORT":
            buckets["supporting"].append(finding)
        elif klass == "CONTRADICTION":
            buckets["contradicting"].append(finding)
        elif klass == "RELATED_CONTEXT":
            buckets["contextual"].append(finding)
        else:
            buckets["unresolved"].append(finding)

    return buckets, counted


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

def _build_methodology(
    research_state: Dict[str, Any],
    state: Dict[str, Any],
    modality: str,
    research_skipped: bool,
    research_skipped_reason: Optional[str],
    modality_specific: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plans = research_state.get("search_plans") or state.get("search_plans") or []
    search_errors = research_state.get("search_errors") or state.get("search_errors") or []
    scrape_failures = research_state.get("scrape_failures") or state.get("scrape_failures") or []
    source_data = research_state.get("source_data") or []
    evidence = research_state.get("evidence_analysis") or []

    queries: List[Dict[str, Any]] = []
    if isinstance(plans, list):
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            query = _clean_text(plan.get("query"))
            if not query:
                continue
            queries.append({
                "label": plan.get("label"),
                "query": query,
                "intent": plan.get("intent"),
                "topic": plan.get("topic"),
            })

    sources_retrieved = len(source_data) if isinstance(source_data, list) else 0
    articles_scraped = 0
    for source in source_data if isinstance(source_data, list) else []:
        if not isinstance(source, dict):
            continue
        if source.get("scrape_status") == "SUCCESS":
            articles_scraped += 1

    nli_evaluated = 0
    for chunk in evidence if isinstance(evidence, list) else []:
        if isinstance(chunk, dict) and chunk.get("nli_status") in ("ok", "shrunk"):
            nli_evaluated += 1

    merged = 0
    for source in source_data if isinstance(source_data, list) else []:
        if isinstance(source, dict) and source.get("duplicates"):
            merged += len(source.get("duplicates") or [])

    stages: List[Dict[str, Any]] = []
    agreement = research_state.get("source_agreement")
    agreement = agreement if isinstance(agreement, dict) else {}

    stages.append({
        "name": "Query planning & web search",
        "status": "skipped" if research_skipped else "completed",
        "detail": (
            f"{len(queries)} search quer{'y' if len(queries) == 1 else 'ies'} issued"
            if queries
            else "No search queries were issued"
        ),
    })
    stages.append({
        "name": "Source retrieval & scraping",
        "status": "skipped" if research_skipped else "completed",
        "detail": (
            f"{sources_retrieved} candidate URL(s) retrieved, {articles_scraped} scraped successfully, "
            f"{len(scrape_failures)} retrieval attempt(s) failed"
            + (
                f"; {agreement.get('total_sources', 0)} distinct source(s) after identity resolution"
                if agreement else ""
            )
            if sources_retrieved or scrape_failures
            else "No sources were retained"
        ),
    })
    stages.append({
        "name": "Evidence extraction & NLI evaluation",
        "status": "skipped" if research_skipped else "completed",
        "detail": (
            f"{counted_passages(evidence)} passage(s) classified, {nli_evaluated} NLI-scored"
            if evidence
            else "No passages were extracted"
        ),
    })

    agreement = research_state.get("source_agreement")
    agreement = agreement if isinstance(agreement, dict) else {}

    if agreement:
        stages.append({
            "name": "Source independence & corroboration",
            "status": "completed",
            "detail": (
                f"{agreement.get('independence_groups', 0)} independent publisher group(s) "
                f"from {agreement.get('total_sources', 0)} distinct source(s); "
                f"{agreement.get('primary_supporting_groups', 0)} primary supporting group(s), "
                f"{agreement.get('primary_contradicting_groups', 0)} primary contradicting group(s); "
                f"{agreement.get('snippet_only_sources', 0)} snippet-only source(s); "
                f"{agreement.get('unresolved_identity_sources', 0)} unresolved source identity/identities; "
                f"agreement {agreement.get('agreement_score')}, "
                f"{agreement.get('independent_supporting', 0)} supporting group(s), "
                f"{agreement.get('independent_contradicting', 0)} contradicting group(s)"
            ),
        })

    if research_state.get("verification"):
        stages.append({
            "name": "Verification agent",
            "status": "completed",
            "detail": "Verification agent analysed the evidence brief and returned a verdict.",
        })
    if research_state.get("feedback"):
        stages.append({
            "name": "Critic agent",
            "status": "completed",
            "detail": "Critic agent reviewed the verification agent's reasoning.",
        })

    final_v = research_state.get("final_verdict")
    if isinstance(final_v, dict) and final_v.get("verdict"):
        stages.append({
            "name": "Deterministic evidence gate",
            "status": "completed",
            "detail": _clean_text(final_v.get("permitted_reason"))
            or f"Adjudicated as {final_v.get('verdict')}",
        })

    if modality == "video":
        stages.extend(_video_diagnostic_stages(state, modality_specific))
    elif modality == "audio":
        stages.extend(_audio_diagnostic_stages(state))

    return {
        "research_skipped": bool(research_skipped),
        "research_skipped_reason": research_skipped_reason,
        "search_queries": queries,
        "search_errors": [_clean_text(e) for e in search_errors if _clean_text(e)],
        "sources_discovered": _as_int(research_state.get("sources_discovered")),
        "sources_retrieved": sources_retrieved,
        "articles_scraped": articles_scraped,
        "scrapes_failed": [
            {
                "url": f.get("url") if _is_valid_url(f.get("url")) else None,
                "domain": f.get("domain"),
                "title": f.get("title"),
                "reason": _clean_text(f.get("reason")),
                "fallback_used": f.get("fallback_used"),
            }
            for f in scrape_failures
            if isinstance(f, dict)
        ],
        "syndicated_sources_merged": merged,
        "evidence_passages": counted_passages(evidence),
        "nli_evaluated": nli_evaluated,
        # --- evidence-policy counters (real, from the fusion layer) ------
        "independent_publishers": agreement.get("independence_groups", 0),
        "primary_supporting_groups": agreement.get("primary_supporting_groups", 0),
        "primary_contradicting_groups": agreement.get("primary_contradicting_groups", 0),
        "article_supporting_groups": agreement.get("article_supporting_groups", 0),
        "article_contradicting_groups": agreement.get("article_contradicting_groups", 0),
        "correction_groups": agreement.get("correction_groups", 0),
        "snippet_only_sources": agreement.get("snippet_only_sources", 0),
        "article_sources": agreement.get("article_sources", 0),
        "unresolved_identity_sources": agreement.get("unresolved_identity_sources", 0),
        "verification_coverage": agreement.get("verification_coverage", 0.0),
        "stages": stages,
    }


def _audio_diagnostic_stages(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Report the audio extraction stages with their real counters."""
    stages: List[Dict[str, Any]] = []

    transcription = state.get("transcription") if isinstance(state.get("transcription"), dict) else {}
    segments = transcription.get("segments") or []
    error = transcription.get("error")
    stages.append({
        "name": "Speech transcription",
        "status": "failed" if error else ("completed" if segments else "skipped"),
        "detail": (
            f"transcription failed: {error}"
            if error
            else f"{len(segments)} segment(s)"
            + (f", language '{transcription.get('language')}'" if transcription.get("language") else "")
        ),
    })

    extraction = state.get("claim_extraction") if isinstance(state.get("claim_extraction"), dict) else {}
    selected = extraction.get("selected_claims") or []
    candidates = extraction.get("all_candidates") or []
    stages.append({
        "name": "Claim extraction",
        "status": "completed" if candidates else "skipped",
        "detail": (
            f"{len(candidates)} candidate claim(s) extracted, {len(selected)} selected for research"
            if candidates
            else "No candidate claims were extracted from the transcript"
        ),
    })
    return stages


def counted_passages(evidence: Any) -> int:
    if not isinstance(evidence, list):
        return 0
    return sum(1 for item in evidence if isinstance(item, dict))


def _video_diagnostic_stages(
    state: Dict[str, Any],
    modality_specific: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Report the video extraction stages with their real counters.

    These explain *why* a video may have produced no verifiable claim, using
    only values the video pipeline actually recorded.
    """
    stages: List[Dict[str, Any]] = []

    segments = state.get("transcription_segments") or []
    language = state.get("transcription_language")
    stages.append({
        "name": "Speech transcription",
        "status": "completed" if segments else "skipped",
        "detail": (
            f"{len(segments)} Whisper segment(s)"
            + (f", language detected as '{language}'" if language else "")
            if segments
            else "No speech segments were produced"
        ),
    })

    ocr_observations: List[Dict[str, Any]] = []
    ocr_frames_with_text = 0
    if isinstance(modality_specific, dict):
        frames = modality_specific.get("frames")
        if isinstance(frames, dict):
            ocr_frames_with_text = _as_int(frames.get("ocr_frames_with_text_count"))
            ocr_observations = frames.get("ocr_observations") or []
    stages.append({
        "name": "On-screen text (OCR)",
        "status": "completed" if ocr_observations else "skipped",
        "detail": (
            f"text detected in {ocr_frames_with_text or len(ocr_observations)} frame(s), "
            f"{len(ocr_observations)} observation(s) retained"
            if ocr_observations
            else "No on-screen text was detected in the sampled frames"
        ),
    })

    total = _as_int(state.get("candidate_claims_count"))
    selected = _as_int(state.get("selected_claims_count"))
    rejected = _as_int(state.get("rejected_claims_count"))
    stages.append({
        "name": "Factual claim filtering",
        "status": "completed" if total else "skipped",
        "detail": (
            f"{total} candidate claim(s) extracted, {selected} passed factual-claim filtering, "
            f"{rejected} rejected"
            if total
            else "No candidate claims were extracted from the speech or on-screen text"
        ),
    })
    return stages


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _build_policy_diagnostics(final_verdict: Dict[str, Any]) -> Dict[str, Any]:
    """
    The deterministic gate's own policy-accounting diagnostics.

    When the evidence policy counters are missing or malformed the gate
    cannot apply its policy and abstains with UNVERIFIED. That abstention is
    an internal evidence-accounting problem and must never be presented as a
    finding about the claim.

    ``gate_ran`` distinguishes "the gate evaluated the policy and the policy
    data was sound" from "the gate never ran for this result" (for example a
    modality with no verifiable claim to research).
    """
    final_verdict = final_verdict or {}

    if "policy_data_valid" not in final_verdict:
        return {
            "gate_ran": False,
            "policy_data_valid": None,
            "policy_data_errors": [],
            "verdict_restricted": False,
            "explanation": (
                "The deterministic evidence gate did not run for this result, so no "
                "evidence policy counters were evaluated."
            ),
        }

    errors = final_verdict.get("policy_data_errors") or []
    if not isinstance(errors, list):
        errors = [errors]
    errors = [str(e) for e in errors]

    valid = bool(final_verdict.get("policy_data_valid")) and not errors

    if valid:
        return {
            "gate_ran": True,
            "policy_data_valid": True,
            "policy_data_errors": [],
            "verdict_restricted": False,
            "explanation": (
                "The evidence policy counters were complete and well-formed, so the "
                "deterministic gate applied the full evidence policy."
            ),
        }

    listed = "; ".join(errors) if errors else "unknown policy fields"
    return {
        "gate_ran": True,
        "policy_data_valid": False,
        "policy_data_errors": errors,
        "verdict_restricted": True,
        "explanation": (
            "The evidence policy data was incomplete or invalid, so the deterministic "
            "gate could not evaluate the evidence policy and restricted the result to "
            f"UNVERIFIED. Unusable policy field(s): {listed}. This is an internal "
            "evidence-accounting issue, NOT proof that the claim is false, and it does "
            "not mean the claim was refuted."
        ),
    }


def _build_agent_status(research_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Which agent stages ran, and which failed.

    A provider failure in an agent stage does not discard the retrieval,
    scraping and NLI work that already completed, but the result is a
    PARTIAL verification and must say so.
    """
    research_state = research_state or {}
    errors = research_state.get("agent_errors") or []
    if not isinstance(errors, list):
        errors = [errors]
    errors = [e for e in errors if isinstance(e, dict)]

    verification = research_state.get("verification_status")
    critic = research_state.get("feedback_status")
    if verification is None and critic is None and not errors:
        return {}

    return {
        "verification_agent": verification or "unknown",
        "critic_agent": critic or "unknown",
        "partial": bool(errors),
        "errors": [
            {"stage": e.get("stage"), "error": e.get("error")} for e in errors
        ],
    }


def _build_executive_summary(
    modality: str,
    primary_claim: Optional[str],
    research_state: Optional[Dict[str, Any]],
    final_verdict: Dict[str, Any],
    findings: Dict[str, List[Dict[str, Any]]],
    methodology: Dict[str, Any],
    research_skipped: bool,
    research_skipped_reason: Optional[str],
) -> Dict[str, Any]:
    verdict = final_verdict.get("verdict")
    confidence = _as_float(final_verdict.get("confidence"))
    if confidence is not None and confidence > 1.0:
        confidence = confidence / 100.0

    reasoning = _clean_text(final_verdict.get("reasoning")) or None
    gate_reason = _clean_text(final_verdict.get("deterministic_gate_reason")) or None

    supporting = findings.get("supporting", [])
    contradicting = findings.get("contradicting", [])
    contextual = findings.get("contextual", [])

    key_findings: List[str] = []
    if supporting:
        names = ", ".join(
            f"{f.get('source_name') or f.get('source_domain')}" for f in supporting[:3]
        )
        key_findings.append(
            f"{len(supporting)} passage(s) from {len({f.get('independence_group') for f in supporting})} "
            f"independent source group(s) directly support the claim ({names})."
        )
    if contradicting:
        names = ", ".join(
            f"{f.get('source_name') or f.get('source_domain')}" for f in contradicting[:3]
        )
        key_findings.append(
            f"{len(contradicting)} passage(s) from {len({f.get('independence_group') for f in contradicting})} "
            f"independent source group(s) directly contradict the claim ({names})."
        )
    if contextual:
        key_findings.append(
            f"{len(contextual)} further passage(s) discuss the subject without establishing or refuting the claim."
        )
    if not key_findings:
        if research_skipped:
            key_findings.append(
                "No web research was performed, so no external evidence was collected."
            )
        else:
            key_findings.append(
                "No retrieved passage was classified as direct support or direct contradiction."
            )

    attempted = _build_attempted(modality, primary_claim, methodology, research_skipped)

    limitations: List[str] = []
    for limit in _build_limitations(final_verdict, methodology, research_skipped, research_skipped_reason):
        if limit and limit not in limitations:
            limitations.append(limit)

    agent_status = _build_agent_status(research_state)
    if agent_status.get("partial"):
        failed = ", ".join(
            str(e.get("stage")) for e in agent_status.get("errors", [])
            if e.get("stage")
        )
        limitations.append(
            f"The {failed or 'agent'} stage could not be completed, so this is a "
            "PARTIAL verification: the retrieved evidence was still analysed and "
            "adjudicated by the deterministic evidence gate, but no agent verdict "
            "was available to be accepted. The result is not a finished "
            "verification of the claim."
        )

    agent_narrative = _extract_section(research_state.get("verification"), "Final Explanation") if research_state else ""
    if not agent_narrative:
        agent_narrative = _clean_text(research_state.get("verification")) if research_state else ""
    critic_review = _extract_section(research_state.get("feedback"), "Final Recommendation") if research_state else ""
    if not critic_review:
        critic_review = _clean_text(research_state.get("feedback")) if research_state else ""

    return {
        "input_claim": primary_claim,
        "modality": modality,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_basis": final_verdict.get("confidence_basis"),
        "confidence_basis_note": (
            "Share of the retrieved material that could actually be read and "
            "classified. It is not a probability that the claim is true or false: "
            "an UNVERIFIED verdict means the evidence did not settle the claim, "
            "not that the claim was refuted."
            if final_verdict.get("confidence_basis") == "verification_coverage"
            else "Strength of the evidence behind a settled verdict, derived from "
                 "independent corroboration, source agreement and the fusion score."
        ),
        "confidence_basis_explanation": (
            "Deterministic confidence computed by the evidence gate from source agreement, "
            "fusion score and independent corroboration counts. It is not an AI-generation score."
        ),
        "policy_diagnostics": _build_policy_diagnostics(final_verdict),
        "agent_status": agent_status,
        "attempted": attempted,
        "key_findings": key_findings,
        "why_this_verdict": reasoning,
        "deterministic_gate_reason": gate_reason,
        "agent_narrative": agent_narrative or None,
        "critic_review": critic_review or None,
        "limitations": limitations,
    }


def _build_attempted(
    modality: str,
    primary_claim: Optional[str],
    methodology: Dict[str, Any],
    research_skipped: bool,
) -> str:
    if research_skipped:
        return (
            "The system extracted the input, but no verifiable factual claim was available "
            "to research, so no web research was performed."
        )
    queries = methodology.get("search_queries") or []
    retrieved = methodology.get("sources_retrieved", 0)
    scraped = methodology.get("articles_scraped", 0)
    passages = methodology.get("evidence_passages", 0)
    return (
        f"The system issued {len(queries)} search quer{'y' if len(queries) == 1 else 'ies'}, "
        f"retrieved {retrieved} candidate source(s) of which {scraped} were scraped successfully "
        f"(resolving to {methodology.get('independent_publishers', 0)} independent publisher group(s)), "
        f"extracted {passages} passage(s), and evaluated each passage against the claim "
        f"with an NLI cross-encoder before applying the deterministic evidence gate."
    )


def _build_limitations(
    final_verdict: Dict[str, Any],
    methodology: Dict[str, Any],
    research_skipped: bool,
    research_skipped_reason: Optional[str],
) -> List[str]:
    limits: List[str] = []

    if research_skipped:
        limits.append(
            research_skipped_reason
            or "Research was skipped because no verifiable claim was extracted from the input."
        )
        limits.append(
            "No external source was consulted, so the factual verdict rests on the absence "
            "of checkable assertions rather than on retrieved evidence."
        )
        return limits

    supporting = _as_int(final_verdict.get("independent_supporting"))
    contradicting = _as_int(final_verdict.get("independent_contradicting"))
    total = _as_int(final_verdict.get("total_sources"))
    permitted = final_verdict.get("permitted_verdicts") or []

    if supporting == 0 and contradicting == 0:
        limits.append(
            "No retrieved source directly addressed the claim, so the verdict reflects an "
            "evidence gap and not a finding of falsity."
        )
    if supporting == 1:
        limits.append(
            "Only one independent publisher corroborated the claim; the evidence gate requires "
            "two independent primary publisher groups before it can permit a corroborated verdict."
        )
    if contradicting == 1:
        limits.append(
            "Only one independent publisher contradicted the claim; a single contradicting "
            "publisher is not enough for the evidence gate to permit a refuted verdict."
        )
    if (
        final_verdict.get("verdict") == "UNVERIFIED"
        and (supporting or contradicting)
    ):
        # Only when the verdict really is UNVERIFIED. UNVERIFIED is always
        # among the permitted verdicts, so testing for its presence would
        # state this limitation even for a settled REAL/FAKE verdict.
        limits.append(
            "The evidence gate permitted UNVERIFIED only, so the reported verdict is a lack of "
            "sufficient independent evidence and must not be read as a finding of falsity."
        )
    if total and not methodology.get("articles_scraped"):
        limits.append(
            "No source page body was scraped successfully; only provider excerpts were available, "
            "which cannot establish or refute a claim on their own."
        )
    if methodology.get("search_errors"):
        limits.append(
            f"{len(methodology['search_errors'])} search request(s) failed during retrieval, "
            "so the candidate source pool was smaller than intended."
        )
    if methodology.get("scrapes_failed"):
        limits.append(
            f"{len(methodology['scrapes_failed'])} retrieved page(s) could not be scraped; "
            "their content was unavailable for passage extraction."
        )
    return limits


# ---------------------------------------------------------------------------
# Claim-level results
# ---------------------------------------------------------------------------

def _build_claim_results(
    modality: str,
    state: Dict[str, Any],
    research_state: Optional[Dict[str, Any]],
    primary_claim: Optional[str],
) -> List[Dict[str, Any]]:
    """One record per claim the backend actually adjudicated."""
    if modality == "video":
        results: List[Dict[str, Any]] = []
        for entry in _video_claim_results(state):
            research = entry.get("research_state") or {}
            final_v = research.get("final_verdict") if isinstance(research, dict) else {}
            final_v = final_v if isinstance(final_v, dict) else {}
            confidence = _as_float(final_v.get("confidence"))
            if confidence is not None and confidence > 1.0:
                confidence = confidence / 100.0
            results.append({
                "claim_text": entry.get("claim") or None,
                "verdict": final_v.get("verdict"),
                "confidence": confidence,
                "evidence_sufficiency": final_v.get("evidence_sufficiency"),
                "reasoning": _clean_text(final_v.get("reasoning")) or None,
                "supporting_evidence": final_v.get("supporting_evidence") or [],
                "contradicting_evidence": final_v.get("contradicting_evidence") or [],
                "source_name": None,
                "source_url": None,
                "timestamps": entry.get("timestamps"),
                "frame_references": entry.get("frame_references"),
                "source_modality": entry.get("source_modality"),
                "selection_reason": entry.get("selection_reason"),
                "priority_score": entry.get("priority_score"),
                "limitations": [],
            })
        return results

    final_v = {}
    if research_state:
        candidate = research_state.get("final_verdict")
        if isinstance(candidate, dict):
            final_v = candidate
    if not final_v:
        return []

    confidence = _as_float(final_v.get("confidence"))
    if confidence is not None and confidence > 1.0:
        confidence = confidence / 100.0

    supporting = final_v.get("supporting_evidence") or []
    contradicting = final_v.get("contradicting_evidence") or []

    def _first_link(items: Any) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(items, list) or not items:
            return None, None
        first = items[0]
        if not isinstance(first, dict):
            return None, None
        url = first.get("url")
        return (url if _is_valid_url(url) else None), (first.get("domain") or None)

    support_url, support_domain = _first_link(supporting)
    contradict_url, contradict_domain = _first_link(contradicting)

    return [{
        "claim_text": primary_claim,
        "verdict": final_v.get("verdict"),
        "confidence": confidence,
        "evidence_sufficiency": final_v.get("evidence_sufficiency"),
        "reasoning": _clean_text(final_v.get("reasoning")) or None,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "source_name": support_domain or contradict_domain,
        "source_url": support_url or contradict_url,
        "timestamps": None,
        "frame_references": None,
        "source_modality": modality,
        "selection_reason": None,
        "priority_score": None,
        "limitations": [],
    }]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_research_report(
    state: Optional[Dict[str, Any]],
    modality: str = "text",
    primary_claim: Optional[str] = None,
    modality_specific: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the point-wise research report for a verification result.

    ``state`` is the raw pipeline state that the API already returns under
    ``raw``. ``primary_claim`` is the claim the backend actually adjudicated.
    ``modality_specific`` is the modality payload the API already exposes.
    """
    state = state if isinstance(state, dict) else {}

    research_state = _text_research_state(state, modality)
    if research_state is None and modality != "video":
        research_state = state if isinstance(state.get("evidence_analysis"), list) else None

    if primary_claim is None:
        primary_claim = _derive_primary_claim(state, modality, research_state)

    final_verdict: Dict[str, Any] = {}
    if research_state:
        candidate = research_state.get("final_verdict")
        if isinstance(candidate, dict):
            final_verdict = candidate

    if modality == "video" and not final_verdict:
        # No research ran, so the video pipeline's own adjudicated verdict is
        # the one that must be reported (e.g. INSUFFICIENT_VERIFIABLE_CLAIMS).
        video_verdict = state.get("final_evidence_verdict")
        if video_verdict:
            final_verdict = {"verdict": video_verdict}

    research_skipped, skip_reason = _research_skipped(state, modality, research_state, final_verdict)

    findings, _ = _build_findings(research_state or {}, primary_claim)
    methodology = _build_methodology(
        research_state or {}, state, modality, research_skipped, skip_reason,
        modality_specific=modality_specific,
    )

    summary = _build_executive_summary(
        modality,
        primary_claim,
        research_state,
        final_verdict,
        findings,
        methodology,
        research_skipped,
        skip_reason,
    )

    claim_results = _build_claim_results(modality, state, research_state, primary_claim)
    gaps = _build_evidence_gaps(findings, methodology, final_verdict, research_skipped, skip_reason)

    return {
        "modality": modality,
        "executive_summary": summary,
        "findings": findings,
        "claims": claim_results,
        "methodology": methodology,
        "evidence_gaps": gaps,
    }


def _derive_primary_claim(
    state: Dict[str, Any],
    modality: str,
    research_state: Optional[Dict[str, Any]],
) -> Optional[str]:
    if modality == "image":
        normalized = state.get("normalized_claim")
        if isinstance(normalized, dict) and normalized.get("claim"):
            return _clean_text(normalized.get("claim"))
        image_claim = state.get("image_claim")
        if isinstance(image_claim, dict) and image_claim.get("ocr_text"):
            return _clean_text(image_claim.get("ocr_text"))[:_MAX_TITLE]
        return None
    if modality == "audio":
        extraction = state.get("claim_extraction")
        if isinstance(extraction, dict) and extraction.get("primary_claim"):
            return _clean_text(extraction.get("primary_claim"))
        return None
    if modality == "video":
        claim = state.get("primary_claim_analyzed")
        if claim:
            return _clean_text(claim)
        for entry in _video_claim_results(state):
            if entry.get("claim"):
                return _clean_text(entry.get("claim"))
        return None
    if research_state:
        claim = research_state.get("claim")
        if claim:
            return _clean_text(claim)
    return None


def _research_skipped(
    state: Dict[str, Any],
    modality: str,
    research_state: Optional[Dict[str, Any]],
    final_verdict: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Detect, from real state, that the research stage never ran."""
    if modality == "video":
        reason = _clean_text(state.get("research_skipped_reason"))
        total = _as_int(state.get("candidate_claims_count"))
        selected = _as_int(state.get("selected_claims_count"))
        detail = ""
        if total:
            detail = (
                f" The video produced {total} candidate claim(s) from speech and on-screen text, "
                f"of which {selected} passed the factual-claim filter."
            )
        if reason:
            return True, f"{reason}{detail}"
        if not state.get("verified_claims"):
            return True, (
                "No candidate claim from the video passed factual-claim filtering, so no "
                "claim was sent to the evidence engine and no web research was performed."
                + detail
            )
        return False, None

    if research_state is None:
        if modality == "audio":
            return True, _audio_skip_reason(state)
        return True, "The evidence engine did not run for this input."

    # Research ran if the retrieval or evidence stage produced anything.
    ran = (
        bool(research_state.get("evidence_analysis"))
        or bool(research_state.get("source_data"))
        or "search_results" in research_state
        or bool(research_state.get("search_plans"))
    )
    if ran:
        return False, None

    verdict = (final_verdict or {}).get("verdict")
    if verdict and verdict != "INSUFFICIENT_VERIFIABLE_CLAIMS":
        return False, None

    return True, (
        "No verifiable claim was extracted from the input, so the evidence engine was not run."
    )


def _audio_skip_reason(state: Dict[str, Any]) -> str:
    """
    Explain, from the audio pipeline's own records, why no claim was researched.
    """
    factual = state.get("factual_verification")
    if isinstance(factual, dict) and factual.get("enabled") is False:
        return (
            "Factual claim research was not enabled for this request, so no claim was "
            "sent to the evidence engine and no web research was performed."
        )

    transcription = state.get("transcription")
    if isinstance(transcription, dict):
        error = transcription.get("error")
        if error:
            return (
                f"Speech transcription failed ({error}), so no transcript was available to "
                "extract claims from and no web research was performed."
            )
        if not (transcription.get("segments") or transcription.get("text")):
            return (
                "The transcription produced no speech segments, so no claim could be "
                "extracted and no web research was performed."
            )

    extraction = state.get("claim_extraction")
    if isinstance(extraction, dict):
        if not extraction.get("primary_claim") and not extraction.get("selected_claims"):
            return (
                "No verifiable claim was extracted from the transcript, so no claim was "
                "sent to the evidence engine and no web research was performed."
            )

    return "The evidence engine did not run for this input."


def _build_evidence_gaps(
    findings: Dict[str, List[Dict[str, Any]]],
    methodology: Dict[str, Any],
    final_verdict: Dict[str, Any],
    research_skipped: bool,
    skip_reason: Optional[str],
) -> List[Dict[str, Any]]:
    """Unresolved questions and evidence gaps, all backed by real counters."""
    gaps: List[Dict[str, Any]] = []

    if research_skipped:
        gaps.append({
            "question": "What does the evidence base actually say about this claim?",
            "status": "NOT_ESTABLISHED",
            "detail": (
                skip_reason
                or "Research was skipped, so no external source was consulted for this input."
            ),
            "what_would_help": (
                "A verifiable factual claim extracted from the input, and web research enabled "
                "so retrieved sources can be evaluated against it."
            ),
        })
        return gaps

    supporting = findings.get("supporting", [])
    contradicting = findings.get("contradicting", [])
    contextual = findings.get("contextual", [])

    if not supporting and not contradicting:
        gaps.append({
            "question": "Is the claim supported or refuted by retrieved reporting?",
            "status": "NOT_ESTABLISHED",
            "detail": (
                f"{len(contextual)} retrieved passage(s) discussed the subject, but none was "
                "classified as direct support or direct contradiction."
            ),
            "what_would_help": (
                "At least two independent primary publisher groups reporting directly on the "
                "claimed event, date and location."
            ),
        })

    if supporting and contradicting:
        gaps.append({
            "question": "Which account should be preferred where sources conflict?",
            "status": "CONTESTED",
            "detail": (
                f"{len(supporting)} supporting passage(s) and {len(contradicting)} contradicting "
                "passage(s) were retrieved from independent source groups."
            ),
            "what_would_help": (
                "A source that resolves the conflict directly, or corroboration of one account "
                "from further independent publishers."
            ),
        })

    snippet_only = [
        f for f in (supporting + contradicting)
        if f.get("evidence_type") == "search_snippet"
    ]
    if snippet_only:
        gaps.append({
            "question": "Would the snippet-level evidence survive full-article reading?",
            "status": "PARTIAL",
            "detail": (
                f"{len(snippet_only)} decisive passage(s) came from provider snippets rather "
                "than a scraped article body."
            ),
            "what_would_help": "Successful retrieval of the full article text for those sources.",
        })

    if methodology.get("search_errors"):
        gaps.append({
            "question": "What would a complete search have returned?",
            "status": "PARTIAL",
            "detail": (
                f"{len(methodology['search_errors'])} search request(s) failed: "
                + "; ".join(methodology["search_errors"][:3])
            ),
            "what_would_help": "A successful retry of the failed search requests.",
        })

    if methodology.get("scrapes_failed"):
        gaps.append({
            "question": "What did the pages that could not be scraped contain?",
            "status": "PARTIAL",
            "detail": (
                f"{len(methodology['scrapes_failed'])} retrieved page(s) failed retrieval, so their "
                "full content was never evaluated."
            ),
            "what_would_help": "Successful retrieval of those pages.",
        })

    return gaps
