"""
independent_llm_analysis.py
===========================
Independent direct-LLM verification stage.

This module sends the ORIGINAL claim/query directly to the configured LLM and
asks it to produce its own independent analysis. It is deliberately separate
from the evidence-based pipeline and from the evidence-reviewing second
opinion:

* It does NOT re-run the evidence engine, search, or scraping.
* It uses the LLM's available knowledge and clearly discloses when live
  verification or reliable evidence is unavailable.
* It never fabricates citations or pretends knowledge-based reasoning is live
  fact-checking.
* It never overrides the deterministic evidence gate.
* Its result is reported under "Independent LLM Analysis", separate from the
  system verdict and from the LLM second opinion.

Verdicts: REAL, FAKE, MISLEADING, UNVERIFIED.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents import invoke_agent, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

VALID_VERDICTS = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}

# Bounded, finite call so it never blocks the pipeline.
INDEPENDENT_LLM_TIMEOUT_SECONDS = 40
INDEPENDENT_LLM_MAX_ATTEMPTS = 2

# Reuse the configured Groq LLM and retry wrapper from agents.py.
_independent_llm_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an independent LLM fact-checker analyzing a claim using your own
available knowledge. You do NOT have access to live web search, retrieved
articles, or an evidence package unless they are explicitly provided below.

TASK: Review the CLAIM and the EXTRACTED TEXT (when available) and produce your OWN independent verification analysis.

The EXTRACTED TEXT is text already pulled from the media (OCR, transcript, on-screen text) by the existing pipelines. It is NOT live web evidence. Do not re-run OCR, transcription, or video analysis. Use only what is supplied.

STRICT RULES:

1. Identify the exact claim being made. Do not merge unrelated statements.
2. Analyze the claim using your available knowledge plus the supplied EXTRACTED TEXT.
3. If reliable live sources or retrieved evidence are NOT provided, clearly state that live verification is unavailable and that your reasoning is knowledge-based, not live fact-checking.
4. Do NOT fabricate citations, URLs, publishers, or quotations.
5. Do NOT claim you independently verified facts without supporting material.
6. Extracted text alone cannot prove an image or video is authentic or detect every deepfake; distinguish text-based analysis from visual/audio/video verification.
7. Do NOT override the deterministic evidence gate: if the available information is insufficient, use UNVERIFIED.
8. Your verdict is separate from the evidence-based system verdict; show agreement or disagreement only as a comparison, never as an override.

VERDICT DEFINITIONS:

REAL:    your knowledge supports the claim as stated.
FAKE:    your knowledge directly refutes a key part of the claim.
MISLEADING: the core event is real but a material detail is wrong or lacks
         context.
UNVERIFIED: there is not enough reliable information to confirm or refute
            the claim.

Confidence: a 0-100 number expressing how well your available knowledge
supports your verdict. It is NOT a probability that the claim is true. Low
confidence is expected when information is thin or conflicting.

Return ONLY this structure (no extra sections):

VERDICT: REAL / FAKE / MISLEADING / UNVERIFIED
CONFIDENCE: <0-100>%
CLAIM_SUMMARY: <one sentence restating the exact claim being checked>
REASONING: <2-4 sentences of knowledge-based analysis>
SUPPORTING_EVIDENCE:
- <url> | <domain> | <short excerpt>   (only if genuinely available)
CONTRADICTING_EVIDENCE:
- <url> | <domain> | <short excerpt>   (only if genuinely available)
LIVE_VERIFICATION_AVAILABLE: yes / no
UNCERTAINTY: <what remains unknown>
LIMITATIONS:
- <limitation>
AGREEMENT_WITH_SYSTEM: <agree / disagree> - <one sentence why>

If a section has no entries, write "none". If no reliable evidence is
available, set VERDICT to UNVERIFIED and LIVE_VERIFICATION_AVAILABLE to no.
""",
        ),
        (
            "human",
            """
============================================================
CLAIM (original query / extracted claim)
============================================================

{claim}

============================================================
EXTRACTED TEXT (already collected by existing pipelines; not live evidence)
============================================================

{extracted_text}

============================================================
EXISTING EVIDENCE-BASED SYSTEM VERDICT (context only - not an override)
============================================================

{existing_verdict}

============================================================
TASK
============================================================

Give your independent analysis of the claim above using your available knowledge and the extracted text. Clearly disclose whether live verification is available. Return the structured review.
""",
        )
    ])
    | llm
    | StrOutputParser()
)


def _extract_section(text: str, heading: str) -> str:
    """Pull a labelled section out of the LLM's structured answer."""
    if not isinstance(text, str) or not text.strip():
        return ""
    pattern = re.compile(
        rf"^{re.escape(heading)}\s*:\s*(.*?)(?=^\s*[A-Z_ /&()'-]+\s*:|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _extract_bool_section(text: str, heading: str) -> Optional[bool]:
    """Extract a yes/no value from a labelled section."""
    body = _extract_section(text, heading)
    if not body:
        return None
    low = body.strip().lower()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    return None


def _extract_list_section(text: str, heading: str) -> List[str]:
    """Pull a bulleted list under a labelled heading."""
    body = _extract_section(text, heading)
    if not body or body.lower() == "none":
        return []
    items = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.lower() == "none":
            continue
        line = re.sub(r"^[-•\*]\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def _parse_evidence_reference(ref: str) -> Optional[Dict[str, str]]:
    """Parse a single evidence reference of the form:
      <url> | <domain> | <excerpt>
    Returns None when it cannot be parsed into at least a URL.
    """
    if not ref:
        return None
    parts = [p.strip() for p in ref.split("|")]
    url = parts[0] if parts else ""
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return None
    domain = parts[1] if len(parts) > 1 else ""
    excerpt = parts[2] if len(parts) > 2 else ""
    return {"url": url, "domain": domain, "excerpt": excerpt}


def _extract_verdict(text: str) -> Optional[str]:
    """Extract the labelled verdict from the LLM answer."""
    if not text:
        return None
    labelled = list(
        re.finditer(
            r"verdict\s*[:\-]\s*\**\s*(REAL|FAKE|MISLEADING|UNVERIFIED)\b",
            text,
            re.IGNORECASE,
        )
    )
    if labelled:
        return labelled[-1].group(1).upper()
    return None


def _extract_confidence(text: str) -> Optional[float]:
    """Extract the confidence percentage from the LLM answer."""
    if not text:
        return None
    match = re.search(
        r"confidence\s*[:\-]?\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%?",
        text,
        re.IGNORECASE,
    )
    if match:
        try:
            value = float(match.group(1))
            if 0 <= value <= 100:
                return value
        except ValueError:
            return None
    return None


def _extract_agreement(text: str) -> Optional[str]:
    """Extract AGREEMENT_WITH_SYSTEM value."""
    if not text:
        return None
    match = re.search(
        r"agreement_with_system\s*[:\-]\s*\**\s*(agree|disagree)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    return None


def _build_independent_llm_result(
    raw_output: str,
    error: Optional[str],
    timeout: bool,
    api_failure: bool,
    existing_verdict: Optional[str],
    no_live: bool,
) -> Dict[str, Any]:
    """Assemble the structured independent LLM result and comparison."""
    verdict = _extract_verdict(raw_output)
    confidence = _extract_confidence(raw_output)
    claim_summary = _extract_section(raw_output, "CLAIM_SUMMARY")
    reasoning = _extract_section(raw_output, "REASONING")
    live_available = _extract_bool_section(raw_output, "LIVE_VERIFICATION_AVAILABLE")
    uncertainty = _extract_section(raw_output, "UNCERTAINTY")
    limitations = _extract_list_section(raw_output, "LIMITATIONS")
    agreement_with_system = _extract_agreement(raw_output)
    supporting = [
        ref
        for ref in (
            _parse_evidence_reference(r)
            for r in _extract_list_section(raw_output, "SUPPORTING_EVIDENCE")
        )
        if ref
    ]
    contradicting = [
        ref
        for ref in (
            _parse_evidence_reference(r)
            for r in _extract_list_section(raw_output, "CONTRADICTING_EVIDENCE")
        )
        if ref
    ]
    raw_output = raw_output or ""

    # Normalise and validate the verdict.
    if verdict is not None:
        verdict = verdict.upper()
        if verdict not in VALID_VERDICTS:
            verdict = None

    # If live verification is unavailable and no knowledge-based verdict could
    # be extracted, default to UNVERIFIED rather than fabricating.
    if verdict is None:
        verdict = "UNVERIFIED" if not live_available else None

    # Agreement / disagreement with the existing system verdict.
    agreement: Optional[bool] = None
    disagreement: Optional[bool] = None
    comparison_summary = ""
    if verdict is None:
        agreement = False
        disagreement = False
        comparison_summary = (
            "No valid independent LLM verdict was returned, so no agreement "
            "comparison could be made."
        )
    elif verdict == existing_verdict:
        agreement = True
        disagreement = False
        comparison_summary = (
            f"The independent LLM analysis agrees with the system verdict "
            f"({existing_verdict})."
        )
    else:
        agreement = False
        disagreement = True
        comparison_summary = (
            f"The independent LLM analysis ({verdict}) disagrees with the "
            f"system verdict ({existing_verdict}). The deterministic evidence "
            f"gate remains authoritative."
        )

    comparison: Dict[str, Any] = {
        "existing_verdict": existing_verdict,
        "llm_verdict": verdict,
        "agreement": agreement,
        "disagreement": disagreement,
        "comparison_summary": comparison_summary,
        "agreement_with_system": agreement_with_system,
        "live_verification_available": live_available,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "claim_summary": claim_summary,
        "reasoning": reasoning,
        "live_verification_available": live_available,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "uncertainty": uncertainty,
        "limitations": limitations,
        "raw_output": raw_output,
        "error": error,
        "timeout": timeout,
        "api_failure": api_failure,
        "comparison": comparison,
    }


# Maximum characters of extracted text sent to the LLM (keeps prompts concise).
INDEPENDENT_LLM_MAX_EXTRACTED_CHARS = 1500


def _truncate(text: Optional[str], max_chars: int) -> str:
    """Truncate text to max_chars, preserving the start; returns '' if empty."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_independent_llm_analysis(
    claim: str,
    existing_verdict: Optional[str],
    extracted_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce an independent direct-LLM analysis of the original claim.

    `extracted_text` is already-collected text from the media (OCR, transcript,
    on-screen text). It is NOT live web evidence. It is bounded to keep the
    prompt concise. No search or scraping is performed here.

    Never blocks the pipeline: on any failure the result is marked
    unavailable and the existing system result stands alone.
    """
    error: Optional[str] = None
    timeout = False
    api_failure = False
    raw_output = ""
    extracted_text = _truncate(extracted_text, INDEPENDENT_LLM_MAX_EXTRACTED_CHARS)

    try:
        raw_output = invoke_agent(
            _independent_llm_chain,
            {
                "claim": claim,
                "extracted_text": extracted_text or "(none available)",
                "existing_verdict": existing_verdict or "(no system verdict)",
            },
            stage="independent_llm_analysis",
        )
    except Exception as exc:
        message = str(exc)
        error = message
        low = message.lower()
        if isinstance(exc, TimeoutError) or "timeout" in low or "408" in low:
            timeout = True
        elif any(
            marker in low
            for marker in ("429", "ratelimit", "500", "502", "503", "504", "401", "403")
        ):
            api_failure = True
        raw_output = ""

    return _build_independent_llm_result(
        raw_output=raw_output,
        error=error,
        timeout=timeout,
        api_failure=api_failure,
        existing_verdict=existing_verdict,
        no_live=True,
    )
