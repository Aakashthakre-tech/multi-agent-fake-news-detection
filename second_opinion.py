"""
second_opinion.py
=================
Independent LLM second-opinion verification stage.

This module adds a separate LLM review of the claim and the evidence package
already collected by the existing evidence pipeline. It is deliberately a
second opinion only:

* it never re-runs search or scraping;
* it never invents sources, URLs, publishers, or quotations;
* it never treats training knowledge as retrieved evidence;
* it cannot override the deterministic evidence gate;
* its verdict is reported alongside, never instead of, the system verdict;
* it is independent of image-origin AI probability (which says nothing about
  whether a news claim is true).

The deterministic evidence gate remains authoritative. If the gate has
restricted the result to UNVERIFIED, the second opinion's verdict is reported
for comparison but never promoted to REAL or FAKE.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from agents import invoke_agent, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

VALID_VERDICTS = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}

# Finite request timeout: the second opinion must not block the pipeline.
SECOND_OPINION_TIMEOUT_SECONDS = 45
SECOND_OPINION_MAX_ATTEMPTS = 3

# Reuse the same LLM provider (Groq) and bounded retry wrapper already
# configured for the verification and critic agents. No new model instance,
# no new API key handling, no duplicated provider setup.
_second_opinion_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an independent SECOND-OPINION fact-checker.

Review the CLAIM and the EVIDENCE PACKAGE below. The EVIDENCE PACKAGE was
already collected, scraped, and classified by a separate evidence pipeline.
You must NOT re-research the claim, and you must NOT use pretrained knowledge,
memory, or general world knowledge.

STRICT RULES:

1. Use ONLY facts present in the EVIDENCE PACKAGE.
2. Do NOT invent sources, URLs, publishers, or quotations.
3. Do NOT treat a claim as verified merely because it sounds plausible.
4. If a fact is not in the EVIDENCE PACKAGE, mark it UNKNOWN.
5. Do NOT override the deterministic evidence gate: if the gate says the
   evidence is insufficient, you may not return REAL or FAKE. Use UNVERIFIED.
6. Only list evidence that actually appears in the EVIDENCE PACKAGE.
7. Image-origin AI probability (whether a photo is AI-generated) is a
   separate signal and says nothing about whether the news claim itself is
   true. Do not let it influence your factual verdict.

VERDICT DEFINITIONS:

REAL:    independent sources in the package directly support the claim,
         including the claimed date and location.
FAKE:    independent reliable sources in the package directly refute a key
         part of the claim.
MISLEADING: the core event is established but a material claimed detail is
         wrong or stripped of context, and that correction is explicitly in
         the package.
UNVERIFIED: the package does not establish the claim (insufficient evidence,
            conflicting sources, single source, or related context only).
            Use this when there is no usable evidence at all.

Confidence: a 0-100 number expressing how well the available EVIDENCE PACKAGE
supports your verdict. It is NOT a probability that the claim is true. Low
confidence is expected when evidence is thin or conflicting.

Return ONLY this structure (no extra sections):

VERDICT: REAL / FAKE / MISLEADING / UNVERIFIED
CONFIDENCE: <0-100>%
CLAIM_SUMMARY: <one sentence restating what is being checked>
REASONING: <2-4 sentences grounded in specific evidence from the package>
SUPPORTING_EVIDENCE:
- <url> | <domain> | <short excerpt>
CONTRADICTING_EVIDENCE:
- <url> | <domain> | <short excerpt>
SOURCE_CREDIBILITY: <assessment of source quality / primary vs snippet>
EVIDENCE_SUFFICIENCY: <sufficient / insufficient / absent>
MISSING_EVIDENCE: <what evidence would be needed to settle the claim>
UNCERTAINTY: <what remains unknown>
LIMITATIONS:
- <limitation>
AGREEMENT_WITH_SYSTEM: <agree / disagree> - <one sentence why>

If a section has no entries, write "none". If there is no usable evidence,
set VERDICT to UNVERIFIED, EVIDENCE_SUFFICIENCY to absent, and state
explicitly that verification is inconclusive.
""",
        ),
        (
            "human",
            """
============================================================
CLAIM
============================================================

{claim}

============================================================
EXISTING SYSTEM VERDICT (context only - do not let it override
your independent reading of the package)
============================================================

{existing_verdict}

============================================================
EVIDENCE PACKAGE (use only this)
============================================================

{evidence_package}

============================================================
TASK
============================================================

Give your independent second opinion on the claim, grounded only in the
EVIDENCE PACKAGE above. Return the structured review.
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


def _condense_evidence_package(
    evidence_analysis: List[Dict[str, Any]],
    source_agreement: Dict[str, Any],
    max_passages: int = 8,
    max_chars: int = 350,
) -> str:
    """
    Build a compact, bounded evidence package for the second-opinion LLM.

    Only real, already-collected evidence is included: URLs, domains, the
    evidence class, and short verbatim excerpts. Nothing is invented.
    """
    if not evidence_analysis:
        return "(no evidence passages were collected)"

    ranked = sorted(
        evidence_analysis,
        key=lambda r: r.get("evidence_score", 0),
        reverse=True,
    )

    lines: List[str] = []
    lines.append(
        f"Independent supporting publishers: "
        f"{source_agreement.get('independent_supporting', 0) if isinstance(source_agreement, dict) else 0}"
    )
    lines.append(
        f"Independent contradicting publishers: "
        f"{source_agreement.get('independent_contradicting', 0) if isinstance(source_agreement, dict) else 0}"
    )
    lines.append("")
    lines.append("TOP EVIDENCE PASSAGES:")

    count = 0
    for result in ranked:
        if count >= max_passages:
            break
        url = result.get("url") or ""
        domain = result.get("domain") or ""
        ev_class = result.get("evidence_class") or "UNKNOWN"
        chunk = (result.get("chunk") or "").strip()
        excerpt = re.sub(r"\s+", " ", chunk)[:max_chars]
        if not url and not excerpt:
            continue
        lines.append(
            f"- [{ev_class}] {url} | {domain} | {excerpt}"
        )
        count += 1

    if count == 0:
        lines.append("(no usable evidence passages)")

    return "\n".join(lines)


def _build_second_opinion_result(
    raw_output: str,
    error: Optional[str],
    timeout: bool,
    api_failure: bool,
    existing_verdict: Optional[str],
    permitted_verdicts: Optional[List[str]],
    no_evidence: bool,
) -> Dict[str, Any]:
    """Assemble the structured second-opinion result and comparison."""
    # When there is no usable evidence at all, short-circuit: report
    # inconclusive UNVERIFIED without fabricating an LLM analysis.
    if no_evidence:
        verdict = "UNVERIFIED"
        confidence = None
        explanation = (
            "No usable evidence was collected for this claim, so verification "
            "is inconclusive. The LLM second opinion was not invoked to avoid "
            "a fabricated analysis."
        )
        supporting = []
        contradicting = []
        uncertainty = "No evidence passages were retrieved or classified."
        limitations = [
            "No search results or scraped article text available to evaluate.",
            "Cannot determine whether the claim is supported or refuted.",
        ]
        source_credibility = "none - no sources retrieved"
        evidence_sufficiency = "absent"
        claim_summary = (existing_verdict and "") or ""
        agreement_with_system = None
        raw_output = ""
    else:
        verdict = _extract_verdict(raw_output)
        confidence = _extract_confidence(raw_output)
        explanation = _extract_section(raw_output, "REASONING")
        claim_summary = _extract_section(raw_output, "CLAIM_SUMMARY")
        source_credibility = _extract_section(raw_output, "SOURCE_CREDIBILITY")
        evidence_sufficiency = _extract_section(raw_output, "EVIDENCE_SUFFICIENCY")
        uncertainty = _extract_section(raw_output, "MISSING_EVIDENCE")
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

    # Gate enforcement: the second opinion may never promote a gate-restricted
    # UNVERIFIED into REAL or FAKE.
    gate_enforced = False
    if (
        existing_verdict == "UNVERIFIED"
        and verdict in ("REAL", "FAKE")
        and permitted_verdicts
        and "REAL" not in permitted_verdicts
        and "FAKE" not in permitted_verdicts
    ):
        gate_enforced = True
        verdict = "UNVERIFIED"

    # Agreement / disagreement with the existing system verdict.
    agreement: Optional[bool] = None
    disagreement: Optional[bool] = None
    comparison_summary = ""
    if verdict is None:
        agreement = False
        disagreement = False
        comparison_summary = (
            "No valid LLM verdict was returned, so no agreement comparison "
            "could be made."
        )
    elif verdict == existing_verdict:
        agreement = True
        disagreement = False
        comparison_summary = (
            f"The LLM second opinion agrees with the system verdict "
            f"({existing_verdict})."
        )
    else:
        agreement = False
        disagreement = True
        comparison_summary = (
            f"The LLM second opinion ({verdict}) disagrees with the system "
            f"verdict ({existing_verdict}). The deterministic evidence gate "
            f"remains authoritative and the reported verdict is "
            f"{existing_verdict}."
        )

    comparison: Dict[str, Any] = {
        "existing_verdict": existing_verdict,
        "llm_verdict": verdict,
        "agreement": agreement,
        "disagreement": disagreement,
        "gate_enforced": gate_enforced,
        "comparison_summary": comparison_summary,
        "agreement_with_system": agreement_with_system,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "claim_summary": claim_summary,
        "explanation": explanation,
        "source_credibility": source_credibility,
        "evidence_sufficiency": evidence_sufficiency,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "uncertainty": uncertainty,
        "limitations": limitations,
        "raw_output": raw_output,
        "error": error,
        "timeout": timeout,
        "api_failure": api_failure,
        "no_evidence": no_evidence,
        "comparison": comparison,
    }


def _call_llm_with_retry(
    claim: str,
    existing_verdict: str,
    evidence_package: str,
) -> str:
    """
    Invoke the second-opinion chain with bounded retries, exponential
    backoff, jitter, and retry-after respect.

    Never retries indefinitely and never blocks the pipeline beyond a fixed
    bound.
    """
    delay = 2.0
    last_error: Optional[Exception] = None
    start = time.monotonic()
    budget = SECOND_OPINION_TIMEOUT_SECONDS

    for attempt in range(1, SECOND_OPINION_MAX_ATTEMPTS + 1):
        try:
            return invoke_agent(
                _second_opinion_chain,
                {
                    "claim": claim,
                    "existing_verdict": existing_verdict,
                    "evidence_package": evidence_package,
                },
                stage="second_opinion",
            )
        except Exception as exc:
            last_error = exc
            message = str(exc)
            low = message.lower()
            elapsed = time.monotonic() - start

            # Non-retryable or budget exhausted: stop.
            if attempt >= SECOND_OPINION_MAX_ATTEMPTS or not _is_retryable(exc, low):
                break

            # Respect the provider's retry-after hint when available.
            wait = _retry_after_seconds(exc)
            if wait is None:
                wait = delay
                delay = min(delay * 2, 20.0)
            # Add jitter so concurrent clients do not retry in lockstep.
            wait = wait + (attempt * 0.25) + (0.5 * (attempt % 3))
            wait = min(max(wait, 0.5), budget - elapsed)
            if wait <= 0:
                break
            time.sleep(wait)

    raise last_error if last_error else RuntimeError("second opinion call failed")


def _is_retryable(exc: Exception, low: str) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    retryable_status = {408, 409, 425, 429, 500, 502, 503, 504}
    if status in retryable_status:
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in ("ratelimit", "timeout", "internalserver", "overloaded")
    )


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """The provider's own retry hint, when it supplies one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            raw = headers.get(key)
            if raw:
                try:
                    return max(0.0, float(str(raw).strip()))
                except (TypeError, ValueError):
                    continue
    return None


def build_second_opinion(
    claim: str,
    evidence_analysis: List[Dict[str, Any]],
    source_agreement: Dict[str, Any],
    existing_verdict: Optional[str],
    permitted_verdicts: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Produce an independent LLM second opinion on the claim.

    Parameters are all already-computed values from the existing pipeline:
    the claim, the classified evidence passages, the source-agreement
    counters, the existing system verdict, and the gate's permitted verdicts.

    No search, scraping, or NLI re-classification is performed here. The LLM
    reviews what is already collected. When there is no usable evidence, the
    result reports inconclusive UNVERIFIED without invoking the LLM.
    """
    evidence_package = _condense_evidence_package(
        evidence_analysis, source_agreement
    )
    no_evidence = not evidence_analysis or "no usable evidence passages" in evidence_package

    error: Optional[str] = None
    timeout = False
    api_failure = False
    raw_output = ""

    if no_evidence:
        # Avoid an unnecessary LLM call; report inconclusive directly.
        return _build_second_opinion_result(
            raw_output="",
            error=None,
            timeout=False,
            api_failure=False,
            existing_verdict=existing_verdict,
            permitted_verdicts=permitted_verdicts or [],
            no_evidence=True,
        )

    try:
        raw_output = _call_llm_with_retry(
            claim=claim,
            existing_verdict=existing_verdict or "(no system verdict)",
            evidence_package=evidence_package,
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

    return _build_second_opinion_result(
        raw_output=raw_output,
        error=error,
        timeout=timeout,
        api_failure=api_failure,
        existing_verdict=existing_verdict,
        permitted_verdicts=permitted_verdicts or [],
        no_evidence=False,
    )
