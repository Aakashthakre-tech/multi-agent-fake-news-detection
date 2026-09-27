"""
final_verdict.py
================
The final decision, with the evidence layer holding the veto.

Previous behaviour: if the Verification Agent and the Critic agreed on a label,
that label was returned regardless of what the evidence layer had actually
found. Two language models agreeing on weak evidence could therefore produce a
confident REAL.

New behaviour: the deterministic layer decides what is *permissible*, and the
agents may only narrow that range.

    no independent support + no independent contradiction -> UNVERIFIED
    one independent supporting source only                -> UNVERIFIED
    >= 2 independent supporting sources, no conflict     -> REAL permitted
    >= 2 independent contradicting sources, no conflict  -> FAKE permitted
    independent sources disagree                         -> UNVERIFIED
    MISLEADING                                          -> requires explicit
                                                           in-evidence correction

Thresholds are never lowered to produce a verdict. UNVERIFIED is a valid and
expected result.
"""

import re


VALID_VERDICTS = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}

# A strong REAL/FAKE requires this many INDEPENDENT publishers.
MIN_INDEPENDENT_SOURCES = 2

# At least this many of the supporting/contradicting publishers must be
# retrieved articles, not search snippets.
MIN_PRIMARY_GROUPS = 1


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _normalize(value):
    return (value or "").strip().upper()


def extract_verdict(text):
    """Pull the verdict label out of an agent response."""
    if not text:
        return None

    match = re.search(
        r"(?:recommended\s+)?verdict\s*[:\-]\s*\**\s*(REAL|FAKE|MISLEADING|UNVERIFIED)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\b(REAL|FAKE|MISLEADING|UNVERIFIED)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def extract_confidence(text):
    """Pull the confidence percentage out of an agent response."""
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

    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if match:
        try:
            value = float(match.group(1))
            if 0 <= value <= 100:
                return value
        except ValueError:
            return None

    return None


def _count_class(results, evidence_class):
    return sum(
        1 for r in results
        if r.get("evidence_class") == evidence_class
    )


def _signals(results):
    """The structural signals the old layer used directly."""
    if not results:
        return {
            "strong_contradictions": 0,
            "strong_entailments": 0,
            "context": 0,
            "primary_source_ratio": 0.0,
        }

    strong_contradictions = sum(
        1 for r in results
        if r.get("evidence_class") == "CONTRADICTION"
        and r.get("evidence_score", 0) >= 0.6
    )

    strong_entailments = sum(
        1 for r in results
        if r.get("evidence_class") == "SUPPORT"
        and r.get("evidence_score", 0) >= 0.6
    )

    context = sum(
        1 for r in results
        if r.get("evidence_class") == "RELATED_CONTEXT"
    )

    article_chunks = [
        r for r in results
        if r.get("evidence_type") == "scraped_article"
    ]

    return {
        "strong_contradictions": strong_contradictions,
        "strong_entailments": strong_entailments,
        "context": context,
        "primary_source_ratio": (
            len(article_chunks) / len(results) if results else 0.0
        ),
    }


def deterministic_gate(source_agreement, results):
    """
    Decide which verdicts the evidence permits, before any agent is consulted.

    Returns a dict describing the permitted set and the reason for the limit.
    """
    agreement = source_agreement or {}

    independent_supporting = agreement.get("independent_supporting", 0)
    independent_contradicting = agreement.get("independent_contradicting", 0)
    primary_supporting = agreement.get("primary_supporting_groups", 0)

    signals = _signals(results)

    gate = {
        "independent_supporting": independent_supporting,
        "independent_contradicting": independent_contradicting,
        "primary_supporting_groups": primary_supporting,
        "permitted_verdicts": {"UNVERIFIED"},
        "permitted_reason": "",
        **signals,
    }

    if independent_supporting and independent_contradicting:
        gate["permitted_reason"] = (
            f"{independent_supporting} independent publisher(s) support the "
            f"claim while {independent_contradicting} contradict it. The claim "
            f"is contested, so no settled verdict is permitted."
        )
        return gate

    if independent_supporting >= MIN_INDEPENDENT_SOURCES and (
        primary_supporting >= MIN_PRIMARY_GROUPS
    ):
        gate["permitted_verdicts"] = {"REAL", "MISLEADING", "UNVERIFIED"}
        gate["permitted_reason"] = (
            f"{independent_supporting} independent publishers, "
            f"{primary_supporting} of them primary articles, directly support "
            f"the claim."
        )
        return gate

    if independent_contradicting >= MIN_INDEPENDENT_SOURCES:
        gate["permitted_verdicts"] = {"FAKE", "UNVERIFIED"}
        gate["permitted_reason"] = (
            f"{independent_contradicting} independent publishers directly "
            f"contradict the claim."
        )
        return gate

    if independent_supporting == 1:
        gate["permitted_reason"] = (
            "Only one independent publisher directly supports the claim. "
            "Independent corroboration is missing, so REAL is not permitted."
        )
        return gate

    if independent_contradicting == 1:
        gate["permitted_reason"] = (
            "Only one independent publisher contradicts the claim. "
            "Independent corroboration is missing, so FAKE is not permitted."
        )
        return gate

    if not results:
        gate["permitted_reason"] = (
            "No passage passed the evidence layer's filters. There is no "
            "evidence to support or refute the claim."
        )
        return gate

    gate["permitted_reason"] = (
        f"Retrieved sources provide related context only "
        f"({signals['context']} context passage(s)); none establishes the "
        f"claimed event with the claimed date."
    )

    return gate


def calculate_final_confidence(verdict, source_agreement, fusion_score, signals):
    """
    Confidence is derived from the evidence, then bounded by what the
    deterministic gate actually allows.
    """
    agreement_score = _clamp(source_agreement.get("agreement_score", 0))
    fusion = _clamp(fusion_score)

    if verdict == "UNVERIFIED":
        # Confidence in "we could not verify", which is high when the evidence
        # is genuinely insufficient and low when the evidence is strong but
        # contested.
        if signals.get("context", 0) > 0 and fusion < 0.62:
            return 70
        if signals.get("independent_contradicting", 0) > 0 and \
                signals.get("independent_supporting", 0) > 0:
            return 55
        return 60

    if verdict == "REAL":
        base = (
            0.55 * agreement_score
            + 0.45 * fusion
        )
        supporting = signals.get("independent_supporting", 0)
        primary = signals.get("primary_supporting_groups", 0)
        bonus = min(0.10, 0.03 * supporting)
        if primary == 0:
            base = min(base, 0.55)
        return round(_clamp(base + bonus) * 100, 1)

    if verdict == "FAKE":
        base = (
            0.55 * agreement_score
            + 0.45 * fusion
        )
        bonus = min(0.08, 0.03 * signals.get("independent_contradicting", 0))
        return round(_clamp(base + bonus) * 100, 1)

    if verdict == "MISLEADING":
        # Deliberately capped: a misleading verdict needs a correction that is
        # explicit in the evidence, and it is the easiest verdict to over-call.
        return round(min(_clamp(0.45 * agreement_score + 0.35 * fusion), 0.70) * 100, 1)

    return 60


def _primary_evidence(verdict, results):
    wanted = "SUPPORT" if verdict in ("REAL", "MISLEADING") else "CONTRADICTION"

    matches = [
        r for r in results
        if r.get("evidence_class") == wanted
    ]
    matches.sort(key=lambda r: r.get("evidence_score", 0), reverse=True)

    return [
        {
            "url": r.get("url"),
            "domain": r.get("domain"),
            "evidence_type": r.get("evidence_type"),
            "source_tier": r.get("source_tier"),
            "date_status": r.get("date_status"),
            "event_match": r.get("event_match"),
            "evidence_score": r.get("evidence_score"),
            "chunk": r.get("chunk", "")[:400],
        }
        for r in matches[:3]
    ]


def generate_final_verdict(
    evidence_results,
    source_agreement,
    fusion_score,
    verification_output="",
    critic_output="",
):
    """
    Deterministic final verdict.

    Agent verdicts are inputs, never the decision: a label is accepted only when
    the evidence gate permits it. Everything else resolves to UNVERIFIED.
    """
    results = evidence_results or []
    agreement = source_agreement or {}

    gate = deterministic_gate(agreement, results)
    permitted = gate["permitted_verdicts"]

    verification_verdict = extract_verdict(verification_output)
    critic_verdict = extract_verdict(critic_output)
    verification_confidence = extract_confidence(verification_output)
    critic_confidence = extract_confidence(critic_output)

    candidates = [v for v in (verification_verdict, critic_verdict) if v]

    agents_agree = (
        verification_verdict is not None
        and verification_verdict == critic_verdict
    )

    requested = (
        candidates[0]
        if agents_agree and verification_verdict
        else (candidates[0] if len(candidates) == 1 else None)
    )

    if requested in permitted:
        verdict = requested
    elif requested == "MISLEADING" and "MISLEADING" in permitted:
        verdict = "MISLEADING"
    else:
        verdict = "UNVERIFIED"

    consistency = {
        "verification_verdict": verification_verdict,
        "critic_verdict": critic_verdict,
        "agents_agree": agents_agree,
        "verdict_overridden_by_evidence": bool(
            requested and requested != verdict
        ),
        "agent_verdict": requested,
    }

    signals = {
        "independent_supporting": gate["independent_supporting"],
        "independent_contradicting": gate["independent_contradicting"],
        "primary_supporting_groups": gate["primary_supporting_groups"],
        "context": gate["context"],
    }

    confidence = calculate_final_confidence(
        verdict, agreement, fusion_score, signals
    )

    if requested and requested != verdict and requested in VALID_VERDICTS:
        # The agents asked for something the evidence cannot justify: report
        # their label, not the replacement, so the override is visible.
        reason = (
            f"Agents proposed {requested}, but the evidence layer permits only "
            f"{sorted(permitted)}. {gate['permitted_reason']}"
        )
    elif verdict == "UNVERIFIED":
        reason = gate["permitted_reason"]
    else:
        reason = gate["permitted_reason"]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_sufficiency": agreement.get("evidence_sufficiency", "none"),
        "source_agreement": agreement.get("agreement_score", 0),
        "fusion_score": _clamp(fusion_score),
        "independent_supporting": gate["independent_supporting"],
        "independent_contradicting": gate["independent_contradicting"],
        "primary_supporting_groups": gate["primary_supporting_groups"],
        "total_sources": agreement.get("total_sources", 0),
        "independence_groups": agreement.get("independence_groups", 0),
        "supporting_sources": agreement.get("supporting_sources", 0),
        "contradicting_sources": agreement.get("contradicting_sources", 0),
        "entailment_score": gate["strong_entailments"],
        "contradiction_score": gate["strong_contradictions"],
        "context_chunks": gate["context"],
        "permitted_verdicts": sorted(permitted),
        "permitted_reason": reason,
        "deterministic_gate_reason": gate["permitted_reason"],
        "verification_verdict": verification_verdict,
        "critic_verdict": critic_verdict,
        "agents_agree": agents_agree,
        "verdict_overridden_by_evidence": consistency["verdict_overridden_by_evidence"],
        "agent_confidence": verification_confidence,
        "critic_confidence": critic_confidence,
        "supporting_evidence": _primary_evidence(verdict, results),
        "contradicting_evidence": _primary_evidence(
            "FAKE" if verdict != "FAKE" else "REAL", results
        ),
        "reasoning": reason,
        "verification_result": verification_output,
        "critic_result": critic_output,
    }
