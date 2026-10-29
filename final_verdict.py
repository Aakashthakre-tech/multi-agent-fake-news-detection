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
# retrieved articles from primary publishers, not search snippets.
MIN_PRIMARY_GROUPS = 1

# A MISLEADING verdict additionally needs an explicit, evidence-backed
# correction of a material claim detail coming from a primary publisher.
MIN_CORRECTION_GROUPS = 1


# The policy counters the deterministic gate needs in order to reason about
# the evidence at all. If any of them is missing or malformed, the gate cannot
# apply the policy safely and must abstain.
REQUIRED_POLICY_FIELDS = (
    "independent_supporting",
    "independent_contradicting",
    "primary_supporting_groups",
    "primary_contradicting_groups",
    "article_supporting_groups",
    "article_contradicting_groups",
    "correction_groups",
    "primary_correction_groups",
)


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _normalize(value):
    return (value or "").strip().upper()


def _count(agreement, key):
    """
    Read a DESCRIPTIVE counter.

    These only feed the explanation text; they never decide the verdict. A
    missing or malformed value is reported as zero so the explanation stays
    truthful about what it does know.
    """
    value = agreement.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if value < 0:
        return 0
    return int(value)


def _policy_counts(agreement):
    """
    Validate and read the policy counters.

    There is deliberately NO fallback here. Substituting one counter for
    another - for example treating every contradicting publisher as a primary
    one, or every correction group as a primary correction group - would let
    an incomplete agreement dict produce a stronger verdict than the evidence
    supports. A missing or invalid policy field is reported instead, and the
    caller abstains.

    Returns (values, errors).
    """
    values = {}
    errors = []

    for key in REQUIRED_POLICY_FIELDS:
        if key not in agreement:
            errors.append(f"missing '{key}'")
            continue

        raw = agreement[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(f"'{key}' is not a number ({raw!r})")
            continue
        if int(raw) != raw:
            errors.append(f"'{key}' is not a whole number ({raw!r})")
            continue
        if raw < 0:
            errors.append(f"'{key}' is negative ({raw!r})")
            continue

        values[key] = int(raw)

    return values, errors


def extract_verdict(text):
    """
    Pull the verdict label out of an agent response.

    The agents are instructed to answer with a ``Verdict:`` /
    ``Recommended Verdict:`` heading, so only a labelled match is trusted. A
    bare label inside an explanation ("it is not REAL to say ...") is NOT a
    verdict, and picking it would let a sentence change the outcome.
    """
    if not text:
        return None

    # A labelled verdict, e.g. "Verdict:\nREAL" or "Recommended Verdict: REAL".
    # The last labelled verdict wins, because a later section is the agent's
    # final position.
    labelled = list(
        re.finditer(
            r"(?:recommended\s+|final\s+)?verdict\s*[:\-]\s*\**\s*"
            r"(REAL|FAKE|MISLEADING|UNVERIFIED)\b",
            text,
            re.IGNORECASE,
        )
    )
    if labelled:
        return labelled[-1].group(1).upper()

    # A verdict on a line of its own, which is the format the prompt asks for.
    standalone = list(
        re.finditer(
            r"^\s*\**\s*(REAL|FAKE|MISLEADING|UNVERIFIED)\s*\**\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    if standalone:
        return standalone[-1].group(1).upper()

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

    The policy is symmetric in both directions:

      * a verdict needs INDEPENDENT publishers, and at least one of them must
        be a primary publisher whose article text was actually retrieved.
        Search snippets alone can never establish or refute a claim.
      * support and contradiction are treated the same way: two independent
        contradicting publishers are not enough unless one of them is primary
        reporting we actually read.
      * MISLEADING additionally requires an explicit, evidence-backed
        correction of a material claim detail from a primary publisher.
        Generic disagreement is not enough.

    Returns a dict describing the permitted set and the reason for the limit.
    """
    agreement = source_agreement if isinstance(source_agreement, dict) else {}

    counts, policy_errors = _policy_counts(agreement)
    signals = _signals(results)

    independent_supporting = counts.get("independent_supporting", 0)
    independent_contradicting = counts.get("independent_contradicting", 0)
    primary_supporting = counts.get("primary_supporting_groups", 0)
    primary_contradicting = counts.get("primary_contradicting_groups", 0)
    article_supporting = counts.get("article_supporting_groups", 0)
    article_contradicting = counts.get("article_contradicting_groups", 0)
    correction_groups = counts.get("correction_groups", 0)
    primary_correction_groups = counts.get("primary_correction_groups", 0)

    gate = {
        "independent_supporting": independent_supporting,
        "independent_contradicting": independent_contradicting,
        "primary_supporting_groups": primary_supporting,
        "primary_contradicting_groups": primary_contradicting,
        "article_supporting_groups": article_supporting,
        "article_contradicting_groups": article_contradicting,
        "correction_groups": correction_groups,
        "primary_correction_groups": primary_correction_groups,
        "permitted_verdicts": {"UNVERIFIED"},
        "permitted_reason": "",
        "policy_data_valid": not policy_errors,
        "policy_data_errors": policy_errors,
        **signals,
    }

    # --------------------------------------------------------
    # 0. Incomplete or invalid policy data
    #
    # The gate cannot apply its policy to counters it does not have, and it
    # must not guess by borrowing a different counter. Abstain, and say why.
    # --------------------------------------------------------
    if policy_errors:
        gate["permitted_reason"] = (
            "The evidence policy data is incomplete or invalid, so no settled "
            "verdict can be permitted. "
            + "; ".join(policy_errors)
            + ". The result is UNVERIFIED by safe default; this is an internal "
            "evidence-accounting problem and is NOT a finding that the claim "
            "is false."
        )
        return gate

    # --------------------------------------------------------
    # 1. Contested evidence
    #
    # Independent disagreement is not settled. The single exception is a
    # disagreement that is explicitly a correction: then the event may be
    # real while a material detail is wrong, which is what MISLEADING is for.
    # --------------------------------------------------------
    every_contradiction_is_a_correction = (
        independent_contradicting > 0
        and correction_groups >= independent_contradicting
    )

    if (
        independent_supporting
        and independent_contradicting
        and not every_contradiction_is_a_correction
    ):
        gate["permitted_reason"] = (
            f"{independent_supporting} independent publisher(s) support the "
            f"claim while {independent_contradicting} contradict it. The claim "
            f"is contested, so no settled verdict is permitted."
        )
        return gate

    # --------------------------------------------------------
    # 2. Support
    # --------------------------------------------------------
    if independent_supporting >= MIN_INDEPENDENT_SOURCES:
        if primary_supporting < MIN_PRIMARY_GROUPS or article_supporting < MIN_PRIMARY_GROUPS:
            gate["permitted_reason"] = (
                f"{independent_supporting} independent publisher(s) appear to support "
                f"the claim, but only {primary_supporting} of them are primary "
                f"publishers whose article text was actually retrieved "
                f"({article_supporting} with retrievable article text). REAL requires "
                f"at least {MIN_INDEPENDENT_SOURCES} independent publishers including "
                f"at least {MIN_PRIMARY_GROUPS} primary article, so REAL is not "
                f"permitted."
            )
            return gate

        gate["permitted_verdicts"] = {"REAL", "UNVERIFIED"}

        if primary_correction_groups >= MIN_CORRECTION_GROUPS:
            gate["permitted_verdicts"].add("MISLEADING")
            gate["permitted_reason"] = (
                f"{independent_supporting} independent publishers, "
                f"{primary_supporting} of them primary publishers with retrieved "
                f"article text, directly support the claim. "
                f"{primary_correction_groups} primary publisher(s) contain an explicit "
                f"correction of a material claim detail, so MISLEADING may also be "
                f"considered."
            )
        else:
            gate["permitted_reason"] = (
                f"{independent_supporting} independent publishers, "
                f"{primary_supporting} of them primary publishers with retrieved "
                f"article text, directly support the claim. No retrieved primary "
                f"evidence contains an explicit correction of a material claim "
                f"detail, so MISLEADING is not permitted."
            )
        return gate

    # --------------------------------------------------------
    # 3. Contradiction
    # --------------------------------------------------------
    if independent_contradicting >= MIN_INDEPENDENT_SOURCES:
        if primary_contradicting < MIN_PRIMARY_GROUPS or article_contradicting < MIN_PRIMARY_GROUPS:
            gate["permitted_reason"] = (
                f"{independent_contradicting} independent publisher(s) contradict the "
                f"claim, but only {primary_contradicting} of them are primary "
                f"publishers whose article text was actually retrieved "
                f"({article_contradicting} with retrievable article text). FAKE "
                f"requires at least {MIN_INDEPENDENT_SOURCES} independent publishers "
                f"including at least {MIN_PRIMARY_GROUPS} primary article, so FAKE is "
                f"not permitted. Search snippets alone can never refute a claim."
            )
            return gate

        gate["permitted_verdicts"] = {"FAKE", "UNVERIFIED"}
        gate["permitted_reason"] = (
            f"{independent_contradicting} independent publishers, "
            f"{primary_contradicting} of them primary publishers with retrieved "
            f"article text, directly contradict the claim."
        )
        return gate

    # --------------------------------------------------------
    # 4. Abstentions
    # --------------------------------------------------------
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

    # A specific, accurate explanation of what the retrieved material was and
    # what a settled verdict would have required.
    supporting_sources = _count(agreement, "supporting_sources")
    contradicting_sources = _count(agreement, "contradicting_sources")
    context_sources = _count(agreement, "related_context_sources")
    non_evidence_sources = _count(agreement, "non_evidence_sources")
    snippet_only_sources = _count(agreement, "snippet_only_sources")
    article_sources = _count(agreement, "article_sources")
    total_sources = _count(agreement, "total_sources")

    requirement = (
        f"A settled verdict requires at least {MIN_INDEPENDENT_SOURCES} "
        f"independent publishers, including at least {MIN_PRIMARY_GROUPS} "
        f"primary publisher whose article text was actually retrieved."
    )

    if snippet_only_sources and not article_sources:
        gate["permitted_reason"] = (
            f"All {snippet_only_sources} retrieved source(s) were search snippets, "
            f"not retrieved articles. A search snippet can never establish or "
            f"refute a claim. {requirement}"
        )
        return gate

    if snippet_only_sources:
        gate["permitted_reason"] = (
            f"{snippet_only_sources} of {total_sources} retrieved source(s) were "
            f"search snippets rather than retrieved articles, and a snippet can "
            f"never establish or refute a claim. {supporting_sources} source(s) "
            f"support the claim and {contradicting_sources} contradict it, which "
            f"does not reach the required number of independent publishers. "
            f"{requirement}"
        )
        return gate

    gate["permitted_reason"] = (
        f"Retrieved sources provide related context only "
        f"({context_sources} context source(s) of {total_sources} retrieved, "
        f"{non_evidence_sources} non-evidence); none establishes the claimed "
        f"event with the claimed date. {requirement}"
    )

    return gate


def calculate_final_confidence(verdict, source_agreement, fusion_score, signals):
    """
    Confidence is derived from the evidence, then bounded by what the
    deterministic gate actually allows.

    For a settled verdict (REAL / FAKE / MISLEADING) the number is an
    evidence-strength score. It is not a calibrated probability that the claim
    is true or false.

    For UNVERIFIED there is no verdict to be confident about, so the number
    reports *verification coverage*: how much of the retrieved material could
    actually be read and classified. A high coverage with an UNVERIFIED
    verdict means "the evidence was inspected and still did not settle the
    claim" - it does NOT mean the claim is probably true or probably false.
    """
    agreement_score = _clamp(source_agreement.get("agreement_score", 0))
    fusion = _clamp(fusion_score)

    if verdict == "UNVERIFIED":
        coverage = _clamp(source_agreement.get("verification_coverage", 0))
        sufficiency = source_agreement.get("evidence_sufficiency", "none")
        supporting = source_agreement.get("independent_supporting", 0) or 0
        contradicting = source_agreement.get("independent_contradicting", 0) or 0

        # No usable evidence at all: confidence must not be presented as a
        # truth probability. Zero coverage means nothing was readable; any
        # coverage with sufficiency "none" means the material was inspected
        # but nothing settled the claim. Neither case justifies a high
        # percentage next to an UNVERIFIED verdict.
        if supporting == 0 and contradicting == 0:
            if sufficiency == "none":
                return 0.0
            # Some material was readable but nothing decisive: cap well below
            # a settled verdict so the number is not mistaken for truth
            # confidence.
            return round(min(coverage * 100, 60.0), 1)

        # Sources disagree (contested): the claim is not settled, so
        # confidence must be capped below a settled verdict to avoid being
        # mistaken for truth confidence.
        if supporting > 0 and contradicting > 0:
            return round(min(coverage * 100, 55.0), 1)

        # Some evidence exists but it is insufficient (single source, or
        # weak coverage): cap well below a settled verdict.
        if supporting == 1 and contradicting == 0:
            return round(min(coverage * 100, 60.0), 1)

        # Some evidence exists but it is insufficient (single source, or
        # conflicting). Report coverage honestly.
        return round(_clamp(coverage) * 100, 1)

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

    return 0.0


def _primary_evidence(evidence_class, results):
    """
    The strongest passages of one evidence class.

    The class is passed explicitly, never derived from the verdict: a
    contradicting passage must never be reported as supporting evidence just
    because the verdict happened to be UNVERIFIED.
    """
    matches = [
        r for r in results
        if r.get("evidence_class") == evidence_class
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
            "is_primary_source": r.get("is_primary_source"),
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

    # What the confidence number actually means. UNVERIFIED reports how much
    # of the retrieved material could be read, not a probability.
    confidence_basis = (
        "verification_coverage"
        if verdict == "UNVERIFIED"
        else "evidence_strength"
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
        "confidence_basis": confidence_basis,
        "evidence_sufficiency": agreement.get("evidence_sufficiency", "none"),
        "source_agreement": agreement.get("agreement_score", 0),
        "fusion_score": _clamp(fusion_score),
        "independent_supporting": gate["independent_supporting"],
        "independent_contradicting": gate["independent_contradicting"],
        "primary_supporting_groups": gate["primary_supporting_groups"],
        "primary_contradicting_groups": gate["primary_contradicting_groups"],
        "article_supporting_groups": gate["article_supporting_groups"],
        "article_contradicting_groups": gate["article_contradicting_groups"],
        "correction_groups": gate["correction_groups"],
        "primary_correction_groups": gate.get("primary_correction_groups", 0),
        "policy_data_valid": gate.get("policy_data_valid", True),
        "policy_data_errors": gate.get("policy_data_errors", []),
        "verification_coverage": agreement.get("verification_coverage", 0.0),
        "unresolved_identity_sources": agreement.get(
            "unresolved_identity_sources", 0
        ),
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
        "supporting_evidence": _primary_evidence("SUPPORT", results),
        "contradicting_evidence": _primary_evidence("CONTRADICTION", results),
        "reasoning": reason,
        "verification_result": verification_output,
        "critic_result": critic_output,
    }
