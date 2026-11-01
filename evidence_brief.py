"""
evidence_brief.py
=================
The single structured evidence document handed to the Verification Agent and
the Critic.

Both agents used to read different excerpts of the raw text (3500 chars for the
verifier, the first 2000 for the critic), which is a direct cause of agents
disagreeing for no reason. They now read the same document, which always
contains:

  * the claim and its parsed attributes (date, location, keywords)
  * the per-source evidence-layer decision and the reason behind it
  * independent corroboration counts
  * the evidence passages themselves, strongest first

The document also states the deterministic floor, so an agent cannot talk itself
into a confident verdict that the evidence layer has already ruled out.
"""

from collections import defaultdict


MAX_PASSAGES = 4
MAX_PASSAGE_CHARS = 600


def _attributes(claim):
    from claim_profile import build_claim_profile

    return build_claim_profile(claim)


def _source_decisions(evidence_results):
    """
    Re-derive per-source decisions for the brief (chunk -> source).

    The grouping must be identical to the one the fusion layer used, otherwise
    the brief and the verdict layer would disagree about how many sources
    there were. The shared helper is used rather than a second implementation.
    """
    from evidence_fusion import _summarise_source, _source_key

    by_source = defaultdict(list)
    resolved = {}
    for index, result in enumerate(evidence_results):
        key, is_resolved = _source_key(result, index)
        by_source[key].append(result)
        resolved[key] = is_resolved

    decisions = []
    for key, results in sorted(by_source.items()):
        decision = _summarise_source(
            results[0].get("source_id"), results,
            identity_resolved=resolved.get(key, True),
        )
        decisions.append((decision, results))

    return decisions


def build_evidence_brief(
    claim,
    sources,
    evidence_results,
    source_agreement,
    fusion_score,
    evidence_diagnostics=None,
):
    """
    Build the evidence document shared by the verification and review agents.
    """
    profile = _attributes(claim)
    agreement = source_agreement or {}

    independent_supporting = agreement.get("independent_supporting", 0)
    independent_contradicting = agreement.get("independent_contradicting", 0)
    sufficiency = agreement.get("evidence_sufficiency", "none")
    total_sources = agreement.get("total_sources", 0)
    total_groups = agreement.get("independence_groups", 0)
    primary_supporting = agreement.get("primary_supporting_groups", 0)
    primary_contradicting = agreement.get("primary_contradicting_groups", 0)
    article_supporting = agreement.get("article_supporting_groups", 0)
    article_contradicting = agreement.get("article_contradicting_groups", 0)
    correction_groups = agreement.get("correction_groups", 0)
    primary_correction_groups = agreement.get("primary_correction_groups", 0)
    snippet_only_sources = agreement.get("snippet_only_sources", 0)
    article_sources = agreement.get("article_sources", 0)
    unresolved_identities = agreement.get("unresolved_identity_sources", 0)
    supporting_sources = agreement.get("supporting_sources", 0)
    contradicting_sources = agreement.get("contradicting_sources", 0)
    neutral_sources = agreement.get("neutral_sources", 0)
    context_sources = agreement.get("related_context_sources", 0)
    non_evidence_sources = agreement.get("non_evidence_sources", 0)

    lines = []
    lines.append("=== CLAIM ===")
    lines.append(claim)

    lines.append("")
    lines.append("=== CLAIM ATTRIBUTES (parsed) ===")
    dates = ", ".join(
        f"{d['day']} {d['month_name']} {d['year']}" for d in profile["dates"]
    ) or "none"
    lines.append(f"Claimed date(s)   : {dates}")
    lines.append(f"Claimed location(s): {', '.join(profile['locations']) or 'none stated'}")
    lines.append(f"Claimed keywords  : {', '.join(profile['keywords']) or 'none'}")
    lines.append(f"Event phrases     : {', '.join(profile['phrases']) or 'none'}")

    lines.append("")
    lines.append("=== EVIDENCE LAYER RESULTS (deterministic) ===")
    lines.append(f"Sources retrieved           : {total_sources}")
    lines.append(f"Sources with article text   : {article_sources}")
    lines.append(f"Sources snippet-only        : {snippet_only_sources}")
    lines.append(f"Independent publishers      : {total_groups}")
    lines.append(f"  supporting               : {supporting_sources}")
    lines.append(f"  contradicting            : {contradicting_sources}")
    lines.append(f"  neutral (conflicting)    : {neutral_sources}")
    lines.append(f"  related context          : {context_sources}")
    lines.append(f"  non-evidence             : {non_evidence_sources}")
    lines.append(f"Unresolved source identities : {unresolved_identities}")
    lines.append(f"Independent SUPPORTING       : {independent_supporting}")
    lines.append(f"Independent CONTRADICTING    : {independent_contradicting}")
    lines.append(f"Primary supporting groups   : {primary_supporting}")
    lines.append(f"Primary contradicting groups: {primary_contradicting}")
    lines.append(f"Groups with a correction    : {correction_groups} "
                 f"({primary_correction_groups} from primary publishers)")
    lines.append(f"Evidence sufficiency        : {sufficiency}")
    lines.append(f"Source agreement score      : {agreement.get('agreement_score', 0)}")
    lines.append(f"Evidence fusion score       : {fusion_score}")

    if not evidence_results:
        lines.append("")
        lines.append(
            "EVIDENCE STATE: ABSENT. No retrieved passage passed the evidence "
            "layer's filters. There is nothing to support or refute the claim, "
            "so the only honest outcome is UNVERIFIED. This is not evidence "
            "that the claim is false."
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("=== PER-SOURCE VERDICTS ===")

    decisions = _source_decisions(evidence_results)

    for decision, results in decisions:
        best = max(results, key=lambda r: r.get("evidence_score", 0))
        lines.append(
            f"- [{decision['decision'].upper()}] {decision['url']}"
        )
        lines.append(
            f"    publisher={decision['independence_group'] or 'UNRESOLVED'} | "
            f"tier={decision['source_tier']} | reliability={decision['source_reliability']} | "
            f"retrieval={decision.get('retrieval_mode', decision['evidence_type'])}"
        )
        if decision.get("is_snippet_only"):
            lines.append(
                "    NOTE: only a search snippet was available for this source; "
                "it was not read in full and cannot establish or refute a claim."
            )
        elif not decision.get("has_article_text", True):
            lines.append(
                "    NOTE: no article text was retrieved for this source; it "
                "cannot establish or refute a claim."
            )
        if not decision.get("is_primary", False):
            lines.append(
                "    NOTE: this publisher is not primary reporting, so it "
                "cannot establish or refute a claim on its own."
            )
        if not decision.get("identity_resolved", True):
            lines.append(
                "    NOTE: the publisher identity could not be resolved for "
                "this source; it is excluded from the independent publisher counts."
            )
        lines.append(
            f"    date_status={best.get('date_status')} ({best.get('date_reason')})"
        )
        lines.append(
            f"    event_match={best.get('event_match')} | "
            f"location_match={best.get('location_match')} | "
            f"claim_detail_coverage={best.get('detail_coverage')}"
        )
        lines.append(
            f"    nli={best.get('nli_label')} {best.get('nli_score')} "
            f"(status={best.get('nli_status')}) | "
            f"evidence_score={best.get('evidence_score')} | "
            f"chunks={decision['chunk_count']}"
        )

    context_groups = [
        g for g in agreement.get("group_decisions", [])
        if g["decision"] == "context"
    ]
    if context_groups:
        lines.append("")
        lines.append(
            "Sources that are only related context (same topic, event/date not "
            "established): "
            + ", ".join(sorted(g["independence_group"] for g in context_groups))
        )

    # ------------------------------------------------------------
    # Evidence passages, strongest first
    # ------------------------------------------------------------
    ranked = sorted(
        evidence_results,
        key=lambda r: r.get("evidence_score", 0),
        reverse=True,
    )

    lines.append("")
    lines.append("=== EVIDENCE PASSAGES (strongest first) ===")

    for index, result in enumerate(ranked[:MAX_PASSAGES], 1):
        lines.append("")
        lines.append(
            f"[{index}] {result.get('evidence_class')} "
            f"score={result.get('evidence_score')} "
            f"({result.get('evidence_type')}, {result.get('source_tier')}, "
            f"date={result.get('date_status')}, event={result.get('event_match')})"
        )
        lines.append(f"URL: {result.get('url')}")
        lines.append(f"TEXT: {result.get('chunk', '')[:MAX_PASSAGE_CHARS]}")

    # ------------------------------------------------------------
    # Evidence consistency check diagnostics
    # ------------------------------------------------------------
    if evidence_diagnostics:
        diag = evidence_diagnostics
        lines.append("")
        lines.append("=== EVIDENCE CONSISTENCY DIAGNOSTICS ===")
        lines.append(
            f"Chunks received      : {diag['total_chunks_received']}"
        )
        lines.append(
            f"Unique articles      : {diag['unique_articles']}"
        )
        lines.append(
            f"Duplicate chunks removed : {diag['duplicate_chunks_removed']}"
        )
        lines.append(
            f"Unique publishers    : {diag['unique_publishers']}"
        )
        lines.append(
            f"Classification changes : {len(diag['classification_changes'])}"
        )
        for change in diag["classification_changes"]:
            lines.append(
                f"  - {change['url']}: {change['from']} -> {change['to']}"
            )
        if diag["related_context_excluded"]:
            lines.append("")
            lines.append("Related context passages excluded (did not meet all gates):")
            for excluded in diag["related_context_excluded"]:
                lines.append(
                    f"  - {excluded['url']} "
                    f"(score={excluded['evidence_score']}, "
                    f"event_match={excluded['event_match']}, "
                    f"nli={excluded['nli_label']} {excluded['nli_score']})"
                )

    # ------------------------------------------------------------
    # Deterministic floor
    #
    # This must agree with final_verdict.deterministic_gate. It states what
    # the evidence actually is, what a settled verdict would require, and
    # whether the result is absent, inaccessible, inconclusive or contested.
    # ------------------------------------------------------------
    lines.append("")
    lines.append("=== DETERMINISTIC EVIDENCE FLOOR ===")

    from final_verdict import (
        MIN_INDEPENDENT_SOURCES,
        MIN_PRIMARY_GROUPS,
        deterministic_gate,
    )

    gate = deterministic_gate(agreement, evidence_results)
    permitted = sorted(gate["permitted_verdicts"])

    lines.append(f"Permitted verdicts         : {', '.join(permitted)}")
    lines.append(f"Reason                     : {gate['permitted_reason']}")

    if independent_supporting and independent_contradicting:
        state = "CONTESTED"
    elif independent_supporting or independent_contradicting:
        state = "INCONCLUSIVE"
    elif supporting_sources or contradicting_sources:
        state = "INCONCLUSIVE"
    elif context_sources or neutral_sources:
        state = "INCONCLUSIVE"
    elif snippet_only_sources and not article_sources:
        state = "INACCESSIBLE"
    else:
        state = "ABSENT"

    lines.append(f"Evidence state             : {state}")

    if state == "INACCESSIBLE":
        lines.append(
            "Only search snippets were available; no article text could be read. "
            "Fallback evidence cannot establish or refute a claim."
        )
    elif state == "ABSENT":
        lines.append(
            "No retrieved passage was usable as evidence for this claim. "
            "This is an evidence gap, not evidence that the claim is false."
        )
    elif state == "CONTESTED":
        lines.append(
            "Independent sources DISAGREE. The claim is genuinely contested and "
            "must not be reported as settled."
        )
    else:
        lines.append(
            "The available evidence does not settle the claim. UNVERIFIED means "
            "the evidence did not justify a verdict; it does NOT mean the claim "
            "is false."
        )

    if primary_supporting < MIN_PRIMARY_GROUPS and independent_supporting >= MIN_INDEPENDENT_SOURCES:
        lines.append(
            f"Requirement not met for REAL: at least {MIN_PRIMARY_GROUPS} of the "
            f"supporting publishers must be primary reporting whose article text "
            f"was actually retrieved (currently {primary_supporting})."
        )
    if primary_contradicting < MIN_PRIMARY_GROUPS and independent_contradicting >= MIN_INDEPENDENT_SOURCES:
        lines.append(
            f"Requirement not met for FAKE: at least {MIN_PRIMARY_GROUPS} of the "
            f"contradicting publishers must be primary reporting whose article text "
            f"was actually retrieved (currently {primary_contradicting})."
        )
    if "MISLEADING" not in permitted:
        lines.append(
            "Requirement not met for MISLEADING: an explicit, evidence-backed "
            "correction of a material claim detail from a primary publisher is "
            "required, and none is present."
        )
    elif "MISLEADING" in permitted:
        lines.append(
            f"{primary_correction_groups} primary publisher(s) contain an explicit "
            f"correction of a material claim detail, so MISLEADING may be "
            f"considered alongside REAL."
        )

    if unresolved_identities:
        lines.append(
            f"{unresolved_identities} source(s) could not be assigned a reliable "
            f"publisher identity. They are reported above for diagnostics but are "
            f"excluded from the independent publisher counts."
        )

    if any(
        r.get("evidence_type") == "search_snippet"
        for r in evidence_results
    ):
        lines.append(
            "At least one source is a search-snippet fallback, not a retrieved "
            "article. Snippet-only evidence can never establish or refute a claim."
        )

    return "\n".join(lines)
