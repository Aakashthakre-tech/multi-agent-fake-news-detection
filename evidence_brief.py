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


MAX_PASSAGES = 6
MAX_PASSAGE_CHARS = 900


def _attributes(claim):
    from claim_profile import build_claim_profile

    return build_claim_profile(claim)


def _source_decisions(evidence_results):
    """Re-derive per-source decisions for the brief (chunk -> source)."""
    from evidence_fusion import _summarise_source

    by_source = defaultdict(list)
    for result in evidence_results:
        by_source[result.get("source_id")].append(result)

    decisions = []
    for source_id, results in sorted(by_source.items()):
        decisions.append((_summarise_source(source_id, results), results))

    return decisions


def build_evidence_brief(
    claim,
    sources,
    evidence_results,
    source_agreement,
    fusion_score,
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
    lines.append(f"Sources analysed            : {total_sources}")
    lines.append(f"Independent publishers      : {total_groups}")
    lines.append(f"Independent SUPPORTING      : {independent_supporting}")
    lines.append(f"Independent CONTRADICTING   : {independent_contradicting}")
    lines.append(f"Evidence sufficiency        : {sufficiency}")
    lines.append(f"Source agreement score      : {agreement.get('agreement_score', 0)}")
    lines.append(f"Evidence fusion score       : {fusion_score}")

    if not evidence_results:
        lines.append("")
        lines.append(
            "No retrieved passage passed the evidence layer's filters. "
            "There is no evidence to support or refute the claim."
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
            f"    publisher={decision['independence_group']} | "
            f"tier={decision['source_tier']} | reliability={decision['source_reliability']} | "
            f"retrieval={decision['evidence_type']}"
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
    # Deterministic floor
    # ------------------------------------------------------------
    lines.append("")
    lines.append("=== DETERMINISTIC EVIDENCE FLOOR ===")

    if sufficiency in ("strong_support", "support"):
        lines.append(
            f"At least {independent_supporting} independent publishers directly "
            f"support the claim, including "
            f"{agreement.get('primary_supporting_groups', 0)} primary article(s) "
            f"that state the claimed date and location."
        )
    elif sufficiency == "single_source":
        lines.append(
            "Only ONE independent publisher directly supports the claim. "
            "Independent corroboration is missing, so a confident REAL verdict "
            "is NOT supported by the evidence layer."
        )
    elif sufficiency in ("strong_contradiction", "single_contradiction"):
        lines.append(
            f"{independent_contradicting} independent publisher(s) directly "
            f"contradict the claim."
        )
    else:
        lines.append(
            "No source directly establishes the claimed event with the claimed "
            "date and location. The retrieved sources are context only. "
            "REAL and FAKE are NOT supported by the evidence layer; the correct "
            "outcome is UNVERIFIED."
        )

    if independent_supporting and independent_contradicting:
        lines.append(
            "Independent sources DISAGREE. The claim is genuinely contested and "
            "must not be reported as settled."
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
