"""
evidence_fusion.py
==================
Source-level aggregation, independence and fusion.

Two rules drive this module:

1. MULTIPLE CHUNKS ARE NOT MULTIPLE SOURCES.
   Every chunk is collapsed into a single decision per source, and sources are
   then collapsed into a single vote per *independence group* (registrable
   domain). Four paragraphs from one article count once; four outlets that
   republish one wire story count once.

2. CORROBORATION IS WHAT COUNTS, NOT VOLUME.
   `independent_supporting` / `independent_contradicting` are the numbers the
   verdict layer is allowed to rely on.
"""

from collections import defaultdict
from urllib.parse import urlparse

from source_reliability import registrable_domain


def _get_domain(url, default_domain=""):
    if default_domain:
        return default_domain.lower()
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        elif netloc.startswith("m."):
            netloc = netloc[2:]
        return netloc
    except Exception:
        return ""


def _independence_group(result):
    """Domain-level voice identifier, resilient to missing fields."""
    group = result.get("independence_group")
    if group:
        return group.lower()
    return registrable_domain(result.get("url", "")) or _get_domain(
        result.get("url", ""), result.get("domain", "")
    )


def _summarise_source(source_id, results):
    """
    Collapse every chunk of one source into a single decision.

    A source supports only if it has a SUPPORT chunk, and contradicts only if it
    has a CONTRADICTION chunk; when both exist the stronger evidence_score wins
    and a near-tie is reported as mixed.
    """
    sample = results[0]

    support = [r for r in results if r.get("evidence_class") == "SUPPORT"]
    contradiction = [r for r in results if r.get("evidence_class") == "CONTRADICTION"]
    context = [r for r in results if r.get("evidence_class") == "RELATED_CONTEXT"]

    best_support = max(support, key=lambda r: r.get("evidence_score", 0), default=None)
    best_contradiction = max(
        contradiction, key=lambda r: r.get("evidence_score", 0), default=None
    )

    support_score = best_support.get("evidence_score", 0) if best_support else 0.0
    contradiction_score = (
        best_contradiction.get("evidence_score", 0) if best_contradiction else 0.0
    )

    if best_support and best_contradiction:
        if support_score > contradiction_score + 0.10:
            decision, best = "entailment", best_support
        elif contradiction_score > support_score + 0.10:
            decision, best = "contradiction", best_contradiction
        else:
            decision = "neutral"
            best = best_support if support_score >= contradiction_score else best_contradiction
    elif best_support:
        decision, best = "entailment", best_support
    elif best_contradiction:
        decision, best = "contradiction", best_contradiction
    elif context:
        decision = "related_context"
        best = max(context, key=lambda r: r.get("evidence_score", 0))
    else:
        decision = "non_evidence"
        best = max(results, key=lambda r: r.get("evidence_score", 0))

    is_snippet_only = all(
        r.get("evidence_type") == "search_snippet" for r in results
    )

    return {
        "source_id": source_id,
        "domain": _get_domain(sample.get("url", ""), sample.get("domain", "")),
        "independence_group": _independence_group(sample),
        "url": sample.get("url", ""),
        "title": sample.get("title", ""),
        "decision": decision,
        "source_score": round(best.get("evidence_score", 0.0), 4),
        "evidence_type": "search_snippet" if is_snippet_only else "scraped_article",
        "source_tier": sample.get("source_tier", "unknown"),
        "source_reliability": sample.get("source_reliability", 0.0),
        "date_status": best.get("date_status", "unknown"),
        "event_match": best.get("event_match", "weak"),
        "chunk_count": len(results),
        "has_support": bool(best_support),
        "has_contradiction": bool(best_contradiction),
        "best_chunk": best.get("chunk", "")[:700],
    }


# ============================================================
# SOURCE AGREEMENT
# ============================================================

def calculate_source_agreement(evidence_results):
    """
    Aggregate chunks into sources, sources into independence groups, and report
    how many *independent* voices support or contradict the claim.
    """
    if not evidence_results:
        return {
            "total_sources": 0,
            "total_chunks": 0,
            "contradictions": 0,
            "entailments": 0,
            "supporting_sources": 0,
            "contradicting_sources": 0,
            "neutral_sources": 0,
            "related_context_sources": 0,
            "non_evidence_sources": 0,
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "independence_groups": 0,
            "primary_supporting_groups": 0,
            "agreement_score": 0.0,
            "evidence_sufficiency": "none",
            "source_decisions": [],
            "group_decisions": [],
        }

    by_source = defaultdict(list)
    for result in evidence_results:
        by_source[result.get("source_id")].append(result)

    source_decisions = [
        _summarise_source(source_id, results)
        for source_id, results in sorted(by_source.items())
    ]

    supporting_sources = sum(1 for s in source_decisions if s["decision"] == "entailment")
    contradicting_sources = sum(1 for s in source_decisions if s["decision"] == "contradiction")
    related_context_sources = sum(1 for s in source_decisions if s["decision"] == "related_context")
    non_evidence_sources = sum(1 for s in source_decisions if s["decision"] == "non_evidence")
    neutral_sources = sum(1 for s in source_decisions if s["decision"] == "neutral")

    # --------------------------------------------------------
    # Collapse sources into independence groups
    # --------------------------------------------------------
    by_group = defaultdict(list)
    for decision in source_decisions:
        by_group[decision["independence_group"]].append(decision)

    group_decisions = []
    for group, decisions in by_group.items():
        entailing = [d for d in decisions if d["decision"] == "entailment"]
        contradicting = [d for d in decisions if d["decision"] == "contradiction"]

        best_support = max(
            entailing, key=lambda d: d["source_score"], default=None
        )
        best_contra = max(
            contradicting, key=lambda d: d["source_score"], default=None
        )

        if best_support and best_contra:
            group_decision = "entailment" if (
                best_support["source_score"] >= best_contra["source_score"]
            ) else "contradiction"
        elif best_support:
            group_decision = "entailment"
        elif best_contra:
            group_decision = "contradiction"
        else:
            group_decision = "context"

        is_snippet_group = all(
            d["evidence_type"] == "search_snippet" for d in decisions
        )

        group_decisions.append({
            "independence_group": group,
            "decision": group_decision,
            "source_count": len(decisions),
            "source_score": round(max(
                (d["source_score"] for d in decisions), default=0.0
            ), 4),
            "domains": sorted({d["domain"] for d in decisions if d["domain"]}),
            "best_url": (
                best_support or best_contra or max(
                    decisions, key=lambda d: d["source_score"]
                )
            )["url"],
            "is_snippet_only": is_snippet_group,
            "is_primary": (best_support or best_contra or {}).get(
                "evidence_type"
            ) == "scraped_article",
        })

    independent_supporting = sum(
        1 for g in group_decisions if g["decision"] == "entailment"
    )
    independent_contradicting = sum(
        1 for g in group_decisions if g["decision"] == "contradiction"
    )
    primary_supporting_groups = sum(
        1 for g in group_decisions
        if g["decision"] == "entailment" and g["is_primary"]
    )

    # --------------------------------------------------------
    # Agreement score
    #
    # Only meaningful evidence counts. One source can never agree with itself,
    # so a single supporting group is capped well below full agreement.
    # --------------------------------------------------------
    meaningful = independent_supporting + independent_contradicting

    if meaningful == 0:
        agreement_score = 0.0
    elif independent_supporting and independent_contradicting:
        dominant = max(independent_supporting, independent_contradicting)
        agreement_score = 0.5 * (dominant / meaningful)
    else:
        dominant = independent_supporting or independent_contradicting
        if independent_supporting:
            # How much of the whole retrieved pool actually supports the claim.
            agreement_score = 0.30 + 0.70 * (dominant / max(1, len(group_decisions)))
        else:
            agreement_score = 0.20 + 0.50 * (dominant / max(1, len(group_decisions)))

    if independent_supporting and primary_supporting_groups == 0:
        # Support exists, but none of it comes from scraped primary reporting.
        agreement_score = min(agreement_score, 0.35)

    if independent_supporting == 1 and independent_contradicting == 0:
        agreement_score = min(agreement_score, 0.65)

    # --------------------------------------------------------
    # Evidence sufficiency: can a strong verdict even be considered?
    # --------------------------------------------------------
    if independent_supporting >= 2 and primary_supporting_groups >= 2:
        sufficiency = "strong_support"
    elif independent_supporting >= 2 and primary_supporting_groups >= 1:
        sufficiency = "support"
    elif independent_supporting == 1:
        sufficiency = "single_source"
    elif independent_contradicting >= 2:
        sufficiency = "strong_contradiction"
    elif independent_contradicting == 1:
        sufficiency = "single_contradiction"
    elif related_context_sources or neutral_sources:
        sufficiency = "context_only"
    else:
        sufficiency = "none"

    return {
        "total_sources": len(source_decisions),
        "total_chunks": len(evidence_results),
        "contradictions": contradicting_sources,
        "entailments": supporting_sources,
        "neutrals": neutral_sources,
        "supporting_sources": supporting_sources,
        "contradicting_sources": contradicting_sources,
        "neutral_sources": neutral_sources,
        "related_context_sources": related_context_sources,
        "non_evidence_sources": non_evidence_sources,
        "independent_supporting": independent_supporting,
        "independent_contradicting": independent_contradicting,
        "independence_groups": len(group_decisions),
        "primary_supporting_groups": primary_supporting_groups,
        "agreement_score": round(min(agreement_score, 1.0), 4),
        "evidence_sufficiency": sufficiency,
        "source_decisions": source_decisions,
        "group_decisions": group_decisions,
    }


# ============================================================
# EVIDENCE FUSION
# ============================================================

def calculate_fusion_score(evidence_results, agreement):
    """
    Fuse source strength with independent corroboration.

    Fusion is driven by the best *independent* voices, not by the average of the
    top chunks: five chunks from one article cannot outvote two independent
    outlets.
    """
    if not evidence_results:
        return 0.0

    if isinstance(agreement, (int, float)):
        agreement_score = float(agreement)
        group_decisions = []
        independent_supporting = 0
        independent_contradicting = 0
    else:
        agreement_score = agreement.get("agreement_score", 0.0)
        group_decisions = agreement.get("group_decisions", [])
        independent_supporting = agreement.get("independent_supporting", 0)
        independent_contradicting = agreement.get("independent_contradicting", 0)

    decisive_groups = [
        g for g in group_decisions if g["decision"] in ("entailment", "contradiction")
    ]

    if not decisive_groups:
        return 0.0

    # Best evidence per independent voice, capped at 0.9.
    group_strengths = [
        min(0.9, g["source_score"]) for g in decisive_groups
    ]
    group_strengths.sort(reverse=True)
    top = group_strengths[:4]

    evidence_strength = sum(top) / len(top)

    # Diminishing returns: a fourth voice adds far less than a second.
    corroboration = min(1.0, (len(decisive_groups) - 1) / 2.0)

    fusion = (
        evidence_strength * 0.55
        + agreement_score * 0.25
        + corroboration * 0.20
    )

    if any(g["is_snippet_only"] for g in decisive_groups) and not any(
        not g["is_snippet_only"] for g in decisive_groups
    ):
        # Everything decisive is a search snippet.
        fusion = min(fusion, 0.45)

    if len(decisive_groups) == 1:
        # A single independent voice cannot be strongly corroborated.
        fusion = min(fusion, 0.62)

    if independent_supporting and independent_contradicting:
        # Genuine disagreement between independent sources must not fuse into
        # a confident number.
        fusion = min(fusion, 0.50)

    if not decisive_groups:
        fusion = 0.0

    return round(max(0.0, min(1.0, fusion)), 4)
