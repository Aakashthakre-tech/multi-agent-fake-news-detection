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
from urllib.parse import urlparse, parse_qsl, urlencode

from evidence_analyzer import classify_evidence
from source_reliability import registrable_domain


# Query parameters that identify a campaign rather than an article. They must
# not split one article into several "sources".
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid",
    "ref", "referrer", "source", "cmpid", "spm", "yclid", "_hsenc", "_hsmi",
    "at_medium", "at_campaign", "CMP", "ito", "ns_campaign", "ns_mchannel",
}


def _canonical_url(url):
    """
    Normalised form of a URL, used as the identity of a source.

    Two URLs that differ only in scheme, ``www.``/``m.`` prefix, trailing
    slash, fragment or campaign parameters are the same article. Query
    parameters that actually select an article (``?id=...``) are kept, so two
    different articles are never merged.
    """
    if not url or not isinstance(url, str):
        return ""

    candidate = url.strip()
    if not candidate:
        return ""

    try:
        parsed = urlparse(candidate)
    except Exception:
        return candidate.lower()

    host = (parsed.netloc or "").lower()
    if "@" in host:
        host = host.split("@", 1)[1]
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break

    path = (parsed.path or "").rstrip("/")

    kept = [
        (key, value)
        for key, value in parse_qsl(parsed.query or "", keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    if not host:
        return candidate.lower()

    return f"{host}{path}" + (f"?{query}" if query else "")


def _source_key(record, fallback_index=None):
    """
    Stable identity for one source, plus whether that identity is trustworthy.

    The canonical URL is the identity of a source. ``source_id`` is only a
    fallback and only when the URL is missing, because a missing or reused
    source_id must never merge two different articles.

    Returns (key, resolved). An unresolved key is still returned so the
    record is preserved for diagnostics, but the caller excludes it from
    independent corroboration.
    """
    url = _canonical_url(record.get("url"))
    if url:
        return f"url:{url}", True

    source_id = record.get("source_id")
    if source_id is not None and source_id != "":
        return f"sid:{source_id}", False

    return f"unresolved:{fallback_index}", False


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
    """
    Domain-level voice identifier, resilient to missing fields.

    Returns "" when no domain can be established. Such a record has no
    resolvable publisher identity and must not be counted as a publisher.
    """
    group = result.get("independence_group")
    if group and str(group).strip():
        return str(group).strip().lower()

    derived = registrable_domain(result.get("url", "")) or _get_domain(
        result.get("url", ""), result.get("domain", "")
    )
    if derived and str(derived).strip():
        return str(derived).strip().lower()

    return ""


# The minimum claim-detail coverage the evidence analyzer itself uses to
# decide that a passage covers a material part of the claim.
_MATERIAL_DETAIL_COVERAGE = 0.30


def _is_material_correction(record):
    """
    True only when this passage is a correction of THIS claim's material
    details.

    A passage that merely uses correction language ("misleading", "actually",
    "fact-check") is not a correction of the claim. For a passage to count it
    must:

      1. be classified as a CONTRADICTION, i.e. it directly addresses and
         contradicts the claim about the claimed event (the analyzer only
         classifies a passage that way when the NLI model reads it as a
         contradiction and the event match is strong);
      2. carry the explicit debunking signal; and
      3. correct something material - either the claimed date, or a material
         part of the claim's own content.

    A passage that contradicts the claim but shares none of its details, or
    that mentions correction language while talking about something else, is
    not a correction and must not satisfy the correction requirement.
    """
    if not isinstance(record, dict):
        return False

    if record.get("evidence_class") != "CONTRADICTION":
        return False

    if not record.get("debunking"):
        return False

    if record.get("date_status") == "different":
        # A dated correction of the claimed event.
        return True

    try:
        coverage = float(record.get("detail_coverage") or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0

    return coverage >= _MATERIAL_DETAIL_COVERAGE


def _summarise_source(source_id, results, identity_resolved=True):
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
    has_article_text = any(
        r.get("evidence_type") in ("scraped_article", "search_extract")
        for r in results
    )

    if is_snippet_only and decision in ("entailment", "contradiction"):
        # A search snippet is provider fallback text, not reporting. The
        # evidence analyzer never classifies one as SUPPORT or CONTRADICTION,
        # and the fusion layer must not let a snippet become one of the
        # publishers that establishes or refutes a claim.
        decision = "related_context"

    # The real retrieval mode, reported honestly. A provider-extracted body is
    # not the same thing as a page we scraped, and neither is a snippet.
    retrieval_modes = {
        r.get("evidence_type")
        for r in results
        if r.get("evidence_type") in ("scraped_article", "search_extract", "search_snippet")
    }
    if not retrieval_modes:
        retrieval_mode = "unknown"
    elif len(retrieval_modes) == 1:
        retrieval_mode = retrieval_modes.pop()
    elif "scraped_article" in retrieval_modes:
        retrieval_mode = "mixed"
    else:
        retrieval_mode = "mixed"

    # "Primary reporting" is a property of the publisher, recorded by
    # source_reliability.get_source_profile and carried on every chunk. It is
    # NOT the same thing as "we managed to scrape the page": a blog post we
    # scraped is still not primary reporting.
    is_primary_reporting = (
        any(bool(r.get("is_primary_source")) for r in results)
        and not is_snippet_only
    )

    # An explicit, material correction of the claim. A passage only counts when
    # it directly contradicts this claim and corrects a material detail of it,
    # not merely because it uses correction language.
    has_correction = any(_is_material_correction(r) for r in results)

    independence_group = _independence_group(sample)

    return {
        "source_id": source_id,
        "domain": _get_domain(sample.get("url", ""), sample.get("domain", "")),
        "independence_group": independence_group,
        "url": sample.get("url", ""),
        "title": sample.get("title", ""),
        "decision": decision,
        "source_score": round(best.get("evidence_score", 0.0), 4),
        "evidence_type": "search_snippet" if is_snippet_only else "scraped_article",
        "retrieval_mode": retrieval_mode,
        "source_tier": sample.get("source_tier", "unknown"),
        "source_reliability": sample.get("source_reliability", 0.0),
        "date_status": best.get("date_status", "unknown"),
        "event_match": best.get("event_match", "weak"),
        "chunk_count": len(results),
        "has_support": bool(best_support),
        "has_contradiction": bool(best_contradiction),
        # --- new, evidence-policy fields -------------------------------
        "is_primary": is_primary_reporting,
        "is_snippet_only": is_snippet_only,
        "has_article_text": has_article_text,
        "has_correction": has_correction,
        "identity_resolved": bool(identity_resolved) and bool(independence_group),
    }


# ============================================================
# SOURCE AGREEMENT
# ============================================================

def calculate_source_agreement(evidence_results):
    """
    Aggregate chunks into sources, sources into independence groups, and report
    how many *independent* voices support or contradict the claim.

    Source identity is the canonical URL. A record whose identity cannot be
    established is kept for diagnostics but excluded from the independent
    corroboration counts, so an absent or reused identifier can never merge
    unrelated reporting into one publisher.
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
            "snippet_only_sources": 0,
            "article_sources": 0,
            "unresolved_identity_sources": 0,
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "independence_groups": 0,
            "primary_supporting_groups": 0,
            "primary_contradicting_groups": 0,
            "article_supporting_groups": 0,
            "article_contradicting_groups": 0,
            "correction_groups": 0,
            "primary_correction_groups": 0,
            "verification_coverage": 0.0,
            "agreement_score": 0.0,
            "evidence_sufficiency": "none",
            "source_decisions": [],
            "group_decisions": [],
        }

    by_source = defaultdict(list)
    resolved_flags = {}
    for index, result in enumerate(evidence_results):
        key, resolved = _source_key(result, index)
        by_source[key].append(result)
        resolved_flags[key] = resolved

    source_decisions = []
    for key, results in sorted(by_source.items()):
        decision = _summarise_source(
            results[0].get("source_id"), results,
            identity_resolved=resolved_flags.get(key, True),
        )
        decision["source_key"] = key
        source_decisions.append(decision)

    supporting_sources = sum(1 for s in source_decisions if s["decision"] == "entailment")
    contradicting_sources = sum(1 for s in source_decisions if s["decision"] == "contradiction")
    related_context_sources = sum(1 for s in source_decisions if s["decision"] == "related_context")
    non_evidence_sources = sum(1 for s in source_decisions if s["decision"] == "non_evidence")
    neutral_sources = sum(1 for s in source_decisions if s["decision"] == "neutral")
    snippet_only_sources = sum(1 for s in source_decisions if s["is_snippet_only"])
    article_sources = sum(1 for s in source_decisions if s["has_article_text"])
    unresolved_identity_sources = sum(
        1 for s in source_decisions if not s["identity_resolved"]
    )

    # --------------------------------------------------------
    # Collapse sources into independence groups
    #
    # A source with no resolvable publisher identity is kept in the
    # diagnostics but cannot join a publisher group.
    # --------------------------------------------------------
    by_group = defaultdict(list)
    unresolved_sources = []
    for decision in source_decisions:
        group = decision["independence_group"]
        if not group or not decision["identity_resolved"]:
            unresolved_sources.append(decision)
            continue
        by_group[group].append(decision)

    group_decisions = []
    for group, decisions in sorted(by_group.items()):
        entailing = [d for d in decisions if d["decision"] == "entailment"]
        contradicting = [d for d in decisions if d["decision"] == "contradiction"]

        best_support = max(
            entailing, key=lambda d: d["source_score"], default=None
        )
        best_contra = max(
            contradicting, key=lambda d: d["source_score"], default=None
        )

        # The same 0.10 margin the source level uses: a group whose sources
        # disagree within that margin is mixed, not a decisive vote.
        if best_support and best_contra:
            if best_support["source_score"] >= best_contra["source_score"] + 0.10:
                group_decision = "entailment"
            elif best_contra["source_score"] >= best_support["source_score"] + 0.10:
                group_decision = "contradiction"
            else:
                group_decision = "neutral"
        elif best_support:
            group_decision = "entailment"
        elif best_contra:
            group_decision = "contradiction"
        elif any(d["decision"] == "related_context" for d in decisions):
            group_decision = "context"
        else:
            group_decision = "non_evidence"

        is_snippet_group = all(d["is_snippet_only"] for d in decisions)
        has_article_group = any(d["has_article_text"] for d in decisions)
        is_primary_group = any(d["is_primary"] for d in decisions)
        # The correction must come from a source that is itself primary
        # reporting with retrieved article text. A primary source elsewhere in
        # the group does not lend its standing to a correcting source that is
        # not primary.
        correcting = [d for d in decisions if d["has_correction"]]
        has_correction = bool(correcting)
        has_primary_correction = any(
            d["is_primary"] and d["has_correction"] for d in decisions
        )

        decisive = best_support or best_contra
        group_decisions.append({
            "independence_group": group,
            "decision": group_decision,
            "source_count": len(decisions),
            "source_score": round(max(
                (d["source_score"] for d in decisions), default=0.0
            ), 4),
            "domains": sorted({d["domain"] for d in decisions if d["domain"]}),
            "best_url": (
                decisive or max(
                    decisions, key=lambda d: d["source_score"]
                )
            )["url"],
            "is_snippet_only": is_snippet_group,
            "has_article_text": has_article_group,
            "is_primary": is_primary_group,
            "has_correction": has_correction,
            "has_primary_correction": has_primary_correction,
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
    primary_contradicting_groups = sum(
        1 for g in group_decisions
        if g["decision"] == "contradiction" and g["is_primary"]
    )
    article_supporting_groups = sum(
        1 for g in group_decisions
        if g["decision"] == "entailment" and g["has_article_text"]
    )
    article_contradicting_groups = sum(
        1 for g in group_decisions
        if g["decision"] == "contradiction" and g["has_article_text"]
    )
    correction_groups = sum(
        1 for g in group_decisions if g["has_correction"]
    )
    primary_correction_groups = sum(
        1 for g in group_decisions if g["has_primary_correction"]
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
        # Support exists, but none of it comes from primary reporting.
        agreement_score = min(agreement_score, 0.35)

    if independent_contradicting and primary_contradicting_groups == 0:
        # Contradiction exists, but only from fallback evidence.
        agreement_score = min(agreement_score, 0.35)

    if independent_supporting == 1 and independent_contradicting == 0:
        agreement_score = min(agreement_score, 0.65)

    # How much of the retrieved pool could actually be read.
    verification_coverage = (
        round(article_sources / len(source_decisions), 4)
        if source_decisions else 0.0
    )

    # --------------------------------------------------------
    # Evidence sufficiency: can a strong verdict even be considered?
    #
    # These labels mirror the deterministic gate exactly: a verdict needs
    # independent publishers AND primary reporting, in both directions.
    # --------------------------------------------------------
    if independent_supporting >= 2 and primary_supporting_groups >= 2:
        sufficiency = "strong_support"
    elif independent_supporting >= 2 and primary_supporting_groups >= 1:
        sufficiency = "support"
    elif independent_supporting == 1:
        sufficiency = "single_source"
    elif independent_contradicting >= 2 and primary_contradicting_groups >= 1:
        sufficiency = "strong_contradiction"
    elif independent_contradicting >= 1:
        # Contradiction exists but rests only on fallback evidence.
        sufficiency = (
            "fallback_contradiction"
            if primary_contradicting_groups == 0
            else "single_contradiction"
        )
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
        "snippet_only_sources": snippet_only_sources,
        "article_sources": article_sources,
        "unresolved_identity_sources": unresolved_identity_sources,
        "independent_supporting": independent_supporting,
        "independent_contradicting": independent_contradicting,
        "independence_groups": len(group_decisions),
        "primary_supporting_groups": primary_supporting_groups,
        "primary_contradicting_groups": primary_contradicting_groups,
        "article_supporting_groups": article_supporting_groups,
        "article_contradicting_groups": article_contradicting_groups,
        "correction_groups": correction_groups,
        "primary_correction_groups": primary_correction_groups,
        "verification_coverage": verification_coverage,
        "agreement_score": round(min(agreement_score, 1.0), 4),
        "evidence_sufficiency": sufficiency,
        "source_decisions": source_decisions,
        "group_decisions": group_decisions,
        "unresolved_source_decisions": unresolved_sources,
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

    if any(g.get("is_snippet_only", False) for g in decisive_groups) and not any(
        g.get("has_article_text", True) for g in decisive_groups
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


# ============================================================
# EVIDENCE CONSISTENCY CHECK
# ============================================================

def reconcile_evidence(evidence_results):
    """
    Post-search consistency check. Runs after evidence classification
    (analyze_evidence) and before source agreement fusion
    (calculate_source_agreement).

    Performs three-level deduplication and re-evaluation:

      1. Chunk level: within one source/URL, keep the best passage per
         evidence class. Multiple chunks from the same article cannot count
         as multiple sources.

      2. Source level: all chunks from one canonical URL collapse into one
         source decision.

      3. Publisher level: all sources from one independence group collapse
         into one publisher decision.

    Re-evaluates RELATED_CONTEXT passages: if the strongest passage from a
    RELATED_CONTEXT source meets all SUPPORT/CONTRADICTION gates (via
    classify_evidence), it is upgraded. This corrects cases where
    classify_evidence was conservative due to marginal gate values.

    Does NOT lower thresholds or force REAL/FAKE. All existing safeguards
    (snippet exclusion, primary source requirement, event_match requirements,
    NLI thresholds) remain intact.

    Returns (reconciled_results, diagnostics, publisher_decisions).

      reconciled_results: list of evidence records (one per source), shaped
          like evidence_analyzer output, suitable for calculate_source_agreement.
      diagnostics: dict with counts, excluded reasons, and classification changes.
      publisher_decisions: list of per-publisher decisions.
    """
    if not evidence_results:
        return [], _empty_diagnostics(), []

    # --- Level 1: Group chunks by canonical URL ---
    by_source = defaultdict(list)
    for result in evidence_results:
        key = _canonical_url(result.get("url", ""))
        by_source[key].append(result)

    source_level = []
    diagnostics = {
        "total_chunks_received": len(evidence_results),
        "unique_articles": 0,
        "unique_publishers": 0,
        "duplicate_chunks_removed": 0,
        "supporting_articles": [],
        "contradicting_articles": [],
        "related_context_excluded": [],
        "classification_changes": [],
    }

    for url_key, chunks in sorted(by_source.items()):
        diagnostics["duplicate_chunks_removed"] += len(chunks) - 1

        # Keep best passage per evidence class
        support_chunks = [
            c for c in chunks if c.get("evidence_class") == "SUPPORT"
        ]
        contradiction_chunks = [
            c for c in chunks if c.get("evidence_class") == "CONTRADICTION"
        ]
        context_chunks = [
            c for c in chunks if c.get("evidence_class") == "RELATED_CONTEXT"
        ]

        best_support = max(
            support_chunks, key=lambda r: r.get("evidence_score", 0.0),
            default=None,
        )
        best_contradiction = max(
            contradiction_chunks, key=lambda r: r.get("evidence_score", 0.0),
            default=None,
        )

        # Source-level decision: SUPPORT > CONTRADICTION > best available
        best = None
        decision = None
        if best_support:
            best = best_support
            decision = "SUPPORT"
        elif best_contradiction:
            best = best_contradiction
            decision = "CONTRADICTION"
        else:
            best = max(chunks, key=lambda r: r.get("evidence_score", 0.0))
            decision = best.get("evidence_class", "NON_EVIDENCE")

        source_level.append({
            "url_key": url_key,
            "url": best.get("url", ""),
            "domain": best.get("domain", ""),
            "title": best.get("title", ""),
            "evidence_type": best.get("evidence_type", ""),
            "best_chunk": best,
            "decision": decision,
            "support_count": len(support_chunks),
            "contradiction_count": len(contradiction_chunks),
            "context_count": len(context_chunks),
            "total_chunks": len(chunks),
        })

    diagnostics["unique_articles"] = len(source_level)

    # --- Level 2: Re-evaluate RELATED_CONTEXT sources ---
    for source in source_level:
        best = source["best_chunk"]

        if best.get("evidence_class") == "RELATED_CONTEXT":
            # Build signals dict from chunk for classify_evidence
            signals = {
                "similarity": best.get("similarity", 0.0),
                "date_status": best.get("date_status", "unknown"),
                "event_match": best.get("event_match", "weak"),
                "raw_nli_label": best.get("raw_nli_label", "neutral"),
                "raw_nli_score": best.get("raw_nli_score", 0.0),
                "nli_label": best.get("nli_label", "neutral"),
                "nli_score": best.get("nli_score", 0.0),
                "nli_status": best.get("nli_status", "ok"),
                "location_match": best.get("location_match", False),
                "detail_coverage": best.get("detail_coverage", 0.0),
                "debunking": best.get("debunking", False),
                "passage_kind": best.get("passage_kind", "paragraph"),
                "claim_type": best.get("claim_type", "other_factual_claim"),
                "event_date_conflict": best.get("event_date_conflict", False),
                "event_date_mismatch": best.get("event_date_mismatch", False),
                "claim_has_location": best.get("claim_has_location", False),
            }
            source_profile = {
                "is_primary": best.get("is_primary_source", False),
                "tier": best.get("source_tier", "unknown"),
            }
            claim_type = best.get("claim_type", "other_factual_claim")

            reclassified = classify_evidence(
                signals, best.get("evidence_type", "scraped_article"),
                source_profile, claim_type=claim_type,
            )

            if reclassified in ("SUPPORT", "CONTRADICTION"):
                best = dict(best)
                best["evidence_class"] = reclassified
                best["consistency_upgraded"] = True
                source["best_chunk"] = best
                source["decision"] = reclassified
                diagnostics["classification_changes"].append({
                    "url": best.get("url", ""),
                    "domain": best.get("domain", ""),
                    "from": "RELATED_CONTEXT",
                    "to": reclassified,
                    "reason": (
                        "Best passage meets all SUPPORT/CONTRADICTION gates "
                        "on re-evaluation",
                    ),
                })
            else:
                diagnostics["related_context_excluded"].append({
                    "url": best.get("url", ""),
                    "domain": best.get("domain", ""),
                    "evidence_score": best.get("evidence_score", 0.0),
                    "nli_label": best.get("nli_label", ""),
                    "nli_score": best.get("nli_score", 0.0),
                    "similarity": best.get("similarity", 0.0),
                    "event_match": best.get("event_match", ""),
                    "date_status": best.get("date_status", ""),
                    "reason": (
                        "Does not meet all SUPPORT/CONTRADICTION gates",
                    ),
                })

        if source["decision"] == "SUPPORT":
            diagnostics["supporting_articles"].append({
                "url": best.get("url", ""),
                "domain": best.get("domain", ""),
                "evidence_score": best.get("evidence_score", 0.0),
            })
        elif source["decision"] == "CONTRADICTION":
            diagnostics["contradicting_articles"].append({
                "url": best.get("url", ""),
                "domain": best.get("domain", ""),
                "evidence_score": best.get("evidence_score", 0.0),
            })

    # --- Level 3: Group by independence group ---
    by_group = defaultdict(list)
    for source in source_level:
        group = source["best_chunk"].get("independence_group", "")
        if group and str(group).strip():
            by_group[str(group).strip().lower()].append(source)
        else:
            by_group[f"unresolved:{source['url_key']}"].append(source)

    diagnostics["unique_publishers"] = len(by_group)

    publisher_decisions = []
    for group, sources in sorted(by_group.items()):
        support_sources = [s for s in sources if s["decision"] == "SUPPORT"]
        contradiction_sources = [
            s for s in sources if s["decision"] == "CONTRADICTION"
        ]

        if support_sources:
            best_source = max(
                support_sources,
                key=lambda s: s["best_chunk"].get("evidence_score", 0.0),
            )
            decision = "SUPPORT"
        elif contradiction_sources:
            best_source = max(
                contradiction_sources,
                key=lambda s: s["best_chunk"].get("evidence_score", 0.0),
            )
            decision = "CONTRADICTION"
        else:
            best_source = max(
                sources,
                key=lambda s: s["best_chunk"].get("evidence_score", 0.0),
            )
            decision = best_source["best_chunk"].get(
                "evidence_class", "NON_EVIDENCE"
            )

        publisher_decisions.append({
            "independence_group": group,
            "decision": decision,
            "source_count": len(sources),
            "best_url": best_source["best_chunk"].get("url", ""),
            "best_domain": best_source["best_chunk"].get("domain", ""),
            "best_score": best_source["best_chunk"].get("evidence_score", 0.0),
        })

    # Build reconciled evidence results (one per source)
    reconciled_results = []
    for source in source_level:
        reconciled_results.append(source["best_chunk"])

    return reconciled_results, diagnostics, publisher_decisions


def _empty_diagnostics():
    return {
        "total_chunks_received": 0,
        "unique_articles": 0,
        "unique_publishers": 0,
        "duplicate_chunks_removed": 0,
        "supporting_articles": [],
        "contradicting_articles": [],
        "related_context_excluded": [],
        "classification_changes": [],
    }
