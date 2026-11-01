"""
scratch/trace_real_evidence.py
==============================
Traces ONE real end-to-end run through every layer of the evidence pipeline:

    retrieval -> scraping -> chunking -> source identity -> classification
    -> source aggregation -> independence groups -> deterministic gate
    -> final verdict

It prints, for every retrieved source, exactly why it ended up where it did,
and whether the final outcome is a genuine evidence gap or a defect.

This makes LIVE network calls (search + scraping). Its output is the evidence
for the audit's "one real failing example" requirement.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLAIM = (
    "The Indian Space Research Organisation launched the Chandrayaan-3 "
    "mission on 14 July 2023."
)

OUT = ROOT / "scratch" / "trace_real_evidence.json"


def main():
    from pipeline import run_research_pipeline
    from evidence_fusion import calculate_source_agreement, calculate_fusion_score
    from final_verdict import deterministic_gate, generate_final_verdict

    print("=" * 78)
    print(f"CLAIM: {CLAIM}")
    print("=" * 78)

    state = run_research_pipeline(CLAIM)

    results = state.get("evidence_analysis") or []
    agreement = calculate_source_agreement(results)
    fusion = calculate_fusion_score(results, agreement)
    gate = deterministic_gate(agreement, results)
    verdict = generate_final_verdict(
        results, agreement, fusion,
        state.get("verification", ""), state.get("feedback", ""),
    )

    print()
    print("=" * 78)
    print("1. RETRIEVAL")
    print("=" * 78)
    plans = state.get("search_plans") or []
    for plan in plans:
        print(f"  query [{plan.get('label')}]: {plan.get('query')}")
    for err in state.get("search_errors") or []:
        print(f"  search error: {err}")
    source_data = state.get("source_data") or []
    print(f"  sources retained : {len(source_data)}")
    print(f"  scrape failures  : {len(state.get('scrape_failures') or [])}")
    for failure in (state.get("scrape_failures") or [])[:5]:
        print(f"    - {failure.get('domain')}: {failure.get('reason')}")

    print()
    print("=" * 78)
    print("2. SOURCE IDENTITY (per source, before classification)")
    print("=" * 78)
    for source in source_data:
        print(f"  id={source_data.index(source) + 1} "
              f"type={source.get('evidence_type'):16s} "
              f"domain={source.get('domain'):28s} "
              f"url={source.get('url')[:70]}")

    print()
    print("=" * 78)
    print("3. CLASSIFICATION (per chunk)")
    print("=" * 78)
    for chunk in results:
        print(
            f"  src={chunk.get('source_id')} "
            f"class={chunk.get('evidence_class'):14s} "
            f"score={chunk.get('evidence_score')} "
            f"nli={chunk.get('nli_label')}/{chunk.get('nli_score')} "
            f"date={chunk.get('date_status')} "
            f"event={chunk.get('event_match')} "
            f"loc={chunk.get('location_match')} "
            f"detail={chunk.get('detail_coverage')}"
        )
        print(f"        {str(chunk.get('chunk'))[:150]}")

    print()
    print("=" * 78)
    print("4. SOURCE AGGREGATION")
    print("=" * 78)
    for decision in agreement["source_decisions"]:
        print(
            f"  [{decision['decision']:14s}] group={decision['independence_group'] or 'UNRESOLVED':24s} "
            f"chunks={decision['chunk_count']} "
            f"retrieval={decision.get('retrieval_mode')} "
            f"primary={decision['is_primary']} "
            f"identity_resolved={decision['identity_resolved']}"
        )

    print()
    print("=" * 78)
    print("5. INDEPENDENCE GROUPS")
    print("=" * 78)
    for group in agreement["group_decisions"]:
        print(
            f"  {group['independence_group']:28s} decision={group['decision']:14s} "
            f"sources={group['source_count']} "
            f"primary={group['is_primary']} "
            f"article={group['has_article_text']} "
            f"snippet_only={group['is_snippet_only']}"
        )
    print(f"  independent_supporting      : {agreement['independent_supporting']}")
    print(f"  independent_contradicting   : {agreement['independent_contradicting']}")
    print(f"  primary_supporting_groups   : {agreement['primary_supporting_groups']}")
    print(f"  primary_contradicting_groups: {agreement['primary_contradicting_groups']}")
    print(f"  snippet_only_sources        : {agreement['snippet_only_sources']}")
    print(f"  article_sources             : {agreement['article_sources']}")
    print(f"  unresolved_identity_sources : {agreement['unresolved_identity_sources']}")
    print(f"  verification_coverage       : {agreement['verification_coverage']}")
    print(f"  evidence_sufficiency        : {agreement['evidence_sufficiency']}")

    print()
    print("=" * 78)
    print("6. DETERMINISTIC GATE")
    print("=" * 78)
    print(f"  permitted : {sorted(gate['permitted_verdicts'])}")
    print(f"  reason    : {gate['permitted_reason']}")

    print()
    print("=" * 78)
    print("7. FINAL VERDICT")
    print("=" * 78)
    print(f"  verdict         : {verdict['verdict']}")
    print(f"  confidence      : {verdict['confidence']} ({verdict.get('confidence_basis')})")
    print(f"  agent verdicts  : verification={verdict['verification_verdict']} "
          f"critic={verdict['critic_verdict']}")
    print(f"  overridden      : {verdict['verdict_overridden_by_evidence']}")
    print(f"  reasoning       : {verdict['reasoning']}")

    OUT.write_text(json.dumps({
        "claim": CLAIM,
        "agreement": {
            k: v for k, v in agreement.items()
            if k not in ("source_decisions", "group_decisions")
        },
        "gate": {k: v for k, v in gate.items()},
        "verdict": {
            k: v for k, v in verdict.items()
            if k not in ("verification_result", "critic_result")
        },
        "chunks": [
            {
                "source_id": c.get("source_id"),
                "url": c.get("url"),
                "domain": c.get("domain"),
                "evidence_class": c.get("evidence_class"),
                "evidence_type": c.get("evidence_type"),
                "evidence_score": c.get("evidence_score"),
                "nli_label": c.get("nli_label"),
                "nli_score": c.get("nli_score"),
                "date_status": c.get("date_status"),
                "event_match": c.get("event_match"),
                "location_match": c.get("location_match"),
                "detail_coverage": c.get("detail_coverage"),
                "independence_group": c.get("independence_group"),
                "is_primary_source": c.get("is_primary_source"),
                "chunk": str(c.get("chunk"))[:400],
            }
            for c in results
        ],
    }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved trace -> {OUT}")


if __name__ == "__main__":
    main()
