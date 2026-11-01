"""
scratch/trace_apollo11.py
=========================
Step 1 + Step 2 of the Apollo 11 investigation.

Runs the exact claim through the real text-verification pipeline and captures
every intermediate value, then re-derives each classification transition by
calling the responsible functions directly on the retrieved text.

Nothing here infers behaviour from log output: every transition is reproduced
by calling the function that makes it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
OUT = ROOT / "scratch" / "apollo11_trace.json"


def main():
    from pipeline import run_research_pipeline
    from evidence_fusion import calculate_source_agreement, calculate_fusion_score
    from final_verdict import deterministic_gate, generate_final_verdict

    # The verification/critic agents need an LLM. Their opinion cannot widen
    # the deterministic gate's permitted set, so for this capture they are
    # stubbed. Every deterministic value below is produced by the real code.
    import pipeline as pipeline_mod

    class _StubChain:
        def invoke(self, _payload):
            return "Verdict:\nUNVERIFIED\n\nConfidence:\n50%\n\nReasoning:\n(stubbed for capture)"

    pipeline_mod.verification_chain = _StubChain()
    pipeline_mod.critic_chain = _StubChain()

    print("=" * 100)
    print(f"CLAIM: {CLAIM}")
    print("=" * 100)

    state = run_research_pipeline(CLAIM)

    results = state.get("evidence_analysis") or []
    agreement = calculate_source_agreement(results)
    fusion = calculate_fusion_score(results, agreement)
    gate = deterministic_gate(agreement, results)
    verdict = generate_final_verdict(
        results, agreement, fusion,
        state.get("verification", ""), state.get("feedback", ""),
    )

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("A. RETRIEVAL - queries issued")
    print("=" * 100)
    for plan in state.get("search_plans") or []:
        print(f"  [{plan.get('label')}] intent={plan.get('intent')} :: {plan.get('query')}")
    for err in state.get("search_errors") or []:
        print(f"  SEARCH ERROR: {err}")

    print()
    print("=" * 100)
    print("B. SOURCE LEDGER (pipeline source_data) - URL normalisation + article vs snippet")
    print("=" * 100)
    print(f"{'#':>2} {'scrape_status':<28} {'evidence_type':<18} {'domain':<26} len(content)")
    for i, src in enumerate(state.get("source_data") or [], 1):
        content = src.get("content") or ""
        snippet = src.get("snippet") or src.get("search_snippet") or ""
        print(f"{i:>2} {str(src.get('scrape_status'))[:27]:<28} "
              f"{str(src.get('evidence_type'))[:17]:<18} {str(src.get('domain'))[:25]:<26} "
              f"content={len(content)} snippet={len(snippet)}")
        print(f"     url={src.get('url')}")

    print()
    print("=" * 100)
    print("C. SCRAPE FAILURES")
    print("=" * 100)
    failures = state.get("scrape_failures") or []
    if not failures:
        print("  (none)")
    for f in failures:
        print(f"  {f.get('domain')}: {f.get('reason')} | fallback={f.get('fallback_used')}")

    # ------------------------------------------------------------------
    # PER-SOURCE DIAGNOSTIC TABLE
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("D. PER-CHUNK DIAGNOSTIC TABLE")
    print("=" * 100)
    header = (f"{'url':<52} {'pub':<20} {'type':<16} {'evt':<8} {'class':<15} "
              f"{'nli':<14} {'nliS':<7} {'raw':<14} {'rawS':<7} {'sim':<6} {'score':<6} {'det':<6} {'date':<18} {'grp'}")
    print(header)
    for c in results:
        print(
            f"{str(c.get('url'))[:51]:<52} "
            f"{str(c.get('domain'))[:19]:<20} "
            f"{str(c.get('evidence_type'))[:15]:<16} "
            f"{str(c.get('event_match'))[:7]:<8} "
            f"{str(c.get('evidence_class'))[:14]:<15} "
            f"{str(c.get('nli_label'))[:13]:<14} "
            f"{str(c.get('nli_score'))[:6]:<7} "
            f"{str(c.get('raw_nli_label'))[:13]:<14} "
            f"{str(c.get('raw_nli_score'))[:6]:<7} "
            f"{str(c.get('similarity'))[:5]:<6} "
            f"{str(c.get('evidence_score'))[:5]:<6} "
            f"{str(c.get('detail_coverage'))[:5]:<6} "
            f"{str(c.get('date_status'))[:17]:<18} "
            f"{str(c.get('independence_group'))[:16]}"
        )

    # ------------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("E. SOURCE AGGREGATION (per source decision)")
    print("=" * 100)
    for d in agreement["source_decisions"]:
        print(f"  [{d['decision']:<15}] pub={d['independence_group'] or 'UNRESOLVED':<24} "
              f"chunks={d['chunk_count']} retrieval={d.get('retrieval_mode'):<15} "
              f"primary={d['is_primary']} article={d['has_article_text']} "
              f"snippet_only={d['is_snippet_only']} correction={d['has_correction']} "
              f"identity_resolved={d['identity_resolved']}")
        print(f"      best_chunk: {str(d.get('best_chunk'))[:130]}")

    print()
    print("=" * 100)
    print("F. INDEPENDENCE GROUPS")
    print("=" * 100)
    for g in agreement["group_decisions"]:
        print(f"  {g['independence_group']:<28} decision={g['decision']:<15} "
              f"sources={g['source_count']} primary={g['is_primary']} "
              f"article={g['has_article_text']} snippet_only={g['is_snippet_only']} "
              f"correction={g['has_correction']}")
        print(f"      domains={g['domains']}  best_url={str(g['best_url'])[:80]}")

    print()
    print("=" * 100)
    print("G. POLICY COUNTERS (source_agreement)")
    print("=" * 100)
    for key in (
        "total_sources", "total_chunks", "independence_groups",
        "independent_supporting", "independent_contradicting",
        "primary_supporting_groups", "primary_contradicting_groups",
        "article_supporting_groups", "article_contradicting_groups",
        "correction_groups", "primary_correction_groups",
        "snippet_only_sources", "article_sources",
        "unresolved_identity_sources", "verification_coverage",
        "agreement_score", "evidence_sufficiency",
        "supporting_sources", "contradicting_sources",
        "related_context_sources", "non_evidence_sources", "neutral_sources",
    ):
        print(f"  {key:<32} {agreement.get(key)}")

    print()
    print("=" * 100)
    print("H. DETERMINISTIC EVIDENCE FLOOR")
    print("=" * 100)
    print(f"  fusion_score        : {fusion}")
    print(f"  permitted verdicts  : {sorted(gate['permitted_verdicts'])}")
    print(f"  reason             : {gate['permitted_reason']}")
    print(f"  policy_data_valid  : {gate.get('policy_data_valid')}")
    print(f"  policy_data_errors : {gate.get('policy_data_errors')}")

    print()
    print("=" * 100)
    print("I. AGENTS")
    print("=" * 100)
    print("  --- verification agent ---")
    print(str(state.get("verification", ""))[:1200])
    print("  --- critic agent ---")
    print(str(state.get("feedback", ""))[:1200])

    print()
    print("=" * 100)
    print("J. FINAL VERDICT")
    print("=" * 100)
    for key in ("verdict", "confidence", "confidence_basis", "evidence_sufficiency",
                "source_agreement", "fusion_score", "independent_supporting",
                "independent_contradicting", "primary_supporting_groups",
                "total_sources", "independence_groups", "supporting_sources",
                "contradicting_sources", "context_chunks", "entailment_score",
                "contradiction_score", "verification_coverage",
                "verdict_overridden_by_evidence", "agents_agree"):
        print(f"  {key:<32} {verdict.get(key)}")
    print(f"  permitted_verdicts         {verdict.get('permitted_verdicts')}")
    print(f"  reasoning                  {str(verdict.get('reasoning'))[:200]}")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    OUT.write_text(json.dumps({
        "claim": CLAIM,
        "search_plans": state.get("search_plans"),
        "source_data": [
            {k: v for k, v in s.items() if k != "content"}
            for s in (state.get("source_data") or [])
        ],
        "scrape_failures": state.get("scrape_failures"),
        "chunks": [
            {
                "url": c.get("url"), "domain": c.get("domain"),
                "source_id": c.get("source_id"),
                "independence_group": c.get("independence_group"),
                "evidence_type": c.get("evidence_type"),
                "evidence_class": c.get("evidence_class"),
                "evidence_score": c.get("evidence_score"),
                "nli_label": c.get("nli_label"), "nli_score": c.get("nli_score"),
                "raw_nli_label": c.get("raw_nli_label"),
                "raw_nli_score": c.get("raw_nli_score"),
                "nli_status": c.get("nli_status"),
                "similarity": c.get("similarity"),
                "date_status": c.get("date_status"),
                "date_reason": c.get("date_reason"),
                "event_match": c.get("event_match"),
                "location_match": c.get("location_match"),
                "claim_has_location": c.get("claim_has_location"),
                "detail_coverage": c.get("detail_coverage"),
                "is_primary_source": c.get("is_primary_source"),
                "source_tier": c.get("source_tier"),
                "source_reliability": c.get("source_reliability"),
                "chunk": str(c.get("chunk"))[:600],
            }
            for c in results
        ],
        "agreement": {k: v for k, v in agreement.items()
                      if k not in ("source_decisions", "group_decisions")},
        "source_decisions": agreement["source_decisions"],
        "group_decisions": agreement["group_decisions"],
        "gate": gate,
        "fusion_score": fusion,
        "verification_output": state.get("verification"),
        "critic_output": state.get("feedback"),
        "verdict": {k: v for k, v in verdict.items()
                    if k not in ("verification_result", "critic_result")},
    }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
