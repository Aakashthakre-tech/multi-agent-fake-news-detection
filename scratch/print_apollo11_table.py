"""scratch/print_apollo11_table.py - the per-source diagnostic table."""
import json
from pathlib import Path

d = json.loads(Path("scratch/apollo11_trace.json").read_text(encoding="utf-8"))

print("=" * 128)
print(f"CLAIM: {d['claim']}")
print("=" * 128)

print()
print("A. SEARCH QUERIES ISSUED")
for p in d.get("search_plans") or []:
    print(f"   [{p.get('label')}] intent={p.get('intent')} :: {p.get('query')}")

print()
print("B. SOURCE LEDGER (retrieval + article vs snippet)")
print(f"{'#':>2}  {'scrape_status':<30} {'evidence_type':<17} {'domain':<24} {'url'}")
for i, s in enumerate(d.get("source_data") or [], 1):
    print(f"{i:>2}  {str(s.get('scrape_status'))[:29]:<30} "
          f"{str(s.get('evidence_type'))[:16]:<17} {str(s.get('domain'))[:23]:<24} {s.get('url')}")

print()
print("C. SCRAPE FAILURES")
for f in d.get("scrape_failures") or []:
    print(f"   {f.get('domain')}: {f.get('reason')} | fallback={f.get('fallback_used')}")

print()
print("D. PER-CHUNK DIAGNOSTIC TABLE")
hdr = (f"{'url':<46} {'pub':<18} {'art?':<5} {'snip':<5} {'evt':<8} {'class':<15} "
       f"{'nli':<13} {'nliS':<7} {'sim':<6} {'score':<6} {'det':<6} {'date':<10} {'group'}")
print(hdr)
print("-" * len(hdr))
for c in d["chunks"]:
    t = c.get("evidence_type") or ""
    print(
        f"{str(c.get('url'))[:45]:<46} "
        f"{str(c.get('domain'))[:17]:<18} "
        f"{('Y' if t != 'search_snippet' else 'n'):<5} "
        f"{('Y' if t == 'search_snippet' else 'n'):<5} "
        f"{str(c.get('event_match'))[:7]:<8} "
        f"{str(c.get('evidence_class'))[:14]:<15} "
        f"{str(c.get('nli_label'))[:12]:<13} "
        f"{str(c.get('nli_score'))[:6]:<7} "
        f"{str(c.get('similarity'))[:5]:<6} "
        f"{str(c.get('evidence_score'))[:5]:<6} "
        f"{str(c.get('detail_coverage'))[:5]:<6} "
        f"{str(c.get('date_status'))[:9]:<10} "
        f"{str(c.get('independence_group'))[:16]}"
    )

print()
print("E. SOURCE DECISIONS (aggregation)")
for s in d.get("source_decisions") or []:
    print(f"   [{s['decision']:<15}] pub={s['independence_group'] or 'UNRESOLVED':<22} "
          f"chunks={s['chunk_count']} retrieval={s.get('retrieval_mode'):<15} "
          f"primary={s['is_primary']} article={s['has_article_text']} "
          f"snippet_only={s['is_snippet_only']}")
    print(f"       best: {str(s.get('best_chunk'))[:120]}")

print()
print("F. INDEPENDENCE GROUPS")
for g in d.get("group_decisions") or []:
    print(f"   {g['independence_group']:<26} decision={g['decision']:<15} "
          f"sources={g['source_count']} primary={g['is_primary']} "
          f"article={g['has_article_text']} snippet_only={g['is_snippet_only']}")

print()
print("G. POLICY COUNTERS")
for k in ("total_sources", "independence_groups", "independent_supporting",
          "independent_contradicting", "primary_supporting_groups",
          "primary_contradicting_groups", "article_supporting_groups",
          "article_contradicting_groups", "correction_groups",
          "primary_correction_groups", "snippet_only_sources", "article_sources",
          "unresolved_identity_sources", "verification_coverage",
          "agreement_score", "evidence_sufficiency", "supporting_sources",
          "contradicting_sources", "related_context_sources", "non_evidence_sources"):
    print(f"   {k:<32} {d['agreement'].get(k)}")

print()
print("H. DETERMINISTIC EVIDENCE FLOOR")
print(f"   fusion_score        : {d['fusion_score']}")
print(f"   permitted verdicts  : {sorted(d['gate']['permitted_verdicts'])}")
print(f"   reason              : {d['gate']['permitted_reason']}")
print(f"   policy_data_valid   : {d['gate'].get('policy_data_valid')}")

print()
print("I. FINAL VERDICT")
for k in ("verdict", "confidence", "confidence_basis", "evidence_sufficiency",
          "source_agreement", "fusion_score", "independent_supporting",
          "independent_contradicting", "primary_supporting_groups",
          "total_sources", "independence_groups", "supporting_sources",
          "contradicting_sources", "context_chunks", "entailment_score",
          "contradiction_score", "verification_coverage"):
    print(f"   {k:<32} {d['verdict'].get(k)}")
print(f"   permitted_verdicts        {d['verdict'].get('permitted_verdicts')}")
print(f"   reasoning                 {str(d['verdict'].get('reasoning'))[:220]}")
