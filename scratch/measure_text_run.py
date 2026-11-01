"""
scratch/measure_text_run.py
===========================
Measurement harness for the text-verification pipeline.

It runs the REAL pipeline in-process and instruments the real functions, so
per-stage timings and call counts are measured, not inferred from logs. No
production module is modified.

Captured per run:
  * wall-clock total
  * retrieval calls and time
  * scrape calls, time, failures and retries
  * NLI calls and time
  * verdict, confidence basis, sufficiency, permitted verdicts
  * supporting / contradicting passages with publisher identity
  * independent-source counts and primary-article counts
  * the evidence-layer decision per source

Usage:  python scratch/measure_text_run.py "<claim>" [<label>]
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "scratch" / "measurements"


def main():
    claim = sys.argv[1] if len(sys.argv) > 1 else (
        "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
    )
    label = sys.argv[2] if len(sys.argv) > 2 else "run"

    import pipeline as pipeline_mod
    import evidence_analyzer as ea_mod
    import tools as tools_mod
    from evidence_fusion import calculate_source_agreement, calculate_fusion_score
    from final_verdict import deterministic_gate, generate_final_verdict

    stats = {
        "retrieval_calls": 0, "retrieval_sec": 0.0,
        "scrape_calls": 0, "scrape_sec": 0.0, "scrape_failures": 0,
        "nli_calls": 0, "nli_sec": 0.0,
    }
    retry_lines = []

    # ---- instrument the real functions -------------------------------
    real_search = pipeline_mod.search_with_diagnostics
    real_scrape_tool = pipeline_mod.scrape_url
    real_scrape = getattr(real_scrape_tool, "func", real_scrape_tool)
    real_nli = ea_mod.check_nli

    def timed_search(claim_text):
        t0 = time.time()
        stats["retrieval_calls"] += 1
        try:
            return real_search(claim_text)
        finally:
            stats["retrieval_sec"] += time.time() - t0

    def timed_scrape(url):
        t0 = time.time()
        stats["scrape_calls"] += 1
        try:
            out = real_scrape(url)
            if isinstance(out, str) and out.startswith("SCRAPE_FAILED"):
                stats["scrape_failures"] += 1
            return out
        finally:
            stats["scrape_sec"] += time.time() - t0

    def timed_nli(premise, hypothesis):
        t0 = time.time()
        stats["nli_calls"] += 1
        try:
            return real_nli(premise, hypothesis)
        finally:
            stats["nli_sec"] += time.time() - t0

    class _TimedTool:
        """Keeps the tool interface (the pipeline calls .invoke)."""

        def __init__(self, tool, fn):
            self._tool = tool
            self._fn = fn

        def invoke(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._tool, name)

    pipeline_mod.search_with_diagnostics = timed_search
    pipeline_mod.scrape_url = _TimedTool(real_scrape_tool, timed_scrape)
    ea_mod.check_nli = timed_nli

    # ---- run ---------------------------------------------------------
    t_start = time.time()
    state = pipeline_mod.run_research_pipeline(claim)
    total = time.time() - t_start

    results = state.get("evidence_analysis") or []
    agreement = calculate_source_agreement(results)
    fusion = calculate_fusion_score(results, agreement)
    gate = deterministic_gate(agreement, results)
    verdict = generate_final_verdict(
        results, agreement, fusion,
        state.get("verification", ""), state.get("feedback", ""),
    )

    summary = {
        "label": label,
        "claim": claim,
        "total_sec": round(total, 2),
        "retrieval_calls": stats["retrieval_calls"],
        "retrieval_sec": round(stats["retrieval_sec"], 2),
        "scrape_calls": stats["scrape_calls"],
        "scrape_sec": round(stats["scrape_sec"], 2),
        "scrape_failures": stats["scrape_failures"],
        "nli_calls": stats["nli_calls"],
        "nli_sec": round(stats["nli_sec"], 2),
        "verdict": verdict["verdict"],
        "confidence": verdict["confidence"],
        "confidence_basis": verdict.get("confidence_basis"),
        "evidence_sufficiency": verdict.get("evidence_sufficiency"),
        "permitted_verdicts": verdict.get("permitted_verdicts"),
        "gate_reason": verdict.get("deterministic_gate_reason"),
        "independent_supporting": verdict.get("independent_supporting"),
        "independent_contradicting": verdict.get("independent_contradicting"),
        "primary_supporting_groups": verdict.get("primary_supporting_groups"),
        "primary_contradicting_groups": verdict.get("primary_contradicting_groups"),
        "article_supporting_groups": verdict.get("article_supporting_groups"),
        "article_contradicting_groups": verdict.get("article_contradicting_groups"),
        "total_sources": verdict.get("total_sources"),
        "independence_groups": verdict.get("independence_groups"),
        "snippet_only_sources": agreement.get("snippet_only_sources"),
        "article_sources": agreement.get("article_sources"),
        "verification_coverage": agreement.get("verification_coverage"),
        "policy_data_valid": verdict.get("policy_data_valid"),
        "policy_data_errors": verdict.get("policy_data_errors"),
        "search_queries": [q.get("query") for q in (state.get("search_plans") or [])],
        "source_decisions": [
            {
                "decision": d["decision"],
                "group": d["independence_group"],
                "domain": d["domain"],
                "url": d["url"],
                "retrieval": d.get("retrieval_mode"),
                "primary": d["is_primary"],
                "article": d["has_article_text"],
                "snippet_only": d["is_snippet_only"],
                "score": d["source_score"],
            }
            for d in agreement["source_decisions"]
        ],
        "supporting_passages": [
            {
                "url": e.get("url"),
                "domain": e.get("domain"),
                "score": e.get("evidence_score"),
                "nli": e.get("chunk", "")[:260],
            }
            for e in verdict.get("supporting_evidence") or []
        ],
        "contradicting_passages": [
            {
                "url": e.get("url"),
                "domain": e.get("domain"),
                "score": e.get("evidence_score"),
                "nli": e.get("chunk", "")[:260],
            }
            for e in verdict.get("contradicting_evidence") or []
        ],
        "class_counts": {
            cls: sum(1 for r in results if r.get("evidence_class") == cls)
            for cls in ("SUPPORT", "CONTRADICTION", "RELATED_CONTEXT", "NON_EVIDENCE")
        },
        "passage_kinds": {
            kind: sum(1 for r in results if r.get("passage_kind") == kind)
            for kind in ("paragraph", "window")
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("=" * 96)
    print(f"[{label}] {claim}")
    print("=" * 96)
    print(f"  VERDICT          : {summary['verdict']}  "
          f"confidence={summary['confidence']} ({summary['confidence_basis']})")
    print(f"  sufficiency      : {summary['evidence_sufficiency']}  "
          f"permitted={summary['permitted_verdicts']}")
    print(f"  publishers       : {summary['independence_groups']} groups / "
          f"{summary['total_sources']} sources  "
          f"(primary supporting {summary['primary_supporting_groups']}, "
          f"primary contradicting {summary['primary_contradicting_groups']})")
    print(f"  article sources  : {summary['article_sources']}  "
          f"snippet-only {summary['snippet_only_sources']}")
    print(f"  classes          : {summary['class_counts']}  "
          f"passage kinds {summary['passage_kinds']}")
    print(f"  TIME total       : {summary['total_sec']}s")
    print(f"       retrieval   : {summary['retrieval_calls']} calls / "
          f"{summary['retrieval_sec']}s")
    print(f"       scraping    : {summary['scrape_calls']} calls / "
          f"{summary['scrape_sec']}s  failures={summary['scrape_failures']}")
    print(f"       NLI         : {summary['nli_calls']} calls / {summary['nli_sec']}s")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
