"""
scratch/smoke_final_text.py
==========================
The final text smoke test: four claims through the LIVE API.

Captures, per claim: final verdict, confidence and its basis, evidence
sufficiency, permitted verdicts, the supporting and contradicting sources with
their URLs, the deterministic gate reason, the processing status, and the
latency.

Nothing is forced: the outcome is whatever the evidence policy produces.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"
OUT = ROOT / "scratch" / "smoke_final_text.json"

CLAIMS = [
    ("apollo_1969", "NASA's Apollo 11 mission landed humans on the Moon in July 1969."),
    ("apollo_1970", "The Apollo 11 mission landed humans on the Moon on July 20, 1970."),
    ("biden_2024", "Joe Biden withdrew from the 2024 United States presidential race in July 2024."),
    ("vague", "NASA's Apollo program was the most successful space program in history."),
]


def verify(claim, timeout=2400):
    req = urllib.request.Request(
        BASE + "/verify/text",
        data=json.dumps({"claim": claim}).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.time() - started


def main():
    results = []
    for key, claim in CLAIMS:
        print("=" * 100, flush=True)
        print(f"[{key}] {claim}", flush=True)
        print("=" * 100, flush=True)
        try:
            payload, elapsed = verify(claim)
        except Exception as exc:
            print(f"  REQUEST FAILED: {exc}", flush=True)
            results.append({"key": key, "claim": claim, "error": str(exc)})
            continue

        fact = payload.get("factual_verification") or {}
        report = payload.get("research_report") or {}
        summary = report.get("executive_summary") or {}
        methodology = report.get("methodology") or {}
        findings = report.get("findings") or {}

        def sources(bucket):
            return [
                {
                    "source": f.get("source_name"),
                    "url": f.get("source_url"),
                    "strength": f.get("evidence_strength"),
                    "nli": f.get("nli_label"),
                }
                for f in (findings.get(bucket) or [])
            ]

        row = {
            "key": key,
            "claim": claim,
            "latency_sec": round(elapsed, 1),
            "status": payload.get("status"),
            "verdict": fact.get("verdict"),
            "confidence": fact.get("confidence"),
            "confidence_basis": fact.get("confidence_basis"),
            "evidence_sufficiency": fact.get("evidence_sufficiency"),
            "permitted_verdicts": fact.get("permitted_verdicts"),
            "independent_supporting": fact.get("independent_supporting"),
            "independent_contradicting": fact.get("independent_contradicting"),
            "primary_supporting_groups": fact.get("primary_supporting_groups"),
            "primary_contradicting_groups": fact.get("primary_contradicting_groups"),
            "article_sources": fact.get("article_supporting_groups"),
            "total_sources": fact.get("total_sources"),
            "entailment_score": fact.get("entailment_score"),
            "contradiction_score": fact.get("contradiction_score"),
            "source_agreement": fact.get("source_agreement"),
            "fusion_score": fact.get("fusion_score"),
            "gate_reason": fact.get("deterministic_gate_reason"),
            "supporting_sources": sources("supporting"),
            "contradicting_sources": sources("contradicting"),
            "search_queries": [q.get("query") for q in (methodology.get("search_queries") or [])],
            "scrape_failures": len(methodology.get("scrapes_failed") or []),
            "agent_status": summary.get("agent_status"),
            "limitations": summary.get("limitations"),
            "verdict_overridden_by_evidence": fact.get("verdict_overridden_by_evidence"),
        }
        results.append(row)

        print(f"  latency            : {row['latency_sec']}s", flush=True)
        print(f"  status             : {row['status']}", flush=True)
        print(f"  verdict            : {row['verdict']}  "
              f"confidence {row['confidence']} ({row['confidence_basis']})", flush=True)
        print(f"  sufficiency        : {row['evidence_sufficiency']}  "
              f"permitted {row['permitted_verdicts']}", flush=True)
        print(f"  independent sup/con: {row['independent_supporting']} / "
              f"{row['independent_contradicting']}  "
              f"(primary {row['primary_supporting_groups']} / "
              f"{row['primary_contradicting_groups']})", flush=True)
        print(f"  scores             : entailment={row['entailment_score']} "
              f"contradiction={row['contradiction_score']} "
              f"agreement={row['source_agreement']} fusion={row['fusion_score']}", flush=True)
        print(f"  gate reason        : {str(row['gate_reason'])[:220]}", flush=True)
        print(f"  supporting sources : {len(row['supporting_sources'])}", flush=True)
        for s in row["supporting_sources"][:4]:
            print(f"      {s['source']}  {s['url']}", flush=True)
        print(f"  contradicting      : {len(row['contradicting_sources'])}", flush=True)
        for s in row["contradicting_sources"][:4]:
            print(f"      {s['source']}  {s['url']}", flush=True)
        print(f"  scrape failures    : {row['scrape_failures']}", flush=True)
        if row.get("limitations"):
            print("  limitations        :", flush=True)
            for lim in row["limitations"][:4]:
                print(f"      - {str(lim)[:160]}", flush=True)
        print(flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'claim':<14}{'verdict':<12}{'conf':<8}{'basis':<22}{'latency':<10}sup/con")
    for r in results:
        if "verdict" not in r:
            print(f"{r['key']:<14}REQUEST FAILED: {r.get('error')}")
            continue
        print(f"{r['key']:<14}{str(r['verdict']):<12}{str(r['confidence']):<8}"
              f"{str(r['confidence_basis']):<22}{str(r['latency_sec']):<10}"
              f"{r['independent_supporting']}/{r['independent_contradicting']}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
