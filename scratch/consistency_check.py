"""
scratch/consistency_check.py
===========================
Section 3: checks that the TERMINAL (in-process pipeline) and the API
response agree, and that the API response's own fields are internally
consistent.

For each claim it compares the measured in-process run against a live
/verify/text call and reports any divergence in:
  * the final verdict
  * the confidence basis
  * the evidence sufficiency
  * the permitted verdicts
  * the independent-source counts
  * the primary-article counts

It also validates the API response against the deterministic gate, i.e. that
the verdict really is inside the permitted set and that the policy diagnostics
are coherent.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MEAS = ROOT / "scratch" / "measurements"
BASE = "http://127.0.0.1:8001"

VALID = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}


def api_verify(claim, timeout=1800):
    req = urllib.request.Request(
        BASE + "/verify/text",
        data=json.dumps({"claim": claim}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main():
    keys = sys.argv[1:] or sorted(p.stem for p in MEAS.glob("*.json"))
    rows = []

    for key in keys:
        path = MEAS / f"{key}.json"
        if not path.exists():
            print(f"[{key}] no measurement file - skipped")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))

        print("=" * 96)
        print(f"[{key}] {m['claim']}")
        print("=" * 96)

        # ---- terminal side -------------------------------------------
        print(f"  TERMINAL verdict={m['verdict']} conf={m['confidence']} "
              f"({m['confidence_basis']}) sufficiency={m['evidence_sufficiency']}")
        print(f"           permitted={m['permitted_verdicts']} "
              f"indep_sup={m['independent_supporting']} "
              f"indep_contra={m['independent_contradicting']} "
              f"primary_sup={m['primary_supporting_groups']} "
              f"article_src={m['article_sources']} snippet_only={m['snippet_only_sources']}")

        # ---- API side ------------------------------------------------
        try:
            d = api_verify(m["claim"])
        except Exception as exc:
            print(f"  API call FAILED: {str(exc)[:160]}")
            rows.append((key, m, None, ["api call failed"]))
            continue

        f = d["factual_verification"]
        print(f"  API      verdict={f.get('verdict')} conf={f.get('confidence')} "
              f"({f.get('confidence_basis')}) sufficiency={f.get('evidence_sufficiency')}")
        print(f"           permitted={f.get('permitted_verdicts')} "
              f"indep_sup={f.get('independent_supporting')} "
              f"indep_contra={f.get('independent_contradicting')} "
              f"primary_sup={f.get('primary_supporting_groups')} "
              f"article_src={f.get('article_supporting_groups')} "
              f"snippet_only={f.get('snippet_only_sources')}")

        problems = []

        # ---- internal consistency of the API response -----------------
        if f.get("verdict") not in VALID:
            problems.append(f"verdict '{f.get('verdict')}' is outside the permitted set")
        permitted = f.get("permitted_verdicts") or []
        if f.get("verdict") and permitted and f["verdict"] not in permitted:
            problems.append(
                f"verdict '{f['verdict']}' is not in the permitted set {permitted}"
            )
        if not f.get("policy_data_valid", True):
            problems.append(f"policy data invalid: {f.get('policy_data_errors')}")

        # Evidence counts must be coherent with the verdict.
        sup = f.get("independent_supporting") or 0
        con = f.get("independent_contradicting") or 0
        if f.get("verdict") == "REAL" and sup < 2:
            problems.append(f"REAL with only {sup} independent supporting publisher(s)")
        if f.get("verdict") == "FAKE" and con < 2:
            problems.append(f"FAKE with only {con} independent contradicting publisher(s)")
        if f.get("verdict") in ("REAL", "FAKE") and f.get("primary_supporting_groups") == 0 \
                and f.get("primary_contradicting_groups") == 0:
            problems.append("a settled verdict with no primary publisher group")

        # The reported confidence must be labelled.
        if f.get("confidence") is not None and not f.get("confidence_basis"):
            problems.append("confidence reported without a confidence_basis")

        # The research report must agree with the top-level verdict.
        es = ((d.get("research_report") or {}).get("executive_summary") or {})
        if es.get("verdict") and f.get("verdict") and es["verdict"] != f["verdict"]:
            problems.append(
                f"report verdict '{es['verdict']}' != factual verdict '{f['verdict']}'"
            )
        pd = es.get("policy_diagnostics") or {}
        if pd:
            if pd.get("verdict_restricted") and f.get("verdict") != "UNVERIFIED":
                problems.append("policy diagnostics say the verdict was restricted, "
                                f"but the verdict is {f.get('verdict')}")

        # Supporting/contradicting passages must exist for the stated verdict.
        sup_passages = ((d.get("research_report") or {}).get("findings") or {}).get("supporting") or []
        con_passages = ((d.get("research_report") or {}).get("findings") or {}).get("contradicting") or []
        if f.get("verdict") == "REAL" and not sup_passages:
            problems.append("REAL but the report shows no supporting passage")
        if f.get("verdict") == "FAKE" and not con_passages:
            problems.append("FAKE but the report shows no contradicting passage")

        # ---- terminal vs API ------------------------------------------
        api_m = {
            "verdict": f.get("verdict"),
            "confidence_basis": f.get("confidence_basis"),
            "evidence_sufficiency": f.get("evidence_sufficiency"),
            "permitted_verdicts": f.get("permitted_verdicts"),
            "independent_supporting": f.get("independent_supporting"),
            "independent_contradicting": f.get("independent_contradicting"),
            "primary_supporting_groups": f.get("primary_supporting_groups"),
        }
        term_m = {
            "verdict": m["verdict"],
            "confidence_basis": m["confidence_basis"],
            "evidence_sufficiency": m["evidence_sufficiency"],
            "permitted_verdicts": m["permitted_verdicts"],
            "independent_supporting": m["independent_supporting"],
            "independent_contradicting": m["independent_contradicting"],
            "primary_supporting_groups": m["primary_supporting_groups"],
        }
        divergent = {k: (term_m[k], api_m[k]) for k in term_m if term_m[k] != api_m[k]}
        if divergent:
            problems.append(f"terminal/API divergence: {divergent}")

        if problems:
            print("  PROBLEMS:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  CONSISTENT: no divergence, no incoherent field.")
        rows.append((key, m, f, problems))

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for key, m, f, problems in rows:
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"  {key:32s} terminal={m['verdict']:<11} "
              f"api={(f or {}).get('verdict', 'n/a'):<11} {status}")
    return rows


if __name__ == "__main__":
    main()
