"""
scratch/apollo11_api_response.py
================================
Runs the Apollo 11 claim through the LIVE API and dumps the fields the UI
renders, then reports every place a 100% or 30% appears in the response.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"
CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
OUT = ROOT / "scratch" / "apollo11_api_response.json"


def find_numbers(obj, path="root", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_numbers(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_numbers(v, f"{path}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if abs(float(obj) - 100.0) < 1e-9 or abs(float(obj) - 30.0) < 1e-9:
            out.append((path, obj))
    return out


def main():
    req = urllib.request.Request(
        BASE + "/verify/text",
        data=json.dumps({"claim": CLAIM}).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    fact = d["factual_verification"]
    print("=" * 88)
    print("API RESPONSE - factual_verification (what the UI renders)")
    print("=" * 88)
    for k in sorted(fact.keys()):
        v = fact[k]
        if isinstance(v, str) and len(v) > 120:
            v = v[:120] + "..."
        print(f"  {k:<34} {v}")

    print()
    print("=" * 88)
    print("EVERY 100 / 30 IN THE RESPONSE")
    print("=" * 88)
    for path, value in find_numbers(d):
        print(f"  {value!r:>8}  at {path}")

    print()
    print("=" * 88)
    print("RESEARCH REPORT - executive summary")
    print("=" * 88)
    es = (d.get("research_report") or {}).get("executive_summary") or {}
    for k in sorted(es.keys()):
        v = es[k]
        if isinstance(v, str) and len(v) > 140:
            v = v[:140] + "..."
        print(f"  {k:<30} {v}")

    print()
    print("=" * 88)
    print("TOP FINDINGS (first 5)")
    print("=" * 88)
    for bucket in ("supporting", "contradicting", "contextual", "unresolved"):
        items = ((d.get("research_report") or {}).get("findings") or {}).get(bucket) or []
        print(f"  {bucket}: {len(items)}")
        for f in items[:2]:
            print(f"     {f.get('source_name')} | strength={f.get('strength_label')} "
                  f"({f.get('evidence_strength')}) nli={f.get('nli_label')}/{f.get('nli_score')} "
                  f"date={f.get('date_status')} evt={f.get('event_match')}")
            print(f"       {str(f.get('excerpt'))[:110]}")
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
