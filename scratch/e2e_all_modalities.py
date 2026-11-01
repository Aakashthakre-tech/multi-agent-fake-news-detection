"""
scratch/e2e_all_modalities.py
============================
Genuine end-to-end run against the live API on http://127.0.0.1:8001.

One real request per modality. Prints what the API actually returned so the
result can be inspected rather than assumed.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"

IMAGE = ROOT / "ai_image_detection" / "benchmark" / "ai" / "ai_1.jpg"
AUDIO = ROOT / "audio_module" / "validation_samples" / "human_03_marathi_report.wav"
VIDEO = ROOT / "video_module" / "ai_generated_data" / "synthetic" / "opensora" / "OpenSora_0.mp4"


def post_json(path, payload, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_multipart(path, field, filename, file_bytes, content_type, timeout=1800):
    boundary = "----TruthLensE2EBoundary"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
    ).encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += file_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def summarise(label, data, elapsed):
    print()
    print("=" * 78)
    print(f"{label}   ({elapsed:.1f}s)")
    print("=" * 78)

    fact = data.get("factual_verification") or {}
    print(f"status              : {data.get('status')}")
    print(f"verdict             : {fact.get('verdict')}")
    print(f"confidence          : {fact.get('confidence')}")
    print(f"evidence_sufficiency: {fact.get('evidence_sufficiency')}")
    print(f"indep support/contra: {fact.get('independent_supporting')} / "
          f"{fact.get('independent_contradicting')}")

    synth = data.get("synthetic_media_detection")
    if synth:
        print(f"ai-generation       : signal={synth.get('signal')} "
              f"model={synth.get('model')} conf={synth.get('confidence')}")
        print(f"manipulation        : {synth.get('manipulation_status') or 'n/a'}")

    vis = data.get("visual_classification")
    if vis:
        print(f"visual classifier   : {vis.get('predicted_label')} "
              f"conf={vis.get('confidence')} available={vis.get('available')}")

    report = data.get("research_report") or {}
    summary = report.get("executive_summary") or {}
    method = report.get("methodology") or {}
    findings = report.get("findings") or {}

    print(f"report modality     : {report.get('modality')}")
    print(f"report verdict      : {summary.get('verdict')}")
    print(f"input claim         : {str(summary.get('input_claim'))[:110]}")
    print(f"attempted           : {str(summary.get('attempted'))[:180]}")
    for i, kf in enumerate(summary.get("key_findings") or [], 1):
        print(f"  key finding {i}     : {kf[:170]}")
    if summary.get("why_this_verdict"):
        print(f"why verdict         : {summary['why_this_verdict'][:200]}")
    if summary.get("agent_narrative"):
        print(f"agent narrative     : {summary['agent_narrative'][:160]}...")
    if summary.get("critic_review"):
        print(f"critic review       : {summary['critic_review'][:160]}")
    for i, lim in enumerate(summary.get("limitations") or [], 1):
        print(f"  limitation {i}      : {lim[:170]}")

    counts = {k: len(v or []) for k, v in findings.items()}
    print(f"finding counts      : {counts}")
    for bucket in ("supporting", "contradicting", "contextual"):
        for f in (findings.get(bucket) or [])[:2]:
            print(f"  [{bucket}] {f.get('source_name')} | {f.get('source_url')}")
            print(f"       strength={f.get('strength_label')} "
                  f"({f.get('evidence_strength')}) nli={f.get('nli_label')} "
                  f"excerpt={str(f.get('excerpt'))[:90]}")

    print(f"search queries      : {len(method.get('search_queries') or [])}")
    for q in (method.get("search_queries") or [])[:5]:
        print(f"   - [{q.get('label')}] {q.get('query')}")
    print(f"sources retrieved   : {method.get('sources_retrieved')}")
    print(f"articles scraped    : {method.get('articles_scraped')}")
    print(f"scrape failures     : {len(method.get('scrapes_failed') or [])}")
    print(f"evidence passages   : {method.get('evidence_passages')}")
    print(f"nli evaluated       : {method.get('nli_evaluated')}")
    print(f"research skipped    : {method.get('research_skipped')} "
          f"{method.get('research_skipped_reason') or ''}")
    for st in method.get("stages") or []:
        print(f"   stage: {st.get('name')} [{st.get('status')}] {st.get('detail')}")

    print(f"claim results       : {len(report.get('claims') or [])}")
    for c in (report.get("claims") or [])[:3]:
        print(f"   claim: {str(c.get('claim_text'))[:90]} -> {c.get('verdict')}")
    for g in report.get("evidence_gaps") or []:
        print(f"   gap [{g.get('status')}]: {g.get('question')}")
    print(f"error               : {data.get('error')}")
    print(f"warnings            : {data.get('warnings')}")


def main():
    results = {}

    print("### TEXT ###")
    t0 = time.time()
    data = post_json("/verify/text", {"claim": "The Indian Space Research Organisation launched the Chandrayaan-3 mission on 14 July 2023."})
    summarise("TEXT", data, time.time() - t0)
    results["text"] = data

    if IMAGE.exists():
        print("\n\n### IMAGE ###")
        t0 = time.time()
        data = post_multipart("/verify/image", "file", IMAGE.name, IMAGE.read_bytes(), "image/jpeg")
        summarise("IMAGE", data, time.time() - t0)
        results["image"] = data
    else:
        print(f"\n[skip] image fixture missing: {IMAGE}")

    if AUDIO.exists():
        print("\n\n### AUDIO ###")
        t0 = time.time()
        data = post_multipart("/verify/audio", "file", AUDIO.name, AUDIO.read_bytes(), "audio/wav")
        summarise("AUDIO", data, time.time() - t0)
        results["audio"] = data
    else:
        print(f"\n[skip] audio fixture missing: {AUDIO}")

    if VIDEO.exists():
        print("\n\n### VIDEO ###")
        t0 = time.time()
        data = post_multipart("/verify/video", "file", VIDEO.name, VIDEO.read_bytes(), "video/mp4")
        summarise("VIDEO", data, time.time() - t0)
        results["video"] = data
    else:
        print(f"\n[skip] video fixture missing: {VIDEO}")

    out = ROOT / "scratch" / "e2e_results.json"
    serialisable = {
        k: {
            "modality": v.get("modality"),
            "status": v.get("status"),
            "verdict": (v.get("factual_verification") or {}).get("verdict"),
            "report": v.get("research_report"),
        }
        for k, v in results.items()
    }
    out.write_text(json.dumps(serialisable, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved condensed results -> {out}")


if __name__ == "__main__":
    main()
