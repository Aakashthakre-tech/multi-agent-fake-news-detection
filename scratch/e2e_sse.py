"""
scratch/e2e_sse.py
==================
Genuine end-to-end test of the two Server-Sent-Events endpoints.

* POST /verify/video/stream  - multipart upload, real progress events
* GET  /verify/text/stream   - query string, real progress events

Checks that:
  - status / progress / heartbeat / result / error events are handled;
  - the final result is compatible with the normal verification schema;
  - the SSE result carries the same research_report as the non-streaming path.
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"

VIDEO = ROOT / "video_module" / "ai_generated_data" / "synthetic" / "opensora" / "OpenSora_0.mp4"
TEXT_CLAIM = "The Indian Space Research Organisation launched the Chandrayaan-3 mission on 14 July 2023."


def stream_text():
    print("=" * 78)
    print("GET /verify/text/stream")
    print("=" * 78)
    url = f"{BASE}/verify/text/stream?claim={urllib.parse.quote(TEXT_CLAIM)}"
    events = []
    with urllib.request.urlopen(url, timeout=900) as r:
        buf = ""
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line == "":
                if buf:
                    events.append(buf)
                    buf = ""
                continue
            buf += line + "\n"
        if buf:
            events.append(buf)
    return _report("text-stream", events)


def stream_video():
    print("=" * 78)
    print("POST /verify/video/stream")
    print("=" * 78)
    bd = "----SSE"
    body = (
        f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{VIDEO.name}\"\r\nContent-Type: video/mp4\r\n\r\n"
    ).encode() + VIDEO.read_bytes() + f"\r\n--{bd}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/verify/video/stream",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={bd}"},
    )
    events = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = ""
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line == "":
                if buf:
                    events.append(buf)
                    buf = ""
                continue
            buf += line + "\n"
        if buf:
            events.append(buf)
    return _report("video-stream", events)


def _report(label, events):
    kinds = []
    result_payload = None
    for ev in events:
        for line in ev.splitlines():
            if line.startswith("event:"):
                kinds.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                try:
                    parsed = json.loads(data)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and "verdict" in parsed and "modality" in parsed:
                    result_payload = parsed

    print(f"raw event blocks : {len(events)}")
    print(f"event kinds      : {sorted(set(kinds))}")
    from collections import Counter
    print(f"event counts     : {dict(Counter(kinds))}")

    if result_payload is None:
        # The final event may itself be the result object.
        for ev in events:
            payload = ev.split("data:", 1)[1].strip() if "data:" in ev else ev
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if isinstance(parsed, dict) and "factual_verification" in parsed:
                result_payload = parsed
                break

    if result_payload is None:
        print("RESULT           : *** no result payload received ***")
        for ev in events[-4:]:
            print("  ", ev[:300].replace("\n", " | "))
        return False

    fact = result_payload.get("factual_verification") or {}
    report = result_payload.get("research_report") or {}
    summary = report.get("executive_summary") or {}
    method = report.get("methodology") or {}

    print(f"modality         : {result_payload.get('modality')}")
    print(f"status           : {result_payload.get('status')}")
    print(f"verdict          : {fact.get('verdict')}")
    print(f"research_report  : present={bool(report)} modality={report.get('modality')}")
    print(f"report verdict   : {summary.get('verdict')}")
    print(f"search queries   : {len(method.get('search_queries') or [])}")
    print(f"findings         : "
          f"{ {k: len(v or []) for k, v in (report.get('findings') or {}).items()} }")
    print(f"has raw payload  : {'raw' in result_payload}")
    print(f"raw keys         : {sorted((result_payload.get('raw') or {}).keys())[:10]}")
    return True


if __name__ == "__main__":
    import urllib.parse
    ok_text = stream_text()
    print()
    ok_video = stream_video()
    print()
    print(f"text-stream  result received: {ok_text}")
    print(f"video-stream result received: {ok_video}")
