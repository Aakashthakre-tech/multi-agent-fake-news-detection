"""scratch/smoke_final.py - final smoke test against the running final code."""
import json
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8001"
ROOT = Path(__file__).resolve().parent.parent


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=60) as r:
        return json.loads(r.read())


def post_multipart(path, filename, data, ctype):
    bd = "----F"
    body = (
        f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{bd}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={bd}"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


h = health()
print("HEALTH:", h["status"], "| keys:", h["api_keys_configured"])

VIDEO = ROOT / "video_module" / "ai_generated_data" / "synthetic" / "opensora" / "OpenSora_0.mp4"
d = post_multipart("/verify/video", VIDEO.name, VIDEO.read_bytes(), "video/mp4")
rr = d.get("research_report") or {}
print("\nVIDEO final code:")
print("  status           :", d["status"])
print("  verdict          :", d["factual_verification"]["verdict"])
print("  report present   :", bool(rr))
print("  report verdict   :", (rr.get("executive_summary") or {}).get("verdict"))
print("  research_skipped :", (rr.get("methodology") or {}).get("research_skipped"))
print("  skip reason      :", (rr.get("methodology") or {}).get("research_skipped_reason"))
print("  visual_classifier:", (d.get("visual_classification") or {}).get("predicted_label"),
      (d.get("visual_classification") or {}).get("confidence"))
print("  findings         :", {k: len(v or []) for k, v in (rr.get("findings") or {}).items()})
print("  gaps             :", len(rr.get("evidence_gaps") or []))
print("  agent model stage:", [s["tool_or_model"] for s in (d.get("pipeline_stages") or [])
                               if "Agent" in s.get("name", "")])
