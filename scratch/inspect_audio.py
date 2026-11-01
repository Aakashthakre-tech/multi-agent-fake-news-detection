"""scratch/inspect_audio.py - inspect the live audio verification response."""
import json
import urllib.request
from pathlib import Path

p = Path("audio_module/validation_samples/human_03_marathi_report.wav")
b = p.read_bytes()
bd = "----B"
body = (
    f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
    "Content-Type: audio/wav\r\n\r\n"
).encode() + b + f"\r\n--{bd}--\r\n".encode()
req = urllib.request.Request(
    "http://127.0.0.1:8001/verify/audio",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={bd}"},
)
d = json.loads(urllib.request.urlopen(req, timeout=900).read())
raw = d.get("raw", {})
print("raw keys:", sorted(raw.keys()))
print()
print("synthetic_media_detection:", json.dumps(d.get("synthetic_media_detection"), ensure_ascii=False)[:500])
print()
print("claim_extraction:", json.dumps(raw.get("claim_extraction"), ensure_ascii=False)[:800])
print()
print("factual_verification:", json.dumps(raw.get("factual_verification"), ensure_ascii=False)[:600])
print()
print("transcription:", str(raw.get("transcription"))[:300])
print()
print("consistency_analysis:", json.dumps(raw.get("consistency_analysis"), ensure_ascii=False)[:500])
print()
print("modality_specific:", json.dumps(d.get("modality_specific"), ensure_ascii=False)[:800])
