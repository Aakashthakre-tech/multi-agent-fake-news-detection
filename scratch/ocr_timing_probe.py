"""Measure real EasyOCR per-frame latency on the 117-frame video's actual frames."""
import sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Installation\AgenticAI\Multi_Agent_System")
sys.path.insert(0, str(ROOT))

import cv2
from image_module.ocr import extract_text, reader

FRAMES = ROOT / "video_module" / "outputs" / "tmpky3cjqt2" / "frames"
manifest = sorted(FRAMES.glob("frame_*.jpg"))
print(f"frames available: {len(manifest)}")

# Warm up + measure a single frame end-to-end (this is what run_ocr_on_frames calls)
t0 = time.time()
r = extract_text(str(manifest[0]))
t1 = time.time()
print(f"frame_000001: {t1 - t0:.2f}s  detections={len(r)}")

# Measure 5 more frames sequentially to estimate 117-frame cost
total = 0.0
for i in range(1, 6):
    t0 = time.time()
    extract_text(str(manifest[i]))
    total += time.time() - t0
avg = total / 5
print(f"avg frames 2-6: {avg:.2f}s/frame")
print(f"estimated 117 frames @ {avg:.2f}s = {117 * avg:.0f}s")

# Check the reader's device / gpu
import torch
print("cuda available:", torch.cuda.is_available())
try:
    print("reader device:", getattr(reader, "device", "?"))
except Exception as e:
    print("reader device err:", e)