"""Generate a short synthetic test video (~10s, 640x360, with audio) for timing runs."""
import sys, subprocess, tempfile
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"C:\Users\ASUS\Installation\AgenticAI\Multi_Agent_System")
OUT = ROOT / "scratch" / "test_short.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w, h, fps, n = 640, 360, 25, 250  # 10s
out = cv2.VideoWriter(str(OUT), fourcc, fps, (w, h))
try:
    for i in range(n):
        val = int((i / n) * 255)
        frame = np.full((h, w, 3), val, dtype=np.uint8)
        # draw some text so OCR has something to chew on
        cv2.putText(frame, f"NEWS {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        out.write(frame)
finally:
    out.release()
print("written", OUT, OUT.stat().st_size)