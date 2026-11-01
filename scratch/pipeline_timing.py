"""End-to-end timing harness for the video pipeline.
Runs process_video_pipeline on a short and a long video, prints per-stage
elapsed times, and exercises the OCR frame budget cap.
"""
import sys, time, json, tempfile, shutil
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Installation\AgenticAI\Multi_Agent_System")
sys.path.insert(0, str(ROOT))

import cv2, numpy as np
from video_module.video_pipeline import process_video_pipeline


def make_video(path: Path, seconds: int, fps: int = 25, w: int = 640, h: int = 360):
    n = seconds * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for i in range(n):
            val = int((i / n) * 255)
            frame = np.full((h, w, 3), val, dtype=np.uint8)
            cv2.putText(frame, f"NEWS headline {i}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            out.write(frame)
    finally:
        out.release()
    return path


def run(name: str, video: Path, **kwargs):
    print(f"\n{'='*70}")
    print(f"RUN: {name}  |  {video.name}  {video.stat().st_size/1e6:.1f}MB")
    print(f"    kwargs: {kwargs}")
    t0 = time.time()
    summary = process_video_pipeline(video_path=video, **kwargs)
    total = time.time() - t0
    print(f"  stage_status: {json.dumps(summary['stage_status'], indent=None)}")
    print(f"  stage_timings: {json.dumps(summary['stage_timings'], indent=None)}")
    print(f"  ocr_summary:  {json.dumps(summary['ocr_summary'], indent=None)}")
    print(f"  TOTAL wall: {total:.1f}s")
    return summary


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tl_timing_"))
    try:
        short = tmp / "short.mp4"
        long_v = tmp / "long.mp4"
        make_video(short, seconds=10)
        make_video(long_v, seconds=120)

        # 1. SHORT video, default OCR budget (15 frames)
        run("SHORT default-budget", short, whisper_model="tiny")

        # 2. SHORT video, OCR cap explicitly small
        run("SHORT max_ocr_frames=3", short, whisper_model="tiny", max_ocr_frames=3)

        # 3. LONG video (120s -> ~120 frames), default budget should cap to 15
        run("LONG default-budget (should cap to 15)", long_v, whisper_model="tiny")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()