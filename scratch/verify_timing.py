"""Time the full verify_video() pipeline including classifier + evidence research."""
import sys, time, json, tempfile, shutil
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Installation\AgenticAI\Multi_Agent_System")
sys.path.insert(0, str(ROOT))

import cv2, numpy as np
from video_module.video_verifier import verify_video


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


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tl_verify_"))
    try:
        short = tmp / "short.mp4"
        make_video(short, seconds=10)

        print("=" * 70)
        print("FULL verify_video() — SHORT video, classifier ON, research ON")
        t0 = time.time()
        report = verify_video(
            video_path=short,
            enable_classifier=True,
            verify_claims=True,
            whisper_model="tiny",
            max_ocr_frames=15,
        )
        print(f"  TOTAL: {time.time()-t0:.1f}s")
        print(f"  step_timings: {json.dumps(report['step_timings'])}")
        print(f"  final_evidence_verdict: {report['final_evidence_verdict']}")
        print(f"  visual available: {report['visual_classification'].get('available')}")
        print(f"  visual error: {report['visual_classification'].get('error')}")
        print(f"  verified_claims: {len(report['verified_claims'])}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()