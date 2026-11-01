"""
Tests for TruthLensAI Video Module - Phase 1.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from video_module.video_pipeline import (
    FFmpegNotFoundError,
    check_ffmpeg_availability,
    deduplicate_ocr_observations,
    detect_disagreements_and_flags,
    extract_audio,
    extract_dates_from_text,
    extract_frames,
    extract_numbers_from_text,
    fuse_whisper_and_ocr,
    get_safe_output_dir,
    get_whisper_recommendations,
    normalize_text_conservative,
    prepare_candidate_claims,
    prepare_truthlens_claim_payload,
    process_video_pipeline,
    run_ocr_on_frames,
    transcribe_audio,
    validate_video,
)


def create_synthetic_video(file_path: Path, num_frames: int = 20, fps: float = 10.0, width: int = 64, height: int = 64) -> Path:
    """Helper to generate a lightweight valid test video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    try:
        for i in range(num_frames):
            # Create a simple colored frame with varying intensity
            val = int((i / max(1, num_frames)) * 255)
            frame = np.full((height, width, 3), val, dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()
    return file_path


class TestVideoPipeline(unittest.TestCase):
    """Test suite for video_pipeline components."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Invalid or missing input path
    def test_invalid_or_missing_input_path(self):
        non_existent = self.temp_path / "does_not_exist.mp4"
        with self.assertRaises(FileNotFoundError):
            validate_video(non_existent)

        # Directory path instead of file
        with self.assertRaises(ValueError):
            validate_video(self.temp_path)

    # 2. Corrupt or unreadable file
    def test_corrupt_media_handling(self):
        corrupt_file = self.temp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"not a valid video payload")
        with self.assertRaises(ValueError):
            validate_video(corrupt_file)

    # 3. Successful validation, frame extraction and timestamp recording
    def test_frame_extraction_and_timestamps(self):
        video_path = self.temp_path / "synthetic.avi"
        create_synthetic_video(video_path, num_frames=20, fps=10.0)

        # Validation
        meta = validate_video(video_path)
        self.assertEqual(meta["width"], 64)
        self.assertEqual(meta["height"], 64)
        self.assertAlmostEqual(meta["fps"], 10.0, places=1)
        self.assertAlmostEqual(meta["duration_sec"], 2.0, places=1)

        # Frame extraction with sample_interval=1.0s (should sample at ~0s and ~1s)
        frames_dir = self.temp_path / "frames"
        manifest = extract_frames(video_path, frames_dir, sample_interval=1.0)

        self.assertGreaterEqual(len(manifest), 2)
        saved_files = list(frames_dir.glob("*.jpg"))
        self.assertEqual(len(saved_files), len(manifest))

        # Check timestamp ordering and filenames
        timestamps = [item["timestamp_sec"] for item in manifest]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(manifest[0]["filename"], "frame_000001.jpg")
        self.assertTrue((frames_dir / manifest[0]["filename"]).exists())

    # 4. Video without audio
    def test_video_without_audio(self):
        video_path = self.temp_path / "no_audio.avi"
        create_synthetic_video(video_path, num_frames=10, fps=10.0)

        audio_dest = self.temp_path / "out_audio.wav"

        # Explicitly marked as has_audio_stream=False
        result = extract_audio(video_path, audio_dest, has_audio_stream=False)
        self.assertIsNone(result)

        # Transcription on None should gracefully skip
        trans_res = transcribe_audio(None)
        self.assertEqual(trans_res["status"], "skipped")
        self.assertEqual(trans_res["text"], "")

    # 5. Clear handling of missing FFmpeg executable
    def test_missing_ffmpeg_handling(self):
        with patch("shutil.which", return_value=None):
            ffmpeg_ok, ffprobe_ok, instructions = check_ffmpeg_availability()
            self.assertFalse(ffmpeg_ok)
            self.assertFalse(ffprobe_ok)
            self.assertIn("winget install Gyan.FFmpeg", instructions)

            video_path = self.temp_path / "test.avi"
            create_synthetic_video(video_path, num_frames=5, fps=10.0)
            audio_dest = self.temp_path / "audio.wav"

            with self.assertRaises(FFmpegNotFoundError):
                extract_audio(video_path, audio_dest, has_audio_stream=True)

    # 6. Audio extraction and transcript output (mocked Whisper & FFmpeg)
    def test_audio_extraction_and_transcription_pipeline(self):
        video_path = self.temp_path / "interview.avi"
        create_synthetic_video(video_path, num_frames=20, fps=10.0)

        out_dir = self.temp_path / "custom_outputs"

        # Mock FFmpeg and FFprobe execution
        def fake_ffmpeg_run(cmd, capture_output=True, text=True, check=False):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stderr = ""
            if cmd[0] == "ffprobe":
                mock_proc.stdout = json.dumps({
                    "streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                    "format": {"duration": "2.0"},
                })
            else:
                # cmd[0] == "ffmpeg"
                audio_path = Path(cmd[-1])
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"RIFF dummy wav content")
                mock_proc.stdout = ""
            return mock_proc

        # Mock Whisper model transcription
        fake_whisper_result = {
            "text": "This is a verified test statement.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.25, "text": "This is a"},
                {"id": 1, "start": 1.25, "end": 2.0, "text": "verified test statement."},
            ],
        }
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = fake_whisper_result

        with patch("shutil.which", return_value="C:\\dummy\\ffmpeg.exe"), \
             patch("subprocess.run", side_effect=fake_ffmpeg_run), \
             patch("video_module.video_pipeline.get_whisper_model", return_value=mock_whisper_model):

            summary = process_video_pipeline(
                video_path=video_path,
                output_dir=out_dir,
                sample_interval=1.0,
                whisper_model="base",
            )

            # Assert pipeline summary fields
            self.assertEqual(summary["stage_status"]["validation"], "success")
            self.assertEqual(summary["stage_status"]["frame_extraction"], "success")
            self.assertEqual(summary["stage_status"]["audio_extraction"], "success")
            self.assertEqual(summary["stage_status"]["transcription"], "success")

            # Check generated files
            self.assertTrue((out_dir / "frames.json").exists())
            self.assertTrue((out_dir / "audio.wav").exists())
            self.assertTrue((out_dir / "transcript.txt").exists())
            self.assertTrue((out_dir / "transcript.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())

            # Verify transcript contents
            txt_content = (out_dir / "transcript.txt").read_text(encoding="utf-8")
            self.assertEqual(txt_content, "This is a verified test statement.")

            json_content = json.loads((out_dir / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(len(json_content), 2)
            self.assertEqual(json_content[0]["text"], "This is a")

    # 7. Safe non-destructive output folder creation
    def test_safe_output_dir_incrementation(self):
        base_dir = self.temp_path / "outputs"
        base_dir.mkdir(parents=True, exist_ok=True)

        first = get_safe_output_dir(base_dir, "my_video")
        (first / "file.txt").write_text("existing")

        second = get_safe_output_dir(base_dir, "my_video")
        self.assertNotEqual(str(first), str(second))
        self.assertTrue(second.name.startswith("my_video_"))

    # 8. Whisper candidate recommendations
    def test_whisper_candidate_recommendations(self):
        recs = get_whisper_recommendations()
        self.assertIn("mr", recs)
        self.assertIn("hi", recs)
        self.assertEqual(recs["mr"]["candidate_model"], "large-v3")
        self.assertEqual(recs["hi"]["candidate_model"], "large-v3")
        self.assertIn("Not measured", recs["mr"]["measured_accuracy"])

    # 9. OCR processing on frames (mocked OCR function)
    def test_run_ocr_on_frames_mocked(self):
        frames_dir = self.temp_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy frame files
        for i in range(1, 4):
            (frames_dir / f"frame_{i:06d}.jpg").write_bytes(b"dummy image bytes")

        manifest = [
            {"frame_index": 1, "filename": "frame_000001.jpg", "timestamp_sec": 0.0},
            {"frame_index": 2, "filename": "frame_000002.jpg", "timestamp_sec": 1.0},
            {"frame_index": 3, "filename": "frame_000003.jpg", "timestamp_sec": 2.0},
        ]

        def fake_ocr(img_path):
            if "frame_000002" in img_path:
                return [
                    {"text": "25 March 2023", "confidence": 0.95, "box": [[0, 0], [10, 0], [10, 5], [0, 5]]},
                    {"text": "LIVE", "confidence": 0.99, "box": [[12, 0], [20, 0], [20, 5], [12, 5]]},
                ]
            return []

        res = run_ocr_on_frames(manifest, frames_dir, ocr_stride=1, ocr_func=fake_ocr)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["processed_frames_count"], 3)
        self.assertEqual(res["frames_with_text_count"], 1)
        self.assertEqual(len(res["observations"]), 1)
        self.assertEqual(res["observations"][0]["frame_index"], 2)
        self.assertEqual(res["observations"][0]["timestamp_sec"], 1.0)
        self.assertIn("25 March 2023", res["observations"][0]["text"])

    # 10. OCR error resilience and missing engine handling
    def test_ocr_error_resilience_and_missing_engine(self):
        frames_dir = self.temp_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_000001.jpg").write_bytes(b"dummy")

        manifest = [{"frame_index": 1, "filename": "frame_000001.jpg", "timestamp_sec": 0.0}]

        # Error in OCR function on individual frame
        def failing_ocr(path):
            raise RuntimeError("OCR hardware driver failure")

        res = run_ocr_on_frames(manifest, frames_dir, ocr_func=failing_ocr)
        self.assertEqual(res["frames_with_text_count"], 0)
        self.assertTrue(any("OCR error" in w for w in res["warnings"]))

        # Missing OCR function altogether
        res_none = run_ocr_on_frames(manifest, frames_dir, ocr_func=None)
        # Should gracefully return skipped or report status without unhandled crash
        self.assertIn(res_none["status"], ["skipped", "partial", "success"])

    # 11. Deduplication of repeated text and preservation of distinct text
    def test_deduplicate_ocr_observations(self):
        raw_obs = [
            {"frame_index": 1, "timestamp_sec": 0.0, "text": "25 March 2023 LIVE abp", "confidence": 0.95, "raw_items": []},
            {"frame_index": 2, "timestamp_sec": 1.0, "text": "25 March 2023 LIVE abp", "confidence": 0.98, "raw_items": []},
            {"frame_index": 3, "timestamp_sec": 2.0, "text": "25 March 2023 LIVE abp", "confidence": 0.96, "raw_items": []},
            {"frame_index": 4, "timestamp_sec": 10.0, "text": "BREAKING: Session Concluded in Assembly", "confidence": 0.90, "raw_items": []},
        ]

        deduped = deduplicate_ocr_observations(raw_obs, similarity_threshold=0.75, time_window_sec=5.0)
        # The first 3 should merge into 1 span, the 4th is distinct text
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["start_sec"], 0.0)
        self.assertEqual(deduped[0]["end_sec"], 2.0)
        self.assertEqual(deduped[0]["frame_references"], [1, 2, 3])
        self.assertEqual(deduped[1]["frame_references"], [4])
        self.assertIn("BREAKING", deduped[1]["text"])

    # 12. Disagreement and review flags detection
    def test_disagreement_and_review_flags(self):
        # Date mismatch
        speech_date = "Speech refers to events on 24 March 2023 in Mumbai."
        ocr_date = "Banner overlay displays 25 March 2023 LIVE."
        disags, flags = detect_disagreements_and_flags(speech_date, ocr_date, time_overlap=True)
        self.assertTrue(any("date_mismatch" in d for d in disags))
        self.assertTrue(any("date_discrepancy" in f for f in flags))

        # Number mismatch
        speech_num = "Govt announces 50 percent hike."
        ocr_num = "Ticker: 25 percent hike approved."
        disags_num, flags_num = detect_disagreements_and_flags(speech_num, ocr_num, time_overlap=True)
        self.assertTrue(any("number_mismatch" in d for d in disags_num))
        self.assertTrue(any("number_discrepancy" in f for f in flags_num))

        # Repetition loop flag
        repeated_speech = "talk talk talk talk talk talk"
        _, rep_flags = detect_disagreements_and_flags(repeated_speech, "normal ocr", time_overlap=True)
        self.assertTrue(any("suspected_whisper_repetition_loop" in f for f in rep_flags))

    # 13. Fusion of Whisper and OCR with provenance attribution
    def test_fuse_whisper_and_ocr(self):
        whisper_res = {
            "status": "success",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "Congress protests today across Maharashtra"},
                {"id": 1, "start": 3.0, "end": 5.0, "text": "Assembly session last day debate"},
            ]
        }
        ocr_res = {
            "status": "success",
            "observations": [
                {"frame_index": 1, "timestamp_sec": 1.0, "text": "25 March 2023 LIVE abp", "confidence": 0.95, "raw_items": []},
                {"frame_index": 2, "timestamp_sec": 3.5, "text": "Assembly session debate in progress", "confidence": 0.92, "raw_items": []},
            ]
        }

        fused = fuse_whisper_and_ocr(whisper_res, ocr_res)
        self.assertEqual(fused["status"], "success")
        self.assertGreaterEqual(fused["fused_entries_count"], 2)

        # Check provenance
        sources = {entry["source"] for entry in fused["fused_entries"]}
        self.assertTrue("whisper" in sources or "both" in sources)
        self.assertEqual(fused["deduplication_summary"]["speech_segments_count"], 2)

    # 14. Candidate claim preparation and TruthLens adapter
    def test_candidate_claim_preparation_and_payload(self):
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "both",
                    "text": "Rahul Gandhi press conference held today regarding disqualification decision.",
                    "start_sec": 4.0,
                    "end_sec": 12.0,
                    "frame_references": [2, 3],
                    "on_screen_context": "25 March 2023 LIVE abp",
                    "disagreements": [],
                    "review_flags": [],
                },
                {
                    "entry_id": "fuse_0002",
                    "source": "whisper",
                    "text": "yes",  # short isolated word
                    "start_sec": 14.0,
                    "end_sec": 15.0,
                    "frame_references": [],
                    "on_screen_context": None,
                    "disagreements": [],
                    "review_flags": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data, video_metadata={"file_name": "sample.mp4"})
        self.assertEqual(claims_res["status"], "success")
        self.assertGreaterEqual(claims_res["candidate_claims_count"], 1)

        claim_1 = claims_res["candidate_claims"][0]
        self.assertEqual(claim_1["status"], "candidate")
        self.assertEqual(claim_1["source_provenance"], "both")
        self.assertEqual(claim_1["context"]["date_reference"], "25 March 2023")

        # Adapter payload check
        payload = prepare_truthlens_claim_payload(claims_res)
        self.assertEqual(len(payload), len(claims_res["candidate_claims"]))
        self.assertIn("claim", payload[0])
        self.assertEqual(payload[0]["status"], "candidate")

    # 15. End-to-end video pipeline with OCR, Fusion, and Claim stages
    def test_full_pipeline_with_ocr_fusion_claims(self):
        video_path = self.temp_path / "full_test.avi"
        create_synthetic_video(video_path, num_frames=20, fps=10.0)

        out_dir = self.temp_path / "full_outputs"

        def fake_ffmpeg_run(cmd, capture_output=True, text=True, check=False):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stderr = ""
            if cmd[0] == "ffprobe":
                mock_proc.stdout = json.dumps({
                    "streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                    "format": {"duration": "2.0"},
                })
            else:
                audio_path = Path(cmd[-1])
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"RIFF dummy wav content")
                mock_proc.stdout = ""
            return mock_proc

        fake_whisper_result = {
            "text": "Assembly session last day final debate.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Assembly session last day final debate."}
            ],
        }
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = fake_whisper_result

        def fake_ocr(path):
            return [{"text": "25 March 2023 LIVE", "confidence": 0.95, "box": [[0, 0], [5, 0], [5, 5], [0, 5]]}]

        with patch("shutil.which", return_value="C:\\dummy\\ffmpeg.exe"), \
             patch("subprocess.run", side_effect=fake_ffmpeg_run), \
             patch("video_module.video_pipeline.get_whisper_model", return_value=mock_whisper):

            summary = process_video_pipeline(
                video_path=video_path,
                output_dir=out_dir,
                sample_interval=1.0,
                whisper_model="base",
                ocr_func=fake_ocr,
            )

            # Assert all 7 stages completed with success
            self.assertEqual(summary["stage_status"]["validation"], "success")
            self.assertEqual(summary["stage_status"]["frame_extraction"], "success")
            self.assertEqual(summary["stage_status"]["audio_extraction"], "success")
            self.assertEqual(summary["stage_status"]["transcription"], "success")
            self.assertEqual(summary["stage_status"]["ocr"], "success")
            self.assertEqual(summary["stage_status"]["fusion"], "success")
            self.assertEqual(summary["stage_status"]["claim_preparation"], "success")

            # Check all generated artifact files
            self.assertTrue((out_dir / "frames.json").exists())
            self.assertTrue((out_dir / "audio.wav").exists())
            self.assertTrue((out_dir / "transcript.txt").exists())
            self.assertTrue((out_dir / "transcript.json").exists())
            self.assertTrue((out_dir / "ocr.json").exists())
            self.assertTrue((out_dir / "fused_transcript.json").exists())
            self.assertTrue((out_dir / "candidate_claims.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())

            # Check summary metadata
            self.assertIn("whisper_recommendations", summary)
            self.assertEqual(summary["claims_summary"]["candidate_claims_count"], 1)


if __name__ == "__main__":
    unittest.main()
