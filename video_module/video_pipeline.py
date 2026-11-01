"""
TruthLensAI Video Module - Phase 1: Video Preprocessing and Timestamped Transcription.

Flow:
video.mp4 -> validation -> frame extraction + audio extraction -> Whisper transcription -> timestamped transcript + processing summary
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

# Safely configure UTF-8 output streams on Windows terminals to prevent cp1252 crashes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path so existing modules (e.g. image_module) can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional reuse of existing OCR implementation from image_module
try:
    from image_module.ocr import extract_text as ocr_extract_text
except Exception:
    ocr_extract_text = None

# Global in-memory cache for loaded Whisper models to avoid reloading
_WHISPER_MODELS: Dict[str, Any] = {}

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}

FFMPEG_INSTALL_INSTRUCTIONS = """
FFmpeg / FFprobe executable was not found on your system PATH.
FFmpeg is required for audio extraction and media inspection.

Windows Installation Instructions:
  Option 1 (via Windows Package Manager - Recommended):
      winget install Gyan.FFmpeg

  Option 2 (via Chocolatey):
      choco install ffmpeg

  Option 3 (via Scoop):
      scoop install ffmpeg

  Option 4 (Manual):
      1. Download FFmpeg build from: https://www.gyan.dev/ffmpeg/builds/
      2. Extract the archive (e.g. to C:\\ffmpeg)
      3. Add 'C:\\ffmpeg\\bin' to your System or User Environment PATH variable.
      4. Reopen your terminal and verify with:
         ffmpeg -version
         ffprobe -version
"""


class FFmpegNotFoundError(FileNotFoundError):
    """Raised when FFmpeg or FFprobe executable is missing from system PATH."""
    pass


def check_ffmpeg_availability() -> Tuple[bool, bool, str]:
    """
    Check if ffmpeg and ffprobe are available on system PATH.

    Returns:
        (ffmpeg_available, ffprobe_available, install_help_message)
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    ffmpeg_ok = ffmpeg_path is not None
    ffprobe_ok = ffprobe_path is not None

    help_msg = ""
    if not (ffmpeg_ok and ffprobe_ok):
        help_msg = FFMPEG_INSTALL_INSTRUCTIONS.strip()

    return ffmpeg_ok, ffprobe_ok, help_msg


def inspect_media_ffprobe(video_path: Path) -> Dict[str, Any]:
    """
    Inspect media stream metadata using ffprobe if available.

    Returns:
        Dictionary with stream details, has_audio flag, codecs, and duration.
    """
    ffmpeg_ok, ffprobe_ok, _ = check_ffmpeg_availability()
    if not ffprobe_ok:
        return {
            "has_audio": None,
            "audio_codec": None,
            "video_codec": None,
            "duration_sec": None,
            "ffprobe_available": False,
        }

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return {
                "has_audio": None,
                "audio_codec": None,
                "video_codec": None,
                "duration_sec": None,
                "ffprobe_available": True,
                "error": proc.stderr.strip(),
            }

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        audio_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
        video_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "video"), None)

        duration = None
        format_info = data.get("format", {})
        if "duration" in format_info:
            try:
                duration = float(format_info["duration"])
            except (ValueError, TypeError):
                duration = None

        return {
            "has_audio": has_audio,
            "audio_codec": audio_codec,
            "video_codec": video_codec,
            "duration_sec": duration,
            "ffprobe_available": True,
        }
    except Exception as exc:
        return {
            "has_audio": None,
            "audio_codec": None,
            "video_codec": None,
            "duration_sec": None,
            "ffprobe_available": True,
            "error": str(exc),
        }


def validate_video(video_path: Path) -> Dict[str, Any]:
    """
    Validate that the video file exists, is readable, and extract basic metadata.

    Args:
        video_path: Path to the target video file.

    Returns:
        Dict containing video properties (fps, resolution, duration, frame_count, has_audio, etc.)

    Raises:
        FileNotFoundError: If the video does not exist.
        ValueError: If format is completely unsupported or file is corrupted/unreadable.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Input path is not a file: {video_path}")

    suffix = video_path.suffix.lower()
    if suffix and suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        # Note: We warn rather than strictly reject if it might be a valid container
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file. Corrupted or unsupported format: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration_sec = 0.0
        if fps > 0 and frame_count > 0:
            duration_sec = frame_count / fps

        # Attempt to read a single frame to verify media integrity
        ret, _ = cap.read()
        if not ret and frame_count > 0:
            raise ValueError(f"Video file appears corrupted; could not decode initial frame: {video_path}")

    finally:
        cap.release()

    # Query ffprobe for audio stream metadata
    ffprobe_meta = inspect_media_ffprobe(video_path)

    # Use ffprobe duration if available and OpenCV duration is zero or unreliable
    if ffprobe_meta.get("duration_sec") is not None and (duration_sec <= 0 or duration_sec is None):
        duration_sec = ffprobe_meta["duration_sec"]

    metadata = {
        "path": str(video_path.resolve()),
        "file_name": video_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps else 0.0,
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 3) if duration_sec else 0.0,
        "has_audio": ffprobe_meta.get("has_audio"),
        "audio_codec": ffprobe_meta.get("audio_codec"),
        "video_codec": ffprobe_meta.get("video_codec"),
        "ffprobe_available": ffprobe_meta.get("ffprobe_available", False),
    }

    return metadata


def extract_frames(
    video_path: Path,
    output_frames_dir: Path,
    sample_interval: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Incrementally sample and save video frames as JPEG images using OpenCV.

    Args:
        video_path: Path to the input video.
        output_frames_dir: Directory where JPEG frames will be saved.
        sample_interval: Interval in seconds between sampled frames (e.g., 1.0 for 1 frame/sec).

    Returns:
        List of frame manifest entries with frame index, filename, and actual timestamp in seconds.
    """
    if sample_interval <= 0:
        raise ValueError(f"Sample interval must be positive, got {sample_interval}")

    output_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for frame extraction: {video_path}")

    manifest: List[Dict[str, Any]] = []

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not (fps == fps):  # handle 0 or NaN
            fps = 25.0

        step_frames = max(1, int(round(fps * sample_interval)))

        frame_idx = 0
        saved_count = 0
        last_saved_timestamp = -1.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate actual timestamp
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec > 0:
                current_timestamp = msec / 1000.0
            else:
                current_timestamp = frame_idx / fps

            # Determine whether this frame matches our sampling interval
            if frame_idx % step_frames == 0:
                rounded_timestamp = round(current_timestamp, 3)

                # Avoid duplicate frames caused by identical timestamp or rounding
                if saved_count == 0 or (rounded_timestamp - last_saved_timestamp) >= (sample_interval * 0.4):
                    saved_count += 1
                    filename = f"frame_{saved_count:06d}.jpg"
                    frame_path = output_frames_dir / filename

                    success = cv2.imwrite(str(frame_path), frame)
                    if not success:
                        raise IOError(f"Failed to write frame to {frame_path}")

                    manifest.append({
                        "frame_index": saved_count,
                        "filename": filename,
                        "timestamp_sec": rounded_timestamp,
                    })
                    last_saved_timestamp = rounded_timestamp

            frame_idx += 1

    finally:
        cap.release()

    return manifest


def extract_audio(
    video_path: Path,
    output_audio_path: Path,
    has_audio_stream: Optional[bool] = None,
) -> Optional[Path]:
    """
    Extract mono 16 kHz PCM WAV audio from video using FFmpeg.

    Args:
        video_path: Path to the input video.
        output_audio_path: Target path for the extracted .wav file.
        has_audio_stream: If pre-inspected, boolean indicating whether audio exists.

    Returns:
        Path to the extracted audio file, or None if the video has no audio stream.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not available on system PATH.
        RuntimeError: If FFmpeg fails unexpectedly.
    """
    if has_audio_stream is False:
        return None

    ffmpeg_ok, _, install_help = check_ffmpeg_availability()
    if not ffmpeg_ok:
        raise FFmpegNotFoundError(install_help)

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Command to extract mono, 16 kHz, 16-bit PCM WAV
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        stderr_lower = proc.stderr.lower()
        # Detect if failure is simply because video has no audio stream
        if (
            "does not contain any stream" in stderr_lower
            or "does not contain any audio stream" in stderr_lower
            or "matches no streams" in stderr_lower
            or "output file does not contain any stream" in stderr_lower
        ):
            if output_audio_path.exists():
                output_audio_path.unlink()
            return None

        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit code {proc.returncode}):\n{proc.stderr.strip()}"
        )

    if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
        if output_audio_path.exists():
            output_audio_path.unlink()
        return None

    return output_audio_path


def get_whisper_model(model_name: str = "base", device: Optional[str] = "cpu"):
    """
    Load and cache OpenAI Whisper model once per process.

    Args:
        model_name: Model identifier (e.g. 'tiny', 'base', 'small', 'medium', 'large').
        device: Device to load model onto ('cpu', 'cuda', etc.). Defaults to 'cpu' to avoid VRAM contention with visual/audio detectors.
    """
    global _WHISPER_MODELS
    target_device = device or "cpu"
    cache_key = f"{model_name}:{target_device}"
    if cache_key not in _WHISPER_MODELS:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. Please install it using: pip install openai-whisper"
            ) from exc

        _WHISPER_MODELS[cache_key] = whisper.load_model(model_name, device=target_device)

    return _WHISPER_MODELS[cache_key]


def _load_pcm_wav(audio_path: Path) -> Optional["np.ndarray"]:
    """
    Load a 16 kHz mono 16-bit PCM WAV file directly into a float32 numpy array.
    Allows Whisper transcription of standard WAV files even if system FFmpeg is absent.
    """
    import wave
    import numpy as np

    try:
        with wave.open(str(audio_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if sampwidth != 2:  # Must be 16-bit PCM
                return None

            raw_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            # Whisper expects 16,000 Hz sample rate
            if framerate != 16000:
                # If sample rate differs, let Whisper's default loader handle it if ffmpeg is present
                return None

            return audio
    except Exception:
        return None


def transcribe_audio(
    audio_path: Optional[Path],
    model_name: str = "base",
) -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper.

    Args:
        audio_path: Path to extracted WAV audio, or None.
        model_name: Whisper model size name.

    Returns:
        Dict with 'text', 'segments' (list of {id, start, end, text}), and status info.
    """
    if audio_path is None or not audio_path.exists():
        return {
            "status": "skipped",
            "reason": "No audio stream available",
            "text": "",
            "segments": [],
        }

    try:
        model = get_whisper_model(model_name)

        # Attempt direct PCM WAV loading to bypass system FFmpeg requirement for transcription
        audio_data = _load_pcm_wav(audio_path)
        input_audio = audio_data if audio_data is not None else str(audio_path)

        # Using fp16=False ensures compatibility across CPU / systems without CUDA
        result = model.transcribe(input_audio, fp16=False)

        raw_segments = result.get("segments", [])
        formatted_segments: List[Dict[str, Any]] = []

        for seg in raw_segments:
            formatted_segments.append({
                "id": seg.get("id"),
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": seg.get("text", "").strip(),
            })

        return {
            "status": "success",
            "reason": None,
            "text": result.get("text", "").strip(),
            "segments": formatted_segments,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "text": "",
            "segments": [],
        }


# ============================================================
# TASK 1: WHISPER TRANSCRIPTION CANDIDATES & COMPARISON
# ============================================================

WHISPER_CANDIDATE_RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "mr": {
        "candidate_model": "large-v3",
        "runtime_config": "faster-whisper int8_float16 (RTX 3050 GPU) or openai-whisper large",
        "rationale": (
            "Large-v3 delivers the highest Devanagari script consistency and accurate named-entity recognition "
            "(e.g., Rahul Gandhi, OBC, BJP, Congress, Maharashtra Kesari, Vaishnavi Patil, Kalyan). "
            "Smaller models ('base', 'small') collapse into repetitive token loops and hallucinated foreign "
            "scripts (Kannada, Bengali), while 'medium' exhibits frequent phonetic distortion."
        ),
        "measured_accuracy": "Not measured (no verified reference transcript available; qualitative comparison only).",
        "known_limitations": "Repetition loops in extended speech segments and minor phonetic variations remain unverified against ground truth."
    },
    "hi": {
        "candidate_model": "large-v3",
        "runtime_config": "faster-whisper int8_float16 with language='hi'",
        "rationale": (
            "Large-v3 with explicit Hindi language constraint produces coherent Devanagari transcription. "
            "Unconstrained small models collapse into repetitive token loops or drift into Urdu/Perso-Arabic script."
        ),
        "measured_accuracy": "Not measured (no verified reference transcript available; qualitative comparison only).",
        "known_limitations": "Occasional script drift into Perso-Arabic or Romanized text if language parameter is unpinned."
    }
}


def get_whisper_recommendations(language: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve transcription candidate recommendations and observed limitations."""
    if language and language in WHISPER_CANDIDATE_RECOMMENDATIONS:
        return {language: WHISPER_CANDIDATE_RECOMMENDATIONS[language]}
    return WHISPER_CANDIDATE_RECOMMENDATIONS


# ============================================================
# TASK 2: OCR EXTRACTION ON VIDEO FRAMES
# ============================================================

def run_ocr_on_frames(
    frame_manifest: List[Dict[str, Any]],
    frames_dir: Path,
    ocr_stride: int = 1,
    max_frames: Optional[int] = None,
    ocr_func: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Run OCR extraction on sampled video frames, preserving frame timestamps and references.
    Reuses existing image_module.ocr if available, or an injected ocr_func for testing.

    Args:
        frame_manifest: List of frame metadata dicts from frames.json.
        frames_dir: Directory containing extracted frame images.
        ocr_stride: Sample every Nth frame (default 1).
        max_frames: Optional upper limit on processed frames.
        ocr_func: Optional custom OCR function accepting an image path string.

    Returns:
        Dict with status, counts, observations with timestamps, and warnings.
    """
    func = ocr_func if ocr_func is not None else ocr_extract_text

    if func is None:
        return {
            "status": "skipped",
            "reason": "OCR engine (image_module.ocr) is not available",
            "processed_frames_count": 0,
            "frames_with_text_count": 0,
            "observations": [],
            "warnings": ["OCR skipped: image_module.ocr is not available in the current environment."],
        }

    if not frame_manifest:
        return {
            "status": "skipped",
            "reason": "No extracted frames found in manifest",
            "processed_frames_count": 0,
            "frames_with_text_count": 0,
            "observations": [],
            "warnings": [],
        }

    # Configurable OCR Frame Budget:
    # A 120s video at 1 fps yields 120 frames. Processing every frame sequentially
    # takes excessive time even on GPU. We enforce a default representative budget of 15 frames
    # uniformly sampled across the video duration if max_frames is not explicitly set.
    total_available = len(frame_manifest)
    effective_max = max_frames if (max_frames is not None and max_frames > 0) else 15
    
    if total_available > effective_max:
        # Uniform sampling across video timeline
        import numpy as np
        indices = np.linspace(0, total_available - 1, effective_max, dtype=int)
        sampled = [frame_manifest[i] for i in indices]
        sampling_policy = f"Uniformly sampled {len(sampled)} frames across {total_available} total frames (budget: {effective_max})."
    else:
        stride = max(1, int(ocr_stride))
        sampled = frame_manifest[::stride]
        sampling_policy = f"Processed {len(sampled)} frames with stride={stride} (within budget of {effective_max})."

    observations: List[Dict[str, Any]] = []
    warnings: List[str] = []

    print(f"      OCR Policy: {sampling_policy}")

    for idx, item in enumerate(sampled, 1):
        frame_idx = item.get("frame_index")
        filename = item.get("filename")
        ts = item.get("timestamp_sec", 0.0)
        frame_path = frames_dir / filename

        if not frame_path.exists():
            warnings.append(f"Frame image not found: {filename}")
            continue

        try:
            t_frame_0 = time.time()
            detected_items = func(str(frame_path))
            t_frame_dur = time.time() - t_frame_0
            if detected_items:
                combined_text = " ".join(
                    d.get("text", "").strip() for d in detected_items if d.get("text")
                ).strip()
                avg_conf = round(
                    sum(float(d.get("confidence", 0.0)) for d in detected_items) / len(detected_items),
                    4
                )
                observations.append({
                    "frame_index": frame_idx,
                    "filename": filename,
                    "timestamp_sec": ts,
                    "text": combined_text,
                    "confidence": avg_conf,
                    "raw_items": detected_items,
                })
                print(f"      [OCR {idx}/{len(sampled)}] Frame {filename} (t={ts:.1f}s) -> \"{combined_text[:40]}...\" in {t_frame_dur:.2f}s")
            else:
                if idx % 5 == 0 or idx == len(sampled):
                    print(f"      [OCR {idx}/{len(sampled)}] Frame {filename} (t={ts:.1f}s) -> [no text] in {t_frame_dur:.2f}s")
        except Exception as exc:
            warnings.append(f"OCR error on frame {filename} (t={ts}s): {str(exc)}")

    status = "success" if observations else ("partial" if sampled else "skipped")
    return {
        "status": status,
        "processed_frames_count": len(sampled),
        "frames_with_text_count": len(observations),
        "observations": observations,
        "sampling_policy": sampling_policy,
        "warnings": warnings,
    }


# ============================================================
# TASK 3: FUSION (SPEECH + OCR) & CONSERVATIVE DEDUPLICATION
# ============================================================

def normalize_text_conservative(text: str) -> str:
    """
    Normalize text conservatively: strips extraneous whitespace while strictly
    preserving Devanagari, Marathi, Hindi, Latin characters, numbers, dates, and negations.
    """
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"[^\S\r\n]+", " ", text).strip()


_DATE_PATTERNS = [
    # E.g. "25 March 2023", "14 Sep 2017", "25th March, 2023"
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s*,?\s*(?:19|20)\d{2}\b",
        re.IGNORECASE
    ),
    # E.g. "March 25, 2023"
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(?:19|20)\d{2}\b",
        re.IGNORECASE
    ),
    # E.g. "2023-03-25" or "2017/09/14"
    re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
]


def extract_dates_from_text(text: str) -> List[str]:
    """Extract standard date representations from text."""
    found: List[str] = []
    for pattern in _DATE_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            clean_m = m.strip()
            if clean_m not in found:
                found.append(clean_m)
    return found


def extract_numbers_from_text(text: str) -> List[str]:
    """Extract significant numbers or percentages from text for consistency checks."""
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    return list(dict.fromkeys(nums))


_NEGATION_WORDS = {
    "नाही", "नसेल", "नकार", "खोटे", "फेक", "विरोधात", "हल्लाबोल", "विरोध",
    "not", "no", "never", "fake", "denies", "denied", "rejected", "false", "refutes"
}


def deduplicate_ocr_observations(
    observations: List[Dict[str, Any]],
    similarity_threshold: float = 0.70,
    time_window_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Deduplicate sequential or nearby OCR observations that share identical or highly similar text
    (e.g., persistent TV news tickers, dates, logos across frames), while strictly preserving
    distinct statements and recording frame references and time spans.
    """
    if not observations:
        return []

    deduped: List[Dict[str, Any]] = []

    for obs in observations:
        text = normalize_text_conservative(obs.get("text", ""))
        if not text:
            continue

        ts = float(obs.get("timestamp_sec", 0.0))
        frame_idx = obs.get("frame_index")

        merged = False
        if deduped:
            last = deduped[-1]
            time_diff = abs(ts - last["end_sec"])
            sim = difflib.SequenceMatcher(None, text.lower(), last["text"].lower()).ratio()

            if (sim >= similarity_threshold or text.lower() == last["text"].lower()) and time_diff <= time_window_sec:
                last["end_sec"] = max(last["end_sec"], ts)
                if frame_idx is not None and frame_idx not in last["frame_references"]:
                    last["frame_references"].append(frame_idx)
                last["confidence"] = round(max(last["confidence"], float(obs.get("confidence", 0.0))), 4)
                if len(text) > len(last["text"]) and sim > 0.85:
                    last["text"] = text
                merged = True

        if not merged:
            deduped.append({
                "source": "ocr",
                "text": text,
                "start_sec": ts,
                "end_sec": ts,
                "frame_references": [frame_idx] if frame_idx is not None else [],
                "confidence": float(obs.get("confidence", 0.0)),
                "raw_items_count": len(obs.get("raw_items", [])),
            })

    return deduped


def detect_disagreements_and_flags(
    speech_text: str,
    ocr_text: str,
    time_overlap: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Check for potential conflicts or review needs between speech and on-screen text.
    Labels cautiously as 'possible_disagreement' or 'needs_review' rather than definitive truth claims.

    Returns:
        (disagreements_list, review_flags_list)
    """
    disagreements: List[str] = []
    review_flags: List[str] = []

    # 1. Date mismatch check
    speech_dates = extract_dates_from_text(speech_text)
    ocr_dates = extract_dates_from_text(ocr_text)
    if speech_dates and ocr_dates:
        if speech_dates[0].lower() != ocr_dates[0].lower():
            disagreements.append(
                f"possible_disagreement: date_mismatch (speech mentions '{speech_dates[0]}' while OCR shows '{ocr_dates[0]}')"
            )
            review_flags.append("needs_review: date_discrepancy")

    # 2. Prominent numbers mismatch in temporally overlapping observations
    if time_overlap:
        speech_nums = set(extract_numbers_from_text(speech_text))
        ocr_nums = set(extract_numbers_from_text(ocr_text))
        if speech_nums and ocr_nums and not (speech_nums & ocr_nums):
            disagreements.append(
                f"possible_disagreement: number_mismatch (speech={sorted(list(speech_nums))} vs ocr={sorted(list(ocr_nums))})"
            )
            review_flags.append("needs_review: number_discrepancy")

    # 3. Negation / Polarity conflict
    speech_tokens = set(speech_text.lower().split())
    ocr_tokens = set(ocr_text.lower().split())
    speech_has_neg = bool(speech_tokens & _NEGATION_WORDS)
    ocr_has_neg = bool(ocr_tokens & _NEGATION_WORDS)
    if speech_has_neg != ocr_has_neg and (len(speech_tokens & ocr_tokens) >= 2):
        disagreements.append(
            "possible_disagreement: potential_polarity_conflict between speech and on-screen text"
        )
        review_flags.append("needs_review: polarity_check")

    # 4. Suspected Whisper repetition loop
    for token in speech_tokens:
        if len(token) > 3 and speech_text.count(token) >= 4:
            review_flags.append("needs_review: suspected_whisper_repetition_loop")
            break

    return disagreements, review_flags


def fuse_whisper_and_ocr(
    transcription_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine Whisper speech transcription and OCR text observations into a structured,
    deduplicated timeline preserving source provenance, timestamps, frame references,
    and review flags.

    Args:
        transcription_result: Dict containing 'segments' from Whisper transcription.
        ocr_result: Dict containing 'observations' from run_ocr_on_frames.

    Returns:
        Structured fusion dict with 'fused_entries', 'deduplication_summary', and status.
    """
    raw_segments = transcription_result.get("segments", [])
    raw_ocr = ocr_result.get("observations", [])

    deduped_ocr = deduplicate_ocr_observations(raw_ocr)

    fused_entries: List[Dict[str, Any]] = []
    entry_counter = 1
    total_review_flags = 0

    # Build entries from speech segments
    for seg in raw_segments:
        seg_text = normalize_text_conservative(seg.get("text", ""))
        if not seg_text:
            continue

        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))

        # Match with OCR observations overlapping this segment time window
        matching_ocr_items: List[Dict[str, Any]] = []
        for ocr_item in deduped_ocr:
            ocr_start = ocr_item["start_sec"]
            ocr_end = ocr_item["end_sec"]
            if not (end_sec < ocr_start - 3.0 or start_sec > ocr_end + 3.0):
                matching_ocr_items.append(ocr_item)

        frame_refs: List[int] = []
        disagreements: List[str] = []
        review_flags: List[str] = []
        source = "whisper"
        combined_ocr_text = ""

        if matching_ocr_items:
            combined_ocr_text = " | ".join(item["text"] for item in matching_ocr_items)
            for item in matching_ocr_items:
                frame_refs.extend(item.get("frame_references", []))

            disags, flags = detect_disagreements_and_flags(seg_text, combined_ocr_text, time_overlap=True)
            disagreements.extend(disags)
            review_flags.extend(flags)

            sim = difflib.SequenceMatcher(None, seg_text.lower(), combined_ocr_text.lower()).ratio()
            if sim >= 0.50:
                source = "both"

        fused_entries.append({
            "entry_id": f"fuse_{entry_counter:04d}",
            "source": source,
            "text": seg_text,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frame_references": sorted(list(set(frame_refs))),
            "on_screen_context": combined_ocr_text if combined_ocr_text else None,
            "disagreements": disagreements,
            "review_flags": review_flags,
        })
        entry_counter += 1
        total_review_flags += len(review_flags)

    # Add standalone OCR observations that had no overlapping speech segment
    for ocr_item in deduped_ocr:
        ocr_start = ocr_item["start_sec"]
        ocr_end = ocr_item["end_sec"]

        covered = any(
            not (entry["end_sec"] < ocr_start - 3.0 or entry["start_sec"] > ocr_end + 3.0)
            for entry in fused_entries
        )

        if not covered:
            fused_entries.append({
                "entry_id": f"fuse_{entry_counter:04d}",
                "source": "ocr",
                "text": ocr_item["text"],
                "start_sec": ocr_start,
                "end_sec": ocr_end,
                "frame_references": ocr_item.get("frame_references", []),
                "on_screen_context": None,
                "disagreements": [],
                "review_flags": [],
            })
            entry_counter += 1

    fused_entries.sort(key=lambda e: (e["start_sec"], e["end_sec"]))

    fusion_status = "success"
    if not raw_segments and not raw_ocr:
        fusion_status = "skipped"
    elif not raw_segments or not raw_ocr:
        fusion_status = "partial"

    return {
        "status": fusion_status,
        "whisper_segments_count": len(raw_segments),
        "ocr_raw_observations_count": len(raw_ocr),
        "ocr_deduped_observations_count": len(deduped_ocr),
        "fused_entries_count": len(fused_entries),
        "fused_entries": fused_entries,
        "deduplication_summary": {
            "raw_ocr_count": len(raw_ocr),
            "deduped_ocr_count": len(deduped_ocr),
            "speech_segments_count": len(raw_segments),
        },
        "review_flags_count": total_review_flags,
    }


# ============================================================
# TASK 4: PREPARE CANDIDATE CLAIMS FOR TRUTHLENS AI
# ============================================================

def prepare_candidate_claims(
    fused_data: Dict[str, Any],
    video_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract structured candidate claims from the fused speech and OCR timeline.
    Preserves multiple topics, source provenance (Whisper, OCR, both),
    timestamps, frame references, and review flags.

    Filters conversational filler, apologies, greetings, promotional slogans,
    and OCR noise, and ranks verifiable factual claims by priority.

    Does NOT perform web search verification or assign truth values.
    """
    entries = fused_data.get("fused_entries", [])
    raw_candidates: List[Dict[str, Any]] = []

    for entry in entries:
        text = entry.get("text", "").strip()
        on_screen_context = entry.get("on_screen_context")
        date_context = None
        dates_in_text = extract_dates_from_text(text)
        if dates_in_text:
            date_context = dates_in_text[0]
        elif on_screen_context:
            dates_in_ocr = extract_dates_from_text(on_screen_context)
            if dates_in_ocr:
                date_context = dates_in_ocr[0]

        if text and len(text) >= 4:
            raw_candidates.append({
                "text": text,
                "source": entry.get("source", "unknown"),
                "start_sec": entry.get("start_sec", 0.0),
                "end_sec": entry.get("end_sec", 0.0),
                "frame_references": entry.get("frame_references", []),
                "on_screen_context": on_screen_context,
                "context": {
                    "date_reference": date_context,
                    "on_screen_context": on_screen_context,
                    "video_file": video_metadata.get("file_name") if video_metadata else None,
                },
                "review_flags": entry.get("review_flags", []),
                "disagreements": entry.get("disagreements", []),
            })

        # Also extract distinct on-screen context when not merged as 'both' and expressing a proposition
        if on_screen_context and entry.get("source") == "whisper":
            sim = difflib.SequenceMatcher(None, text.lower(), on_screen_context.lower()).ratio()
            if sim < 0.50 and len(on_screen_context.strip()) >= 6:
                from video_module.claim_ranker import detect_predicate_and_verbs
                has_pred, _ = detect_predicate_and_verbs(on_screen_context)
                if has_pred:
                    dates_in_ocr = extract_dates_from_text(on_screen_context)
                    ocr_date = dates_in_ocr[0] if dates_in_ocr else None
                    raw_candidates.append({
                        "text": on_screen_context.strip(),
                        "source": "ocr",
                        "start_sec": entry.get("start_sec", 0.0),
                        "end_sec": entry.get("end_sec", 0.0),
                        "frame_references": entry.get("frame_references", []),
                        "on_screen_context": None,
                        "context": {
                            "date_reference": ocr_date,
                            "on_screen_context": None,
                            "video_file": video_metadata.get("file_name") if video_metadata else None,
                        },
                        "review_flags": [],
                        "disagreements": [],
                    })

    from video_module.claim_ranker import process_and_rank_candidate_claims
    ranked_result = process_and_rank_candidate_claims(raw_candidates, video_metadata=video_metadata)

    status = "success" if ranked_result["candidate_claims"] else ("skipped" if not entries else "partial")
    ranked_result["status"] = status

    return ranked_result


def prepare_truthlens_claim_payload(candidate_claims_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Adapter function preparing structured candidate claims into the format
    expected by TruthLensAI's pipeline (e.g. pipeline.run_research_pipeline).
    Leaves live verification uncalled.
    """
    claims = candidate_claims_data.get("candidate_claims", [])
    payload: List[Dict[str, Any]] = []

    for c in claims:
        base_text = c["claim_text"]
        date_ref = c.get("context", {}).get("date_reference")
        full_statement = base_text
        if date_ref and date_ref.lower() not in base_text.lower():
            full_statement = f"{base_text.rstrip('.?!')} (Dated: {date_ref})."

        is_selected = c.get("selection_status") == "selected"

        payload.append({
            "claim_id": c["claim_id"],
            "claim": full_statement,
            "original_text": c.get("original_text", base_text),
            "source_provenance": c["source_provenance"],
            "source_modality": c.get("source_modality", "speech" if c["source_provenance"] == "whisper" else c["source_provenance"]),
            "timestamps": c["timestamps"],
            "frame_references": c["frame_references"],
            "date_reference": date_ref,
            "status": c.get("status", "candidate"),
            "selection_status": c.get("selection_status", "selected" if is_selected else "rejected"),
            "priority_score": c.get("priority_score", 0.5),
            "selection_reason": c.get("selection_reason", ""),
            "is_verifiable": c.get("is_verifiable", is_selected),
            "is_factual": c.get("is_factual", is_selected),
            "needs_manual_review": c["needs_manual_review"],
            "review_flags": c["review_flags"],
            "ready_for_pipeline": is_selected and not c["needs_manual_review"],
        })

    return payload


def get_safe_output_dir(base_parent: Path, video_stem: str) -> Path:
    """
    Generate a non-destructive output folder name.
    If 'base_parent / video_stem' exists and has content, appends an incrementing counter.
    """
    candidate = base_parent / video_stem
    if not candidate.exists() or not any(candidate.iterdir()):
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    counter = 1
    while True:
        candidate = base_parent / f"{video_stem}_{counter:03d}"
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


def process_video_pipeline(
    video_path: Path,
    output_dir: Optional[Path] = None,
    sample_interval: float = 1.0,
    whisper_model: str = "base",
    ocr_stride: int = 1,
    max_ocr_frames: Optional[int] = None,
    skip_ocr: bool = False,
    ocr_func: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Execute Video Preprocessing, Transcription, OCR, Fusion, and Claim Preparation Pipeline:
    Validation -> Frame Extraction -> Audio Extraction -> Transcription -> OCR -> Fusion -> Claim Preparation -> Summary.

    Args:
        video_path: Path to target video file.
        output_dir: Optional custom output directory.
        sample_interval: Sampling interval for video frames in seconds.
        whisper_model: Whisper model size name.
        ocr_stride: Sample stride for OCR processing over extracted frames (default: 1).
        max_ocr_frames: Optional max number of frames to run OCR on.
        skip_ocr: If True, skips the OCR extraction stage.
        ocr_func: Optional custom OCR function for dependency injection or testing.

    Returns:
        Dict representing summary.json content.
    """
    video_path = Path(video_path).resolve()
    warnings: List[str] = []
    errors: List[str] = []

    stage_status: Dict[str, str] = {
        "validation": "pending",
        "frame_extraction": "pending",
        "audio_extraction": "pending",
        "transcription": "pending",
        "ocr": "pending",
        "fusion": "pending",
        "claim_preparation": "pending",
    }
    stage_timings: Dict[str, float] = {}

    # 1. Validation
    t_stage = time.time()
    print(f"[1/7] [{time.strftime('%H:%M:%S')}] Validating video: {video_path.name}")
    try:
        metadata = validate_video(video_path)
        stage_status["validation"] = "success"
        stage_timings["validation"] = round(time.time() - t_stage, 2)
        print(f"      Validation completed in {stage_timings['validation']}s")
        if metadata.get("ffprobe_available") is False:
            warnings.append("FFprobe not available on system PATH; audio stream inspection was skipped.")
    except Exception as exc:
        stage_status["validation"] = "failed"
        stage_timings["validation"] = round(time.time() - t_stage, 2)
        errors.append(f"Validation failed: {str(exc)}")
        raise

    # Determine safe output directory
    if output_dir is not None:
        target_output_dir = Path(output_dir).resolve()
        target_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        default_outputs_root = Path(__file__).parent / "outputs"
        target_output_dir = get_safe_output_dir(default_outputs_root, video_path.stem)

    frames_dir = target_output_dir / "frames"
    frames_manifest_path = target_output_dir / "frames.json"
    audio_wav_path = target_output_dir / "audio.wav"
    transcript_txt_path = target_output_dir / "transcript.txt"
    transcript_json_path = target_output_dir / "transcript.json"
    ocr_json_path = target_output_dir / "ocr.json"
    fused_transcript_path = target_output_dir / "fused_transcript.json"
    candidate_claims_path = target_output_dir / "candidate_claims.json"
    summary_json_path = target_output_dir / "summary.json"

    # 2. Frame Extraction
    t_stage = time.time()
    print(f"[2/7] [{time.strftime('%H:%M:%S')}] Extracting frames (1 frame every {sample_interval}s)...")
    frame_manifest: List[Dict[str, Any]] = []
    try:
        frame_manifest = extract_frames(
            video_path=video_path,
            output_frames_dir=frames_dir,
            sample_interval=sample_interval,
        )
        stage_status["frame_extraction"] = "success"
        stage_timings["frame_extraction"] = round(time.time() - t_stage, 2)
        with open(frames_manifest_path, "w", encoding="utf-8") as f:
            json.dump(frame_manifest, f, indent=2)
        print(f"      Extracted {len(frame_manifest)} frames -> {frames_dir.name}/ in {stage_timings['frame_extraction']}s")
    except Exception as exc:
        stage_status["frame_extraction"] = "failed"
        stage_timings["frame_extraction"] = round(time.time() - t_stage, 2)
        errors.append(f"Frame extraction failed: {str(exc)}")
        print(f"      Frame extraction error: {exc}")

    # 3. Audio Extraction
    t_stage = time.time()
    print(f"[3/7] [{time.strftime('%H:%M:%S')}] Extracting audio stream...")
    extracted_audio: Optional[Path] = None
    try:
        extracted_audio = extract_audio(
            video_path=video_path,
            output_audio_path=audio_wav_path,
            has_audio_stream=metadata.get("has_audio"),
        )
        stage_timings["audio_extraction"] = round(time.time() - t_stage, 2)
        if extracted_audio is not None:
            stage_status["audio_extraction"] = "success"
            print(f"      Audio extracted -> {audio_wav_path.name} in {stage_timings['audio_extraction']}s")
        else:
            stage_status["audio_extraction"] = "skipped"
            warnings.append("No audio stream detected or video is silent; skipped audio extraction.")
            print("      No audio stream found in video; skipped audio extraction.")
    except FFmpegNotFoundError as exc:
        stage_status["audio_extraction"] = "failed"
        stage_timings["audio_extraction"] = round(time.time() - t_stage, 2)
        errors.append(str(exc))
        print("      FFmpeg not found! Cannot extract audio.")
    except Exception as exc:
        stage_status["audio_extraction"] = "failed"
        stage_timings["audio_extraction"] = round(time.time() - t_stage, 2)
        errors.append(f"Audio extraction failed: {str(exc)}")
        print(f"      Audio extraction error: {exc}")

    # 4. Speech Transcription
    t_stage = time.time()
    print(f"[4/7] [{time.strftime('%H:%M:%S')}] Transcribing audio with Whisper ('{whisper_model}')...")
    transcription_result: Dict[str, Any] = {
        "status": "skipped",
        "reason": "Audio extraction was not performed or failed",
        "text": "",
        "segments": [],
    }

    if stage_status["audio_extraction"] == "success" and extracted_audio is not None:
        try:
            transcription_result = transcribe_audio(
                audio_path=extracted_audio,
                model_name=whisper_model,
            )
            stage_status["transcription"] = transcription_result["status"]
            stage_timings["transcription"] = round(time.time() - t_stage, 2)
            if transcription_result["status"] == "success":
                with open(transcript_txt_path, "w", encoding="utf-8") as f:
                    f.write(transcription_result["text"])
                with open(transcript_json_path, "w", encoding="utf-8") as f:
                    json.dump(transcription_result["segments"], f, indent=2, ensure_ascii=False)
                print(f"      Transcription completed ({len(transcription_result['segments'])} segments) in {stage_timings['transcription']}s.")
            else:
                warnings.append(f"Transcription status: {transcription_result.get('reason')}")
                print(f"      Transcription status: {transcription_result.get('reason')}")
        except Exception as exc:
            stage_status["transcription"] = "failed"
            stage_timings["transcription"] = round(time.time() - t_stage, 2)
            errors.append(f"Transcription failed: {str(exc)}")
            print(f"      Transcription error: {exc}")
    else:
        stage_status["transcription"] = "skipped"
        stage_timings["transcription"] = 0.0
        print("      Transcription skipped (no extracted audio).")

    # 5. OCR Extraction on Frames
    t_stage = time.time()
    print(f"[5/7] [{time.strftime('%H:%M:%S')}] Running OCR on extracted video frames...")
    ocr_result: Dict[str, Any] = {
        "status": "skipped",
        "reason": "Skipped or no frames available",
        "processed_frames_count": 0,
        "frames_with_text_count": 0,
        "observations": [],
        "warnings": [],
    }

    if not skip_ocr and stage_status["frame_extraction"] == "success" and frame_manifest:
        try:
            ocr_result = run_ocr_on_frames(
                frame_manifest=frame_manifest,
                frames_dir=frames_dir,
                ocr_stride=ocr_stride,
                max_frames=max_ocr_frames,
                ocr_func=ocr_func,
            )
            stage_status["ocr"] = ocr_result["status"]
            stage_timings["ocr"] = round(time.time() - t_stage, 2)
            warnings.extend(ocr_result.get("warnings", []))
            with open(ocr_json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_result["observations"], f, indent=2, ensure_ascii=False)
            print(f"      OCR processed {ocr_result['processed_frames_count']} frames (detected text in {ocr_result['frames_with_text_count']} frames) in {stage_timings['ocr']}s.")
        except Exception as exc:
            stage_status["ocr"] = "failed"
            stage_timings["ocr"] = round(time.time() - t_stage, 2)
            errors.append(f"OCR stage failed: {str(exc)}")
            print(f"      OCR stage error: {exc}")
    else:
        stage_status["ocr"] = "skipped"
        stage_timings["ocr"] = 0.0
        if skip_ocr:
            print("      OCR skipped (--skip-ocr specified).")
        else:
            print("      OCR skipped (no frames available).")

    # 6. Fusion (Whisper Speech + On-Screen OCR)
    t_stage = time.time()
    print(f"[6/7] [{time.strftime('%H:%M:%S')}] Fusing speech transcript and on-screen text...")
    fusion_result: Dict[str, Any] = {
        "status": "skipped",
        "whisper_segments_count": 0,
        "ocr_raw_observations_count": 0,
        "ocr_deduped_observations_count": 0,
        "fused_entries_count": 0,
        "fused_entries": [],
        "deduplication_summary": {},
        "review_flags_count": 0,
    }

    try:
        fusion_result = fuse_whisper_and_ocr(
            transcription_result=transcription_result,
            ocr_result=ocr_result,
        )
        stage_status["fusion"] = fusion_result["status"]
        stage_timings["fusion"] = round(time.time() - t_stage, 2)
        with open(fused_transcript_path, "w", encoding="utf-8") as f:
            json.dump(fusion_result, f, indent=2, ensure_ascii=False)
        print(f"      Fusion completed ({fusion_result['fused_entries_count']} fused entries, {fusion_result['review_flags_count']} review flags) in {stage_timings['fusion']}s.")
    except Exception as exc:
        stage_status["fusion"] = "failed"
        stage_timings["fusion"] = round(time.time() - t_stage, 2)
        errors.append(f"Fusion stage failed: {str(exc)}")
        print(f"      Fusion stage error: {exc}")

    # 7. Candidate Claim Preparation
    t_stage = time.time()
    print(f"[7/7] [{time.strftime('%H:%M:%S')}] Preparing candidate claims for TruthLens AI...")
    claims_result: Dict[str, Any] = {
        "status": "skipped",
        "candidate_claims_count": 0,
        "claims_flagged_for_review_count": 0,
        "candidate_claims": [],
    }

    try:
        claims_result = prepare_candidate_claims(
            fused_data=fusion_result,
            video_metadata=metadata,
        )
        stage_status["claim_preparation"] = claims_result["status"]
        stage_timings["claim_preparation"] = round(time.time() - t_stage, 2)
        with open(candidate_claims_path, "w", encoding="utf-8") as f:
            json.dump(claims_result, f, indent=2, ensure_ascii=False)
        print(f"      Prepared {claims_result['candidate_claims_count']} candidate claims ({claims_result['claims_flagged_for_review_count']} flagged for manual review) in {stage_timings['claim_preparation']}s.")
    except Exception as exc:
        stage_status["claim_preparation"] = "failed"
        stage_timings["claim_preparation"] = round(time.time() - t_stage, 2)
        errors.append(f"Candidate claim preparation failed: {str(exc)}")
        print(f"      Claim preparation error: {exc}")

    # 8. Build Summary
    summary: Dict[str, Any] = {
        "input_video_path": str(video_path),
        "video_metadata": metadata,
        "sample_interval_sec": sample_interval,
        "frames_extracted_count": len(frame_manifest),
        "frame_manifest_path": str(frames_manifest_path) if frames_manifest_path.exists() else None,
        "audio_output_path": str(audio_wav_path) if audio_wav_path.exists() else None,
        "transcript_text_path": str(transcript_txt_path) if transcript_txt_path.exists() else None,
        "transcript_json_path": str(transcript_json_path) if transcript_json_path.exists() else None,
        "ocr_json_path": str(ocr_json_path) if ocr_json_path.exists() else None,
        "fused_transcript_path": str(fused_transcript_path) if fused_transcript_path.exists() else None,
        "candidate_claims_path": str(candidate_claims_path) if candidate_claims_path.exists() else None,
        "whisper_model": whisper_model,
        "whisper_recommendations": WHISPER_CANDIDATE_RECOMMENDATIONS,
        "stage_status": stage_status,
        "stage_timings": stage_timings,
        "ocr_summary": {
            "processed_frames": ocr_result.get("processed_frames_count", 0),
            "frames_with_text": ocr_result.get("frames_with_text_count", 0),
            "observations_count": len(ocr_result.get("observations", [])),
        },
        "fusion_summary": {
            "total_fused_entries": fusion_result.get("fused_entries_count", 0),
            "deduplication_summary": fusion_result.get("deduplication_summary", {}),
            "review_flags_count": fusion_result.get("review_flags_count", 0),
        },
        "claims_summary": {
            "candidate_claims_count": claims_result.get("candidate_claims_count", 0),
            "claims_flagged_for_review": claims_result.get("claims_flagged_for_review_count", 0),
        },
        "warnings": warnings,
        "errors": errors,
        # Structured sub-results exposed verbatim for the UI/report layer
        "transcript": transcription_result,
        "ocr_result": ocr_result,
        "fusion_result": fusion_result,
        "candidate_claims_result": claims_result,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nProcessing completed!")
    print(f"Outputs saved to: {target_output_dir}")
    print(f" - Frames Directory:   {frames_dir}")
    print(f" - Frame Manifest:     {frames_manifest_path}")
    if audio_wav_path.exists():
        print(f" - Audio File:         {audio_wav_path}")
    if transcript_txt_path.exists():
        print(f" - Transcript Text:    {transcript_txt_path}")
    if transcript_json_path.exists():
        print(f" - Transcript JSON:    {transcript_json_path}")
    if ocr_json_path.exists():
        print(f" - OCR JSON:           {ocr_json_path}")
    if fused_transcript_path.exists():
        print(f" - Fused Transcript:   {fused_transcript_path}")
    if candidate_claims_path.exists():
        print(f" - Candidate Claims:   {candidate_claims_path}")
    print(f" - Summary File:       {summary_json_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="TruthLensAI Video Preprocessing, Transcription, OCR & Claim Preparation"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (e.g. video.mp4, video.mov, video.mkv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional destination directory for extracted frames, audio, and transcripts.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds between extracted frames (default: 1.0).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="OpenAI Whisper model size name (default: 'base').",
    )
    parser.add_argument(
        "--ocr-stride",
        type=int,
        default=1,
        help="Sample every Nth frame for OCR processing (default: 1).",
    )
    parser.add_argument(
        "--max-ocr-frames",
        type=int,
        default=None,
        help="Optional upper limit on number of frames to run OCR on.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR processing stage entirely.",
    )
    return parser


def main() -> None:
    """CLI entry point for video_pipeline.py."""
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.video_path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        process_video_pipeline(
            video_path=input_path,
            output_dir=output_dir,
            sample_interval=args.sample_interval,
            whisper_model=args.whisper_model,
            ocr_stride=args.ocr_stride,
            max_ocr_frames=args.max_ocr_frames,
            skip_ocr=args.skip_ocr,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline aborted: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
