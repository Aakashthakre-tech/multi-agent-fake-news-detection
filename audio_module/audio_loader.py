"""
audio_loader.py
===============
Robust audio validation, decoding, and resampling for TruthLensAI.
Supports WAV, MP3, M4A, FLAC, AAC, and OGG.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .config import (
    SUPPORTED_AUDIO_EXTENSIONS,
    TARGET_SAMPLE_RATE,
    MIN_AUDIO_DURATION_SEC,
    SILENCE_RMS_THRESHOLD,
)


class AudioModuleError(Exception):
    """Base exception for audio module errors."""
    pass


class AudioFileNotFoundError(AudioModuleError, FileNotFoundError):
    """Raised when the specified audio file cannot be located."""
    pass


class UnsupportedAudioFormatError(AudioModuleError, ValueError):
    """Raised when the audio format is not supported."""
    pass


class CorruptAudioError(AudioModuleError, RuntimeError):
    """Raised when an audio file cannot be decoded or is corrupt."""
    pass


class SilentAudioError(AudioModuleError, ValueError):
    """Raised when the audio contains only silence or near-zero amplitude."""
    pass


def validate_audio_file(audio_path: Union[str, Path]) -> Path:
    """
    Validate that the file exists, has a supported extension, and has non-trivial size.
    """
    path = Path(audio_path).resolve()
    if not path.exists() or not path.is_file():
        raise AudioFileNotFoundError(f"Audio file does not exist: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}'. Supported formats: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    if path.stat().st_size < 128:
        raise CorruptAudioError(f"Audio file is too small or truncated ({path.stat().st_size} bytes): {path}")

    return path


def validate_audio_path(audio_path: Union[str, Path]) -> Dict[str, Union[str, int, bool, Path]]:
    """
    Validate audio path and return metadata dictionary.
    """
    path = validate_audio_file(audio_path)
    return {
        "path": path,
        "format": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "exists": True,
    }


def _decode_with_ffmpeg(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Decode audio to 16 kHz mono float32 numpy array via system FFmpeg.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", str(path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0:
            return None

        raw = proc.stdout
        if not raw:
            return None

        # Convert 16-bit PCM buffer to normalized float32 (-1.0 to 1.0)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _decode_with_wave(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Zero-dependency fallback: load standard 16-bit PCM WAV using Python's built-in wave module.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if sampwidth != 2:  # Only 16-bit PCM supported directly
                return None

            raw_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            if framerate != target_sr:
                # Simple nearest/linear resample if standard library wave
                duration = len(audio) / framerate
                new_len = int(duration * target_sr)
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

            return audio
    except Exception:
        return None


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    allow_silent: bool = False,
) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
    """
    Load, validate, and decode an audio file into a 1D float32 numpy array (mono, target_sr).

    Returns:
        audio: 1D np.ndarray float32
        metadata: Dict with duration_seconds, sample_rate, format, file_size_bytes, rms
    """
    path = validate_audio_file(audio_path)

    # 1. Try decoding with FFmpeg first (supports all formats)
    audio = _decode_with_ffmpeg(path, target_sr=target_sr)

    # 2. Fallback to built-in wave if it's a WAV file and FFmpeg failed or was unavailable
    if audio is None and path.suffix.lower() == ".wav":
        audio = _decode_with_wave(path, target_sr=target_sr)

    if audio is None or len(audio) == 0:
        raise CorruptAudioError(f"Could not decode audio stream from: {path.name}")

    duration = float(len(audio)) / float(target_sr)
    if duration < MIN_AUDIO_DURATION_SEC:
        raise CorruptAudioError(
            f"Audio duration ({duration:.2f}s) is shorter than minimum required ({MIN_AUDIO_DURATION_SEC}s)."
        )

    # Calculate RMS to detect silence
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < SILENCE_RMS_THRESHOLD and not allow_silent:
        raise SilentAudioError(
            f"Audio file appears silent or contains near-zero signal (RMS: {rms:.6f})."
        )

    metadata = {
        "file_name": path.name,
        "format": path.suffix.lower().lstrip("."),
        "file_size_bytes": path.stat().st_size,
        "sample_rate": target_sr,
        "channels": 1,
        "duration_seconds": round(duration, 3),
        "rms": round(rms, 6),
        "samples_count": len(audio),
    }

    return audio, metadata
