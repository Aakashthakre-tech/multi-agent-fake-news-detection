"""
transcriber.py
==============
Speech transcription interface reusing TruthLensAI's Whisper engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Reuse Whisper singleton from video_module to share model in memory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from video_module.video_pipeline import get_whisper_model
except ImportError:
    # Standalone fallback if video_module is not imported
    _WHISPER_MODELS = {}

    def get_whisper_model(model_name: str = "base", device: Optional[str] = "cpu"):
        global _WHISPER_MODELS
        target_device = device or "cpu"
        cache_key = f"{model_name}:{target_device}"
        if cache_key not in _WHISPER_MODELS:
            import whisper
            _WHISPER_MODELS[cache_key] = whisper.load_model(model_name, device=target_device)
        return _WHISPER_MODELS[cache_key]

from .config import DEFAULT_WHISPER_MODEL


def transcribe_speech(
    audio: Union[np.ndarray, Path, str],
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    device: Optional[str] = "cpu",
) -> Dict[str, Any]:
    """
    Transcribe speech using Whisper.

    Args:
        audio: 1D float32 numpy array (16 kHz mono) or file path.
        model_name: Whisper model size ('tiny', 'base', 'small', etc.).
        language: Optional language code (e.g. 'en', 'hi', 'fr').
        device: Target device for whisper ('cpu', 'cuda', etc.). Defaults to 'cpu'.

    Returns:
        Dict containing:
          - text: full transcribed text
          - language: detected or specified language
          - segments: list of timestamped segments
          - status: 'success' | 'empty' | 'failed'
          - error: error message if failed
    """
    try:
        model = get_whisper_model(model_name, device=device)

        if isinstance(audio, (str, Path)):
            input_data = str(audio)
        elif isinstance(audio, np.ndarray):
            input_data = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")

        transcribe_kwargs: Dict[str, Any] = {"fp16": False}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(input_data, **transcribe_kwargs)

        full_text = result.get("text", "").strip()
        detected_language = result.get("language", language or "unknown")

        raw_segments = result.get("segments", [])
        formatted_segments: List[Dict[str, Any]] = []

        for seg in raw_segments:
            seg_text = seg.get("text", "").strip()
            if seg_text:
                formatted_segments.append({
                    "id": seg.get("id"),
                    "start": round(float(seg.get("start", 0.0)), 2),
                    "end": round(float(seg.get("end", 0.0)), 2),
                    "text": seg_text,
                })

        status = "success" if full_text else "empty"

        return {
            "status": status,
            "text": full_text,
            "language": detected_language,
            "segments": formatted_segments,
            "segment_count": len(formatted_segments),
            "whisper_model": model_name,
            "error": None,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "text": "",
            "language": language or "unknown",
            "segments": [],
            "segment_count": 0,
            "whisper_model": model_name,
            "error": str(exc),
        }


# Alias for consistent naming
transcribe_audio = transcribe_speech
