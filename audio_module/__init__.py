"""
TruthLensAI Audio Module
========================
Dedicated audio verification module providing:
1. Robust multi-format audio decoding (WAV, MP3, M4A, FLAC, AAC, OGG).
2. Deepfake / AI-generated speech detection (mo-thecreator/Deepfake-audio-detection).
3. Speech-to-text transcription via Whisper.
4. Factual claim extraction, filtering, and priority ranking.
5. Integration with live factual verification pipeline (run_research_pipeline).
6. Multi-signal consistency matrix separating Provenance from Veracity.
"""

from audio_module.audio_loader import (
    AudioFileNotFoundError,
    CorruptAudioError,
    SilentAudioError,
    UnsupportedAudioFormatError,
    load_audio,
    validate_audio_path,
)
from audio_module.audio_pipeline import verify_audio
from audio_module.claim_extractor import extract_claims_from_transcript
from audio_module.consistency import analyze_audio_consistency
from audio_module.synthetic_detector import get_synthetic_detector, predict_synthetic_speech
from audio_module.transcriber import transcribe_audio

__all__ = [
    "verify_audio",
    "load_audio",
    "validate_audio_path",
    "transcribe_audio",
    "extract_claims_from_transcript",
    "predict_synthetic_speech",
    "get_synthetic_detector",
    "analyze_audio_consistency",
    "AudioFileNotFoundError",
    "UnsupportedAudioFormatError",
    "CorruptAudioError",
    "SilentAudioError",
]
