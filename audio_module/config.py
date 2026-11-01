"""
config.py
=========
Central configuration for TruthLensAI Audio Module.
"""

from pathlib import Path

# Paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

# Supported audio extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

# Audio processing standards
TARGET_SAMPLE_RATE = 16000       # 16 kHz required by both Whisper and Wav2Vec2
MIN_AUDIO_DURATION_SEC = 0.5    # Minimum length for meaningful acoustic analysis
SILENCE_RMS_THRESHOLD = 1e-4    # Root Mean Square threshold below which audio is considered silent

# Synthetic speech detector constants
SYNTHETIC_SPEECH_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
SYNTHETIC_SPEECH_MODEL_NAME = "Wav2Vec2 Deepfake Audio Detector (mo-thecreator)"
SYNTHETIC_SPEECH_LICENSE = "Apache-2.0"

# Calibration thresholds for synthetic speech detection
# We intentionally do not use a harsh 0.5 binary cutoff to prevent false certainties.
PROB_FAKE_THRESHOLD_HIGH = 0.65   # >= 0.65 -> AI_GENERATED (Spoof)
PROB_FAKE_THRESHOLD_LOW  = 0.35   # <= 0.35 -> REAL (Bonafide Human)
# Between 0.35 and 0.65 -> UNCERTAIN

LABEL_MAP = {
    0: "AI_GENERATED",  # model output 0 is "fake"
    1: "REAL",          # model output 1 is "real"
}

# Whisper settings
DEFAULT_WHISPER_MODEL = "base"

# Verification settings
DEFAULT_MAX_CLAIMS_TO_VERIFY = 1
