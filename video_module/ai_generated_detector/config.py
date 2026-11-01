"""
config.py
=========
Central configuration for the AI-generated video detection module.
All paths, hyperparameters, and constants are defined here.
"""

from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────────
MODULE_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
VIDEO_MODULE = MODULE_DIR.parent

DATA_DIR      = VIDEO_MODULE / "ai_generated_data"
REAL_DIR      = DATA_DIR / "real"
SYNTH_DIR     = DATA_DIR / "synthetic"
SPLITS_DIR    = DATA_DIR / "splits"
CKPT_DIR      = VIDEO_MODULE / "ai_generated_checkpoints"
OUTPUTS_DIR   = VIDEO_MODULE / "outputs"

# ── Dataset constants ────────────────────────────────────────────────────────
CLASS_MAP     = {"REAL": 0, "AI_GENERATED": 1}
INV_CLASS_MAP = {0: "REAL", 1: "AI_GENERATED"}
CLASS_NAMES   = ["REAL", "AI_GENERATED"]

# Pilot budget
PILOT_REAL_TARGET  = 250
PILOT_FAKE_TARGET  = 250
PILOT_TOTAL_TARGET = 500

# Generators included in the pilot
PILOT_GENERATORS = ["videocrafter2", "opensora"]   # two synthetic families
HOLDOUT_GENERATOR = "opensora"                     # reserved for OOD test

# ── Frame sampling ───────────────────────────────────────────────────────────
NUM_FRAMES    = 8           # frames per video (reduced for memory/speed)
FRAME_SIZE    = 224         # height and width (EfficientNet-B0 native)

# ── Feature extraction ──────────────────────────────────────────────────────
BACKBONE_NAME = "efficientnet_b0.ra_in1k"   # timm model ID
FEATURE_DIM   = 1280                         # EfficientNet-B0 embedding dim
EXTRACT_BATCH = 16                           # videos per extraction batch

# ── Model head ──────────────────────────────────────────────────────────────
GRU_HIDDEN    = 256
GRU_LAYERS    = 1
GRU_BIDIR     = True
DROPOUT       = 0.3
NUM_CLASSES   = 2

# ── Training ─────────────────────────────────────────────────────────────────
SEED          = 42
TRAIN_EPOCHS  = 60
BATCH_SIZE    = 16          # feature-cache batch size (tiny; fits easily in RAM)
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15          # early-stopping patience
GRAD_CLIP     = 1.0

# ── Splits ───────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ── Checkpoint / cache file names ────────────────────────────────────────────
CACHE_TRAIN  = CKPT_DIR / "features_train.pt"
CACHE_VAL    = CKPT_DIR / "features_val.pt"
CACHE_TEST   = CKPT_DIR / "features_test.pt"
CACHE_OOD    = CKPT_DIR / "features_ood.pt"
BEST_MODEL   = CKPT_DIR / "best_model.pt"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
TRAIN_CSV    = SPLITS_DIR / "train.csv"
VAL_CSV      = SPLITS_DIR / "val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"
OOD_CSV      = SPLITS_DIR / "ood_generator.csv"
EXP_LOG      = CKPT_DIR / "experiment_log.json"

# ── Model metadata ────────────────────────────────────────────────────────────
MODEL_NAME    = "TruthLensAI AI-Gen Detector (EfficientNet-B0 + BiGRU)"
MODEL_VERSION = "1.0-pilot"
EXPERIMENTAL_WARNING = (
    "PILOT EXPERIMENT: Trained on ~500 videos from 2 generators. "
    "Results are preliminary. Do not treat as production-grade detection."
)
