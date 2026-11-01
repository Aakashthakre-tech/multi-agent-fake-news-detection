# TruthLensAI Baseline Fake-vs-Real Video Classifier

This directory contains the initial baseline model for binary video classification (`fake` vs `real`) on the TruthLensAI Fake Video Corpus (FVC).

## Architecture
- **Frame Backbone:** Pretrained `EfficientNet-B0` (ImageNet-1k weights), extracting 1280-dimensional feature embeddings across 12 uniformly sampled video frames.
- **Temporal Aggregation Head:** Multi-branch pooling combining learned Temporal Attention Pooling + Temporal Mean Pooling + Temporal Max Pooling ($3 \times 1280 = 3840$ features).
- **Classifier:** 2-layer MLP with LayerNorm, ReLU, and Dropout ($0.35$).
- **Loss:** Class-weighted Cross-Entropy to handle class distribution balance.

## Directory Structure
- `model.py` — Architecture definitions for `TemporalVideoClassifier` and `TemporalAttentionPooling`.
- `dataset.py` — Resilient frame-sampling module with uniform interval indexing and invalid frame fallbacks.
- `train.py` — Two-stage feature extraction, training, validation checkpoint selection, and test evaluation pipeline.
- `evaluate.py` — Single-video inference and evaluation CLI.
- `best_model.pt` — Saved PyTorch checkpoint of the best validation model.
- `config.json` — Hyperparameters and configuration.
- `class_mapping.json` — Index-to-label dictionary (`0: real`, `1: fake`).
- `test_metrics.json` — Final metrics on the held-out test split.
- `training_history.json` — Epoch-by-epoch loss, accuracy, and macro-F1 logs.
- `confusion_matrix.png` — Visualized confusion matrix plot on the test set.

## How to Run Training
```powershell
.venv\Scripts\python.exe video_module\baseline_classifier\train.py
```

## How to Run Single-Video Inference
```powershell
.venv\Scripts\python.exe video_module\baseline_classifier\evaluate.py path\to\video.mp4
```
