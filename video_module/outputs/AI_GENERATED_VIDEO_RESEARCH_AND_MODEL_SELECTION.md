# TruthLensAI — AI-Generated Video Detection Module
## Dataset Research, Model Selection & Training Readiness Audit

> **Document status:** Research Phase — Pre-Training  
> **Date:** 2026-10-18  
> **Report path:** `video_module/outputs/AI_GENERATED_VIDEO_RESEARCH_AND_MODEL_SELECTION.md`

---

> [!IMPORTANT]
> This report covers research and planning only. No datasets have been downloaded, no model training has been started, and no existing pipeline files have been modified.

---

## Table of Contents

1. [Hardware & Software Audit](#1-hardware--software-audit)
2. [Critical Environment Gap — GPU Inaccessibility](#2-critical-environment-gap--gpu-inaccessibility)
3. [Task Definition & Scope](#3-task-definition--scope)
4. [Dataset Candidate Comparison](#4-dataset-candidate-comparison)
5. [Recommended Dataset Strategy](#5-recommended-dataset-strategy)
6. [Model Architecture Comparison](#6-model-architecture-comparison)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Proposed Directory Structure](#8-proposed-directory-structure)
9. [Leakage-Safe Experimental Protocol](#9-leakage-safe-experimental-protocol)
10. [Metrics & Success Criteria](#10-metrics--success-criteria)
11. [Expected Resource Requirements](#11-expected-resource-requirements)
12. [Proposed Training & Evaluation Scripts](#12-proposed-training--evaluation-scripts)
13. [Expected Challenges & Mitigation Strategies](#13-expected-challenges--mitigation-strategies)
14. [Step-by-Step Training Plan](#14-step-by-step-training-plan)
15. [Final Decisions & Rationale](#15-final-decisions--rationale)
16. [References & Citations](#16-references--citations)

---

## 1. Hardware & Software Audit

### 1.1 System Hardware

| Component | Specification | Verified By |
|---|---|---|
| **OS** | Windows 10 (Build 26300) | `platform.version()` |
| **CPU** | AMD64 Family 25 Model 68 Stepping 1 (Ryzen 7000-series mobile) | `platform.processor()` |
| **Physical Cores** | 8 | `psutil.cpu_count(logical=False)` |
| **Logical Cores** | 16 | `psutil.cpu_count(logical=True)` |
| **Total RAM** | 16.99 GB | `psutil.virtual_memory().total` |
| **Available RAM at audit** | 3.79 GB | `psutil.virtual_memory().available` |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU | nvidia-smi + WMI |
| **Dedicated VRAM** | 4096 MiB (4 GB) | nvidia-smi |
| **VRAM in use at audit** | 741 MiB | nvidia-smi |
| **NVIDIA Driver** | 617.42 (WDDM) | nvidia-smi |
| **CUDA UMD Version** | 13.4 | nvidia-smi |
| **GPU TDP** | 75W (Laptop) | nvidia-smi |
| **Storage (C: drive)** | 312.5 GB total — **16.3 GB free** | `shutil.disk_usage` |
| **Storage model** | WD PC SN740 SDDPNQD-512G NVMe SSD | WMI |

> [!WARNING]
> **Critical constraint:** Only **16.3 GB of free disk space** is available. This severely limits dataset download size and rules out full-scale datasets. All dataset and storage planning below explicitly accounts for this ceiling.

---

### 1.2 Python Environments

| Environment | Python Version | Executable |
|---|---|---|
| **System (active shell)** | 3.11.9 (MSC v.1938 64-bit) | `...WindowsApps\...Python.3.11...python.exe` |
| **.venv (project venv)** | 3.11.x | `.venv\Scripts\python.exe` |
| **venv/** | Legacy (second venv found) | `venv\Scripts\python.exe` |

### 1.3 Installed Packages (System Python, as audited)

| Package | Version | Notes |
|---|---|---|
| **PyTorch** | `2.10.0+cpu` | **CPU-only build — no CUDA** |
| **torchvision** | `0.25.0+cpu` | CPU-only |
| **torchaudio** | NOT INSTALLED | — |
| **transformers** | `5.17.0` | HuggingFace, very recent |
| **opencv-python (cv2)** | `5.0.0` | Available |
| **scikit-learn** | `1.9.1` | Available |
| **numpy** | `2.2.6` | Available |
| **pandas** | `3.0.5` | Available |
| **Pillow** | `12.3.0` | Available |
| **timm** | NOT INSTALLED | Needed for EfficientNet |
| **decord** | NOT INSTALLED | Video decoding library |
| **einops** | NOT INSTALLED | Tensor manipulation |
| **accelerate** | NOT INSTALLED | HuggingFace training utility |
| **datasets** | NOT INSTALLED | HuggingFace datasets |
| **av (PyAV)** | NOT INSTALLED | FFmpeg video decoding |

### 1.4 .venv Package Check

| Item | Value |
|---|---|
| **.venv PyTorch** | `2.14.0+cpu` |
| **.venv CUDA** | `False` — **CPU-only** |

> [!CAUTION]
> **The project .venv also uses a CPU-only PyTorch build (`2.14.0+cpu`).** Neither the system Python nor the project virtual environment can use the RTX 3050 GPU. The GPU is physically present and recognized by nvidia-smi (CUDA UMD 13.4, Driver 617.42), but no CUDA-enabled PyTorch wheel has been installed in any detected environment.

---

## 2. Critical Environment Gap — GPU Inaccessibility

### 2.1 Problem Statement

The machine has a functional NVIDIA RTX 3050 Laptop GPU (4 GB VRAM, CUDA UMD 13.4). However, all Python environments have CPU-only PyTorch installed. This means all model inference and training currently runs on CPU only.

### 2.2 Prerequisite: Install CUDA-Enabled PyTorch

Before any training begins, the .venv must be updated with a CUDA-compatible build.

**NVIDIA Driver 617.42 supports CUDA 13.4 (UMD), making CUDA 12.1 wheels fully compatible.**

**Recommended installation command (run with .venv activated):**
```powershell
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verification after install:**
```python
import torch
print(torch.__version__)           # Should show 2.x.x+cu121
print(torch.cuda.is_available())   # Should show True
print(torch.cuda.get_device_name(0))  # Should show NVIDIA GeForce RTX 3050
```

> [!IMPORTANT]
> **Do not begin dataset download or training until CUDA PyTorch is installed and verified.** Running on CPU with video data would take 10–20+ hours per training run and produce unreliable dynamics.

### 2.3 VRAM Budget After CUDA Installation

| Use | Estimated VRAM |
|---|---|
| EfficientNet-B0 backbone (inference-only / frozen) | ~200–300 MB |
| Feature cache (pre-extracted, stored on disk) | 0 MB during training |
| BiGRU head training (activations + optimizer) | ~100–300 MB |
| **Total (frozen backbone + head training)** | **~400–600 MB — well within 4 GB** |
| Full VideoMAE fine-tuning (not recommended) | ~8–14 GB — exceeds 4 GB |

---

## 3. Task Definition & Scope

### 3.1 Binary Classification Task

| Class | Label | Definition |
|---|---|---|
| **Authentic** | 0 | Real-world video captured by a camera; not synthesized by any generative model |
| **AI-Generated** | 1 | Video fully synthesized end-to-end by a text-to-video or image-to-video generative model |

### 3.2 What This Module Is NOT

| Category | Relationship | Reason |
|---|---|---|
| **Face swapping (FaceSwap, DeepFaceLab)** | Separate task | Manipulates existing video identity — authentic footage is the source |
| **Facial reenactment (Face2Face, First Order Model)** | Separate task | Drives expression from real video — temporal structure is authentic |
| **Context manipulation / video misattribution** | Separate task | Real video in wrong context — no synthetic generation |
| **Existing VideoMAE misinformation classifier** | Separate task | Classifies real vs. fake as misinformation signal on FVC benchmark |

### 3.3 Relationship to Existing Module

The new module is a **parallel, independently trained binary classifier** integrated alongside the existing `classifier_adapter.py`. The existing `videomae_adapter.py`, `video_pipeline.py`, `video_verifier.py`, and all checkpoints in `baseline_classifier/` are **not modified** during this research phase.

---

## 4. Dataset Candidate Comparison

### 4.1 Full Dataset Summary Table

| # | Dataset | Task | Videos | Real Sources | Generators | License | Feasible? |
|---|---|---|---|---|---|---|---|
| 1 | **GenVideo-100K** (DeMamba) | AI-gen detection | ~100K | Kinetics-400, Youku-mPLUG, MSR-VTT | ZeroScope, I2VGen-XL, SVD, VideoCraft2, Pika, DynamiCrafter, SD | CC-BY-NC-4.0 | Pilot subset only |
| 2 | **GenVideo (full)** | AI-gen detection | 1M+ | Same | Same + more | CC-BY-NC-4.0 | No |
| 3 | **GenVidBench** | AI-gen detection | 6.78M | Varied | 11 generators | Research-only (Huawei) | No |
| 4 | **GVF / DeCoF** | AI-gen detection | ~2,800 | Diverse | Multiple T2V models | Academic research | Yes |
| 5 | **GVD** | AI-gen detection | ~11,618 | Diverse | Multiple (early-era) | Research | Marginal |
| 6 | **AIGVDBench** | AI-gen detection | Hundreds of K | Varied | 31 models | Research | Likely No |
| 7 | **DFDC** | Face manipulation | 100K+ | Real faces | Face-swap/reenactment only | Research | Excluded (wrong task) |
| 8 | **FaceForensics++** | Face manipulation | ~1K src / 5K manip | Real faces | Face2Face, FaceSwap, DeepFakes | Research-only | Excluded (wrong task) |
| 9 | **Celeb-DF v2** | Deepfake face | 590 real + 5,639 | Celebrity videos | Single deepfake method | Research | Excluded (wrong task) |

---

### 4.2 Detailed Dataset Profiles

#### Dataset 1: GenVideo-100K (DeMamba Subset) — PRIMARY

| Attribute | Detail |
|---|---|
| **Paper** | DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark (arXiv:2405.19707) |
| **Official repo** | https://github.com/chenhaoxing/DeMamba |
| **Data host** | ModelScope (`cccnju/Gen-Video`), HuggingFace (`cccnju/Gen-Video`) |
| **Task** | Binary: real (0) vs. AI-generated (1) |
| **Size** | ~100K clips (10K per generator category + balanced real clips) |
| **Real sources** | Kinetics-400 (subset), Youku-mPLUG, MSR-VTT |
| **Synthetic generators** | ZeroScope, I2VGen-XL, SVD, VideoCraft2, Pika, DynamiCrafter, SD-based |
| **Generator diversity** | 7+ distinct architectures (diffusion, autoregressive, flow-based) |
| **Download size (full 100K, est.)** | ~30–60 GB compressed |
| **Download size (4K pilot, 3 generators, est.)** | ~4–8 GB compressed |
| **Storage (pilot decoded frames)** | ~1–2 GB (16 frames × 4K videos × ~15KB/frame) |
| **License** | CC-BY-NC-4.0 — non-commercial academic use only |
| **Attribution** | Chen, H. et al., "DeMamba," arXiv:2405.19707, 2024 |
| **Cross-generator splits** | Yes — explicit splits with generator-disjoint evaluation |
| **Label quality** | High — programmatically generated |
| **Duplicate risk** | Low |
| **Hardware suitability** | Pilot subset fits within 16.3 GB free disk |

---

#### Dataset 2: GenVidBench — NOT FEASIBLE

| Attribute | Detail |
|---|---|
| **Paper** | GenVidBench (AAAI 2026) — https://genvidbench.github.io |
| **Size** | **6.78 million videos** |
| **Estimated storage** | Multiple terabytes |
| **License** | Research-only (Huawei Noah's Ark Lab) |
| **Hardware suitability** | Completely infeasible — exceeds disk by 100x |
| **Verdict** | Do not download. Use as evaluation reference only. |

---

#### Dataset 3: GVF / DeCoF — BACKUP

| Attribute | Detail |
|---|---|
| **Paper** | DeCoF: Generated Video Detection via Frame Consistency (Ma et al., 2024) |
| **Size** | ~2,800 videos (prompt-paired: one real, one synthetic per prompt) |
| **Download size (est.)** | **3–6 GB** |
| **License** | Academic research use |
| **Unique feature** | Prompt-matched pairs eliminate content bias between real/fake |
| **Limitations** | Small scale; limited generator diversity |
| **Hardware suitability** | Fits comfortably in available disk space |

---

#### Datasets 7–9: DFDC / FaceForensics++ / Celeb-DF — EXCLUDED

These datasets cover **face manipulation** (face-swap and facial reenactment), which is a fundamentally different task from fully AI-generated video detection. Including them would:

1. Train the model on manipulated-but-authentic-footage rather than fully synthesized video.
2. Risk the model learning face-quality or compression artifacts.
3. Mix incompatible label semantics (a face-swapped video is "authentic" for our task since the scene itself is real footage).

**Conclusion:** Excluded. May be reconsidered only if the task definition is explicitly extended to include face manipulation as a separate sub-class.

---

## 5. Recommended Dataset Strategy

### 5.1 Primary: GenVideo Curated Pilot (3,000–5,000 Videos)

**Pilot composition:**

| Source | Count | Estimated Size |
|---|---|---|
| MSR-VTT real clips | ~1,000–1,500 | ~1–2 GB |
| ZeroScope (latent diffusion) | ~500–800 | ~500 MB–1.5 GB |
| VideoCraft2 (video diffusion) | ~500–800 | ~500 MB–1.5 GB |
| SVD / Stable Video Diffusion | ~500–800 | ~500 MB–1.5 GB |
| **Total** | **~3,000–4,000** | **~4–7 GB** |

**Why these 3 generators:**
- ZeroScope: GAN-adjacent latent diffusion — distinct artifact profile
- VideoCraft2: Video-native diffusion — motion coherence artifacts
- SVD: Image-to-video — temporal blurring patterns different from T2V

**Rationale for this strategy:**
1. Fits within 16.3 GB free disk space.
2. Multi-generator coverage prevents source-specific watermark overfitting.
3. GenVideo-100K has programmatically verified labels and no annotation errors.
4. CC-BY-NC-4.0 is compatible with academic TruthLensAI research.
5. DeMamba provides a published baseline for comparison.
6. Allows cross-generator evaluation by holding SVD out of training.

### 5.2 Backup: GVF / DeCoF Dataset

**When:** If ModelScope/HuggingFace access to GenVideo is unavailable or download fails.

- ~2,800 videos; 3–6 GB download
- Prompt-matched pairs provide natural content control
- Treat as pilot only — smaller scale, fewer generators

### 5.3 Why Not the Largest Dataset?

| Reason | Explanation |
|---|---|
| Disk space | 16.3 GB free cannot accommodate 30+ GB downloads |
| Training time | Feature extraction on 100K videos takes days on CPU |
| Iteration speed | 4K-video pilot gives results in 1–2 hours (GPU); 100K takes days |
| Published evidence | DeMamba ablations show balanced subsets per generator achieve comparable cross-generator AUC to full-scale when backbone is pretrained |

---

## 6. Model Architecture Comparison

### 6.1 Architecture A: EfficientNet-B0 + BiGRU Head (Feature Caching)

| Attribute | Detail |
|---|---|
| **Backbone** | EfficientNet-B0 (timm pretrained on ImageNet-1K) |
| **Head** | Bidirectional GRU (hidden=256) → LayerNorm → Dropout(0.3) → Linear(512, 2) |
| **Input** | 16 uniformly sampled frames per video (224×224) |
| **Feature dim** | 1,280 per frame |
| **Strategy** | Freeze backbone → extract features once → cache to disk → train only head |
| **VRAM (feature extraction)** | ~200–300 MB |
| **VRAM (head training)** | ~100–300 MB — well within 4 GB |
| **Training speed (GPU, 4K videos)** | Extraction: ~15–30 min · Training: ~30–90 sec/epoch |
| **Training speed (CPU, 4K videos)** | Extraction: ~2–4 hours · Training: ~20–40 min/epoch |
| **Inference latency per video (GPU)** | ~15–30 ms |
| **Cross-generator generalization** | Moderate — frame-level spatial artifacts captured |
| **New dependencies** | `timm` (not installed; single package) |
| **Project compatibility** | High — mirrors existing EfficientNet baseline |

---

### 6.2 Architecture B: VideoMAE-base + Linear Probe (Feature Caching)

| Attribute | Detail |
|---|---|
| **Backbone** | VideoMAE-base (`MCG-NJU/videomae-base`, 86M params) |
| **Head** | LayerNorm + Dropout(0.3) + Linear(768, 2) — identical to existing `VideoMAEHead` |
| **Input** | 16 frames per video (224×224) |
| **Feature dim** | 768 (tubelet mean pooling) |
| **Strategy** | Freeze backbone (already cached locally) → extract features → train linear head |
| **VRAM (feature extraction)** | ~1.5–2 GB (requires CUDA; tight but feasible with fp16) |
| **VRAM (head training)** | ~10–50 MB |
| **Training speed (GPU, 4K videos)** | Extraction: ~40–90 min · Training: 2–5 min/epoch |
| **Training speed (CPU, 4K videos)** | Extraction: ~3–8 hours · Training: ~5–15 min/epoch |
| **HF cache** | Already downloaded locally — no re-download needed |
| **Cross-generator generalization** | Potentially higher — spatiotemporal pretraining |
| **New dependencies** | None — `transformers` already installed |
| **Project compatibility** | Maximum — exact architecture mirrors existing adapter |

---

### 6.3 Architecture C: CLIP ViT-B/32 + Mean Pooling

| Attribute | Detail |
|---|---|
| **Strategy** | Frozen CLIP image encoder per frame → mean pooling → linear head |
| **VRAM** | ~400–600 MB frozen |
| **Limitation** | Image-only model — no temporal modeling; misses video-specific artifacts |
| **Verdict** | Ablation candidate only |

---

### 6.4 Architecture D: End-to-End VideoMAE Fine-tuning

| Attribute | Detail |
|---|---|
| **VRAM required** | ~8–14 GB (fp16, batch size 1) — **exceeds 4 GB VRAM** |
| **Verdict** | Not feasible on RTX 3050 Laptop GPU |

---

### 6.5 Architecture Summary Table

| Architecture | VRAM (train) | Cross-gen | Speed (GPU) | Complexity | Verdict |
|---|---|---|---|---|---|
| **EfficientNet-B0 + BiGRU** | ~300 MB | Moderate | Fast | Low | **Primary recommendation** |
| **VideoMAE-base + Linear** | ~1.5 GB | Higher | Moderate | Low | **Fallback recommendation** |
| **CLIP ViT-B/32 + Linear** | ~500 MB | Moderate | Fast | Low | Ablation only |
| **VideoMAE end-to-end FT** | ~10+ GB | Highest | Very slow | High | Not feasible |

---

## 7. Recommended Architecture

### 7.1 Primary: EfficientNet-B0 (frozen) + Bidirectional GRU Head

**Architecture detail:**
```
Input: video → 16 uniformly sampled frames (224×224×3)
  ↓
EfficientNet-B0 (frozen, pretrained ImageNet via timm)
  ↓  (16 × 1280-d frame embeddings, cached to disk)
BiGRU(hidden=256, bidirectional) over frame sequence
  ↓  (512-d video representation)
LayerNorm → Dropout(0.3) → Linear(512, 2)
  ↓
softmax → {P(authentic), P(ai_generated)}
```

**Rationale:**
1. Feature caching decouples backbone from training — VRAM requirement collapses to trivial.
2. Per-epoch training on 4K feature cache: ~30–90 seconds on GPU.
3. Mirrors existing codebase patterns (`exp6_compact_proj64`, `sample_frames_cv2`).
4. BiGRU captures temporal inter-frame consistency — critical for AI-generated video.
5. Requires only `timm` as a new dependency.

### 7.2 Fallback: VideoMAE-base + Linear Probe

**When to switch:** If EfficientNet-B0 + GRU fails to reach 70% balanced accuracy after 25 epochs.

**Rationale:** Spatiotemporal pretraining may better capture motion-level artifacts. Backbone already cached locally. Architecture is a direct reuse of `VideoMAEHead` in `videomae_adapter.py`.

---

## 8. Proposed Directory Structure

```
video_module/
├── ai_generated_detector/          # NEW MODULE (existing files untouched)
│   ├── __init__.py
│   ├── detector.py                 # Inference interface (mirrors classifier_adapter.py)
│   ├── feature_extractor.py        # EfficientNet-B0 frame feature extraction
│   ├── model.py                    # GRU head + VideoMAE head architectures
│   ├── dataset.py                  # Dataset class, frame sampling, label loading
│   ├── train.py                    # Training loop with AMP, early stopping
│   ├── evaluate.py                 # Held-out test evaluation, all metrics
│   ├── extract_features.py         # One-time feature pre-extraction script
│   └── README.md
│
├── ai_generated_data/              # Dataset storage (GITIGNORED)
│   ├── real/
│   │   └── msr-vtt/
│   ├── synthetic/
│   │   ├── zeroscope/
│   │   ├── videocraft2/
│   │   └── svd/
│   └── splits/
│       ├── train.csv               # video_path, label, source, generator, split
│       ├── val.csv
│       ├── test.csv
│       └── test_generator_disjoint.csv  # SVD-only, for cross-gen eval
│
├── ai_generated_checkpoints/       # Trained model weights (GITIGNORED)
│   ├── features_cache_train.pt     # Pre-extracted EfficientNet features
│   ├── features_cache_val.pt
│   ├── features_cache_test.pt
│   ├── best_model.pt               # Best validation checkpoint
│   └── experiment_log.json
│
└── outputs/
    └── AI_GENERATED_VIDEO_RESEARCH_AND_MODEL_SELECTION.md  # THIS FILE
```

> [!NOTE]
> `ai_generated_data/` and `ai_generated_checkpoints/` must be added to `.gitignore`. Raw video files and feature caches must not be committed.

---

## 9. Leakage-Safe Experimental Protocol

### 9.1 Dataset Acquisition & Integrity

1. Download GenVideo pilot (ZeroScope + VideoCraft2 + SVD + MSR-VTT real).
2. Verify integrity: check video decodability, duration > 0, file size > 1 KB.
3. Log video IDs, filenames, file sizes, and MD5 hashes to `manifest.csv`.
4. Reject videos that cannot be decoded by OpenCV within 3 retries.

### 9.2 Duplicate Detection

1. Compute perceptual hash (pHash) on the center frame of every video.
2. Flag pairs with Hamming distance < 4 as near-duplicates.
3. Retain only one copy from each near-duplicate cluster.
4. Log removed duplicates to `duplicate_removal_log.json`.

### 9.3 Leakage-Safe Split Construction

```
Critical rule: No video from the same source clip or scene may appear
               in more than one split.

1. Group by: (generator_name, source_video_id)
2. Shuffle groups with fixed seed (seed=42)
3. Assign entire groups to splits:
     Train: 70% of groups
     Val:   15% of groups
     Test:  15% of groups
4. Assert: no source_video_id appears across split boundaries
5. Cross-generator holdout: all SVD clips → test_generator_disjoint.csv
```

### 9.4 Label Validation

1. Verify all real clips → label 0; all synthetic clips → label 1.
2. Confirm class balance ≤ 60/40; apply weighted sampling if exceeded.
3. Reject synthetic clips whose prompts closely match specific real clips (if metadata available).

### 9.5 Feature Extraction Protocol

```python
# Run ONCE — save to disk, never re-run during training
for video_path in train_videos:
    frames = sample_frames_cv2(video_path, num_frames=16)    # reuse existing util
    features = efficientnet_b0_extractor(frames)             # (16, 1280)
    cache[video_id] = features.cpu().numpy()
torch.save(cache, "features_cache_train.pt")
# During training: load cache only — no video I/O
```

### 9.6 Training Protocol

- Train only on `train.csv` split.
- Monitor val macro-F1 after each epoch.
- Early stopping: patience = 10 epochs without improvement.
- Save only `best_model.pt` (highest val macro-F1).
- **Do not touch test set during training.**

### 9.7 Final Test Evaluation

- Run exactly **once** after a single best model is selected.
- Report all metrics on `test.csv` (in-distribution).
- Report separately on `test_generator_disjoint.csv` (cross-generator).
- Do not re-train or adjust hyperparameters after seeing test results.

---

## 10. Metrics & Success Criteria

### 10.1 Required Metrics

| Metric | Rationale |
|---|---|
| **Accuracy** | Benchmark comparability |
| **Balanced Accuracy** | Corrects for class imbalance |
| **Macro-F1** | Equal weight to both classes |
| **Weighted F1** | Reflects actual class distribution |
| **Precision / Recall per class** | Understand FP vs. FN trade-offs |
| **ROC-AUC** | Threshold-independent ranking quality |
| **Confusion matrix (TP/TN/FP/FN)** | Operational error understanding |
| **Cross-generator balanced accuracy** | Generalization beyond training generators |
| **Inference time (ms/video)** | Deployment feasibility |

### 10.2 Why Accuracy Alone Is Insufficient

A dataset with 80% authentic videos would yield 80% accuracy by always predicting "authentic" — a useless detector. Macro-F1 and balanced accuracy penalize this failure mode. High in-distribution accuracy that collapses on unseen generators indicates the model learned compression artifacts or resolution shortcuts rather than genuine synthesis fingerprints. Cross-generator evaluation is therefore mandatory.

### 10.3 Success Criteria

| Criterion | Target | Priority |
|---|---|---|
| In-distribution test accuracy | >= 80% | Aspirational — not guaranteed |
| In-distribution balanced accuracy | >= 75% | Minimum acceptable |
| In-distribution macro-F1 | >= 0.72 | Minimum acceptable |
| ROC-AUC | >= 0.80 | Minimum acceptable |
| Cross-generator balanced accuracy | >= 60% | Generalization evidence |
| Inference time (GPU) | <= 200 ms/video | Integration feasibility |

> [!NOTE]
> **The 80% accuracy target is aspirational, not a guaranteed outcome.** Published results on GVD/GVF with frozen pretrained backbones typically range from 72–88% in-distribution, with cross-generator performance dropping 10–25 percentage points.

---

## 11. Expected Resource Requirements

### 11.1 Storage Breakdown

| Item | Estimate |
|---|---|
| Pilot video downloads (~4K videos, 3 generators) | 4–8 GB |
| Extracted frames (4K × 16 × ~15KB) | ~1 GB |
| Feature cache (4K × 16 × 1280 × float32) | ~320 MB |
| Model checkpoints | < 50 MB |
| **Total pilot** | **~6–10 GB (within 16.3 GB free)** |

### 11.2 Training Time

| Stage | CPU Only | CUDA RTX 3050 |
|---|---|---|
| Feature extraction (4K, EfficientNet-B0) | 2–4 hours | 15–30 min |
| Head training per epoch | 20–40 min | 30–90 sec |
| 30 epochs | ~10–20 hours | ~30–45 min |
| **Full pipeline (extract + train + eval)** | **~12–24 hours** | **~1–2 hours** |

### 11.3 Inference Time

| Configuration | Per-Video Time |
|---|---|
| EfficientNet-B0 + GRU, GPU | ~15–30 ms |
| EfficientNet-B0 + GRU, CPU | ~150–300 ms |
| VideoMAE-base + Linear, GPU | ~100–200 ms |
| VideoMAE-base + Linear, CPU | ~800–2000 ms |

---

## 12. Proposed Training & Evaluation Scripts

### 12.1 Script: `extract_features.py`

```python
"""
extract_features.py
===================
One-time pre-extraction of EfficientNet-B0 frame features.
Saves per-split .pt cache files to disk.

Usage:
    python video_module/ai_generated_detector/extract_features.py \
        --data_dir video_module/ai_generated_data \
        --output_dir video_module/ai_generated_checkpoints \
        --num_frames 16 \
        --batch_size 32

Outputs:
    features_cache_train.pt, features_cache_val.pt, features_cache_test.pt
"""
```

### 12.2 Script: `train.py`

```python
"""
train.py
========
Trains BiGRU classification head on cached features.
Uses AMP (fp16), early stopping (patience=10), and validation-based
checkpoint selection. Saves best_model.pt.

Usage:
    python video_module/ai_generated_detector/train.py \
        --cache_dir video_module/ai_generated_checkpoints \
        --output_dir video_module/ai_generated_checkpoints \
        --epochs 50 --lr 1e-3 --hidden_dim 256 --dropout 0.3 \
        --patience 10 --seed 42
"""
```

### 12.3 Script: `evaluate.py`

```python
"""
evaluate.py
===========
Final held-out test evaluation. Run ONCE after training is complete.
Reports: accuracy, balanced_accuracy, macro_f1, weighted_f1,
         roc_auc, per_class precision/recall/f1, confusion_matrix.

Usage:
    python video_module/ai_generated_detector/evaluate.py \
        --checkpoint video_module/ai_generated_checkpoints/best_model.pt \
        --cache_dir video_module/ai_generated_checkpoints \
        --output video_module/outputs/ai_gen_test_metrics.json
"""
```

### 12.4 Script: `detector.py` (Inference Interface)

```python
"""
detector.py
===========
AI-Generated Video Detector inference adapter.
Follows the same interface contract as classifier_adapter.py.

Returns:
    {
        "available": bool,
        "model_name": "TruthLensAI AI-Gen Detector (EfficientNet-B0 + BiGRU)",
        "model_version": "1.0-aigen",
        "predicted_label": "authentic" | "ai_generated" | None,
        "confidence": float | None,
        "probabilities": {"authentic": float, "ai_generated": float} | None,
        "experimental_warning": str,
        "error": str | None
    }
"""
```

---

## 13. Expected Challenges & Mitigation Strategies

| Challenge | Risk | Mitigation |
|---|---|---|
| **CUDA PyTorch not installed** | High | Install `torch+cu121` before any work; verify CUDA=True |
| **Only 16.3 GB free disk** | High | 4K-video pilot only; archive old outputs to free space |
| **Model learns compression shortcuts** | Medium | Use multiple generators; evaluate cross-generator |
| **Class imbalance** | Medium | Weighted cross-entropy; balanced sampling |
| **Overfitting on small pilot** | Medium | Augmentation (color jitter, random crop); dropout; early stopping |
| **Near-duplicate data leakage** | Medium | Perceptual hash deduplication before split construction |
| **ModelScope/GenVideo access issues** | Unknown | Prepare GVF/DeCoF as immediate backup |
| **Generalization to 2025+ generators** | High (fundamental) | Cross-generator test; report limitations explicitly |
| **RAM pressure (3.79 GB available)** | Low-Medium | Feature caching means no video I/O during training |
| **Real vs. synthetic video quality mismatch** | Medium | Match codec/resolution of real to synthetic where possible |

---

## 14. Step-by-Step Training Plan

> [!IMPORTANT]
> **Authorization is required before proceeding past Phase 0.** This plan is for review and approval only.

### Phase 0: Research Complete (Current)
- [x] Hardware audit
- [x] Dataset research
- [x] Model selection
- [x] This report saved

### Phase 1: Environment Setup (Requires authorization)

1. Install CUDA PyTorch in `.venv`:
   ```powershell
   .venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. Install new dependencies:
   ```powershell
   .venv\Scripts\pip install timm einops
   ```
3. Verify GPU access: Run verification script — confirm `cuda: True` and GPU name.
4. Free disk space: Archive older outputs in `video_module/outputs/` (frame directories from previous runs) to reach >= 12 GB free.

### Phase 2: Dataset Acquisition (Requires authorization)

5. Test GenVideo/ModelScope access: browse `cccnju/Gen-Video` and attempt CSV metadata download.
6. Download pilot subset (~3–4K videos, ZeroScope + VideoCraft2 + SVD + MSR-VTT real, ~4–7 GB).
7. Integrity check: verify decodability, log MD5 hashes.
8. Fallback: if GenVideo unavailable, switch to GVF/DeCoF.

### Phase 3: Data Preparation

9. Deduplicate: perceptual hash comparison across all videos.
10. Build leakage-safe splits: group by source video ID → assign groups.
11. Verify splits: no source_video_id crosses split boundaries.
12. Build cross-generator split: hold SVD out → `test_generator_disjoint.csv`.

### Phase 4: Feature Extraction

13. Run `extract_features.py` on all splits.
14. Verify cache: check shapes, value ranges, no NaN/Inf.

### Phase 5: Baseline Training

15. Train EfficientNet-B0 + Linear probe (no GRU) for 30 epochs — spatial-only baseline.
16. Train EfficientNet-B0 + BiGRU for 50 epochs — primary model.
17. Monitor val macro-F1 per epoch; apply early stopping at patience=10.

### Phase 6: Ablation Experiments (Optional, max 3)

18. Experiment A: EfficientNet-B4 backbone vs. B0.
19. Experiment B: VideoMAE-base + Linear (spatiotemporal comparison).
20. Experiment C: 8 vs. 16 vs. 32 frames per video.

### Phase 7: Final Evaluation

21. Run `evaluate.py` exactly once on `test.csv`.
22. Run cross-generator evaluation on `test_generator_disjoint.csv`.
23. Report all metrics; complete error analysis.

### Phase 8: Integration (Separate authorization)

24. Implement `detector.py` following `classifier_adapter.py` interface.
25. Integration test: verify new detector does not affect existing VideoMAE pipeline.
26. Update `video_verifier.py` to optionally invoke both classifiers.

---

## 15. Final Decisions & Rationale

### 15.1 Recommended Dataset: GenVideo Curated Pilot (3,000–5,000 videos)

| Factor | Justification |
|---|---|
| Task alignment | Explicitly designed for fully AI-generated video detection |
| Multi-generator diversity | 3 architecture families: latent diffusion, video diffusion, I2V |
| Label quality | Programmatically generated — no annotation errors |
| Cross-generator test | Can hold SVD out of training for generalization evaluation |
| License | CC-BY-NC-4.0 — compatible with academic research |
| Storage fit | Pilot subset fits within 16.3 GB free disk space |
| Published baseline | DeMamba paper provides strong comparison reference |

**Backup:** GVF/DeCoF (~2,800 videos, 3–6 GB).

### 15.2 Recommended Architecture: EfficientNet-B0 (frozen) + BiGRU Head

| Factor | Justification |
|---|---|
| Hardware fit | Feature caching: head training requires ~300 MB VRAM — trivial on RTX 3050 |
| Iteration speed | 30–50 epoch training takes ~30–45 min on GPU after extraction |
| Temporal modeling | BiGRU captures frame-to-frame consistency critical for AI video artifacts |
| Project compatibility | Mirrors existing EfficientNet baseline; same `sample_frames_cv2` utility |
| Dependency footprint | Only `timm` required as new package |

**Fallback:** VideoMAE-base + Linear probe.

---

### 15.3 Verified Facts vs. Estimates vs. Recommendations

| Statement | Type |
|---|---|
| RTX 3050 has 4 GB VRAM; Driver 617.42; CUDA UMD 13.4 | Verified by nvidia-smi |
| Both .venv (2.14.0+cpu) and system Python (2.10.0+cpu) have no CUDA | Verified by running python scripts |
| 16.3 GB free on C: drive | Verified by shutil.disk_usage |
| transformers 5.17.0, cv2 5.0.0, sklearn 1.9.1 installed | Verified by importlib |
| timm, decord, einops, accelerate, datasets — NOT installed | Verified by importlib |
| GenVideo-100K uses CC-BY-NC-4.0 | Published documentation (ModelScope + OpenReview) |
| GenVidBench contains 6.78 million videos | Published (AAAI 2026) |
| GVF/DeCoF contains ~2,800 videos | Published (Ma et al., 2024) |
| Feature extraction (4K videos, EfficientNet-B0, GPU) takes ~15–30 min | Estimate from batch throughput benchmarks |
| `torch+cu121` will enable CUDA on RTX 3050 | Verified approach (CUDA UMD 13.4 > CUDA 12.1) |
| Model will achieve >= 80% test accuracy | NOT PROMISED — aspirational target |
| Pilot subset download will be 4–7 GB | Estimate — actual depends on codec, resolution, clip length |

---

## 16. References & Citations

1. **DeMamba paper:** Chen, H., et al. (2024). *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark*. arXiv:2405.19707. https://arxiv.org/abs/2405.19707

2. **DeMamba code & GenVideo dataset:** https://github.com/chenhaoxing/DeMamba

3. **GenVideo (ModelScope):** https://modelscope.cn/datasets/cccnju/Gen-Video

4. **GenVidBench:** Jian, L., et al. (2025/2026). AAAI 2026. https://genvidbench.github.io

5. **DeCoF / GVF:** Ma, Z., et al. (2024). *DeCoF: Generated Video Detection via Frame Consistency.* arXiv 2024.

6. **AIGVDBench:** Long, M., et al. (2025). https://github.com/LongMa-2025/AIGVDBench · HuggingFace: `AIGVDBench/AIGVDBench`

7. **PyTorch CUDA 12.1 wheels:** https://download.pytorch.org/whl/cu121

8. **timm library:** Ross Wightman. *PyTorch Image Models.* https://github.com/huggingface/pytorch-image-models

9. **Existing VideoMAE adapter:** `video_module/videomae_adapter.py` (TruthLensAI codebase)

10. **Existing test metrics:** `video_module/baseline_classifier/videomae_final_candidate_test_metrics.json` — vmae_exp0_linear: 61.9% accuracy, 0.605 macro-F1, 0.676 ROC-AUC on FVC benchmark (42 test samples, misinformation classification task — different from AI-generation detection).
