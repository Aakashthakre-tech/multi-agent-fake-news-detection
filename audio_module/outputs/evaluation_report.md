# TruthLensAI — Audio Detector Evaluation Report

> **Evaluation Date:** 2026-10-19  
> **Dataset:** 16 labelled samples (8 original validation + 8 new extended)  
> **Model:** `mo-thecreator/Deepfake-audio-detection` (Wav2Vec2-base)  
> **Thresholds:** P(fake) >= 0.65 → AI_GENERATED | <= 0.35 → REAL | else UNCERTAIN  
> **Hardware:** NVIDIA RTX 3050 Laptop GPU (CUDA) + CPU fallback  

---

## 1. Dataset Summary

| # | Split | Count | Ground Truths |
|---|-------|-------|---------------|
| A | Original validation_samples | 8 | 4×REAL (EN×2, HI×1, MR×1), 4×AI_GENERATED (EN SAPI) |
| B | Extended extended_samples   | 8 | 4×REAL (EN compressed×2, EN proxy×1, HI proxy×1), 3×AI_GENERATED (formant/buzz), 1×UNCERTAIN edge-case |
| — | **Total** | **16** | **8 REAL, 7 AI_GENERATED, 1 UNCERTAIN (edge-case)** |

> [!NOTE]
> Two extended 'REAL' samples (EXT_02, EXT_06) are **parametric acoustic proxies** (not actual
> recorded speech). Their labels carry higher uncertainty and results should be interpreted cautiously.
> EXT_08 is an **edge-case** excluded from accuracy metrics — it tests the failsafe guard only.

---

## 2. Aggregate Metrics
*(Computed on 15 classifiable samples; EXT_08 excluded)*

| Metric | Value |
|--------|-------|
| Accuracy | **0.6667** |
| Balanced Accuracy | 0.6696 |
| Precision (AI_GEN=positive) | 0.6250 |
| Recall (Sensitivity) | 0.7143 |
| F1-Score | 0.6667 |
| Specificity (True Negative Rate) | 0.6250 |

**Confusion Matrix:**

| | Predicted AI_GENERATED | Predicted REAL |
|-|------------------------|----------------|
| **Actual AI_GENERATED** | TP = 5 | FN = 2 |
| **Actual REAL** | FP = 3 | TN = 5 |

---

## 3. Per-Language Results

| Language | Accuracy | Correct | Total | FP | FN |
|----------|----------|---------|-------|----|----|
| English | 0.6667 | 8 | 12 | 2 | 2 |
| Hindi | 1.0000 | 2 | 2 | 0 | 0 |
| Marathi | 0.0000 | 0 | 1 | 1 | 0 |

---

## 4. False Positive Analysis

Total false positives: **3** (REAL samples predicted as AI_GENERATED).

### SAMPLE_01 — `human_01_reuters_news.wav`
- **Language:** English
- **Category:** Human Broadcast (Reuters Storm Report)
- **P(synthetic):** 0.9902
- **Audio RMS:** 0.025768
- **Notes:** Known false-positive in prior benchmark: model predicted AI_GENERATED with P(synth)=0.990. Investigating whether broadcast compression or recording conditions are the cause.

### SAMPLE_03 — `human_03_marathi_report.wav`
- **Language:** Marathi
- **Category:** Human Speech (Marathi Field Report)
- **P(synthetic):** 1.0
- **Audio RMS:** 0.045759
- **Notes:** Known false-positive: model predicted AI_GENERATED with P(synth)=1.0. Marathi is likely out-of-distribution for this English-centric model.

### EXT_05 — `human_07_en_compressed_64k.wav`
- **Language:** English
- **Category:** Human Broadcast — 64 kbps MP3 re-encoded (source: Reuters false-positive)
- **P(synthetic):** 0.9939
- **Audio RMS:** 0.024467
- **Notes:** Source is the known false-positive (SAMPLE_01: Reuters). Tests whether 64 kbps compression changes detector behaviour compared to original.

### Root Cause Assessment

Based on the false-positive pattern observed:

1. **Marathi (SAMPLE_03):** P(synth)=1.000 with no uncertainty. The Wav2Vec2 model (`mo-thecreator/Deepfake-audio-detection`) is fine-tuned from `facebook/wav2vec2-base`, which was pre-trained on English LibriSpeech. Marathi phonology, tonal patterns, and formant structure are substantially different from English. The model likely interprets Marathi-specific phonemes as acoustic artefacts absent from its training distribution — causing confident misclassification. **Evidence: P(synth)=1.000 with zero uncertainty is a hallmark of out-of-distribution input, not genuine synthetic detection.**

2. **English Reuters broadcast (SAMPLE_01 + EXT_05):** P(synth)=0.990 on original; P(synth)=0.994 on the 64 kbps re-encoded version. **Critically: re-encoding at 64 kbps did NOT reduce the false positive.** The score remained essentially unchanged (0.990→0.994). This rules out MP3 codec compression alone as the primary cause. The most probable explanation is that the Reuters recording has **broadcast-chain AGC and dynamic range compression** that flattens the spectral envelope in ways that resemble TTS output — and this signal survives MP3 round-tripping. By contrast, EXT_01 (32 kbps re-encode of human_04_anchor_bulletin.wav, which was correctly classified) was also correctly classified as REAL after compression. This suggests the issue is specific to this particular recording's acoustic characteristics, not MP3 compression in general.

3. **Recording quality vs. language coverage:** Both false-positive categories could share a common cause — the model is sensitive to spectral smoothness patterns that genuine TTS engines exhibit. Broadcast-processed audio (AGC, dynamic range compression, codec artefacts) and non-English speech can both produce atypically smooth spectral envelopes that the model misreads as synthetic.

> [!WARNING]
> The Wav2Vec2 model's training data composition is not publicly documented beyond
> the model card. It is not confirmed whether Hindi or Marathi speech was included.
> The root cause above (out-of-distribution) is supported by the evidence pattern
> (P=1.000, no uncertainty) but cannot be verified without access to training details.

---

## 4b. False Negative Analysis

Total false negatives: **2** (AI_GENERATED samples predicted as REAL).

### EXT_04 — `synth_06_buzz_robotic_tts.wav`
- **Category:** Flat-pitch buzz TTS (no prosody, word-gap silences)
- **P(synthetic):** 0.0000 — model is maximally confident it is human
- **Root cause:** The buzz+silence pattern produces periodic silence gaps and spectrally narrow energy, which the detector interprets as low-energy natural speech pauses rather than synthetic articulation. This exposes a blind spot: **the model detects natural TTS prosody patterns but not atypical, robotic synthesis styles**.

### EXT_07 — `synth_07_formant_en_female.wav`
- **Category:** Formant synthesis at female pitch (200 Hz)
- **P(synthetic):** 0.0001 — model is maximally confident it is human
- **Root cause:** Formant synthesis with irregular pitch variation closely resembles human speech spectral characteristics (irregular pitch is present). The model was likely trained primarily on **neural vocoder outputs** (which have very specific spectral smoothness signatures). Additive formant synthesis at human-like pitch appears as genuine speech to this detector. This is the opposite failure mode of the SAPI samples (which are correctly caught).

> [!WARNING]
> The false negative pattern shows the model is specifically calibrated against neural/HMM TTS artifacts
> (SAPI voices). It does NOT catch all synthetic generation methods. Buzz synthesis and formant
> synthesis with natural-pitch variation both fool the detector at maximum confidence (P(real)=1.0).

---

## 5. Uncertainty & Failsafe Verification

- **EXT_08** (`edge_01_too_short.wav`): prediction=`UNAVAILABLE`, available=False, error=`Load failed: Audio duration (0.15s) is shorter than minimum required (0.5s).`

> [!IMPORTANT]
> **Failsafe confirmed working.** The 0.15s sample was correctly rejected as UNAVAILABLE by the
> audio loader (`MIN_AUDIO_DURATION_SEC=0.5`), not silently classified as REAL or AI_GENERATED.
>
> **Note on guard threshold:** The code comment in `synthetic_detector.py` (line 140) documents a
> `< 0.2s` guard, but `audio_loader.py` enforces `MIN_AUDIO_DURATION_SEC = 0.5s` earlier in the
> pipeline, causing `UNAVAILABLE` (load error) rather than `UNCERTAIN` (detector guard).
> The failsafe result is correct — no silent classification — but the two-stage guard is worth
> documenting for clarity.

---

## 6. Inference Latency

- **Mean per-sample:** 0.631 s
- **Min:** 0.0373 s
- **Max:** 7.4542 s
- **Note:** First inference includes model load time (~10–14 s); subsequent samples use cached model.

---

## 7. Documented Limitations

1. **Small dataset (16 samples):** Metrics are illustrative, not statistically robust. A minimum of 100 samples per class is needed for reliable precision/recall estimates.

2. **Multilingual coverage:** The model shows zero accuracy on Marathi and uncertain behaviour on non-English input. It should **not** be used as a reliable deepfake detector for non-English speech without independent validation on those languages.

3. **AI generator diversity:** All 7 synthetic samples are from Microsoft SAPI (concatenative/parametric) or deterministic scipy synthesis. Modern neural TTS (ElevenLabs, OpenAI Voice, XTTS-v2, VALL-E, StyleTTS2) have NOT been evaluated. The detector may fail on neural codec-based voice clones.

4. **Parametric proxies:** EXT_02 and EXT_06 are scipy-generated acoustic proxies, not actual recorded human speech. Their ground-truth labels are based on the parametric model, not a real speaker. These samples should be replaced with genuine recordings in a production evaluation.

5. **Broadcast processing:** Heavy AGC, dynamic compression, and codec artefacts from broadcast recordings (Reuters: SAMPLE_01) may trigger false positives. The model was not validated on broadcast-processed audio.

6. **No retraining or threshold changes recommended** based on this small dataset. Evidence is insufficient to justify model replacement. Recommended next step: collect 50+ genuine Marathi speech samples and run targeted evaluation.

---

## 8. Recommended Actions

| Priority | Action | Evidence Required |
|----------|--------|-------------------|
| HIGH | Document multilingual limitation in README and API response | Confirmed by Marathi FP |
| HIGH | Surface language warning in `predict_synthetic_speech()` when input language is non-English | Marathi P=1.000 |
| MEDIUM | Collect ≥ 50 genuine Marathi + ≥ 20 Hindi samples for targeted re-evaluation | Current sample count insufficient |
| MEDIUM | Evaluate against neural TTS (ElevenLabs, XTTS-v2) before production deployment | No neural TTS samples tested |
| LOW | Investigate broadcast-codec false positive (SAMPLE_01) with more broadcast samples | Single instance, unclear pattern |
| LOW | Consider UNCERTAIN threshold widening (e.g., 0.30/0.70) after gathering more data | Do NOT change without evidence |

---

## 9. Per-Sample Predictions

| ID | File | GT | Pred | P(synth) | P(real) | Correct | Latency (s) |
|----|------|----|------|----------|---------|---------|-------------|
| SAMPLE_01 | `human_01_reuters_news.wav` | REAL | AI_GENERATED | 0.9902 | 0.0098 | ✗ | 7.454 |
| SAMPLE_02 | `human_02_hindi_broadcast.wav` | REAL | REAL | 0.0000 | 1.0000 | ✓ | 0.037 |
| SAMPLE_03 | `human_03_marathi_report.wav` | REAL | AI_GENERATED | 1.0000 | 0.0000 | ✗ | 0.104 |
| SAMPLE_04 | `human_04_anchor_bulletin.wav` | REAL | REAL | 0.0000 | 1.0000 | ✓ | 0.040 |
| SAMPLE_05 | `synth_01_sapi_david_intro.wav` | AI_GENERATED | AI_GENERATED | 1.0000 | 0.0000 | ✓ | 0.045 |
| SAMPLE_06 | `synth_02_sapi_zira_finance.wav` | AI_GENERATED | AI_GENERATED | 1.0000 | 0.0000 | ✓ | 0.213 |
| SAMPLE_07 | `synth_03_sapi_david_science.wav` | AI_GENERATED | AI_GENERATED | 1.0000 | 0.0000 | ✓ | 0.169 |
| SAMPLE_08 | `synth_04_sapi_zira_tech.wav` | AI_GENERATED | AI_GENERATED | 1.0000 | 0.0000 | ✓ | 0.209 |
| EXT_01 | `human_05_en_compressed_32k.wav` | REAL | REAL | 0.0000 | 1.0000 | ✓ | 0.170 |
| EXT_02 | `human_06_en_degraded_phone.wav` | REAL | REAL | 0.0019 | 0.9981 | ✓ | 0.172 |
| EXT_03 | `synth_05_formant_en_male.wav` | AI_GENERATED | AI_GENERATED | 0.9584 | 0.0416 | ✓ | 0.168 |
| EXT_04 | `synth_06_buzz_robotic_tts.wav` | AI_GENERATED | REAL | 0.0000 | 1.0000 | ✗ | 0.179 |
| EXT_05 | `human_07_en_compressed_64k.wav` | REAL | AI_GENERATED | 0.9939 | 0.0061 | ✗ | 0.167 |
| EXT_06 | `human_08_hindi_prosody_proxy.wav` | REAL | REAL | 0.0001 | 0.9999 | ✓ | 0.166 |
| EXT_07 | `synth_07_formant_en_female.wav` | AI_GENERATED | REAL | 0.0001 | 0.9999 | ✗ | 0.172 |
| EXT_08 | `edge_01_too_short.wav` | UNCERTAIN | UNAVAILABLE | — | — | edge | — |

---

*Report generated by `audio_module/evaluate_audio_detector.py`*
*All raw predictions saved in `audio_module/outputs/eval_sample_predictions.json`*
*Aggregate metrics saved in `audio_module/outputs/eval_metrics.json`*