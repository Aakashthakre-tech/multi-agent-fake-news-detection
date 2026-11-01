::: {align="center"}
# TruthLensAI

### Multimodal Misinformation Detection & Claim Verification

**See the facts beyond the noise.**

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Project](https://img.shields.io/badge/Status-Academic%20Prototype-orange)](#project-status)

*A multimodal research prototype for analyzing text, images, videos, and
audio using extraction models, web evidence, source analysis, and
AI-assisted reasoning.*
:::

------------------------------------------------------------------------

## Contents

-   [Overview](#overview)
-   [Problem Statement](#problem-statement)
-   [Features](#key-features)
-   [Architecture](#system-architecture)
-   [Workflow](#verification-workflow)
-   [Technology Stack](#technology-stack)
-   [Modules and Models](#modules-and-models)
-   [Image Model Training](#image-model-training)
-   [Image Model Evaluation](#image-model-evaluation)
-   [Confusion Matrix](#confusion-matrix)
-   [Verdict Logic](#final-verdict-logic)
-   [API](#api-overview)
-   [Setup](#installation-and-setup)
-   [Testing](#testing)
-   [Applications](#applications)
-   [Limitations](#limitations-and-responsible-use)
-   [Future Work](#future-improvements)

## Overview

TruthLensAI is a multimodal misinformation-analysis prototype that
accepts **text, images, videos, and audio**. It extracts claims and
relevant signals, researches claims using available web sources,
analyzes evidence, and returns a structured assessment.

The system deliberately separates two questions:

1.  Is the media itself likely to be AI-generated?
2.  Is the factual claim shown or spoken in that media supported by
    evidence?

An AI-generated image can contain a true statement, and an authentic
photograph can accompany a false claim. Media-origin classification is
therefore reported separately from factual claim verification.

> **Project status:** Academic/research prototype. Results must be
> interpreted within the evaluation conditions and limitations described
> below.

## Problem Statement

Misinformation spreads through posts, screenshots, photographs, short
videos, and voice recordings. Single-format verification tools may miss
relevant information or confuse media authenticity with factual
accuracy. TruthLensAI explores a unified workflow that extracts
information from different media types, researches claims, evaluates
evidence, and communicates uncertainty.

## Key Features

-   Multimodal input: text, image, video, and audio.
-   OCR for text embedded in images and video frames.
-   Speech-to-text for audio and video.
-   Web search and source retrieval for claim research.
-   Evidence classification: supporting, contradicting, neutral, and
    contextual.
-   Source reliability and publisher-independence checks.
-   Natural Language Inference (NLI) signals for claim/evidence
    comparison.
-   AI-assisted analysis and a separate LLM second opinion.
-   Deterministic evidence gate for final verdict decisions.
-   Separate AI-image-origin assessment.
-   Explicit `UNVERIFIED` handling when evidence is insufficient or
    contested.

## System Architecture

``` mermaid
flowchart TD
    A[User Input] --> B{Input Type}
    B -->|Text| C[Text Claim Pipeline]
    B -->|Image| D[Image Pipeline]
    B -->|Video| E[Video Preprocessing]
    B -->|Audio| F[Audio Pipeline]
    D --> G[OCR / Object Detection / Image-Origin Model]
    E --> H[Frame Extraction / Audio Extraction]
    H --> I[Whisper Transcription / Frame OCR]
    F --> J[Whisper Transcription / Wav2Vec2 Analysis]
    C --> K[Claim Extraction and Research]
    G --> K
    I --> K
    J --> K
    K --> L[Web Search and Source Retrieval]
    L --> M[Article/Text Extraction]
    M --> N[Evidence Analysis and NLI]
    N --> O[Source Reliability and Independence]
    O --> P[Evidence Fusion]
    P --> Q[Deterministic Verdict Gate]
    Q --> R[Final Report and Explanation]
```

### Processing layers

1.  **Frontend:** accepts input and displays progress, evidence,
    verdicts, and model outputs.
2.  **Backend API:** validates requests and routes them to the correct
    modality pipeline.
3.  **Modality processing:** extracts text, frames, speech, objects, and
    media-origin signals.
4.  **Research and verification:** retrieves sources, compares evidence
    with claims, and evaluates source quality.
5.  **Decision/reporting:** fuses evidence signals, applies
    deterministic rules, and returns a structured result.

## Verification Workflow

For a text claim:

1.  Extract or normalize the claim.
2.  Generate search queries and retrieve relevant sources.
3.  Scrape or extract available article text.
4.  Compare evidence passages with the claim.
5.  Classify passages as supporting, contradicting, neutral, or
    contextual.
6.  Evaluate source reliability and publisher independence.
7.  Run AI-assisted analysis and an independent LLM assessment where
    enabled.
8.  Apply the deterministic evidence gate.
9.  Return verdict, explanation, evidence references, and limitations.

For image, audio, and video inputs, the corresponding extraction stage
runs first. Extracted text is used for claim research where applicable;
it does not by itself prove that the original media is authentic.

## Technology Stack

  ------------------------------------------------------------------------------------------
  Area                    Technology                                 Role
  ----------------------- ------------------------------------------ -----------------------
  Backend                 Python, FastAPI                            API and processing
                                                                     orchestration

  Frontend                React, TypeScript, Vite                    User interface

  Styling                 Tailwind CSS                               UI styling

  LLM                     Groq-hosted language models                AI-assisted analysis
                                                                     and second opinion

  Web research            Tavily API                                 Search and source
                                                                     discovery

  Evidence reasoning      DeBERTa-based NLI                          Entailment,
                                                                     contradiction, neutral
                                                                     signals

  Image OCR               EasyOCR                                    Text extraction from
                                                                     images

  Object detection        YOLO (pretrained)                          Visual object detection

  Image-origin model      `forensic_residual_cnn`                    REAL / AI-GENERATED
                                                                     classification

  Speech transcription    OpenAI Whisper                             Audio-to-text

  Audio-origin model      Wav2Vec2-based                             Deepfake-audio
                          `mo-thecreator/Deepfake-audio-detection`   classification

  Video                   OpenCV, OCR, VideoMAE (experimental)       Frame extraction and
                                                                     video analysis

  Testing                 pytest, Vitest                             Backend and frontend
                                                                     tests
  ------------------------------------------------------------------------------------------

*The stack lists documented components. It does not imply that every
model was trained from scratch by this project.*

## Modules and Models

### Text verification

-   Extracts the main factual claim.
-   Searches for relevant sources and extracts article text.
-   Uses NLI and evidence-analysis logic to compare passages with the
    claim.
-   Checks source reliability, duplication, and publisher independence.
-   Uses LLM reasoning as an additional signal, not as a replacement for
    evidence or deterministic rules.

### Image analysis

-   **EasyOCR:** extracts headlines, captions, and visible text.
-   **YOLO (pretrained):** detects relevant objects.
-   **`forensic_residual_cnn`:** estimates whether an image is REAL or
    AI-GENERATED, with an uncertainty region.
-   The extracted claim is sent through the factual verification
    pipeline separately.

### Video analysis

-   Validates and preprocesses video.
-   Extracts frames at configured intervals.
-   Extracts the audio stream.
-   Uses Whisper for speech transcription.
-   Runs OCR on selected frames.
-   Fuses spoken text and on-screen text.
-   Prepares candidate claims for verification.
-   Uses a visual classifier signal and an experimental VideoMAE
    component where configured.

### Audio analysis

-   Validates and preprocesses audio.
-   Uses Whisper for speech-to-text.
-   Uses the Wav2Vec2-based `mo-thecreator/Deepfake-audio-detection`
    model for audio-origin/deepfake classification.
-   Extracts claims from transcripts for evidence research where
    applicable.

Whisper is the transcription model; it is not itself the deepfake-audio
classifier.

## Image Model Training

The documented primary trained image classifier is
**`forensic_residual_cnn`**, a residual CNN designed to distinguish real
images from AI-generated images.

### Dataset composition

  Class               Images
  -------------- -----------
  REAL                 2,000
  AI-GENERATED         2,000
  **Total**        **4,000**

Documented data sources: - CIFAKE - HEMG / AI-or-Not - Real news
photographs and historical photographs - AI-generated news-style images
and historical recreations - AI composites and images containing
text/headlines - Images generated using tools such as Midjourney and
DALL·E

### Dataset split

  Split          Percentage      Images
  ------------ ------------ -----------
  Training              70%       2,800
  Validation            15%         600
  Test                  15%         600
  **Total**        **100%**   **4,000**

Each split is documented as class-balanced.

### Training configuration

  Setting            Value
  ------------------ ----------------------
  Architecture       Residual CNN
  Input              RGB image, 224 × 224
  Optimizer          Adam
  Loss               Binary Cross-Entropy
  Calibration        Temperature scaling
  Temperature        4.75
  Final threshold    0.470
  Uncertainty band   ±0.100

Training-only augmentations recorded: JPEG compression, resizing,
cropping, brightness/contrast changes, blur, noise, screenshot-like
transformations, text/headline overlays, and social-media-style
compression.

### Example training progress

These are the epoch examples recorded in the project image overview, not
a complete training log.

    Epoch   Train loss   Validation loss   Validation accuracy
  ------- ------------ ----------------- ---------------------
        1         0.68              0.61                  0.64
        5         0.42              0.39                  0.72
       10         0.31              0.33                  0.76
       20         0.24              0.28                  0.79
       30         0.21              0.27                  0.80

## Image Model Evaluation

The documented locked-test evaluation reports:

  --------------------------------------------------------------------------------------------
  Test            Decided   Precision     Recall         F1   Macro F1   Uncertain    Coverage
  subset         accuracy                                                     rate 
  ---------- ------------ ----------- ---------- ---------- ---------- ----------- -----------
  CIFAKE           80.00%        0.81       0.80       0.80       0.80       32.0%       68.0%

  HEMG             71.10%        0.72       0.71       0.71       0.71       36.0%       64.0%

  Combined     **75.55%**    **0.76**   **0.76**   **0.76**   **0.76**   **34.0%**   **66.0%**
  --------------------------------------------------------------------------------------------

**Interpretation:** Decided accuracy is measured on samples classified
as REAL or AI-GENERATED. Coverage is the fraction receiving a decided
prediction. The uncertain rate is reported separately, so accuracy
should not be read without coverage.

## Confusion Matrix

The project evaluation overview records this combined test-set confusion
matrix for **decided images**:

  Actual  Predicted     REAL   AI-GENERATED
  ------------------- ------ --------------
  **REAL**               420             80
  **AI-GENERATED**        86            414

``` text
                         PREDICTED
                    REAL       AI-GENERATED
ACTUAL REAL          420             80
ACTUAL AI             86            414
```

Reported combined metrics: - Decided accuracy: **75.55%** - AI recall:
**76.30%** - REAL recall: **74.80%** - False positive rate: **16.0%** -
False negative rate: **17.2%** - Uncertain rate: **34.0%** - Coverage:
**66.0%**

> **Evaluation-data consistency note:** The saved image-module overview
> lists a 600-image test split (15% of 4,000), while the
> confusion-matrix counts above sum to 1,000 decided samples. These
> cannot describe the same single 600-image test split as written. The
> figures are preserved as separately reported project results;
> reconcile the original evaluation export/logs before presenting them
> as one internally consistent experiment.

## Video and Audio Evaluation

The project records identify the model names and processing stages, but
do not establish a verified final confusion matrix or sufficiently
documented test accuracy for the video and audio modules.

  ------------------------------------------------------------------------------------------
  Module                  Models / tools                             Main outputs
  ----------------------- ------------------------------------------ -----------------------
  Video                   OpenCV, Whisper, OCR, VideoMAE             Frames, transcript,
                          (experimental), visual classifier          on-screen text, visual
                                                                     signal, candidate
                                                                     claims

  Audio                   Whisper, Wav2Vec2-based                    Speech transcript and
                          `mo-thecreator/Deepfake-audio-detection`   audio-origin/deepfake
                                                                     signal
  ------------------------------------------------------------------------------------------

A visual classifier output such as `real (0.5567)` is not a verdict
about whether the video's factual claim is true. Evidence-based claim
verification remains separate.

## Final Verdict Logic

TruthLensAI uses four claim-verification outcomes:

  -----------------------------------------------------------------------
  Verdict                             Meaning
  ----------------------------------- -----------------------------------
  **REAL**                            Available evidence sufficiently
                                      supports the claim under the
                                      system's evidence rules.

  **FAKE**                            Available evidence sufficiently
                                      refutes the claim under the
                                      system's evidence rules.

  **MISLEADING**                      The claim contains a true element
                                      but omits or distorts important
                                      context.

  **UNVERIFIED**                      Evidence is insufficient,
                                      ambiguous, or contested.
  -----------------------------------------------------------------------

Evidence matching, NLI signals, source reliability, publisher
independence, evidence sufficiency, and source agreement contribute to
the assessment. A deterministic gate enforces minimum evidence
requirements.

The independent LLM opinion is displayed separately. Agreement between
an LLM and the main system is not itself independent evidence.

### Separate media-origin result

Image-origin analysis reports whether the image appears REAL or
AI-GENERATED. It does not determine whether the factual claim is true.
Video and audio classifier signals are also not substitutes for
evidence-based claim verification.

## API Overview

The documented backend routes include:

  -----------------------------------------------------------------------
  Endpoint                            Purpose
  ----------------------------------- -----------------------------------
  `GET /health`                       Backend health check

  `POST /verify/text`                 Verify a text claim

  `POST /verify/image`                Process and verify an image

  `POST /verify/video`                Process and verify a video

  `POST /verify/audio`                Process and verify audio

  `POST /verify`                      Unified verification entry point

  `/verify/*/stream`                  Streaming verification endpoints
                                      for supported modalities
  -----------------------------------------------------------------------

Check the current FastAPI route definitions for exact request schemas
and streaming response formats.

## Project Structure

``` text
TruthLensAI/
├── api/
│   └── server.py
├── frontend/
│   └── src/
│       └── components/
├── ai_image_detection/
├── audio_module/
├── video_module/
├── image_pipeline.py
├── multimodal_router.py
├── pipeline.py
├── evidence_analyzer.py
├── evidence_fusion.py
├── final_verdict.py
├── research_report.py
├── query_planner.py
├── claim_profile.py
├── source_reliability.py
├── tools.py
├── tests/
├── requirements.txt
└── README.md
```

*This is a simplified map of documented components; the current checkout
may contain additional files.*

## Installation and Setup

### Prerequisites

-   Python compatible with the pinned project dependencies.
-   Node.js and npm.
-   Git.
-   API credentials for external services enabled in the local
    configuration.
-   Optional CUDA-compatible GPU for supported inference workloads.

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TruthLensAI
```

### 2. Set up the backend

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

### 3. Configure environment variables

If `.env.example` exists:

``` powershell
Copy-Item .env.example .env
```

Add required API keys and configuration values locally. Never commit
`.env`, API keys, tokens, or private credentials.

### 4. Run the backend

The documented development command is:

``` bash
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```

If the current checkout uses a different ASGI import path, use the entry
point in its server configuration.

### 5. Run the frontend

``` bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite.

## Testing

Backend:

``` bash
pytest
```

Frontend production build:

``` bash
cd frontend
npm run build
```

Run the frontend test command configured in `frontend/package.json` (for
example, `npm run test` if available).

Test all four modalities with known examples. Check extraction output,
evidence references, verdict mapping, error handling, and separation of
claim truth from media origin.

*Test counts and pass rates are not stated because the exact current
test run has not been verified for this README revision.*

## Applications

-   Screening suspicious claims shared through messaging and social
    platforms.
-   Checking headlines and claims embedded in screenshots.
-   Extracting and researching claims from short news videos.
-   Transcribing spoken claims from audio recordings.
-   Separately assessing whether an image may be AI-generated.
-   Supporting journalists, researchers, students, and media-literacy
    activities with evidence summaries.
-   Demonstrating multimodal AI pipelines and evidence-based reasoning
    in academic projects.

TruthLensAI is an assistance tool, not an authority that replaces
professional fact-checking.

## Limitations and Responsible Use

-   Search results may be incomplete, unavailable, duplicated, or
    unrelated to the exact event/date.
-   Search snippets are weaker evidence than successfully retrieved
    article text.
-   NLI scores are model signals, not proof.
-   LLM analysis can be incorrect or out of date and must not override
    evidence requirements.
-   OCR and speech recognition can misread names, dates, places, and
    multilingual content.
-   Short, noisy, compressed, or non-English audio may reduce
    transcription and classification quality.
-   Frame-based video analysis may miss events between sampled frames.
-   Image-origin classifiers can make mistakes on compressed, edited, or
    unfamiliar image styles.
-   `UNVERIFIED` means available evidence is insufficient or contested;
    it does not mean the claim is automatically false.
-   Model probabilities are not guaranteed to be calibrated truth
    probabilities.
-   Do not use the system alone for legal, medical, financial, or other
    high-impact decisions.

## Future Improvements

-   Stronger multilingual OCR and speech recognition, including Hindi
    and Marathi.
-   More robust video temporal-consistency analysis.
-   Expanded and independently audited audio/video evaluation datasets.
-   Reproducible per-model training scripts and locked test-set reports.
-   Better source provenance, deduplication, and event/date matching.
-   Improved calibration and uncertainty reporting.
-   Exportable verification reports and case history.
-   Additional adversarial testing for manipulated and synthetic media.

## Acknowledgements

TruthLensAI combines open-source machine-learning tools, pretrained
models, web-research services, and custom verification logic. Dataset
and model authors retain their respective licenses and attribution
requirements. Review the licenses of all datasets, checkpoints, and
external APIs before redistribution or commercial use.

------------------------------------------------------------------------

::: {align="center"}
**TruthLensAI --- Verify the claim. Examine the evidence. Respect
uncertainty.**

*Built as an academic and research prototype.*
:::
