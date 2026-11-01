# TruthLensAI — Video Module: Phase 1

This module implements **Phase 1 of the TruthLensAI Video Module**: video ingestion, validation, frame sampling, audio extraction, and timestamped speech transcription.

It operates strictly as a preprocessing pipeline. It does **not** perform authenticity classification, deepfake detection, claim verification, or model training.

---

## 1. Pipeline Architecture

```text
Input Video (.mp4, .mov, .avi, .mkv)
           │
           ▼
 [ Stage 1: Validation ]
  - Verify file readability & container integrity
  - Extract video metadata (resolution, fps, duration) via OpenCV
  - Inspect stream codecs & audio availability via FFprobe
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
 [ Stage 2: Frame Extraction ]             [ Stage 3: Audio Extraction ]
  - OpenCV incremental frame decoding       - FFmpeg stream extraction
  - Configurable sampling interval          - 16 kHz mono 16-bit PCM WAV
  - Save sequential JPEG frames             - Handles audio-less videos gracefully
  - Record frame manifest (frames.json)              │
                                                     ▼
                                           [ Stage 4: Transcription ]
                                            - OpenAI Whisper model (cached)
                                            - Preserves exact segment timestamps
                                            - Exports UTF-8 text & JSON formats
                                                     │
           ┌─────────────────────────────────────────┘
           ▼
 [ Stage 5: Summary Generation ]
  - Outputs summary.json with stage statuses, metadata, and artifact paths
```

---

## 2. Requirements & Dependencies

### Python Dependencies
Installed via `video_module/requirements.txt`:
- `opencv-python>=4.8.0` — Incremental video decoding and frame capture
- `openai-whisper>=20231117` — Speech-to-text recognition with segment timestamps
- `pytest>=7.0.0` — Automated unit testing

### System Dependencies
- **FFmpeg & FFprobe**: Required for audio stream extraction and stream metadata inspection.

#### Windows Installation Instructions:
FFmpeg must be installed and accessible on your system `PATH`. Choose one of the following methods:

- **Option 1 (Windows Package Manager - Recommended)**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **Option 2 (Chocolatey)**:
  ```powershell
  choco install ffmpeg
  ```
- **Option 3 (Scoop)**:
  ```powershell
  scoop install ffmpeg
  ```
- **Option 4 (Manual Installation)**:
  1. Download the release build from [gyan.dev FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/) or [ffmpeg.org](https://ffmpeg.org/download.html).
  2. Extract the archive (for example, to `C:\ffmpeg`).
  3. Add `C:\ffmpeg\bin` to your System or User `PATH` environment variable.
  4. Restart your terminal and verify:
     ```powershell
     ffmpeg -version
     ffprobe -version
     ```

---

## 3. Installation

From the project root:

```bash
# Activate your virtual environment
.venv\Scripts\activate   # Windows PowerShell / CMD

# Install dependencies
pip install -r video_module/requirements.txt
```

---

## 4. CLI Usage

Run the pipeline from the project root:

```bash
python video_module/video_pipeline.py "path/to/video.mp4"
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `video_path` | Positional | Required | Path to input video (`.mp4`, `.mov`, `.avi`, `.mkv`, etc.) |
| `--sample-interval` | Float | `1.0` | Frame sampling interval in seconds (e.g. `0.5` = 2 frames/sec, `2.0` = 1 frame every 2s) |
| `--whisper-model` | String | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--output-dir` | Path | Auto | Custom destination folder (defaults to `video_module/outputs/<video_name>/`) |

### Examples

```bash
# Default run (1 frame per second, Whisper base model)
python video_module/video_pipeline.py "sample.mp4"

# Sample 2 frames per second with a lighter Whisper model
python video_module/video_pipeline.py "sample.mp4" --sample-interval 0.5 --whisper-model tiny

# Save outputs to a specific directory
python video_module/video_pipeline.py "sample.mp4" --output-dir "video_module/outputs/custom_run"
```

---

## 5. Output Organization

For an input named `interview.mp4`, the pipeline creates:

```text
video_module/
└── outputs/
    └── interview/
        ├── frames/
        │   ├── frame_000001.jpg
        │   ├── frame_000002.jpg
        │   └── ...
        ├── frames.json
        ├── audio.wav
        ├── transcript.txt
        ├── transcript.json
        └── summary.json
```

### Output Files Specification

1. **`frames/`**: Sequential JPEG images extracted at the specified sample interval.
2. **`frames.json`**: Frame manifest mapping frame numbers to filenames and timestamps:
   ```json
   [
     {
       "frame_index": 1,
       "filename": "frame_000001.jpg",
       "timestamp_sec": 0.0
     },
     {
       "frame_index": 2,
       "filename": "frame_000002.jpg",
       "timestamp_sec": 1.0
     }
   ]
   ```
3. **`audio.wav`**: Mono 16 kHz 16-bit PCM WAV audio extracted directly from the video.
4. **`transcript.txt`**: Complete transcribed speech as UTF-8 text.
5. **`transcript.json`**: Detailed speech segments preserving Whisper timestamps:
   ```json
   [
     {
       "id": 0,
       "start": 0.0,
       "end": 2.5,
       "text": "Good morning and welcome to the briefing."
     }
   ]
   ```
6. **`summary.json`**: Processing report containing input metadata, execution status, and file paths:
   ```json
   {
     "input_video_path": "interview.mp4",
     "video_metadata": {
       "width": 1920,
       "height": 1080,
       "fps": 30.0,
       "frame_count": 900,
       "duration_sec": 30.0,
       "has_audio": true,
       "audio_codec": "aac",
       "video_codec": "h264"
     },
     "sample_interval_sec": 1.0,
     "frames_extracted_count": 30,
     "frame_manifest_path": ".../frames.json",
     "audio_output_path": ".../audio.wav",
     "transcript_text_path": ".../transcript.txt",
     "transcript_json_path": ".../transcript.json",
     "whisper_model": "base",
     "stage_status": {
       "validation": "success",
       "frame_extraction": "success",
       "audio_extraction": "success",
       "transcription": "success"
     },
     "warnings": [],
     "errors": []
   }
   ```

---

## 6. Testing

Run automated tests using `pytest`:

```bash
pytest video_module/tests/test_video_pipeline.py -v
```

Or using standard Python `unittest`:

```bash
python -m unittest video_module/tests/test_video_pipeline.py
```

### Covered Test Cases
- Missing or invalid input paths.
- Corrupted media container detection.
- Frame sampling rate, JPEG creation, and timestamp monotonicity.
- Handling videos without audio streams (skipping transcription without crash).
- Handling missing system FFmpeg executables with clear instructions.
- Audio extraction and Whisper transcription pipeline with segment verification.
- Safe, non-destructive output folder creation.

---

## 7. Multi-Modal Video Verification & Baseline Classifier Integration

TruthLensAI provides an integrated multi-modal verification orchestrator in [`video_module/video_verifier.py`](video_verifier.py) that couples:
1. Video preprocessing, frame extraction, speech transcription, and OCR claim extraction (`video_module/video_pipeline.py`).
2. Supporting visual fake-vs-real classifier (`video_module/classifier_adapter.py`).
3. Multi-agent evidence research and verification engine (`pipeline.py`).

### Architectural Safety Principles
- **Evidence-Driven Truth**: The ultimate fact-checking verdict (`SUPPORTED`, `CONTRADICTED`, `INCONCLUSIVE`) is driven by verified evidence.
- **Experimental Classifier Constraint**: The baseline visual classifier achieved 52.38% test accuracy on the FVC benchmark (re-contextualized authentic video footage). It serves strictly as a supporting heuristic signal.
- **Divergence Diagnostics**: When the visual classifier and fact-checking evidence disagree, it triggers a diagnostic explanation (e.g. *out-of-context video misattribution* or *visual anomaly flagged*) rather than overriding the factual verdict.
- **Resilient & Optional**: The pipeline functions cleanly if the classifier is disabled (`--disable-classifier`) or if model loading fails.

### CLI Usage

#### Run Multi-Modal Verification on a Video:
```powershell
.venv\Scripts\python.exe video_module\video_verifier.py path\to\video.mp4
```

#### Run Offline Verification (Preprocess + Visual Classifier, Skip Live Web Search):
```powershell
.venv\Scripts\python.exe video_module\video_verifier.py path\to\video.mp4 --skip-research
```

#### Run with Visual Classifier Disabled:
```powershell
.venv\Scripts\python.exe video_module\video_verifier.py path\to\video.mp4 --disable-classifier
```

#### Run Standalone Classifier Prediction:
```powershell
.venv\Scripts\python.exe -c "from video_module.classifier_adapter import predict_video; print(predict_video('path/to/video.mp4'))"
```

---

## 8. Integration Test Suite

Run the full integration and regression test suites:
```powershell
.venv\Scripts\python.exe -m unittest discover -s video_module/tests -p "test_*.py"
```

