import os
import subprocess
from pathlib import Path

sample_dir = Path("audio_module/validation_samples")
sample_dir.mkdir(parents=True, exist_ok=True)

# 1. Synthesize known AI-generated TTS samples using Windows Speech API (SAPI)
tts_specs = [
    {
        "filename": "synth_01_sapi_david_intro.wav",
        "voice": "Microsoft David Desktop",
        "rate": 0,
        "text": "Welcome to TruthLens AI. This is a computer generated synthetic speech verification sample.",
    },
    {
        "filename": "synth_02_sapi_zira_finance.wav",
        "voice": "Microsoft Zira Desktop",
        "rate": 0,
        "text": "The central bank decided to hold benchmark interest rates steady at five point two five percent amidst cooling inflation.",
    },
    {
        "filename": "synth_03_sapi_david_science.wav",
        "voice": "Microsoft David Desktop",
        "rate": 1,
        "text": "Astronomers observing the outer solar system have detected an unusual trajectory in a distant Kuiper Belt object.",
    },
    {
        "filename": "synth_04_sapi_zira_tech.wav",
        "voice": "Microsoft Zira Desktop",
        "rate": -1,
        "text": "Deep learning neural networks can now synthesize photorealistic imagery and natural sounding human speech.",
    },
]

print("Generating synthetic speech (TTS) samples...")
for spec in tts_specs:
    out_path = sample_dir / spec["filename"]
    ps_cmd = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $synth.SelectVoice('{spec['voice']}')
}} catch {{}}
$synth.Rate = {spec['rate']}
$synth.SetOutputToWaveFile('{str(out_path.resolve())}')
$synth.Speak('{spec['text']}')
$synth.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    print(f"Generated TTS: {out_path} ({out_path.stat().st_size} bytes)")

# 2. Extract genuine human speech samples from existing video audio extracts
human_sources = [
    {
        "filename": "human_01_reuters_news.wav",
        "src": "video_module/outputs/r179/audio.wav",
        "start": 0.0,
        "duration": 6.0,
    },
    {
        "filename": "human_02_hindi_broadcast.wav",
        "src": "video_module/outputs/hindi_test/audio.wav",
        "start": 0.0,
        "duration": 6.0,
    },
    {
        "filename": "human_03_marathi_report.wav",
        "src": "video_module/outputs/marathi_test/audio.wav",
        "start": 0.0,
        "duration": 6.0,
    },
    {
        "filename": "human_04_anchor_bulletin.wav",
        "src": "video_module/outputs/BREAKING/audio.wav",
        "start": 0.0,
        "duration": 6.0,
    },
]

print("Extracting authentic human speech samples...")
for spec in human_sources:
    src_path = Path(spec["src"])
    out_path = sample_dir / spec["filename"]
    if src_path.exists():
        cmd = [
            "ffmpeg", "-y", "-ss", str(spec["start"]), "-i", str(src_path),
            "-t", str(spec["duration"]), "-ar", "16000", "-ac", "1", str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Extracted Human Speech: {out_path} ({out_path.stat().st_size} bytes)")
    else:
        print(f"Source not found: {src_path}")
