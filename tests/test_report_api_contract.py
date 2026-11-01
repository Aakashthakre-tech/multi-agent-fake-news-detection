"""
tests/test_report_api_contract.py
=================================
API-level regression tests for the `research_report` contract and for the
integration defects that were confirmed and fixed:

* the audio endpoint defaulted `verify_claims` to False, so audio claim
  research never ran;
* the audio synthetic-speech detector reported `model_name` / `model_id`, but
  the API read `model`, so the response never named the detector;
* the pipeline crashed with a UnicodeEncodeError on a Windows console when it
  printed its progress glyphs, aborting the whole research run;
* the pipeline stages advertised a Groq model that was not the one running.

These are FIXTURE tests (the API is exercised with mocked pipelines). The
genuine live run is covered by `scratch/e2e_all_modalities.py`.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. The research_report is present and serialises for every modality
# ---------------------------------------------------------------------------

@patch("pipeline.run_research_pipeline")
def test_text_response_contains_research_report(mock_research):
    mock_research.return_value = {
        "claim": "Sky is blue",
        "search_plans": [{"label": "exact", "query": "sky is blue", "intent": "exact"}],
        "search_errors": [],
        "scrape_failures": [],
        "source_data": [
            {
                "url": "https://news.example/sky",
                "domain": "news.example",
                "title": "Sky is blue",
                "scrape_status": "SUCCESS",
                "search_query": "sky is blue",
            }
        ],
        "evidence_analysis": [
            {
                "source_id": 1,
                "url": "https://news.example/sky",
                "domain": "news.example",
                "title": "Sky is blue",
                "evidence_type": "scraped_article",
                "chunk": "The sky appears blue because of Rayleigh scattering.",
                "nli_label": "entailment",
                "nli_score": 0.9,
                "nli_status": "ok",
                "source_reliability": 0.9,
                "source_tier": "major_outlet",
                "source_tier_reason": "established outlet",
                "independence_group": "news.example",
                "evidence_score": 0.85,
                "evidence_class": "SUPPORT",
                "date_status": "match",
                "event_match": "strong",
                "search_query": "sky is blue",
            }
        ],
        "source_agreement": {"agreement_score": 1.0, "independent_supporting": 1,
                             "independent_contradicting": 0},
        "verification": "Verdict: REAL\nConfidence: 90%\nFinal Explanation: Corroborated.",
        "feedback": "Final Recommendation: Accept.",
        "final_verdict": {
            "verdict": "REAL",
            "confidence": 90.0,
            "evidence_sufficiency": "strong_support",
            "independent_supporting": 1,
            "independent_contradicting": 0,
            "total_sources": 1,
            "permitted_verdicts": ["REAL", "UNVERIFIED"],
            "permitted_reason": "One publisher corroborates the claim.",
            "reasoning": "One publisher corroborates the claim.",
            "supporting_evidence": [
                {"url": "https://news.example/sky", "domain": "news.example",
                 "chunk": "The sky appears blue because of Rayleigh scattering."}
            ],
            "contradicting_evidence": [],
        },
    }

    response = client.post("/verify/text", json={"claim": "Sky is blue"})
    assert response.status_code == 200
    data = response.json()

    report = data["research_report"]
    assert report["modality"] == "text"

    summary = report["executive_summary"]
    assert summary["verdict"] == "REAL"
    assert summary["input_claim"] == "Sky is blue"
    assert summary["agent_narrative"] == "Corroborated."
    assert summary["critic_review"] == "Accept."

    findings = report["findings"]
    assert len(findings["supporting"]) == 1
    finding = findings["supporting"][0]
    assert finding["source_url"] == "https://news.example/sky"
    assert finding["excerpt"].startswith("The sky appears blue")
    assert finding["evidence_strength"] == 0.85
    assert findings["contradicting"] == []

    method = report["methodology"]
    assert method["research_skipped"] is False
    assert method["search_queries"][0]["query"] == "sky is blue"
    assert method["sources_retrieved"] == 1
    assert method["articles_scraped"] == 1
    assert method["evidence_passages"] == 1
    assert method["nli_evaluated"] == 1

    assert report["claims"][0]["verdict"] == "REAL"
    assert report["claims"][0]["supporting_evidence"]


@patch("image_pipeline.run_image_pipeline")
def test_image_response_contains_research_report(mock_img):
    mock_img.return_value = {
        "image_path": "temp.png",
        "normalized_claim": {"claim": "Alien spotted in Times Square"},
        "detected_objects": [],
        "ocr_results": [],
        "ai_generation": {"available": True, "ai_probability": 0.9, "real_probability": 0.1,
                          "confidence": 0.9, "model": "forensic_residual_cnn", "verdict": "AI-GENERATED"},
        "verification": {
            "final_verdict": {"verdict": "FAKE", "confidence": 0.95},
            "evidence_analysis": [],
            "source_data": [],
        },
        "ai_generation_note": "AI image detected.",
    }

    img = io.BytesIO(b"sample_png_data")
    response = client.post(
        "/verify/image", files={"file": ("test.png", img, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()

    report = data["research_report"]
    assert report["modality"] == "image"
    assert report["executive_summary"]["verdict"] == "FAKE"
    # AI-generation probability must not become a factual verdict.
    assert data["synthetic_media_detection"]["signal"] == "AI_GENERATED"
    assert data["synthetic_media_detection"]["manipulation_status"] == "NOT_ASSESSED"


# ---------------------------------------------------------------------------
# 2. The audio endpoint must run claim research by default
# ---------------------------------------------------------------------------

@patch("audio_module.verify_audio")
def test_audio_endpoint_enables_claim_research_by_default(mock_audio):
    """`verify_claims` must default to True, as it does for text and video."""
    mock_audio.return_value = {
        "status": "success",
        "audio_metadata": {},
        "synthetic_speech_detection": {
            "label": "REAL", "confidence": 0.88, "available": True,
            "model_name": "Wav2Vec2 Deepfake Audio Detector (mo-thecreator)",
            "model_id": "mo-thecreator/Deepfake-audio-detection",
            "model_license": "Apache-2.0",
        },
        "transcription": {"text": "The council approved the plan on 4 March 2020."},
        "claim_extraction": {"primary_claim": "The council approved the plan on 4 March 2020."},
        "factual_verification": {"verdict": "UNVERIFIED", "confidence": 0.4},
        "consistency_analysis": {},
    }

    import inspect
    from api import server as server_mod

    signature = inspect.signature(server_mod.endpoint_verify_audio)
    declared = signature.parameters["verify_claims"].default
    # FastAPI wraps the default in a Form(...) object.
    default = getattr(declared, "default", declared)
    assert default is True, (
        "the audio endpoint must enable claim research by default; "
        f"otherwise audio never produces a factual research report (got {default!r})"
    )

    aud = io.BytesIO(b"sample_wav_data")
    response = client.post("/verify/audio", files={"file": ("test.wav", aud, "audio/wav")})
    assert response.status_code == 200
    data = response.json()

    # The detector's real identity must reach the API.
    assert data["synthetic_media_detection"]["model"] == "Wav2Vec2 Deepfake Audio Detector (mo-thecreator)"
    assert data["synthetic_media_detection"]["model_id"] == "mo-thecreator/Deepfake-audio-detection"


@patch("audio_module.verify_audio")
def test_audio_response_reports_real_detector_model(mock_audio):
    mock_audio.return_value = {
        "status": "success",
        "audio_metadata": {},
        "synthetic_speech_detection": {
            "label": "AI_GENERATED", "confidence": 1.0, "available": True,
            "model_name": "Wav2Vec2 Deepfake Audio Detector (mo-thecreator)",
            "model_id": "mo-thecreator/Deepfake-audio-detection",
        },
        "transcription": {"text": "", "status": "failed",
                          "error": "openai-whisper is not installed."},
        "claim_extraction": {"primary_claim": None, "selected_claims": []},
        "factual_verification": {"enabled": True, "verdict": "INSUFFICIENT_VERIFIABLE_CLAIMS"},
        "consistency_analysis": {},
    }

    aud = io.BytesIO(b"sample_wav_data")
    response = client.post("/verify/audio", files={"file": ("test.wav", aud, "audio/wav")})
    data = response.json()

    assert data["synthetic_media_detection"]["model"] == "Wav2Vec2 Deepfake Audio Detector (mo-thecreator)"

    # The report must explain the skip using the real transcription error.
    report = data["research_report"]
    assert report["methodology"]["research_skipped"] is True
    assert "openai-whisper is not installed" in report["methodology"]["research_skipped_reason"]
    assert report["findings"]["supporting"] == []
    assert report["evidence_gaps"]


# ---------------------------------------------------------------------------
# 3. Video: research skipped must be reported, not faked
# ---------------------------------------------------------------------------

@patch("video_module.video_verifier.verify_video")
def test_video_response_reports_skipped_research(mock_video):
    mock_video.return_value = {
        "status": "success",
        "primary_claim_analyzed": None,
        "candidate_claims": [],
        "candidate_claims_count": 32,
        "selected_claims_count": 0,
        "rejected_claims_count": 32,
        "verified_claims": [],
        "research_skipped_reason": (
            "No verifiable factual claims passed filtering. All candidates were "
            "conversational filler, greetings, apologies, or noise. Web research skipped."
        ),
        "final_evidence_verdict": "INSUFFICIENT_VERIFIABLE_CLAIMS",
        "transcription_segments": [],
        "transcription_language": "mr",
        "consistency_analysis": {},
        "visual_classification": {"available": True, "predicted_label": "fake", "confidence": 0.6413},
        "summary_artifacts": {},
        "warnings": [],
        "errors": [],
    }

    vid = io.BytesIO(b"sample_mp4_data")
    response = client.post("/verify/video", files={"file": ("test.mp4", vid, "video/mp4")})
    assert response.status_code == 200
    data = response.json()

    report = data["research_report"]
    assert report["methodology"]["research_skipped"] is True
    assert "32 candidate claim(s)" in report["methodology"]["research_skipped_reason"]
    assert report["executive_summary"]["verdict"] == "INSUFFICIENT_VERIFIABLE_CLAIMS"
    assert all(bucket == [] for bucket in report["findings"].values())
    assert report["evidence_gaps"]

    # The experimental visual classifier must stay a separate signal.
    assert data["visual_classification"]["predicted_label"] == "fake"
    assert data["factual_verification"]["verdict"] == "INSUFFICIENT_VERIFIABLE_CLAIMS"


# ---------------------------------------------------------------------------
# 4. The pipeline stages must not name a model that is not running
# ---------------------------------------------------------------------------

@patch("pipeline.run_research_pipeline")
def test_pipeline_stages_report_the_actual_agent_model(mock_research):
    mock_research.return_value = {
        "claim": "Sky is blue",
        "final_verdict": {"verdict": "REAL", "confidence": 0.99},
    }

    response = client.post("/verify/text", json={"claim": "Sky is blue"})
    data = response.json()
    stages = data["pipeline_stages"]
    assert stages, "text runs must report their pipeline stages"

    agent_stages = [s for s in stages if "Agent" in s["name"]]
    assert agent_stages, "the verification and critic agent stages must be reported"
    for stage in agent_stages:
        assert "LLaMA" not in stage["tool_or_model"], (
            "the stage must not advertise a model that is not the one running"
        )
        assert stage["tool_or_model"]


# ---------------------------------------------------------------------------
# 5. The pipeline must survive a console that cannot encode its glyphs
# ---------------------------------------------------------------------------

def test_pipeline_configures_utf8_stdout():
    """
    The pipeline prints progress glyphs that a cp1252 console cannot encode.

    Without the UTF-8 reconfiguration the research run aborts with
    UnicodeEncodeError part-way through.
    """
    import pipeline as pipeline_mod

    assert hasattr(pipeline_mod, "_force_utf8_stdout")
    # It must be safe to call again (the server calls it too).
    pipeline_mod._force_utf8_stdout()

    encoding = getattr(sys.stdout, "encoding", "") or ""
    errors = getattr(sys.stdout, "errors", "") or ""
    assert errors in ("replace", "backslashreplace") or encoding.lower().startswith("utf-8"), (
        f"stdout must tolerate unencodable characters (encoding={encoding!r}, errors={errors!r})"
    )
