"""
scratch/build_frontend_fixture.py
=================================
Turn the saved genuine API response into a frontend test fixture.

The fixture is the real report the backend produced, so the frontend render
test asserts against real data rather than hand-written values.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scratch" / "e2e_results.json"
OUT_DIR = ROOT / "frontend" / "src" / "test-fixtures"
OUT = OUT_DIR / "realVerificationResponses.json"

data = json.loads(SRC.read_text(encoding="utf-8"))

fixture = {}
for modality, payload in data.items():
    report = payload.get("report") or {}
    summary = report.get("executive_summary") or {}
    fixture[modality] = {
        "modality": payload.get("modality"),
        "status": payload.get("status"),
        "factual_verification": {
            "verdict": payload.get("verdict"),
            "confidence": summary.get("confidence"),
            "confidence_basis": summary.get("confidence_basis"),
            "evidence_sufficiency": summary.get("evidence_sufficiency"),
            "independent_supporting": 0,
            "independent_contradicting": 0,
            "contradiction_score": None,
            "source_agreement": None,
            "fusion_score": None,
        },
        "synthetic_media_detection": None,
        "visual_classification": None,
        "transcription": None,
        "transcription_segments": None,
        "primary_claim": (report.get("executive_summary") or {}).get("input_claim"),
        "consistency_analysis": None,
        "case_summary": None,
        "pipeline_stages": [],
        "modality_specific": None,
        "research_report": report,
        "raw": {},
        "error": None,
        "warnings": [],
    }

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
for modality, payload in fixture.items():
    report = payload["research_report"]
    counts = {k: len(v or []) for k, v in (report.get("findings") or {}).items()}
    print(f"  {modality}: verdict={report['executive_summary'].get('verdict')} findings={counts}")
