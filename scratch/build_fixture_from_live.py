"""scratch/build_fixture_from_live.py - build the frontend fixture from the CURRENT live response."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scratch" / "apollo11_api_response.json"
OUT_DIR = ROOT / "frontend" / "src" / "test-fixtures"
OUT = OUT_DIR / "realVerificationResponses.json"

d = json.loads(SRC.read_text(encoding="utf-8"))
report = d.get("research_report") or {}
summary = report.get("executive_summary") or {}

fixture = {
    "text": {
        "modality": d.get("modality"),
        "status": d.get("status"),
        "factual_verification": {
            "verdict": d["factual_verification"].get("verdict"),
            "confidence": d["factual_verification"].get("confidence"),
            "confidence_basis": d["factual_verification"].get("confidence_basis"),
            "evidence_sufficiency": d["factual_verification"].get("evidence_sufficiency"),
            "independent_supporting": d["factual_verification"].get("independent_supporting", 0),
            "independent_contradicting": d["factual_verification"].get("independent_contradicting", 0),
            "contradiction_score": None,
            "source_agreement": None,
            "fusion_score": None,
        },
        "synthetic_media_detection": None,
        "visual_classification": None,
        "transcription": None,
        "transcription_segments": None,
        "primary_claim": summary.get("input_claim"),
        "consistency_analysis": None,
        "case_summary": None,
        "pipeline_stages": [],
        "modality_specific": None,
        "research_report": report,
        "raw": {},
        "error": None,
        "warnings": [],
    }
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Merge into the existing multi-modality fixture: only the text entry is
# refreshed from the current live response, so the image/audio/video cases
# stay covered.
existing = {}
if OUT.exists():
    existing = json.loads(OUT.read_text(encoding="utf-8"))
existing["text"] = fixture["text"]

OUT.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}  (modalities: {', '.join(sorted(existing))})")
print("verdict            :", fixture["text"]["factual_verification"]["verdict"])
print("confidence         :", fixture["text"]["factual_verification"]["confidence"],
      fixture["text"]["factual_verification"]["confidence_basis"])
print("supporting findings:", len((report.get("findings") or {}).get("supporting") or []))
print("policy diagnostics :", (summary.get("policy_diagnostics") or {}).get("policy_data_valid"))
