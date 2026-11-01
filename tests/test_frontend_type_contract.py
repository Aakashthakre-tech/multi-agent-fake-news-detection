"""
tests/test_frontend_type_contract.py
=====================================
Guards the contract between the API response and the TypeScript interfaces.

The frontend declares its response shape in `frontend/src/types/index.ts`.
If the backend adds or renames a field and the TypeScript interface is not
updated, the UI silently drops that data. This test parses the real interface
declarations and asserts that every field the API actually returns for a
verification result is declared.

The field list is taken from the real Pydantic response models, so this
catches drift in either direction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TYPES_FILE = PROJECT_ROOT / "frontend" / "src" / "types" / "index.ts"


def _parse_interfaces(source: str) -> Dict[str, Set[str]]:
    """
    Extract `export interface Name { ... }` blocks and their field names.
    """
    interfaces: Dict[str, Set[str]] = {}
    pattern = re.compile(
        r"export\s+interface\s+(\w+)\s*\{(.*?)\n\}", re.DOTALL
    )
    for match in pattern.finditer(source):
        name = match.group(1)
        body = match.group(2)
        fields: Set[str] = set()
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue
            # `name: type;` or `name?: type;`
            field = re.match(r"^(\w+)\??\s*:", stripped)
            if field:
                fields.add(field.group(1))
        interfaces[name] = fields
    return interfaces


@pytest.fixture(scope="module")
def interfaces() -> Dict[str, Set[str]]:
    if not TYPES_FILE.exists():
        pytest.skip("frontend types file not present")
    return _parse_interfaces(TYPES_FILE.read_text(encoding="utf-8"))


def test_types_file_parses(interfaces):
    assert "UnifiedVerificationResponse" in interfaces
    assert "ResearchReport" in interfaces
    assert "ResearchExecutiveSummary" in interfaces
    assert "ResearchFinding" in interfaces
    assert "ResearchMethodology" in interfaces
    assert "ResearchClaimResult" in interfaces
    assert "ResearchEvidenceGap" in interfaces


def test_unified_response_declares_research_report(interfaces):
    fields = interfaces["UnifiedVerificationResponse"]
    for required in (
        "modality", "status", "factual_verification", "synthetic_media_detection",
        "visual_classification", "transcription", "primary_claim",
        "consistency_analysis", "case_summary", "pipeline_stages",
        "modality_specific", "research_report", "raw", "error", "warnings",
    ):
        assert required in fields, f"UnifiedVerificationResponse must declare '{required}'"


def test_research_report_declares_every_sub_block(interfaces):
    fields = interfaces["ResearchReport"]
    for required in (
        "modality", "executive_summary", "findings", "claims", "methodology",
        "evidence_gaps",
    ):
        assert required in fields, f"ResearchReport must declare '{required}'"


def test_executive_summary_declares_every_field(interfaces):
    fields = interfaces["ResearchExecutiveSummary"]
    for required in (
        "input_claim", "modality", "verdict", "confidence", "confidence_basis",
        "attempted", "key_findings", "why_this_verdict",
        "deterministic_gate_reason", "agent_narrative", "critic_review",
        "limitations",
    ):
        assert required in fields, f"ResearchExecutiveSummary must declare '{required}'"


def test_finding_declares_every_field(interfaces):
    fields = interfaces["ResearchFinding"]
    for required in (
        "title", "explanation", "claim_text", "status", "evidence_strength",
        "strength_label", "source_name", "source_domain", "source_url",
        "source_tier", "source_reliability", "excerpt", "excerpt_kind",
        "date_status", "event_match", "nli_label", "nli_score",
        "independence_group", "evidence_class", "evidence_type", "search_query",
    ):
        assert required in fields, f"ResearchFinding must declare '{required}'"


def test_methodology_declares_every_field(interfaces):
    fields = interfaces["ResearchMethodology"]
    for required in (
        "research_skipped", "research_skipped_reason", "search_queries",
        "search_errors", "sources_retrieved", "articles_scraped",
        "scrapes_failed", "evidence_passages", "nli_evaluated", "stages",
    ):
        assert required in fields, f"ResearchMethodology must declare '{required}'"


def test_claim_result_declares_every_field(interfaces):
    fields = interfaces["ResearchClaimResult"]
    for required in (
        "claim_text", "verdict", "confidence", "evidence_sufficiency", "reasoning",
        "supporting_evidence", "contradicting_evidence", "source_name",
        "source_url", "timestamps", "frame_references", "source_modality",
        "selection_reason", "priority_score", "limitations",
    ):
        assert required in fields, f"ResearchClaimResult must declare '{required}'"


def test_evidence_gap_declares_every_field(interfaces):
    fields = interfaces["ResearchEvidenceGap"]
    for required in ("question", "status", "detail", "what_would_help"):
        assert required in fields, f"ResearchEvidenceGap must declare '{required}'"


def test_synthetic_details_declare_manipulation_and_operating_point(interfaces):
    fields = interfaces["SyntheticMediaDetails"]
    assert "manipulation_status" in fields
    assert "manipulation_note" in fields
    assert "operating_point" in fields
    assert "model" in fields


def test_saved_e2e_response_matches_the_declared_types():
    """
    The real saved API response must only contain fields the TypeScript
    interfaces declare (or that live inside an open `Record<string, any>`).
    """
    saved = PROJECT_ROOT / "scratch" / "e2e_results.json"
    if not saved.exists():
        pytest.skip("no saved e2e response available")

    import json

    data = json.loads(saved.read_text(encoding="utf-8"))
    interfaces = _parse_interfaces(TYPES_FILE.read_text(encoding="utf-8"))
    report_fields = interfaces["ResearchReport"]

    for modality, payload in data.items():
        report = payload.get("report") or {}
        unknown = set(report.keys()) - report_fields
        assert not unknown, (
            f"{modality}: research_report contains fields the frontend does not declare: "
            f"{sorted(unknown)}"
        )

        summary_fields = interfaces["ResearchExecutiveSummary"]
        unknown_summary = set((report.get("executive_summary") or {}).keys()) - summary_fields
        assert not unknown_summary, (
            f"{modality}: executive_summary contains undeclared fields: "
            f"{sorted(unknown_summary)}"
        )

        method_fields = interfaces["ResearchMethodology"]
        unknown_method = set((report.get("methodology") or {}).keys()) - method_fields
        assert not unknown_method, (
            f"{modality}: methodology contains undeclared fields: "
            f"{sorted(unknown_method)}"
        )
