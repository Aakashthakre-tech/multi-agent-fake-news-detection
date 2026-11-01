"""
tests/test_evidence_policy_guards.py
====================================
Focused regression tests for the two confirmed high-priority logic risks.

1. ``final_verdict.deterministic_gate`` used to substitute one counter for
   another when a required policy field was missing. That let an incomplete
   agreement dict produce a stronger verdict than the evidence supports.
   There are now no fallbacks: missing or invalid policy fields make the gate
   abstain with a diagnostic.

2. ``evidence_fusion`` counted ANY passage that used correction language as a
   correction. A correction now only counts when it directly addresses this
   claim, corrects a material detail of it, and contradicts it.

Every test here uses controlled fixtures. No network call and no LLM call is
made, and nothing here proves anything about real-world detection accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_fusion import (  # noqa: E402
    _is_material_correction,
    calculate_source_agreement,
)
from final_verdict import (  # noqa: E402
    REQUIRED_POLICY_FIELDS,
    deterministic_gate,
    generate_final_verdict,
)

REAL_AGENT = "Verdict:\nREAL\n\nConfidence:\n94%\n\nReasoning:\nConfirmed."
FAKE_AGENT = "Verdict:\nFAKE\n\nConfidence:\n88%\n\nReasoning:\nNever happened."


def chunk(
    source_id,
    url,
    evidence_class="SUPPORT",
    score=0.9,
    evidence_type="scraped_article",
    group=None,
    primary=True,
    tier="established_wire",
    debunking=False,
    date_status="match",
    event_match="strong",
    detail_coverage=0.9,
    nli_label=None,
    **overrides,
) -> dict:
    record = {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2] if url else "",
        "title": "title",
        "evidence_type": evidence_type,
        "chunk": "evidence passage",
        "similarity": 0.80,
        "nli_label": nli_label or (
            "entailment" if evidence_class == "SUPPORT"
            else "contradiction" if evidence_class == "CONTRADICTION"
            else "neutral"
        ),
        "nli_score": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.9,
        "source_reliability": 0.9,
        "source_tier": tier,
        "independence_group": group if group is not None else (
            url.split("/")[2].replace("www.", "") if url else ""
        ),
        "is_primary_source": primary,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": date_status,
        "date_reason": "claimed date matched",
        "event_match": event_match,
        "location_match": True,
        "detail_coverage": detail_coverage,
        "claim_has_location": True,
        "nli_status": "ok",
        "debunking": debunking,
    }
    record.update(overrides)
    return record


def complete_agreement(**overrides) -> dict:
    """A valid, complete policy payload - what the fusion layer produces."""
    agreement = {
        "independent_supporting": 0,
        "independent_contradicting": 0,
        "primary_supporting_groups": 0,
        "primary_contradicting_groups": 0,
        "article_supporting_groups": 0,
        "article_contradicting_groups": 0,
        "correction_groups": 0,
        "primary_correction_groups": 0,
    }
    agreement.update(overrides)
    return agreement


# ===========================================================================
# 1. Missing policy fields cannot create a stronger verdict
# ===========================================================================

@pytest.mark.parametrize("missing", REQUIRED_POLICY_FIELDS)
def test_missing_policy_field_abstains(missing):
    agreement = complete_agreement(
        independent_supporting=3,
        primary_supporting_groups=2,
        article_supporting_groups=2,
    )
    del agreement[missing]

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert missing in gate["permitted_reason"]
    assert "REAL" not in gate["permitted_verdicts"]
    assert "FAKE" not in gate["permitted_verdicts"]
    assert "MISLEADING" not in gate["permitted_verdicts"]


def test_missing_primary_contradicting_does_not_fall_back_to_independent():
    """
    The unsafe legacy fallback: without ``primary_contradicting_groups`` the
    gate used to treat every contradicting publisher as a primary one, so two
    publishers could refute a claim on their own.
    """
    agreement = {
        "independent_supporting": 0,
        "independent_contradicting": 3,
        "primary_supporting_groups": 0,
        "article_supporting_groups": 0,
        "article_contradicting_groups": 0,
        "correction_groups": 0,
        "primary_correction_groups": 0,
        # primary_contradicting_groups deliberately absent
    }

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "primary_contradicting_groups" in gate["permitted_reason"]


def test_missing_primary_correction_does_not_fall_back_to_correction_groups():
    """
    The unsafe legacy fallback: without ``primary_correction_groups`` the gate
    used to treat every correction group as a primary correction, which could
    permit MISLEADING on a non-primary correction.
    """
    agreement = complete_agreement(
        independent_supporting=3,
        primary_supporting_groups=2,
        article_supporting_groups=2,
        correction_groups=1,
    )
    del agreement["primary_correction_groups"]

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "primary_correction_groups" in gate["permitted_reason"]


def test_missing_article_supporting_does_not_fall_back_to_primary():
    agreement = {
        "independent_supporting": 3,
        "independent_contradicting": 0,
        "primary_supporting_groups": 2,
        "primary_contradicting_groups": 0,
        "article_contradicting_groups": 0,
        "correction_groups": 0,
        "primary_correction_groups": 0,
        # article_supporting_groups deliberately absent
    }

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert "REAL" not in gate["permitted_verdicts"]


def test_empty_agreement_abstains():
    gate = deterministic_gate({}, [])
    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert gate["policy_data_errors"]


def test_none_agreement_abstains():
    gate = deterministic_gate(None, [])
    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}


# ===========================================================================
# 2. Invalid policy fields fail safely
# ===========================================================================

@pytest.mark.parametrize("bad_value", [
    "2",              # a string
    None,             # null
    True,             # a bool is not a count
    -1,               # negative
    1.5,              # not a whole number
    [2],              # a list
    {"count": 2},     # a dict
])
def test_invalid_policy_field_abstains(bad_value):
    agreement = complete_agreement(
        independent_supporting=3,
        primary_supporting_groups=2,
        article_supporting_groups=2,
    )
    agreement["primary_supporting_groups"] = bad_value

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "REAL" not in gate["permitted_verdicts"]
    assert gate["policy_data_errors"]


def test_invalid_correction_counter_cannot_permit_misleading():
    agreement = complete_agreement(
        independent_supporting=3,
        primary_supporting_groups=2,
        article_supporting_groups=2,
        correction_groups="many",
    )

    gate = deterministic_gate(agreement, [])

    assert gate["policy_data_valid"] is False
    assert "MISLEADING" not in gate["permitted_verdicts"]


def test_failed_safe_gate_survives_the_full_verdict_path():
    """
    The abstention must hold all the way to the final verdict, even when
    both agents confidently ask for REAL.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    agreement = complete_agreement(
        independent_supporting=2,
        primary_supporting_groups=2,
        article_supporting_groups=2,
    )
    del agreement["article_supporting_groups"]

    verdict = generate_final_verdict(
        results, agreement, 0.8, REAL_AGENT, REAL_AGENT
    )

    assert verdict["verdict"] == "UNVERIFIED"
    assert verdict["policy_data_valid"] is False
    assert verdict["policy_data_errors"]
    assert "REAL" not in verdict["permitted_verdicts"]
    # The agents asked for REAL; the override must be reported.
    assert verdict["verdict_overridden_by_evidence"] is True
    assert verdict["agents_agree"] is True


# ===========================================================================
# 3. Unrelated correction evidence does not satisfy the correction requirement
# ===========================================================================

def test_correction_language_on_an_unrelated_passage_is_not_a_correction():
    """
    A passage that merely uses correction language is not a correction of
    this claim.
    """
    unrelated = chunk(
        1, "https://www.reuters.com/a", "RELATED_CONTEXT", 0.2,
        debunking=True, date_status="match", detail_coverage=0.0,
    )
    assert _is_material_correction(unrelated) is False

    agreement = calculate_source_agreement([unrelated])
    assert agreement["correction_groups"] == 0
    assert agreement["primary_correction_groups"] == 0


def test_correction_language_on_a_supporting_passage_is_not_a_correction():
    """
    Support and correction language in the same passage: the passage supports
    the claim, so it cannot also be the correction of it.
    """
    supporting = chunk(
        1, "https://www.reuters.com/a", "SUPPORT", 0.9,
        debunking=True, date_status="match", detail_coverage=0.9,
    )
    assert _is_material_correction(supporting) is False

    agreement = calculate_source_agreement([supporting])
    assert agreement["correction_groups"] == 0


def test_a_contradiction_without_correction_language_is_not_a_correction():
    generic = chunk(
        1, "https://www.reuters.com/a", "CONTRADICTION", 0.86,
        debunking=False, date_status="match", detail_coverage=0.9,
    )
    assert _is_material_correction(generic) is False

    agreement = calculate_source_agreement([generic])
    assert agreement["correction_groups"] == 0
    assert agreement["primary_correction_groups"] == 0


def test_a_contradiction_with_no_material_detail_is_not_a_correction():
    """
    It contradicts the claim, but shares none of the claim's own content and
    does not correct the date, so it corrects nothing material.
    """
    thin = chunk(
        1, "https://www.reuters.com/a", "CONTRADICTION", 0.86,
        debunking=True, date_status="match", detail_coverage=0.05,
    )
    assert _is_material_correction(thin) is False

    agreement = calculate_source_agreement([thin])
    assert agreement["correction_groups"] == 0


def test_unrelated_correction_cannot_permit_misleading():
    """
    The full path: unrelated correction language must not unlock MISLEADING.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://www.ndtv.com/c", "RELATED_CONTEXT", 0.2,
              debunking=True, date_status="match", detail_coverage=0.0),
    ]
    agreement = calculate_source_agreement(results)

    assert agreement["correction_groups"] == 0
    assert agreement["primary_correction_groups"] == 0

    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" not in gate["permitted_verdicts"]


def test_a_non_primary_correction_is_not_a_primary_correction():
    """
    The correcting source is a blog, so it cannot unlock MISLEADING even
    though the group also contains a primary supporting source.
    """
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://blog.example.org/c", "CONTRADICTION", 0.86,
              primary=False, tier="weak/blog",
              debunking=True, date_status="different"),
    ]
    agreement = calculate_source_agreement(results)

    assert agreement["correction_groups"] == 1
    assert agreement["primary_correction_groups"] == 0

    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" not in gate["permitted_verdicts"]


# ===========================================================================
# 4. Relevant correction evidence still works
# ===========================================================================

def test_a_dated_correction_of_the_claimed_event_is_a_correction():
    correction = chunk(
        1, "https://www.ndtv.com/c", "CONTRADICTION", 0.86,
        debunking=True, date_status="different", detail_coverage=0.4,
    )
    assert _is_material_correction(correction) is True

    agreement = calculate_source_agreement([correction])
    assert agreement["correction_groups"] == 1
    assert agreement["primary_correction_groups"] == 1


def test_a_correction_that_covers_a_material_detail_is_a_correction():
    correction = chunk(
        1, "https://www.ndtv.com/c", "CONTRADICTION", 0.86,
        debunking=True, date_status="match", detail_coverage=0.75,
    )
    assert _is_material_correction(correction) is True


def test_relevant_correction_still_permits_misleading():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.86,
              debunking=True, date_status="different", detail_coverage=0.5),
    ]
    agreement = calculate_source_agreement(results)

    assert agreement["correction_groups"] == 1
    assert agreement["primary_correction_groups"] == 1

    gate = deterministic_gate(agreement, results)
    assert "MISLEADING" in gate["permitted_verdicts"]
    assert "REAL" in gate["permitted_verdicts"]
    assert "correction" in gate["permitted_reason"].lower()


# ===========================================================================
# 5. Existing valid verdict behaviour is unchanged
# ===========================================================================

def test_valid_complete_input_still_permits_real():
    agreement = complete_agreement(
        independent_supporting=2,
        primary_supporting_groups=2,
        article_supporting_groups=2,
    )
    gate = deterministic_gate(agreement, [])
    assert gate["policy_data_valid"] is True
    assert gate["policy_data_errors"] == []
    assert gate["permitted_verdicts"] == {"REAL", "UNVERIFIED"}


def test_valid_complete_input_still_permits_fake():
    agreement = complete_agreement(
        independent_contradicting=2,
        primary_contradicting_groups=2,
        article_contradicting_groups=2,
    )
    gate = deterministic_gate(agreement, [])
    assert gate["policy_data_valid"] is True
    assert gate["permitted_verdicts"] == {"FAKE", "UNVERIFIED"}


def test_valid_complete_input_still_abstains_when_under_supported():
    agreement = complete_agreement(independent_supporting=1)
    gate = deterministic_gate(agreement, [])
    assert gate["policy_data_valid"] is True
    assert gate["permitted_verdicts"] == {"UNVERIFIED"}
    assert "Only one independent publisher" in gate["permitted_reason"]


def test_real_fusion_output_is_always_valid():
    """
    The layer that produces the agreement must always produce a payload the
    gate can consume. This is the contract between the two modules.
    """
    for results in (
        [],
        [chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9)],
        [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        ],
        [
            chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9),
            chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88),
        ],
        [chunk(1, "https://www.reuters.com/a", "RELATED_CONTEXT", 0.2)],
    ):
        agreement = calculate_source_agreement(results)
        gate = deterministic_gate(agreement, results)
        assert gate["policy_data_valid"] is True, (
            f"the fusion layer produced an unusable agreement for {results}"
        )
        assert gate["policy_data_errors"] == []


def test_end_to_end_real_verdict_is_unchanged():
    results = [
        chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
        chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
    ]
    agreement = calculate_source_agreement(results)
    verdict = generate_final_verdict(
        results, agreement, 0.8, REAL_AGENT, REAL_AGENT
    )
    assert verdict["verdict"] == "REAL"
    assert verdict["policy_data_valid"] is True
    assert verdict["policy_data_errors"] == []


def test_end_to_end_fake_verdict_is_unchanged():
    results = [
        chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9),
        chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88),
    ]
    agreement = calculate_source_agreement(results)
    verdict = generate_final_verdict(
        results, agreement, 0.8, FAKE_AGENT, FAKE_AGENT
    )
    assert verdict["verdict"] == "FAKE"
    assert verdict["policy_data_valid"] is True
