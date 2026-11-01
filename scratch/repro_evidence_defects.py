"""
scratch/repro_evidence_defects.py
================================
Reproduce the suspected evidence-pipeline defects with controlled fixtures.

Every case here is synthetic. None of it proves anything about real-world
detection accuracy; it only shows what the current code does with a given
evidence shape.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_fusion import calculate_source_agreement, calculate_fusion_score
from final_verdict import (
    deterministic_gate,
    extract_verdict,
    generate_final_verdict,
)

FAKE_AGENT = "Verdict:\nFAKE\n\nConfidence:\n88%\n\nReasoning:\nNever happened."
REAL_AGENT = "Verdict:\nREAL\n\nConfidence:\n94%\n\nReasoning:\nConfirmed."


def chunk(source_id, url, evidence_class, score, evidence_type="scraped_article",
          group=None, primary=True, tier="established_wire"):
    return {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2],
        "title": "title",
        "evidence_type": evidence_type,
        "chunk": "evidence passage",
        "similarity": 0.80,
        "nli_label": (
            "entailment" if evidence_class == "SUPPORT"
            else "contradiction" if evidence_class == "CONTRADICTION"
            else "neutral"
        ),
        "nli_score": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.9,
        "source_reliability": 0.9,
        "source_tier": tier,
        "independence_group": group or url.split("/")[2].replace("www.", ""),
        "is_primary_source": primary,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": "match",
        "event_match": "strong",
        "location_match": True,
        "detail_coverage": 0.9,
        "claim_has_location": True,
        "nli_status": "ok",
        "debunking": False,
    }


def decide(results, verification=REAL_AGENT, critic=REAL_AGENT):
    agreement = calculate_source_agreement(results)
    return generate_final_verdict(
        results, agreement, calculate_fusion_score(results, agreement),
        verification, critic,
    )


def hdr(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
hdr("DEFECT 1: two snippet-only CONTRADICTING publishers -> FAKE?")
# ---------------------------------------------------------------------------
snippet_contradiction = [
    chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9,
          evidence_type="search_snippet"),
    chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88,
          evidence_type="search_snippet"),
]
agreement = calculate_source_agreement(snippet_contradiction)
print("  independent_contradicting :", agreement["independent_contradicting"])
print("  primary_supporting_groups  :", agreement["primary_supporting_groups"])
print("  group is_snippet_only      :",
      [g["is_snippet_only"] for g in agreement["group_decisions"]])
gate = deterministic_gate(agreement, snippet_contradiction)
print("  gate permitted             :", sorted(gate["permitted_verdicts"]))
print("  gate reason                :", gate["permitted_reason"])
verdict = decide(snippet_contradiction, FAKE_AGENT, FAKE_AGENT)
print("  FINAL VERDICT              :", verdict["verdict"])
print()
print("  >>> snippet-only evidence must NEVER refute a claim:",
      "REPRODUCED" if verdict["verdict"] == "FAKE" else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 2: two snippet-only SUPPORTING publishers -> REAL?")
# ---------------------------------------------------------------------------
snippet_support = [
    chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9,
          evidence_type="search_snippet"),
    chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88,
          evidence_type="search_snippet"),
]
verdict = decide(snippet_support)
print("  FINAL VERDICT              :", verdict["verdict"], "(expected UNVERIFIED)")

# ---------------------------------------------------------------------------
hdr("DEFECT 3: MISSING source_id merges unrelated sources")
# ---------------------------------------------------------------------------
missing_ids = [
    {**chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9), "source_id": None},
    {**chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88), "source_id": None},
    {**chunk(3, "https://www.ndtv.com/c", "SUPPORT", 0.86), "source_id": None},
]
agreement = calculate_source_agreement(missing_ids)
print("  total_sources              :", agreement["total_sources"],
      "(3 distinct URLs)")
print("  independence_groups        :", agreement["independence_groups"],
      "(3 distinct publishers)")
print("  independent_supporting     :", agreement["independent_supporting"])
verdict = decide(missing_ids)
print("  FINAL VERDICT              :", verdict["verdict"])
print("  >>> unrelated sources merged:", "REPRODUCED"
      if agreement["total_sources"] == 1 else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 4: same source_id reused across different URLs")
# ---------------------------------------------------------------------------
reused = [
    chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
    chunk(1, "https://www.bbc.com/b", "SUPPORT", 0.88),
]
agreement = calculate_source_agreement(reused)
print("  total_sources              :", agreement["total_sources"],
      "(2 distinct URLs)")
print("  independence_groups        :", agreement["independence_groups"],
      "(2 distinct publishers)")
print("  >>> unrelated URLs merged   :", "REPRODUCED"
      if agreement["total_sources"] == 1 else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 5: 'primary' means 'scraped', not 'primary reporting'")
# ---------------------------------------------------------------------------
not_primary = [
    chunk(1, "https://blog.example.com/a", "SUPPORT", 0.9,
          primary=False, tier="weak/blog"),
    chunk(2, "https://other.example.org/b", "SUPPORT", 0.88,
          primary=False, tier="weak/blog"),
]
agreement = calculate_source_agreement(not_primary)
print("  is_primary_source on chunks:",
      [c["is_primary_source"] for c in not_primary])
print("  group is_primary           :",
      [g["is_primary"] for g in agreement["group_decisions"]])
print("  primary_supporting_groups  :", agreement["primary_supporting_groups"])
verdict = decide(not_primary)
print("  FINAL VERDICT              :", verdict["verdict"],
      "(non-primary/blog sources must not yield REAL)")
print("  >>> scraped treated as primary:", "REPRODUCED"
      if agreement["primary_supporting_groups"] == 2 else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 6: MISLEADING permitted with no evidence-backed correction")
# ---------------------------------------------------------------------------
# The agreement payload carries every policy counter: the gate no longer
# substitutes one counter for a missing one, it abstains instead.
gate = deterministic_gate(
    {"independent_supporting": 3, "independent_contradicting": 0,
     "primary_supporting_groups": 3, "primary_contradicting_groups": 0,
     "article_supporting_groups": 3, "article_contradicting_groups": 0,
     "correction_groups": 0, "primary_correction_groups": 0},
    [],
)
print("  gate permitted             :", sorted(gate["permitted_verdicts"]))
print("  >>> MISLEADING permitted without a correction:",
      "REPRODUCED" if "MISLEADING" in gate["permitted_verdicts"] else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 7: UNVERIFIED mislabels supporting_evidence as contradiction")
# ---------------------------------------------------------------------------
mixed = [
    chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
    chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.9),
]
verdict = decide(mixed)
print("  FINAL VERDICT              :", verdict["verdict"])
print("  supporting_evidence classes:",
      [e.get("url") for e in verdict["supporting_evidence"]])
print("  contradicting_evidence     :",
      [e.get("url") for e in verdict["contradicting_evidence"]])
print("  >>> supporting list holds contradiction chunks:",
      "REPRODUCED"
      if verdict["supporting_evidence"]
      and verdict["supporting_evidence"][0]["url"].endswith("/b")
      else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 8: agent verdict extraction picks an incidental label")
# ---------------------------------------------------------------------------
incidental = (
    "The source article discusses the claim in depth. It is not REAL to say "
    "the event happened; the record suggests FAKE. "
    "Final Explanation: The evidence does not establish the claim."
)
print("  extract_verdict(...)       :", extract_verdict(incidental))
print("  >>> incidental label selected:", "REPRODUCED"
      if extract_verdict(incidental) in ("REAL", "FAKE") else "not reproduced")

# ---------------------------------------------------------------------------
hdr("DEFECT 9: UNVERIFIED confidence is a high heuristic constant")
# ---------------------------------------------------------------------------
context_only = [
    chunk(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.2),
    chunk(2, "https://www.ndtv.com/b", "RELATED_CONTEXT", 0.2),
]
verdict = decide(context_only)
print("  FINAL VERDICT              :", verdict["verdict"])
print("  confidence                 :", verdict["confidence"])
print("  confidence_basis           :", verdict.get("confidence_basis"))
# The number is now the verification coverage and is labelled as such, so a
# high value next to UNVERIFIED is no longer read as "70% sure".
print("  >>> unlabelled heuristic confidence on UNVERIFIED:",
      "REPRODUCED"
      if verdict.get("confidence_basis") not in ("verification_coverage", "evidence_strength")
      else "not reproduced")
