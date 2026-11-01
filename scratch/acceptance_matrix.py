"""
Text-claim matrix acceptance test.
Runs fixture-based evidence analysis through analyze_evidence +
generate_final_verdict without hitting external search APIs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_analyzer import analyze_evidence
from evidence_fusion import calculate_source_agreement, calculate_fusion_score
from final_verdict import generate_final_verdict

NASA_TEXT = (
    "Apollo 11 was the mission that landed the first humans on the Moon. "
    "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the "
    "Sea of Tranquility on July 20, 1969, while Michael Collins remained in "
    "lunar orbit aboard the Columbia."
)

SMITHSONIAN_TEXT = (
    "On July 20, 1969, Apollo 11 landed on the Moon and Neil Armstrong and "
    "Edwin 'Buzz' Aldrin became the first humans to step onto the lunar "
    "surface."
)

BBC_TEXT = (
    "Apollo 11 was the spaceflight that first landed humans on the Moon. "
    "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the "
    "Eagle lunar module on July 20, 1969."
)

IRRELEVANT_TEXT = (
    "The Saturn V rocket was used for the Skylab and Apollo-Soyuz missions. "
    "The Apollo Applications Program was cancelled in 1968."
)

VAGUE_TEXT = (
    "This is some news about things happening in the world today."
)

CONFLICT_TEXT_1 = (
    "Apollo 11 was the spaceflight that first landed humans on the Moon. "
    "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the "
    "Eagle lunar module on July 20, 1969, while Michael Collins remained in "
    "lunar orbit aboard the Columbia command module."
)

CONFLICT_TEXT_2 = (
    "The Apollo 11 mission touched down on the lunar surface on July 20, "
    "1970, according to the mission log. Armstrong and Aldrin exited the "
    "Eagle module and walked on the Moon that day."
)


def _source(url, content, evidence_type="scraped_article", scrape_status="SUCCESS"):
    return {
        "url": url,
        "domain": url.split("//")[1].split("/")[0],
        "title": "Test Source",
        "content": content,
        "evidence_type": evidence_type,
        "scrape_status": scrape_status,
    }


def run_case(name, claim, sources, verdict_agent, confidence_agent):
    print(f"\n{'='*70}")
    print(f"CASE: {name}")
    print(f"CLAIM: {claim}")
    print(f"{'='*70}")

    results = analyze_evidence(claim, sources, verbose=False)

    agreement = calculate_source_agreement(results)
    fusion = calculate_fusion_score(results, agreement)

    verdict = generate_final_verdict(
        results,
        agreement,
        fusion,
        f"Verdict:\n{verdict_agent}\n\nConfidence:\n{confidence_agent}%\n\nReasoning:\nAgent reasoning.",
        f"Verdict:\n{verdict_agent}\n\nConfidence:\n{confidence_agent}%\n\nReasoning:\nCritic reasoning.",
    )

    print(f"  Actual verdict:      {verdict.get('verdict')}")
    print(f"  Confidence:          {verdict.get('confidence')}%  (basis: {verdict.get('confidence_basis')})")
    print(f"  Evidence sufficiency: {verdict.get('evidence_sufficiency')}")
    print(f"  Independent supporting: {verdict.get('independent_supporting')}")
    print(f"  Independent contradicting: {verdict.get('independent_contradicting')}")
    print(f"  Permitted verdicts:   {verdict.get('permitted_verdicts')}")
    print(f"  Reasoning:            {str(verdict.get('reasoning'))[:200]}")
    print(f"  Verdict overridden:   {verdict.get('verdict_overridden_by_evidence')}")
    print(f"  Agents agree:         {verdict.get('agents_agree')}")

    return verdict, results


print("\n########################################################################")
print("TEXT-CLAIM MATRIX ACCEPTANCE TESTS")
print("########################################################################")

# Case 1: True claim — supported by two independent primary sources
v1, r1 = run_case(
    "1. TRUE CLAIM — Apollo 11 landed July 20, 1969",
    "NASA's Apollo 11 mission landed humans on the Moon on July 20, 1969.",
    [
        _source("https://www.nasa.gov/apollo-11", NASA_TEXT),
        _source("https://www.si.edu/apollo-11", SMITHSONIAN_TEXT),
    ],
    "REAL", 90,
)

# Case 2: False claim — wrong date contradicted by two independent sources
v2, r2 = run_case(
    "2. FALSE CLAIM — Apollo 11 landed July 20, 1970",
    "NASA's Apollo 11 mission landed humans on the Moon on July 20, 1970.",
    [
        _source("https://www.nasa.gov/apollo-11", CONFLICT_TEXT_1),
        _source("https://www.si.edu/apollo-11", NASA_TEXT),
    ],
    "UNVERIFIED", 50,
)

# Case 3: Incorrect date — single contradicting source (not enough for FAKE)
v3, r3 = run_case(
    "3. INCORRECT DATE — single contradicting publisher",
    "Apollo 11 landed humans on the Moon on July 20, 1970.",
    [
        _source("https://www.nasa.gov/apollo-11", CONFLICT_TEXT_1),
    ],
    "UNVERIFIED", 50,
)

# Case 4: Ambiguous claim — vague, no verifiable content
v4, r4 = run_case(
    "4. AMBIGUOUS CLAIM — 'This news is misleading.'",
    "This news is misleading.",
    [
        _source("https://example.com/news1", VAGUE_TEXT),
    ],
    "UNVERIFIED", 50,
)

# Case 5: Insufficient evidence — no relevant sources
v5, r5 = run_case(
    "5. INSUFFICIENT EVIDENCE — irrelevant article only",
    "Apollo 11 landed humans on the Moon on July 20, 1969.",
    [
        _source("https://example.com/irrelevant", IRRELEVANT_TEXT),
    ],
    "UNVERIFIED", 50,
)

# Case 6: Conflicting sources — one supports, one contradicts
v6, r6 = run_case(
    "6. CONFLICTING SOURCES — support vs contradiction",
    "Apollo 11 landed humans on the Moon on July 20, 1969.",
    [
        _source("https://www.nasa.gov/apollo-11", NASA_TEXT),
        _source("https://example.com/fake-claim", CONFLICT_TEXT_2),
    ],
    "UNVERIFIED", 50,
)

# Summary
print(f"\n{'='*70}")
print("MATRIX SUMMARY")
print(f"{'='*70}")

def check(name, verdict, expected, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {name} -> verdict={verdict.get('verdict')} (expected={expected}) {detail}")

check("True claim", v1, "REAL",
      v1["verdict"] == "REAL" and v1["independent_supporting"] >= 2,
      "supported by two independent publishers")
check("False claim (wrong date)", v2, "UNVERIFIED",
      v2["verdict"] == "UNVERIFIED" and v2["independent_contradicting"] >= 2,
      "two independent contradicting publishers")
check("Incorrect date (single source)", v3, "UNVERIFIED",
      v3["verdict"] == "UNVERIFIED" and v3["independent_contradicting"] == 1,
      "single contradicting publisher cannot force FAKE")
check("Ambiguous claim", v4, "UNVERIFIED",
      v4["verdict"] == "UNVERIFIED",
      "vague claim produces UNVERIFIED")
check("Insufficient evidence", v5, "UNVERIFIED",
      v5["verdict"] == "UNVERIFIED" and v5["independent_supporting"] == 0,
      "no relevant evidence produces UNVERIFIED")
check("Conflicting sources", v6, "UNVERIFIED",
      v6["verdict"] == "UNVERIFIED",
      "support + contradiction balance to UNVERIFIED")

print(f"\n{'='*70}")
print("All matrix cases recorded above.")
print(f"{'='*70}")
