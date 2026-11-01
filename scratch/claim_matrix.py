"""
scratch/claim_matrix.py
=======================
The documented test matrix for the text-verification validation.

`reference_outcome` is the FACTUAL label established from reputable reference
sources, and is stated independently of what the system returns. It is NOT a
prediction of what TruthLensAI will answer - the whole point of the matrix is
to compare the two.
"""

# Each entry:
#   key            - short id
#   claim          - the exact text sent to /verify/text
#   reference      - the factual outcome, and the reputable sources that
#                    establish it
#   expected_shape - what the evidence policy implies for a correct run
#                    (independent of whether the system finds the evidence)

MATRIX = [
    {
        "key": "apollo_r1",
        "repeat_of": "apollo",
        "claim": "NASA's Apollo 11 mission landed humans on the Moon in July 1969.",
        "reference_outcome": "TRUE",
        "reference_sources": [
            "NASA, 'July 20, 1969: One Giant Leap For Mankind' "
            "(nasa.gov/history/july-20-1969-one-giant-leap-for-mankind)",
            "Smithsonian National Air and Space Museum, 'Apollo 11' "
            "(si.edu/spotlight/apollo-11)",
        ],
        "reference_note": (
            "Apollo 11 landed on 20 July 1969; Armstrong and Aldrin spent "
            "about 21.5 hours on the lunar surface."
        ),
        "expected_shape": "REAL, or an honest UNVERIFIED if two independent "
                          "primary publishers with article text are not both found.",
    },
    {
        "key": "apollo_r2",
        "repeat_of": "apollo",
        "claim": "NASA's Apollo 11 mission landed humans on the Moon in July 1969.",
        "reference_outcome": "TRUE",
        "reference_sources": [
            "NASA, 'July 20, 1969: One Giant Leap For Mankind'",
            "Smithsonian National Air and Space Museum, 'Apollo 11'",
        ],
        "reference_note": "Repeat of the same claim to measure run-to-run variance.",
        "expected_shape": "Same as apollo_r1; differences must be explained by "
                          "retrieval variance, not by policy.",
    },
    {
        "key": "apollo_r3",
        "repeat_of": "apollo",
        "claim": "NASA's Apollo 11 mission landed humans on the Moon in July 1969.",
        "reference_outcome": "TRUE",
        "reference_sources": [
            "NASA, 'July 20, 1969: One Giant Leap For Mankind'",
            "Smithsonian National Air and Space Museum, 'Apollo 11'",
        ],
        "reference_note": "Third repeat of the same claim.",
        "expected_shape": "Same as apollo_r1.",
    },
    {
        "key": "obama_true",
        "claim": "Barack Obama was inaugurated as President of the United States on 20 January 2009.",
        "reference_outcome": "TRUE",
        "reference_sources": [
            "White House / Barack Obama Presidential Library, inauguration of "
            "Barack Obama, 20 January 2009",
            "U.S. Senate, 'About the Inauguration' (senate.gov)",
        ],
        "reference_note": "Obama was sworn in as the 44th President on 20 January 2009.",
        "expected_shape": "REAL, or an honest UNVERIFIED.",
    },
    {
        "key": "armstrong_mars_false",
        "claim": "Neil Armstrong was the first human to walk on Mars in July 1969.",
        "reference_outcome": "FALSE",
        "reference_sources": [
            "NASA, 'Apollo 11' and 'Mars Exploration Rover' mission pages "
            "(no human has landed on Mars)",
        ],
        "reference_note": (
            "Armstrong walked on the MOON in July 1969. No human has ever "
            "landed on Mars. The claim substitutes Mars for Moon, so the "
            "entity and the event do not match."
        ),
        "expected_shape": "FAKE if a credible source is found that says the "
                          "1969 landing was on the Moon, otherwise UNVERIFIED. "
                          "It must NOT be REAL.",
    },
    {
        "key": "apollo_misleading_duration",
        "claim": ("The Apollo 11 mission landed humans on the Moon in July 1969, "
                  "and the astronauts stayed on the lunar surface for three days."),
        "reference_outcome": "PARTLY FALSE (material detail wrong)",
        "reference_sources": [
            "NASA, 'Apollo 11' mission page: Armstrong and Aldrin spent about "
            "21.5 hours on the surface, of which about 2.5 hours outside the "
            "lunar module",
            "Smithsonian National Air and Space Museum, 'Apollo 11' timeline",
        ],
        "reference_note": (
            "The landing and the month/year are correct; 'three days on the "
            "surface' is wrong. The distinction is between a false claim (the "
            "event did not happen) and a misleading one (the event happened "
            "but a detail is wrong) - the latter is what MISLEADING is for."
        ),
        "expected_shape": "MISLEADING if an explicit, evidence-backed "
                          "correction of the duration is found; otherwise "
                          "UNVERIFIED, or FAKE if the source directly refutes "
                          "the whole claim. It must NOT be a clean REAL.",
    },
    {
        "key": "covid_lab_conflict",
        "claim": ("COVID-19 was engineered in a laboratory and is not of "
                  "natural origin."),
        "reference_outcome": "CONTESTED / NOT ESTABLISHED",
        "reference_sources": [
            "WHO, 'Origin of SARS-CoV-2' (who.int) - evidence points to a "
            "zoonotic origin",
            "US ODNI / intelligence agencies, 'Updated Coordinated Medical "
            "Intelligence Bulletin' (April 2023) - concluded the origin was "
            "not conclusively determined and that no specific foreign "
            "government or entity was supported as the cause",
        ],
        "reference_note": (
            "Credible sources disagree on whether the question is settled. "
            "This is the case where an honest UNVERIFIED is the CORRECT "
            "answer, and a decisive verdict would be over-claiming."
        ),
        "expected_shape": "UNVERIFIED (contested). A confident REAL or FAKE "
                          "here would be an over-claim.",
    },
    {
        "key": "biden_withdrawal_recent",
        "claim": "Joe Biden withdrew from the 2024 United States presidential race in July 2024.",
        "reference_outcome": "TRUE",
        "reference_sources": [
            "White House, 'President Biden to Step Down' statement, 21 July 2024",
            "Associated Press / BBC / Reuters coverage of the 21 July 2024 announcement",
        ],
        "reference_note": (
            "A recent, well-documented event: Biden announced his withdrawal on "
            "21 July 2024, less than a week before the Democratic National "
            "Convention. Requires fresh sources."
        ),
        "expected_shape": "REAL, or an honest UNVERIFIED if fresh sources are "
                          "not reachable.",
    },
    {
        "key": "apollo_vague_opinion",
        "claim": "NASA's Apollo program was the most successful space program in history.",
        "reference_outcome": "UNVERIFIABLE (opinion, no falsifiable event)",
        "reference_sources": [
            "No reputable source establishes 'most successful' as a measurable "
            "property of the Apollo program; it is a comparative judgement "
            "with no stated metric, baseline or date.",
        ],
        "reference_note": (
            "A vague comparative with no event, date or measurable quantity. "
            "There is nothing to verify, so an honest UNVERIFIED is the correct "
            "answer and anything else would be fabrication."
        ),
        "expected_shape": "UNVERIFIED. It must NOT be REAL or FAKE.",
    },
]
