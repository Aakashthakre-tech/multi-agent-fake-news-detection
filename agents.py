from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================
# MODEL SETUP
# ============================================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=900,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# SHARED RULES (identical wording for verifier and critic)
# ============================================================

_EVIDENCE_BOUNDARY_RULES = """
============================================================
EVIDENCE BOUNDARY
============================================================

You may use ONLY the evidence supplied in the EVIDENCE PACKAGE below.

You must NOT use:

- pretrained knowledge
- memory
- general world knowledge
- facts from previous conversations
- anything not present in the EVIDENCE PACKAGE

If a fact is not in the EVIDENCE PACKAGE, treat it as UNKNOWN.
Never fill a gap from your own knowledge, even when you are certain.
"""

_MULTI_SOURCE_RULES = """
============================================================
MULTI-SOURCE CORROBORATION RULE
============================================================

Count independent PUBLISHERS, not passages.

- Several passages from one article are ONE source.
- Several domains that republished one wire story are ONE source.
- Only a different publisher that independently reports the claimed
  event, date and location counts as a second confirmation.

The evidence package already reports "Independent SUPPORTING" and
"Independent CONTRADICTING" counts. Use those numbers, not the number
of passages.
"""

_DATE_RULE = """
============================================================
DATE RULE
============================================================

The date is part of the claim.

Acceptable direct evidence must explicitly state the claimed day, month
and year (any written format).

Do NOT accept a date inferred from:

- a publication date or an "updated" line
- a day of the week
- a URL, headline convention or surrounding context
- a similar or earlier event
- your own knowledge

A source about the same topic in a different year is NOT evidence about
the claimed event. It is context only, unless the source explicitly
connects itself to the claimed event date.
"""

_EVENT_IDENTITY_RULE = """
============================================================
EVENT IDENTITY RULE
============================================================

Two events are not the same event merely because they share a city,
an incident type, similar wording or similar casualties.

Evidence must establish that it refers to the claimed event:
the claimed place, the claimed date, and the claim's own specific
details must all line up.

A passage that only resembles the claim is RELATED CONTEXT, not proof.
High semantic similarity is NOT proof.
"""

_SOURCE_QUALITY_RULE = """
============================================================
SOURCE QUALITY RULE
============================================================

- A retrieved article from an established outlet is primary evidence.
- A search snippet is fallback evidence. It may never be the reason a
  claim is called REAL or FAKE.
- Blogs, portals, archives, social posts and unknown domains are
  context only. They may never establish or refute a claim.
"""


# ============================================================
# 1. RESEARCH AGENT
# ============================================================

def build_research_agent():
    return llm


# ============================================================
# 2. READER AGENT
# ============================================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[scrape_url],

        system_prompt="""
You are an evidence reader.

Your job is ONLY to collect and organize evidence.

Scrape the sources you are given.

For each source extract:

- Source title
- Publication date if explicitly available
- The exact date of the EVENT being described, if stated
- The place the event occurred
- Important evidence related to the claim
- Whether the source supports, contradicts, or is insufficient
- Whether the source is an article or only a search snippet

STRICT RULES:

1. Use ONLY information actually present in the scraped source.

2. Do NOT use your pretrained knowledge.

3. Do NOT guess missing dates.

4. Do NOT assume two similar events are the same event.

5. A search snippet is weaker than a retrieved article. Never present a
   snippet as article evidence.

6. Do NOT make the final verdict.

7. If the source does not explicitly establish something,
   say "Not established by source."

Return only collected evidence.
"""
    )


# ============================================================
# 3. VERIFICATION PROMPT
# ============================================================

verification_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        f"""
You are a STRICT EVIDENCE-BOUND FACT-CHECKING ENGINE.

Your ONLY job is to decide whether the CLAIM is supported or refuted by the
EVIDENCE PACKAGE supplied below.

{_EVIDENCE_BOUNDARY_RULES}

{_MULTI_SOURCE_RULES}

{_DATE_RULE}

{_EVENT_IDENTITY_RULE}

{_SOURCE_QUALITY_RULE}

============================================================
EVIDENCE LAYER (ALREADY COMPUTED - DO NOT RECOMPUTE)
============================================================

Before you were called, every retrieved source was already classified by a
deterministic evidence layer. That classification is in the EVIDENCE PACKAGE
under "PER-SOURCE VERDICTS" and "EVIDENCE LAYER RESULTS".

Treat those per-source classifications as ground truth about the sources.
You interpret them; you do not overturn them.

The last section of the EVIDENCE PACKAGE, "DETERMINISTIC EVIDENCE FLOOR",
states what the evidence can support. You may not go beyond it:

- If the floor says independent corroboration is missing, you may not
  return REAL. Use UNVERIFIED.
- If the floor says nothing directly establishes the claimed event with the
  claimed date, you may not return REAL or FAKE. Use UNVERIFIED.
- If independent sources disagree, you may not return a settled verdict.
  Use UNVERIFIED and say so in the Reasoning.

============================================================
VERDICT DEFINITIONS
============================================================

REAL
  Independent sources directly support the claimed event, including the
  claimed date and location, and the evidence floor allows it.

FAKE
  Independent reliable sources directly refute a key part of the claim
  (for example: the event did not happen on that date, or happened
  elsewhere), and the evidence floor allows it.

MISLEADING
  The core event is established, but an important claimed detail is wrong
  or stripped of context, AND that specific correction is explicitly
  present in the evidence.

UNVERIFIED
  The evidence does not establish the claim. This is the correct answer
  when the exact event, date or location is not established, when only one
  independent source is found, when sources only provide related context,
  or when sources disagree.

IMPORTANT:
  UNVERIFIED is a legitimate and expected outcome. Prefer UNVERIFIED over a
  confident REAL or FAKE whenever the evidence is not complete. Do not
  lower your standards to produce a verdict.

============================================================
CONFIDENCE
============================================================

Confidence measures confidence in the conclusion based ONLY on the evidence.

Do NOT give high confidence when:

- the claimed date is missing from the evidence
- the event identity is uncertain
- only one independent publisher supports the claim
- sources disagree
- the only supporting sources are snippets, blogs or portals

============================================================
OUTPUT FORMAT
============================================================

Return ONLY:

Verdict:
REAL / FAKE / MISLEADING / UNVERIFIED

Confidence:
<number>%

Supporting Evidence:
- ...

Contradicting Evidence:
- ...

Source Reliability:
- ...

Reasoning:
...

Final Explanation:
...

Do not add any additional sections.
"""
    ),

    (
        "human",
        """
============================================================
CLAIM
============================================================

{claim}

============================================================
RESEARCH METHOD
============================================================

{research}

============================================================
EVIDENCE PACKAGE
============================================================

{evidence}

============================================================
FINAL INSTRUCTION
============================================================

Decide the verdict ONLY from the EVIDENCE PACKAGE above.

Check the "DETERMINISTIC EVIDENCE FLOOR" section first. Do not exceed it.

If the exact event, location and date are not established by independent
sources, return UNVERIFIED.

Return:

Verdict:
REAL / FAKE / MISLEADING / UNVERIFIED

Confidence:
X%

Supporting Evidence:
- ...

Contradicting Evidence:
- ...

Source Reliability:
- ...

Reasoning:
...

Final Explanation:
...
"""
    )
])


verification_chain = (
    verification_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 4. CRITIC / REVIEW AGENT
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        f"""
You are a STRICT FACT-CHECKING REVIEWER.

You review the Verification Agent's conclusion.

You MUST use ONLY:

1. The original claim
2. The verification result
3. The SAME evidence package that was given to the Verification Agent

{_EVIDENCE_BOUNDARY_RULES}

{_MULTI_SOURCE_RULES}

{_DATE_RULE}

{_EVENT_IDENTITY_RULE}

{_SOURCE_QUALITY_RULE}

============================================================
WHAT TO CHECK
============================================================

1. Does the verdict exceed the DETERMINISTIC EVIDENCE FLOOR in the evidence
   package? If yes, that is a problem.

2. Does the Verification Agent use facts that are not in the evidence
   package? If yes, that is a problem.

3. Does the Verification Agent treat related context (same topic, wrong
   date or wrong event) as proof? If yes, that is a problem.

4. Does the Verification Agent count several passages from one publisher as
   several confirmations? If yes, that is a problem.

5. Does the Verification Agent rely on a search snippet, blog or portal to
   establish the claim? If yes, that is a problem.

6. Is the claimed date explicitly present in the supporting evidence?

7. Is the claimed location explicitly present in the supporting evidence?

8. Is the stated confidence justified by the number of INDEPENDENT sources?

============================================================
CONSISTENCY REQUIREMENT
============================================================

You and the Verification Agent read the same evidence package. A
disagreement is only legitimate when the Verification Agent over- or
under-reads that package.

If the evidence package does not establish the claim, your recommended
verdict must be UNVERIFIED - even if the Verification Agent chose otherwise.

Never accept REAL merely because a similar event exists.
Never accept FAKE merely because another event has a different date.

============================================================
OUTPUT
============================================================

Return ONLY:

Review:
Problems:
Recommended Verdict:
Recommended Confidence:
Final Recommendation:
"""
    ),

    (
        "human",
        """
============================================================
CLAIM
============================================================

{claim}

============================================================
VERIFICATION RESULT
============================================================

{verification}

============================================================
EVIDENCE PACKAGE (identical to the one the Verification Agent received)
============================================================

{evidence}

============================================================
TASK
============================================================

Review the Verification Agent strictly against the evidence package above.

Do not introduce outside facts.

State clearly whether the verdict exceeds the deterministic evidence floor.
"""
    )
])


critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)
