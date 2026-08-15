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
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# 1. RESEARCH AGENT
# ============================================================

def build_research_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# ============================================================
# 2. EVIDENCE READER AGENT
# ============================================================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# ============================================================
# 3. VERIFICATION PROMPT
# ============================================================

verification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional fact-checking agent.

Your task is to analyze a news claim using the research and
evidence provided.

Do not blindly trust a source.

Look for:
- Supporting evidence
- Contradicting evidence
- Source credibility
- Conflicting information
- Missing or insufficient evidence

Classify the claim as one of:
REAL
FAKE
MISLEADING
UNVERIFIED

Be factual and evidence-based."""
    ),

    (
        "human",
        """Analyze the following news claim.

Claim:
{claim}

Research:
{research}

Detailed Evidence:
{evidence}

Provide:

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
..."""
    )
])


verification_chain = verification_prompt | llm | StrOutputParser()


# ============================================================
# 4. CRITIC / REVIEW AGENT
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict fact-checking reviewer.

Review the proposed verification result.

Check whether:
1. The evidence actually supports the verdict.
2. Important contradictory evidence was ignored.
3. Sources appear credible.
4. The confidence level is justified.
5. The conclusion contains unsupported assumptions.

If the evidence is insufficient, recommend UNVERIFIED."""
    ),

    (
        "human",
        """News Claim:
{claim}

Verification Result:
{verification}

Research Evidence:
{evidence}

Respond in this format:

Review:
...

Problems:
- ...
- ...

Recommended Verdict:
REAL / FAKE / MISLEADING / UNVERIFIED

Recommended Confidence:
X%

Final Recommendation:
..."""
    )
])


critic_chain = critic_prompt | llm | StrOutputParser()