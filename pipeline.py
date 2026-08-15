from agents import (
    build_reader_agent,
    build_research_agent,
    verification_chain,
    critic_chain
)


def run_research_pipeline(claim: str) -> dict:

    state = {}

    # ============================================================
    # STEP 1 — RESEARCH AGENT
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 1 - Research Agent is working...")
    print("=" * 50)

    search_agent = build_research_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""Investigate the following news claim.

Claim:
{claim}

Find recent, reliable and relevant information.
Look for both supporting and contradicting evidence."""
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSearch Results:\n")
    print(state["search_results"])


    # ============================================================
    # STEP 2 — EVIDENCE READER AGENT
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 2 - Evidence Reader Agent is working...")
    print("=" * 50)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""From the following research, identify the most
relevant source URL and scrape it for deeper evidence.

News Claim:
{claim}

Research Results:
{state["search_results"][:4000]}
"""
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped Evidence:\n")
    print(state["scraped_content"])


    # ============================================================
    # STEP 3 — VERIFICATION CHAIN
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 3 - Verification Agent is evaluating the claim...")
    print("=" * 50)

    state["verification"] = verification_chain.invoke({
        "claim": claim,
        "research": state["search_results"],
        "evidence": state["scraped_content"]
    })

    print("\nVerification Result:\n")
    print(state["verification"])


    # ============================================================
    # STEP 4 — CRITIC / REVIEW
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 4 - Critic Agent is reviewing the verification...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "claim": claim,
        "verification": state["verification"],
        "evidence": (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"SCRAPED EVIDENCE:\n{state['scraped_content']}"
        )
    })

    print("\nCritic Review:\n")
    print(state["feedback"])


    # ============================================================
    # RETURN COMPLETE PIPELINE STATE
    # ============================================================

    return state


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    claim = input("\nEnter a news claim: ")

    result = run_research_pipeline(claim)