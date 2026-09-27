import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from agents import (
    verification_chain,
    critic_chain
)

from evidence_fusion import (
    calculate_source_agreement,
    calculate_fusion_score
)

from evidence_analyzer import analyze_evidence
from evidence_brief import build_evidence_brief
from source_reliability import (
    get_source_profile,
    is_near_duplicate,
    registrable_domain,
)
from tools import web_search, scrape_url
from final_verdict import generate_final_verdict


# ============================================================
# CANDIDATE EXTRACTION & DEDUPLICATION HELPERS
# ============================================================

# How many candidates are retrieved. Retrieval is wider than this; the limit
# exists so a run cannot fan out without bound.
MAX_SOURCES = 8

# How many candidates are actually fetched. Scraping failures are the single
# biggest cause of lost evidence, so we fetch more than we report and keep the
# ones that yield usable text.
MAX_SCRAPE_ATTEMPTS = 12

# Minimum characters of provider-extracted text before it is treated as real
# article text rather than a snippet.
MIN_EXTRACT_CHARS = 600

# Candidates from these retrieval classes are always analysed first, and are
# preferred when the pool has to be trimmed.
PRIMARY_RETRIEVAL_CLASSES = ("exact_date", "factcheck", "wrong_date")

_SEARCH_FIELDS = (
    "URL", "Domain", "Published Date", "Found Via", "Source Tier",
    "Reliability", "Match Signals", "Retrieval Class", "Content Kind",
    "Search Score", "Search Snippet",
)


def normalize_url(url: str) -> str:
    """Normalize URL by stripping tracking parameters, anchors, and trailing slashes."""
    try:
        parsed = urlparse(url)
        tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid"}
        query_dict = parse_qs(parsed.query)
        cleaned_query = {k: v for k, v in query_dict.items() if k.lower() not in tracking_params}
        new_query = urlencode(cleaned_query, doseq=True)
        path = parsed.path.rstrip("/")
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.params, new_query, ""))
    except Exception:
        return url.strip().rstrip("/")


def extract_domain(url: str) -> str:
    """Extract clean domain name without www or mobile prefixes."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        elif netloc.startswith("m."):
            netloc = netloc[2:]
        return netloc
    except Exception:
        return ""


def parse_search_candidates(search_output: str) -> list:
    """
    Parse structured candidate sources from web_search tool output.
    """
    candidates = []
    blocks = re.split(r"(?=SOURCE\s+\d+)", search_output)
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith("SOURCE"):
            continue

        def field(name, text=block):
            lookahead = "|".join(re.escape(f) for f in _SEARCH_FIELDS if f != name)
            match = re.search(
                rf"{name}:\s*\n*(.*?)(?=\n\s*(?:{lookahead})|\Z)",
                text,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
            match = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE)
            return match.group(1).strip() if match else ""

        url = field("URL")
        if not url:
            continue

        title = field("Title")
        domain = field("Domain") or extract_domain(url)
        snippet = field("Search Snippet")
        published = field("Published Date")
        found_via = field("Found Via")
        tier_line = field("Source Tier")
        signals = field("Match Signals")
        retrieval_class = field("Retrieval Class")
        content_kind_line = field("Content Kind")

        content_kind = "snippet"
        if content_kind_line.lower().startswith("full_text"):
            content_kind = "full_text"

        try:
            score = float(field("Search Score") or 0)
        except ValueError:
            score = 0.0

        try:
            reliability = float(field("Reliability") or 0)
        except ValueError:
            reliability = 0.0

        candidates.append({
            "url": url,
            "title": title,
            "domain": domain,
            "score": score,
            "snippet": snippet,
            "content_kind": content_kind,
            "published_date": "" if published.startswith("Not stated") else published,
            "found_via": [
                label.strip()
                for label in found_via.split(",")
                if label.strip() and label.strip() != "n/a"
            ],
            "source_tier": tier_line.split("(")[0].strip() or "unknown",
            "retrieval_class": retrieval_class.strip() or "context",
            "match_signals": signals.strip(),
            "search_reliability": reliability,
        })

    return candidates


def merge_syndicated_sources(sources, threshold=0.62):
    """
    Collapse sources that are copies of one another.

    Two candidates carrying the same story (wire copy, mirrored article) are one
    voice. The highest-priority source survives; the copies are recorded on it
    so the evidence log still shows what was found.
    """
    ordered = sorted(
        sources,
        key=lambda s: s.get("retrieval_rank", 999),
    )

    kept = []
    for candidate in ordered:
        body = candidate.get("content", "")
        duplicate_of = None

        for existing in kept:
            if not body or not existing.get("content"):
                continue
            if is_near_duplicate(body, existing["content"], threshold):
                duplicate_of = existing["url"]
                break

        if duplicate_of:
            candidate["merged_into"] = duplicate_of
            existing = next(s for s in kept if s["url"] == duplicate_of)
            existing.setdefault("duplicates", []).append(candidate["url"])
            continue

        kept.append(candidate)

    return kept


def run_research_pipeline(claim: str) -> dict:

    state = {}

    # ============================================================
    # STEP 1 - RESEARCH
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 1 - RESEARCH")
    print("=" * 60)

    state["claim"] = claim

    print("\nClaim:")
    print(claim)

    state["search_results"] = web_search.invoke(claim)

    print("\nSearch Results:\n")
    print(state["search_results"])


    # ============================================================
    # STEP 2 - CANDIDATE SELECTION & EVIDENCE COLLECTION
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 2 - COLLECTING EVIDENCE")
    print("=" * 60)

    # Parse candidate sources
    candidates = parse_search_candidates(state["search_results"])

    # Fallback to URL regex if structured blocks could not be parsed
    if not candidates:
        raw_urls = re.findall(r"URL:\s*(https?://\S+)", state["search_results"])
        for u in raw_urls:
            candidates.append({
                "url": u,
                "title": "",
                "domain": extract_domain(u),
                "score": 0.5,
                "snippet": "",
                "found_via": [],
                "source_tier": "unknown",
                "retrieval_class": "context",
                "match_signals": "",
                "published_date": "",
            })

    # Deduplicate candidate sources by normalized URL
    seen_urls = set()
    deduped_candidates = []
    for rank, cand in enumerate(candidates, 1):
        norm = normalize_url(cand["url"])
        if norm in seen_urls:
            continue
        seen_urls.add(norm)
        cand["retrieval_rank"] = rank
        # Primary retrieval classes first, then search rank.
        cand["_priority"] = (
            0 if cand.get("retrieval_class") in PRIMARY_RETRIEVAL_CLASSES else 1,
            rank,
        )
        deduped_candidates.append(cand)

    deduped_candidates.sort(key=lambda c: c["_priority"])
    scrape_queue = deduped_candidates[:MAX_SCRAPE_ATTEMPTS]

    source_data = []
    evidence_package_blocks = []
    active_source_id = 0

    # ------------------------------------------------------------
    # Process candidate sources
    # ------------------------------------------------------------

    for candidate in scrape_queue:
        url = candidate["url"]
        domain = candidate["domain"] or extract_domain(url)
        title = candidate["title"]
        snippet = candidate["snippet"]
        retrieval_class = candidate.get("retrieval_class", "context")
        content_kind = candidate.get("content_kind", "snippet")

        print(f"\nEvaluating Candidate Source (rank {candidate['retrieval_rank']})...")
        print(f"URL: {url}")
        print(f"Domain: {domain}")
        print(f"Retrieval Class: {retrieval_class}")
        print(f"Retrieved Text: {content_kind} ({len(snippet)} chars)")

        content = scrape_url.invoke(url)

        # Determine scrape success
        is_scrape_success = bool(
            content
            and not content.startswith("SCRAPE_FAILED:")
            and not content.startswith("Could not scrape URL:")
            and len(content.strip()) >= 180
        )

        if is_scrape_success:
            active_source_id += 1
            print("Scrape Status: SUCCESS")
            print("Evidence Type: SCRAPED_ARTICLE (Primary Evidence)")

            primary_text = content[:12000]

            source_data.append({
                "url": url,
                "domain": domain,
                "title": title,
                "content": primary_text,
                "evidence_type": "scraped_article",
                "scrape_status": "SUCCESS",
                "retrieval_class": retrieval_class,
                "found_via": candidate.get("found_via", []),
                "published_date": candidate.get("published_date", ""),
                "independence_group": registrable_domain(url) or domain,
            })

            evidence_package_blocks.append(
                f"SOURCE {active_source_id}\n"
                f"EVIDENCE TYPE: SCRAPED ARTICLE (PRIMARY EVIDENCE)\n"
                f"RETRIEVAL CLASS: {retrieval_class}\n"
                f"URL: {url}\n"
                f"DOMAIN: {domain}\n"
                f"TITLE: {title}\n"
                f"EVIDENCE:\n{primary_text}"
            )

        else:
            # Scraping failed. The search provider may still have returned the
            # page's article text, which is real reporting and may corroborate a
            # claim. A short snippet may not.
            fail_reason = content.strip() if content else "Empty response"
            print(f"Scrape Status: FAILED ({fail_reason})")

            has_full_text = (
                content_kind == "full_text"
                and len(snippet.strip()) >= MIN_EXTRACT_CHARS
            )

            if has_full_text:
                active_source_id += 1
                print(
                    "Evidence Type: SEARCH_EXTRACT "
                    "(Provider-Extracted Article Text)"
                )

                fallback_text = (
                    f"TITLE: {title}\nPUBLICATION DATE: "
                    f"{candidate.get('published_date') or 'Not stated'}\n"
                    f"ARTICLE TEXT:\n{snippet}"
                )

                source_data.append({
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "content": fallback_text,
                    "evidence_type": "search_extract",
                    "scrape_status": f"FAILED ({fail_reason})",
                    "retrieval_class": retrieval_class,
                    "found_via": candidate.get("found_via", []),
                    "published_date": candidate.get("published_date", ""),
                    "independence_group": registrable_domain(url) or domain,
                })

                evidence_package_blocks.append(
                    f"SOURCE {active_source_id}\n"
                    f"EVIDENCE TYPE: SEARCH_EXTRACT (PROVIDER-EXTRACTED "
                    f"ARTICLE TEXT)\n"
                    f"RETRIEVAL CLASS: {retrieval_class}\n"
                    f"URL: {url}\n"
                    f"DOMAIN: {domain}\n"
                    f"TITLE: {title}\n"
                    f"ARTICLE TEXT:\n{snippet}"
                )

            elif snippet and len(snippet.strip()) >= 50:
                active_source_id += 1
                print("Evidence Type: SEARCH_SNIPPET_FALLBACK (Using Tavily Snippet)")

                fallback_text = f"TITLE: {title}\nSNIPPET: {snippet}" if title else snippet

                source_data.append({
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "content": fallback_text,
                    "evidence_type": "search_snippet",
                    "scrape_status": f"FAILED ({fail_reason})",
                    "retrieval_class": retrieval_class,
                    "found_via": candidate.get("found_via", []),
                    "published_date": candidate.get("published_date", ""),
                    "independence_group": registrable_domain(url) or domain,
                })

                evidence_package_blocks.append(
                    f"SOURCE {active_source_id}\n"
                    f"EVIDENCE TYPE: SEARCH_SNIPPET_FALLBACK (WEAKER FALLBACK "
                    f"EVIDENCE)\n"
                    f"RETRIEVAL CLASS: {retrieval_class}\n"
                    f"URL: {url}\n"
                    f"DOMAIN: {domain}\n"
                    f"TITLE: {title}\n"
                    f"FALLBACK SNIPPET:\n{snippet}"
                )
            else:
                print("No usable fallback text available. Candidate discarded.")

    # ------------------------------------------------------------
    # Syndicated / mirrored copies are one voice, not several
    # ------------------------------------------------------------
    before_merge = len(source_data)
    source_data = merge_syndicated_sources(source_data)
    if before_merge != len(source_data):
        print(
            f"\nMerged {before_merge - len(source_data)} duplicated/syndicated "
            f"source(s); they count as one voice."
        )

    state["scraped_content"] = (
        "\n\n--------------------\n\n".join(
            evidence_package_blocks
        )
    )

    state["source_data"] = source_data

    print(
        "\nEvidence Collection Complete."
    )
    print(
        f"\nSources Prepared for Analysis: {len(source_data)}"
    )

    for i, source in enumerate(source_data, 1):
        ev_type = (
            "SCRAPED_ARTICLE"
            if source["evidence_type"] == "scraped_article"
            else "SEARCH_SNIPPET_FALLBACK"
        )
        print(f"{i}. [{ev_type}] {source['url']} (class: {source.get('retrieval_class')})")


    # ============================================================
    # STEP 3 - EVIDENCE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 3 - ANALYZING EVIDENCE")
    print("=" * 60)

    state["evidence_analysis"] = analyze_evidence(
        claim,
        source_data
    )

    # Detailed evidence layer logging
    print("\n" + "=" * 60)
    print("EVIDENCE LAYER SUMMARY")
    print("=" * 60)

    for i, src in enumerate(source_data, 1):
        print(f"\nSOURCE {i}")
        print(f"Domain: {src['domain'] or 'N/A'}")
        print(f"Scrape Status: {src['scrape_status']}")
        ev_label = {
            "scraped_article": "SCRAPED_ARTICLE",
            "search_extract": "SEARCH_EXTRACT",
        }.get(src["evidence_type"], "SEARCH_SNIPPET_FALLBACK")
        print(f"Evidence Type: {ev_label}")

        src_results = [
            r for r in state["evidence_analysis"]
            if r.get("source_id") == i
        ]

        if src_results:
            best_r = src_results[0]
            print(f"Similarity: {best_r['similarity']}")
            print(f"Date Status: {best_r.get('date_status', 'unknown')}")
            print(f"Event Match: {best_r.get('event_match', 'weak')}")
            print(f"NLI: {best_r['nli_label']} ({best_r.get('nli_status')})")
            print(f"Evidence Class: {best_r['evidence_class']}")
            print(f"Evidence Score: {best_r['evidence_score']}")
        else:
            print("No evidence chunk passed the pre-filters for this source.")


    print(
        "\nTop Evidence Results:"
    )

    for i, result in enumerate(
        state["evidence_analysis"][:5],
        1
    ):
        print(
            f"\nEvidence {i}"
        )
        print(
            "Source:",
            result["source_id"]
        )
        print(
            "Evidence Type:",
            result.get("evidence_type", "scraped_article")
        )
        print(
            "URL:",
            result["url"]
        )
        print(
            "Similarity:",
            result["similarity"]
        )
        print(
            "NLI:",
            result["nli_label"],
            result["nli_score"],
            f"({result.get('nli_status')})"
        )
        print(
            "Date Status:",
            result.get("date_status", "unknown")
        )
        print(
            "Event Match:",
            result.get("event_match", "weak")
        )
        print(
            "Source Tier:",
            result.get("source_tier", "unknown"),
            result["source_reliability"]
        )
        print(
            "Evidence Class:",
            result.get("evidence_class", "NON_EVIDENCE")
        )
        print(
            "Evidence Score:",
            result["evidence_score"]
        )


    # ============================================================
    # STEP 4 - SOURCE AGREEMENT + EVIDENCE FUSION
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 4 - SOURCE AGREEMENT & EVIDENCE FUSION")
    print("=" * 60)

    state["source_agreement"] = (
        calculate_source_agreement(
            state["evidence_analysis"]
        )
    )

    state["fusion_score"] = (
        calculate_fusion_score(
            state["evidence_analysis"],
            state["source_agreement"]
        )
    )

    print(
        "\nSource Agreement:"
    )
    print(
        f"  Sources analysed      : {state['source_agreement']['total_sources']}"
    )
    print(
        f"  Independence groups   : {state['source_agreement']['independence_groups']}"
    )
    print(
        f"  Independent supporting: {state['source_agreement']['independent_supporting']}"
    )
    print(
        f"  Independent contra    : {state['source_agreement']['independent_contradicting']}"
    )
    print(
        f"  Evidence sufficiency  : {state['source_agreement']['evidence_sufficiency']}"
    )
    print(
        f"  Agreement score       : {state['source_agreement']['agreement_score']}"
    )

    print(
        "\nEvidence Fusion Score:",
        state["fusion_score"]
    )


    # ============================================================
    # STEP 5 - SHARED EVIDENCE BRIEF
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 5 - BUILDING EVIDENCE BRIEF")
    print("=" * 60)

    state["evidence_brief"] = build_evidence_brief(
        claim,
        state["source_data"],
        state["evidence_analysis"],
        state["source_agreement"],
        state["fusion_score"],
    )

    print(f"\nBrief length: {len(state['evidence_brief'])} characters")
    print("\n" + state["evidence_brief"][:2500])


    # ============================================================
    # STEP 6 - VERIFICATION AGENT
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 6 - VERIFICATION")
    print("=" * 60)

    state["verification"] = (
        verification_chain.invoke({
            "claim":
                claim,
            "research":
                "Multiple targeted searches were run (exact wording, exact date, "
                "location, alternative wording, fact-check and official angles). "
                "Scraped articles are primary evidence. Search snippets are "
                "fallback evidence only and cannot confirm or refute a claim.",
            "evidence":
                state["evidence_brief"]
        })
    )

    print(
        "\nVerification Result:\n"
    )
    print(
        state["verification"]
    )


    # ============================================================
    # STEP 7 - CRITIC / REVIEW
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 7 - CRITIC REVIEW")
    print("=" * 60)

    # The critic sees exactly the same evidence the verifier saw, plus the
    # verifier's own reasoning. It never gets a shorter or different excerpt.
    state["feedback"] = (
        critic_chain.invoke({
            "claim":
                claim,
            "verification":
                state["verification"],
            "evidence":
                state["evidence_brief"]
        })
    )

    print(
        "\nCritic Review:\n"
    )
    print(
        state["feedback"]
    )


    # ============================================================
    # STEP 8 - FINAL VERDICT
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 8 - FINAL VERDICT")
    print("=" * 60)


    state["final_verdict"] = (
        generate_final_verdict(

            state["evidence_analysis"],

            state["source_agreement"],

            state["fusion_score"],

            state["verification"],

            state["feedback"]
        )
    )


    # ============================================================
    # FINAL RESULT
    # ============================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "TRUTHLENS AI - FINAL ASSESSMENT"
    )

    print(
        "=" * 60
    )


    print(
        "\nCLAIM:"
    )

    print(
        claim
    )


    print(
        "\nVERDICT:"
    )

    print(
        state["final_verdict"].get(
            "verdict",
            "UNVERIFIED"
        )
    )


    print(
        "\nCONFIDENCE:"
    )

    print(
        state["final_verdict"].get(
            "confidence",
            0
        ),
        "%"
    )


    print(
        "\nEVIDENCE SUFFICIENCY:"
    )

    print(
        state["final_verdict"].get(
            "evidence_sufficiency",
            "none"
        )
    )


    print(
        "\nINDEPENDENT SUPPORTING SOURCES:"
    )

    print(
        state["final_verdict"].get(
            "independent_supporting",
            0
        )
    )


    print(
        "\nINDEPENDENT CONTRADICTING SOURCES:"
    )

    print(
        state["final_verdict"].get(
            "independent_contradicting",
            0
        )
    )


    print(
        "\nCONTRADICTION SCORE:"
    )

    print(
        state["final_verdict"].get(
            "contradiction_score",
            0
        )
    )


    print(
        "\nENTAILMENT SCORE:"
    )

    print(
        state["final_verdict"].get(
            "entailment_score",
            0
        )
    )


    print(
        "\nSOURCE AGREEMENT:"
    )

    print(
        state["final_verdict"].get(
            "source_agreement",
            0
        )
    )


    print(
        "\nFUSION SCORE:"
    )

    print(
        state["final_verdict"].get(
            "fusion_score",
            0
        )
    )


    # ============================================================
    # EVIDENCE SUMMARY
    # ============================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "EVIDENCE SUMMARY"
    )

    print(
        "-" * 60
    )


    support_chunks = [
        result for result in state["evidence_analysis"]
        if result.get("evidence_class") == "SUPPORT"
    ]

    contradiction_chunks = [
        result for result in state["evidence_analysis"]
        if result.get("evidence_class") == "CONTRADICTION"
    ]

    print(
        "\nSupporting Chunks:",
        len(support_chunks)
    )

    print(
        "Contradicting Chunks:",
        len(contradiction_chunks)
    )


    # ============================================================
    # SUPPORTING EVIDENCE
    # ============================================================

    if support_chunks:

        print(
            "\nTOP SUPPORTING EVIDENCE:"
        )

        for result in support_chunks[:3]:

            print(
                "\nSource:",
                result["url"]
            )

            print(
                "\nEvidence:"
            )

            print(
                result["chunk"][:500]
            )


    # ============================================================
    # CONTRADICTING EVIDENCE
    # ============================================================

    if contradiction_chunks:

        print(
            "\nTOP CONTRADICTING EVIDENCE:"
        )

        for result in contradiction_chunks[:3]:

            print(
                "\nSource:",
                result["url"]
            )

            print(
                "\nEvidence:"
            )

            print(
                result["chunk"][:500]
            )


    # ============================================================
    # RETURN COMPLETE PIPELINE STATE
    # ============================================================

    return state


# ================================================================
# TEST PIPELINE
# ================================================================

if __name__ == "__main__":

    claim = input(
        "\nEnter a news claim: "
    )

    result = run_research_pipeline(
        claim
    )
