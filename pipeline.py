import re
import sys
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from agents import (
    verification_chain,
    critic_chain,
    invoke_agent,
)
from research_report import redact_sensitive

from claim_profile import build_claim_profile
from evidence_fusion import (
    calculate_source_agreement,
    calculate_fusion_score,
    reconcile_evidence,
)

from evidence_analyzer import analyze_evidence
from evidence_brief import build_evidence_brief
from source_reliability import (
    get_source_profile,
    is_near_duplicate,
    registrable_domain,
)
from tools import search_with_diagnostics, scrape_url, scrape_url_detailed, query_google_fact_check
from final_verdict import generate_final_verdict, deterministic_gate
from second_opinion import build_second_opinion
from independent_llm_analysis import build_independent_llm_analysis
from research_report import redact_sensitive


def _scientific_fallback_url(url: str) -> str | None:
    """Return a verified HTTPS alternative for this specific known endpoint."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if (
        host == "mail.iapws.org"
        and parsed.path.rstrip("/") == "/faq1/boil.html"
    ):
        return "https://iapws.org/faqs/faq1"

    return None

def _force_utf8_stdout() -> None:
    """
    Make console output UTF-8 safe.

    The pipeline prints progress markers such as the check mark and arrow
    glyphs. On a Windows console whose code page is cp1252, printing them
    raises UnicodeEncodeError and aborts the whole research run. Replacing
    unencodable characters keeps the diagnostics readable and the run alive.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


_force_utf8_stdout()


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
    "Title", "URL", "Domain", "Published Date", "Search Query", "Query Intent",
    "Found Via", "Source Tier",
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
            "query": "" if field("Search Query").lower() == "n/a" else field("Search Query"),
            "query_intent": field("Query Intent"),
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


def _diagnose_unverified_reason(state: dict) -> str:
    """
    Return a concise, data-backed explanation of why the verdict is UNVERIFIED.

    The reason is derived from actual execution evidence, not guesswork:
    - agent failures (Groq 429, timeout, etc.)
    - NLI model unavailable
    - no usable sources
    - only snippets / non-primary sources
    - primary sources found but nothing classified as support/contradiction
    - too few independent supporting sources
    - conflicting reliable evidence
    """
    agent_errors = state.get("agent_errors", []) or []
    agreement = state.get("source_agreement", {}) or {}
    analysis = state.get("evidence_analysis", []) or []

    supporting = sum(
        1 for r in analysis if r.get("evidence_class") == "SUPPORT"
    )
    contradicting = sum(
        1 for r in analysis if r.get("evidence_class") == "CONTRADICTION"
    )
    sufficiency = agreement.get("evidence_sufficiency", "none")
    independent_supporting = agreement.get("independent_supporting", 0) or 0
    independent_contradicting = agreement.get("independent_contradicting", 0) or 0

    # Agent failures take priority — the pipeline couldn't complete.
    for err in agent_errors:
        stage = err.get("stage", "agent")
        error_text = redact_sensitive(str(err.get("error", "")))
        if "rate_limit" in error_text.lower() or "429" in error_text:
            return f"Groq service rate-limited ({stage}); retry budget exhausted."
        if "auth" in error_text.lower() or "401" in error_text or "403" in error_text:
            return f"Groq authentication failure ({stage}); API key invalid or revoked."
        if "timeout" in error_text.lower() or "503" in error_text or "502" in error_text:
            return f"Groq service unavailable ({stage}); transient provider error."
        return f"Agent failure ({stage}): {error_text[:120]}."

    # NLI model unavailable.
    for r in analysis:
        if r.get("nli_status") in ("unavailable", "error"):
            return "NLI model unavailable; cannot classify evidence passages."

    # No sources at all.
    total = agreement.get("total_sources", 0) or 0
    if total == 0:
        return "No usable sources found; search returned no candidates."

    # Only snippets or non-primary sources.
    has_primary_full = any(
        r.get("evidence_type") in ("scraped_article", "search_extract")
        and r.get("source_tier", "unknown") not in ("weak/blog", "user_generated")
        for r in analysis
    )
    if not has_primary_full:
        return "Only search snippets or non-primary sources found; no full article evidence."

    # Primary sources found but nothing classified as support/contradiction.
    if supporting == 0 and contradicting == 0:
        return (
            "Primary sources retrieved but no passage directly supports or "
            "refutes the claim; evidence is related context only."
        )

    # Too few independent supporting sources.
    if independent_supporting == 0 and independent_contradicting == 0:
        return "No independent publisher corroborates or refutes the claim."

    # Conflicting evidence.
    if independent_supporting > 0 and independent_contradicting > 0:
        return (
            "Independent sources conflict: some support and some contradict "
            "the claim; verdict cannot be settled."
        )

    # Insufficient independent support (single source).
    if independent_supporting == 1:
        return "Only one independent publisher supports the claim; corroboration required."

    return "Evidence insufficient to reach a settled verdict."


def run_research_pipeline(claim: str, extracted_text: Optional[str] = None) -> dict:

    state = {}

    # ============================================================
    # STEP 1 - RESEARCH
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 1 - RESEARCH")
    print("=" * 60)

    state["claim"] = claim
    state["claim_type"] = build_claim_profile(claim).get("claim_type", "other_factual_claim")

    print("\nClaim:")
    print(claim)
    print(f"Claim type: {state['claim_type']}")

    search_outcome = search_with_diagnostics(claim)
    state["search_results"] = search_outcome["output"]
    # Preserve the real retrieval plan (the actual queries issued) and any
    # provider-level failures so the API can report genuine research activity.
    state["search_plans"] = search_outcome.get("plans", [])
    state["search_errors"] = search_outcome.get("search_errors", [])

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
    scrape_failures = []

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

        
        scrape_result = scrape_url_detailed(url)
        content = scrape_result["content"]
        scrape_diag = {
            "url": url,
            "domain": domain,
            "title": title,
            "scrape_status": scrape_result["scrape_status"],
            "extraction_method": scrape_result["extraction_method"],
            "char_count": scrape_result["char_count"],
            "http_status": scrape_result["http_status"],
            "retry_count": scrape_result["retry_count"],
            "error_category": scrape_result["error_category"],
        }

        if (
            not content
            or scrape_result["scrape_status"] == "failed"
        ):
            fallback_url = _scientific_fallback_url(url)

            if fallback_url:
                fallback_result = scrape_url_detailed(fallback_url)
                fallback_content = fallback_result["content"]

                if (
                    fallback_url
                    and fallback_content
                    and fallback_result["scrape_status"] == "success"
                    and len(fallback_content.strip()) >= 180
                ):
                    url = fallback_url
                    domain = extract_domain(url)
                    title = "FAQs About Water and Steam"
                    content = fallback_content
                    scrape_diag = {
                        "url": url,
                        "domain": domain,
                        "title": title,
                        "scrape_status": fallback_result["scrape_status"],
                        "extraction_method": fallback_result["extraction_method"],
                        "char_count": fallback_result["char_count"],
                        "http_status": fallback_result["http_status"],
                        "retry_count": fallback_result["retry_count"],
                        "error_category": fallback_result["error_category"],
                    }


        # Determine scrape success
        is_scrape_success = bool(
            content
            and scrape_result["scrape_status"] == "success"
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
                "search_query": candidate.get("query", ""),
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
                "search_query": candidate.get("query", ""),
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

                scrape_failures.append({
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "reason": fail_reason,
                    "fallback_used": "search_extract",
                })

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
                "search_query": candidate.get("query", ""),
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

                scrape_failures.append({
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "reason": fail_reason,
                    "fallback_used": "search_snippet",
                })
            else:
                print("No usable fallback text available. Candidate discarded.")
                # A discarded candidate is still real research activity: the URL
                # was found and the retrieval was attempted. Record it so the
                # report can show what failed instead of silently dropping it.
                scrape_failures.append({
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "reason": fail_reason,
                    "fallback_used": False,
                })

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
    state["scrape_failures"] = scrape_failures

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

    # ============================================================
    # STEP 3.25 - GOOGLE FACT CHECK API
    # ============================================================
    #
    # Calls the Google Fact Check Tools API as a supplementary evidence
    # source. Results are normalized into the same evidence format and
    # merged into evidence_analysis before reconciliation, so they go
    # through the same dedup, source-independence, and verdict gates as
    # web-scraped evidence. The API is supplementary: it never overrides
    # the deterministic verdict gate.
    #
    state["google_fact_check"] = None
    state["google_fact_check_error"] = None

    try:
        gc_result = query_google_fact_check(claim)
        state["google_fact_check"] = gc_result

        if gc_result.get("status") == "success" and gc_result.get("claims"):
            print(f"\nGoogle Fact Check: {len(gc_result['claims'])} claim(s) found")

            google_evidence = []
            for gc_claim in gc_result["claims"]:
                for review in gc_claim.get("claimReview", []):
                    publisher = review.get("publisher", {})
                    publisher_name = publisher.get("name", "Unknown")
                    review_url = review.get("url", "")
                    rating = (review.get("textualRating") or "").strip()

                    # Map fact-check rating to evidence class
                    rating_lower = rating.lower()
                    if rating_lower in ("true", "mostly true"):
                        ev_class = "SUPPORT"
                        nli_label = "entailment"
                        nli_score = 0.90
                    elif rating_lower in ("false", "mostly false"):
                        ev_class = "CONTRADICTION"
                        nli_label = "contradiction"
                        nli_score = 0.90
                    elif rating_lower in ("mixture", "misleading", "half true", "half-true"):
                        ev_class = "RELATED_CONTEXT"
                        nli_label = "neutral"
                        nli_score = 0.50
                    else:
                        ev_class = "RELATED_CONTEXT"
                        nli_label = "neutral"
                        nli_score = 0.50

                    # Get source profile for independence grouping
                    source_profile = get_source_profile(review_url, "scraped_article")

                    google_evidence.append({
                        "source_id": f"gfc-{len(state['evidence_analysis']) + len(google_evidence) + 1}",
                        "url": review_url,
                        "domain": extract_domain(review_url),
                        "title": publisher_name,
                        "content": (
                            f"CLAIM: {gc_claim.get('text', '')}\n"
                            f"PUBLISHER: {publisher_name}\n"
                            f"RATING: {rating}\n"
                            f"REVIEW: {review.get('textualRating', '')}"
                        ),
                        "evidence_type": "fact_check",
                        "scrape_status": "SUCCESS",
                        "retrieval_class": "factcheck",
                        "found_via": ["google_fact_check"],
                        "search_query": claim,
                        "published_date": gc_claim.get("claimDate", ""),
                        "independence_group": registrable_domain(review_url) or extract_domain(review_url),
                        "similarity": 0.85,
                        "raw_nli_label": nli_label,
                        "raw_nli_score": nli_score,
                        "nli_label": nli_label,
                        "nli_score": nli_score,
                        "nli_status": "ok",
                        "source_reliability": source_profile.get("reliability", 0.5),
                        "source_tier": source_profile.get("tier", "unknown"),
                        "evidence_score": nli_score * source_profile.get("reliability", 0.5),
                        "evidence_class": ev_class,
                        "date_status": "match",
                        "event_match": "strong",
                        "location_match": True,
                        "detail_coverage": 0.7,
                        "debunking": False,
                        "passage_kind": "paragraph",
                        "claim_type": state.get("claim_type", "other_factual_claim"),
                        "event_date_conflict": False,
                        "event_date_mismatch": False,
                        "is_primary_source": source_profile.get("is_primary_source", False),
                        "claim_has_location": False,
                        "chunk": (
                            f"FACT-CHECK: {gc_claim.get('text', '')}\n"
                            f"Publisher: {publisher_name}\n"
                            f"Rating: {rating}\n"
                            f"Review: {review.get('textualRating', '')}"
                        ),
                    })

            state["evidence_analysis"].extend(google_evidence)
            print(f"  Added {len(google_evidence)} Google Fact Check evidence item(s)")
        else:
            err_msg = gc_result.get("error_message", "No results returned")
            print(f"\nGoogle Fact Check: {err_msg}")
            state["google_fact_check_error"] = err_msg

    except Exception as e:
        state["google_fact_check"] = {
            "status": "error",
            "error_category": "request_error",
            "error_message": str(e)[:200],
            "claims": [],
        }
        state["google_fact_check_error"] = str(e)[:200]
        print(f"  [Google Fact Check] Error: {str(e)[:160]}")

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
    # STEP 3.5 - EVIDENCE CONSISTENCY CHECK
    # ============================================================
    #
    # Runs after evidence classification and before source agreement fusion.
    # Deduplicates chunks at chunk/article/publisher levels, re-evaluates
    # RELATED_CONTEXT passages, and generates diagnostics. Replaces
    # evidence_analysis with a reconciled list (one record per source) so
    # that multiple chunks from the same article or publisher cannot inflate
    # corroboration counts.

    print("\n" + "=" * 60)
    print("STEP 3.5 - EVIDENCE CONSISTENCY CHECK")
    print("=" * 60)

    state["evidence_analysis"], state["evidence_diagnostics"], state["publisher_decisions"] = (
        reconcile_evidence(state["evidence_analysis"])
    )

    diag = state["evidence_diagnostics"]
    print(
        f"  Chunks received     : {diag['total_chunks_received']}"
    )
    print(
        f"  Unique articles     : {diag['unique_articles']}"
    )
    print(
        f"  Duplicate chunks removed : {diag['duplicate_chunks_removed']}"
    )
    print(
        f"  Unique publishers   : {diag['unique_publishers']}"
    )
    print(
        f"  Classification changes: {len(diag['classification_changes'])}"
    )
    for change in diag["classification_changes"]:
        print(
            f"    - {change['url']}: {change['from']} -> {change['to']}"
        )
    for excluded in diag["related_context_excluded"]:
        print(
            f"    - Excluded RELATED_CONTEXT: {excluded['url']} "
            f"(score={excluded['evidence_score']}, "
            f"event_match={excluded['event_match']}, "
            f"nli={excluded['nli_label']} {excluded['nli_score']})"
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
        evidence_diagnostics=state.get("evidence_diagnostics"),
    )

    print(f"\nBrief length: {len(state['evidence_brief'])} characters")
    print("\n" + state["evidence_brief"][:2500])


    # ============================================================
    # STEP 6 - VERIFICATION AGENT
    # ============================================================

    # A provider failure in an agent stage must not discard the retrieval,
    # scraping and NLI work that has already been completed. The stage failure
    # is recorded, the deterministic evidence decision stays authoritative, and
    # the result is reported as partial rather than as a finished verification.
    agent_errors = []
    state["agent_errors"] = agent_errors

    print("\n" + "=" * 60)
    print("STEP 6 - VERIFICATION")
    print("=" * 60)

    try:
        state["verification"] = invoke_agent(
            verification_chain,
            {
                "claim":
                    claim,
                "claim_type":
                    state["claim_type"],
                "research":
                    "Multiple targeted searches were run (exact wording, exact date, "
                    "location, alternative wording, fact-check and official angles). "
                    "Scraped articles are primary evidence. Search snippets are "
                    "fallback evidence only and cannot confirm or refute a claim.",
                "evidence":
                    state["evidence_brief"]
            },
            stage="verification_agent",
        )
        state["verification_status"] = "completed"
    except Exception as agent_error:
        state["verification"] = ""
        state["verification_status"] = "failed"
        agent_errors.append({
            "stage": "verification_agent",
            "error": redact_sensitive(str(agent_error)),
        })
        print(f"  [verification_agent] unavailable: {str(agent_error)[:160]}")

    print(
        "\nVerification Result:\n"
    )
    print(
        state["verification"] or "(verification agent unavailable)"
    )


    # ============================================================
    # STEP 7 - CRITIC / REVIEW
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 7 - CRITIC REVIEW")
    print("=" * 60)

    # The critic checks the verifier's verdict against the deterministic
    # evidence floor. It receives a condensed package (claim, verifier
    # verdict, permitted verdicts and reason, independent-source counts,
    # top passages) rather than the full brief, to reduce input tokens.
    # The critic call is never skipped: the floor already bounds the
    # permitted verdicts, but the critic's review is retained as audit.
    gate = deterministic_gate(state["source_agreement"], state["evidence_analysis"])

    top_passages = sorted(
        state["evidence_analysis"],
        key=lambda r: r.get("evidence_score", 0),
        reverse=True,
    )[:3]

    passage_lines = []
    for p in top_passages:
        passage_lines.append(
            f"- [{p.get('evidence_class')}] {p.get('url')}"
            f" | nli={p.get('nli_label')} {p.get('nli_score')}"
            f" | similarity={p.get('similarity')}"
            f" | coverage={p.get('detail_coverage')}"
            f" | {str(p.get('chunk', ''))[:250]}"
        )

    condensed_evidence = "\n".join([
        "=== CLAIM ===",
        claim,
        "",
        "=== VERIFIER VERDICT ===",
        state["verification"] or "(verifier unavailable)",
        "",
        "=== DETERMINISTIC EVIDENCE FLOOR ===",
        f"Permitted verdicts: {', '.join(sorted(gate['permitted_verdicts']))}",
        f"Reason: {gate['permitted_reason']}",
        f"Independent supporting: {gate['independent_supporting']}",
        f"Independent contradicting: {gate['independent_contradicting']}",
        "",
        "=== TOP EVIDENCE PASSAGES ===",
        "\n".join(passage_lines) if passage_lines else "(none)",
    ])

    try:
        state["feedback"] = invoke_agent(
            critic_chain,
            {
                "claim":
                    claim,
                "claim_type":
                    state["claim_type"],
                "verification":
                    state["verification"],
                "evidence":
                    condensed_evidence
            },
            stage="critic_agent",
        )
        state["feedback_status"] = "completed"
    except Exception as agent_error:
        state["feedback"] = ""
        state["feedback_status"] = "failed"
        agent_errors.append({
            "stage": "critic_agent",
            "error": redact_sensitive(str(agent_error)),
        })
        print(f"  [critic_agent] unavailable: {str(agent_error)[:160]}")

    print(
        "\nCritic Review:\n"
    )
    print(
        state["feedback"] or "(critic agent unavailable)"
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

            state["feedback"],

            state["google_fact_check"],
        )
    )

    # ============================================================
    # STEP 8.5 - LLM SECOND OPINION (independent review)
    # ============================================================
    #
    # A separate LLM reviews the claim and the evidence package already
    # collected. It never re-runs search/scraping and never overrides the
    # deterministic evidence gate. Its verdict is reported alongside the
    # system verdict for comparison only.
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 8.5 - LLM SECOND OPINION")
    print("=" * 60)

    final_verdict = state.get("final_verdict", {}) or {}
    permitted = final_verdict.get("permitted_verdicts") or []
    if isinstance(permitted, str):
        permitted = [permitted]

    state["llm_second_opinion"] = build_second_opinion(
        claim=claim,
        evidence_analysis=state["evidence_analysis"],
        source_agreement=state["source_agreement"],
        existing_verdict=final_verdict.get("verdict"),
        permitted_verdicts=permitted,
    )

    print(
        "\nLLM Second Opinion:\n"
    )
    print(
        state["llm_second_opinion"].get("comparison", {}).get(
            "comparison_summary",
            "(second opinion unavailable)",
        )
    )

    # ============================================================
    # STEP 8.6 - INDEPENDENT LLM ANALYSIS (direct LLM call)
    # ============================================================
    # A separate direct LLM analysis of the original claim using the LLM's
    # own knowledge. It never re-runs search/scraping and never overrides the
    # deterministic evidence gate. Reported separately from the system verdict
    # and from the LLM second opinion.
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 8.6 - INDEPENDENT LLM ANALYSIS")
    print("=" * 60)

    state["independent_llm_analysis"] = build_independent_llm_analysis(
        claim=claim,
        existing_verdict=final_verdict.get("verdict"),
        extracted_text=extracted_text or claim,
    )

    print(
        "\nIndependent LLM Analysis:\n"
    )
    print(
        state["independent_llm_analysis"].get("comparison", {}).get(
            "comparison_summary",
            "(independent analysis unavailable)",
        )
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

    # Structured diagnostic explaining the verdict reason (data-backed).
    state["diagnostic_summary"] = _diagnose_unverified_reason(state)

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
