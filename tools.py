from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
import re
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode, urlunparse
from dotenv import load_dotenv

from claim_profile import (
    build_claim_profile,
    extract_dates,
    extract_years,
    strip_publication_boilerplate,
)
from query_planner import plan_queries, _quote
from source_reliability import (
    classify_tier,
    get_source_profile,
    is_near_duplicate,
    TIER_UGC,
    TIER_AGGREGATOR,
    TIER_WEAK,
    TIER_OFFICIAL,
)

load_dotenv()


# ============================================================
# TAVILY
# ============================================================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)



# ============================================================
# DATE-WINDOWED NEWS SEARCH
# ============================================================

def _date_window(claim_date, lead_days, follow_days):
    """
    ISO bounds for a news search restricted to the claimed event window.

    A news index is dominated by what happened today. Without a window, a query
    about a July 2024 event returns September 2026 articles about the same city,
    which are exactly the "right topic, wrong event" results that produce weak,
    misleading evidence.
    """
    if not claim_date:
        return None, None

    try:
        anchor = datetime(claim_date["year"], claim_date["month"], claim_date["day"])
        start = anchor - timedelta(days=lead_days)
        end = anchor + timedelta(days=follow_days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    except Exception:
        return None, None


# Rolling news pages cover many days, so a single day inside them is not
# evidence about one event. They are ranked below single-event reports.
_LIVEBLOG_MARKERS = (
    "liveblog", "live-blog", "live-updates", "live_updates", "newsblog",
    "news-blog", "livenews", "as-it-happened", "latest-news",
)


def date_variants_for(claim_date):
    """Textual spellings of a parsed claim date."""
    from claim_profile import date_variants

    return date_variants(claim_date)


def is_rolling_page(url, title=""):
    lowered = f"{url or ''} {title or ''}".lower()
    return any(marker in lowered for marker in _LIVEBLOG_MARKERS)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Retrieval is deliberately wide; filtering is what keeps it honest.
MAX_QUERIES = 8
RESULTS_PER_QUERY = 8
MAX_CANDIDATES = 10

# Below this many strict candidates, the pool is widened: a verdict needs two
# independent sources, so a pool of three cannot produce one.
MIN_POOL_BEFORE_WIDENING = 6

# Widened pool size.
WIDENED_POOL_SIZE = 14

MIN_TAVILY_SCORE = 0.10

# Retrieved text this long or longer is real article text extracted by the
# search provider, not a search snippet. The distinction matters: extracted
# article text may corroborate a claim, a snippet may not.
MIN_FULL_TEXT_CHARS = 600

# Retrieval classes that can establish or refute a claim. A claim that gets
# fewer than two of these is retried with an exact-phrase search.
PRIMARY_RETRIEVAL_CLASSES = ("exact_date", "factcheck", "wrong_date")

# Near-duplicate (syndicated) copy detection threshold.
SYNDICATION_JACCARD = 0.62

EXCLUDED_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "threads.net",
    "quora.com",
]

_FACTCHECK_RE = re.compile(
    r"\b(fact[\s\-]?check|debunk|falsely\s+claimed|false\s+claim|"
    r"viral\s+claim|hoax|misleading|rumou?r|untrue|"
    r"no\s+such\s+incident|clarified\s+that\s+no|"
    r"did\s+not\s+happen|did\s+not\s+occur|"
    r"actually\s+took\s+place|actually\s+occurred|"
    r"old\s+video|old\s+photo|old\s+image|misinformation|"
    r"claim\s+is\s+false|claim\s+is\s+misleading|"
    r"fact\s+vs\s+fiction|truth\s+behind)\b",
    re.IGNORECASE,
)

_INCIDENT_MARKERS = [
    "broke out", "fire", "killed", "injured", "engulfed", "erupted",
    "occurred", "happened", "destroyed", "damaged", "blaze", "gutted",
    "accident", "incident", "collapse", "blast", "crash", "protest",
    "arrest", "struck", "victims", "casualties", "flood", "earthquake",
    "derailed", "attack", "resigned", "elected", "verdict", "verdicted",
]

_POLICY_MARKERS = [
    "safety regulations", "safety guidelines", "compliance rules",
    "safety norms", "building codes", "advisory guidelines",
    "safety manual", "regulations for", "guidelines issued",
]


# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    """Normalize URL by stripping tracking parameters, anchors, and trailing slashes."""
    try:
        parsed = urlparse(url)
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term",
            "utm_content", "ref", "fbclid", "gclid",
        }
        query_dict = parse_qs(parsed.query)
        cleaned_query = {
            k: v for k, v in query_dict.items()
            if k.lower() not in tracking_params
        }
        new_query = urlencode(cleaned_query, doseq=True)
        path = parsed.path.rstrip("/")
        return urlunparse((
            parsed.scheme.lower(), parsed.netloc.lower(), path,
            parsed.params, new_query, ""
        ))
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


def has_fact_check_signal(text):
    """True when text contains clear fact-check / debunking language."""
    return bool(_FACTCHECK_RE.search(text or ""))


# Index/listing pages mention the right words and the right date without ever
# reporting the claimed event. They must never enter the evidence pool.
_LISTING_MARKERS = (
    "click here", "| sr-no", "sr no", "bid document", "tender", "vacancy",
    "notification", "download the", "list of ", "table of contents",
    "s.no", "serial no", "global listings", "property listings",
    "all categories", "browse our", "site map", "search results",
    "read more stories", "more from", "sign up for our newsletter",
    # property / classifieds / job-board furniture
    "for sale", "buy now", "contact seller", "contact us to book",
    "price on request", "emi", "sq ft", "sq.ft", "sqm", "bhk", "possession",
    "carpet area", "plot area", "facing", "car parking", "lift out",
)

_LISTING_URL_MARKERS = (
    "/property", "/properties", "/real-estate", "/buy", "/rent",
    "/classified", "/directory", "/listing", "/plots", "/apartments",
    "/residential", "/commercial-property", "/jobs", "/job", "/career",
    "/category/", "/tag/", "/search?", "/archive", "/press-release-",
    "/index", "/sitemap", "showbid", "/tender", "/vacanc",
)


def is_listing_or_index_page(text, url=""):
    """
    Detect index, tender, directory, property and job-listing pages.

    These pages match on keywords and dates but contain no event reporting, so
    admitting them is the fastest way to manufacture false corroboration.
    """
    body = str(text or "")
    lowered = body.lower()
    url_lower = str(url or "").lower()

    distinct_dates = {
        (d["day"], d["month"], d["year"]) for d in extract_dates(body)
    }
    if len(distinct_dates) >= 5:
        return True

    if any(marker in url_lower for marker in _LISTING_URL_MARKERS):
        return True

    marker_hits = sum(1 for marker in _LISTING_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True

    return False


# ============================================================
# EVENT-DATE MATCHING
# ============================================================

def check_text_date_match(combined_text, claim_date):
    """
    Date signals for one candidate source.

    claim_date is a claim_profile date dict (or None).

    Returns:
        exact_date       claim's exact day/month/year appears in the text
        same_year        claim year appears in the text
        different_date   an explicit full date of another day/month/year appears
        claim_year_absent claim year never appears
    """
    if not claim_date:
        return {
            "exact_date": False,
            "same_year": False,
            "different_date": False,
            "claim_year_absent": False,
        }

    text = strip_publication_boilerplate(combined_text)

    claim_key = (claim_date["day"], claim_date["month"], claim_date["year"])
    claim_year = claim_date["year"]

    text_dates = extract_dates(text)
    text_keys = {(d["day"], d["month"], d["year"]) for d in text_dates}

    exact_date = claim_key in text_keys

    years = extract_years(text)
    same_year = claim_year in years
    claim_year_absent = bool(years) and claim_year not in years

    # A different full date only matters when it is the same year (a different
    # day of the claimed year) or when the claimed year is absent entirely.
    different_date = any(
        (key[2] == claim_year and key[:2] != claim_key[:2]) or
        (key[2] != claim_year and claim_year_absent)
        for key in text_keys
    )

    return {
        "exact_date": exact_date,
        "same_year": same_year,
        "different_date": different_date,
        "claim_year_absent": claim_year_absent,
    }


# ============================================================
# CANDIDATE RELEVANCE
# ============================================================

def _text_of(candidate):
    return " ".join(str(candidate.get(field) or "") for field in ("title", "content", "url"))


def _match_signals(candidate, profile):
    """Keyword / phrase / location hits of a candidate against the claim."""
    text = _text_of(candidate).lower()

    keywords = [k.lower() for k in profile["keywords"]]
    locations = [loc.lower() for loc in profile["locations"]]
    location_terms = [t.lower() for t in profile["location_terms"]]
    phrases = [p.lower() for p in profile["phrases"]]

    keyword_hits = [k for k in keywords if re.search(rf"\b{re.escape(k)}\b", text)]
    phrase_hits = [p for p in phrases if p in text]
    location_hits = [loc for loc in locations if loc in text]
    term_hits = [t for t in location_terms if re.search(rf"\b{re.escape(t)}\b", text)]

    keyword_ratio = (len(keyword_hits) / len(keywords)) if keywords else 0.0

    return {
        "keyword_hits": keyword_hits,
        "keyword_count": len(keyword_hits),
        "keyword_ratio": round(keyword_ratio, 4),
        "phrase_hits": phrase_hits,
        "location_hits": location_hits,
        "location_match": bool(location_hits),
        "term_match": bool(term_hits),
    }


def is_result_relevant(result, profile, date_signals=None, claim_date=None):
    """
    Strict layered candidate filter.

    A Tavily score alone can never promote an unrelated or wrong-year article:
    for dated claims an explicit different event date is a hard reject, and the
    claimed location must appear.
    """
    score = result.get("score", 0)
    if score < MIN_TAVILY_SCORE:
        return False, {"reason": "Below minimum search score."}

    profile = profile or {}
    keywords = [k.lower() for k in profile.get("keywords", [])]
    locations = [loc.lower() for loc in profile.get("locations", [])]
    phrases = [p.lower() for p in profile.get("phrases", [])]
    claim_has_date = bool(profile.get("has_date"))

    combined_lower = _text_of(result).lower()
    signals = _match_signals(result, profile)

    if claim_has_date and date_signals is None:
        date_signals = check_text_date_match(_text_of(result), claim_date)

    # ----------------------------------------------------------
    # Layer 1: never retrieve user-generated platforms as evidence
    # ----------------------------------------------------------
    tier, _ = classify_tier(result.get("url", ""))
    if tier == TIER_UGC:
        return False, {"reason": "User-generated platform."}

    # ----------------------------------------------------------
    # Layer 1b: reject index / listing / tender / directory pages
    # ----------------------------------------------------------
    if is_listing_or_index_page(_text_of(result), result.get("url", "")):
        return False, {"reason": "Index or listing page, not event reporting."}

    # ----------------------------------------------------------
    # Layer 2: pure policy/regulation pages are not incident reports
    # ----------------------------------------------------------
    claim_text = profile.get("date_stripped", "").lower()
    claim_has_incident = any(m in claim_text for m in _INCIDENT_MARKERS)
    has_policy_focus = any(p in combined_lower for p in _POLICY_MARKERS)
    has_incident_report = any(m in combined_lower for m in _INCIDENT_MARKERS)

    if (
        claim_has_incident
        and has_policy_focus
        and not has_incident_report
        and score < 0.65
    ):
        return False, {"reason": "Policy/regulations page for an incident claim."}

    # ----------------------------------------------------------
    # Layer 2b: incident claims need incident reporting
    #
    # A page that never describes an incident (property listings, jobs,
    # tourism, schedules) can match every claim keyword and still say nothing
    # about the claimed event.
    # ----------------------------------------------------------
    if claim_has_incident and not has_incident_report:
        if not has_fact_check_signal(_text_of(result)):
            return False, {
                "reason": "No incident reporting for an incident claim."
            }


    # ----------------------------------------------------------
    # Layer 3: event relevance (a search score cannot substitute)
    # ----------------------------------------------------------
    has_event_relevance = (
        signals["keyword_count"] >= 1
        or bool(signals["phrase_hits"])
        or bool(signals["location_hits"])
    )

    if not has_event_relevance:
        if not (score >= 0.75 and not claim_has_date):
            return False, {"reason": "No event keyword, phrase or location overlap."}

    # ----------------------------------------------------------
    # Layer 4: claimed location must be present for dated claims
    # ----------------------------------------------------------
    if locations and not signals["location_match"] and not signals["term_match"]:
        if claim_has_date:
            return False, {"reason": "Claimed location absent from candidate."}
        if score < 0.70:
            return False, {"reason": "No location alignment and weak score."}

    # ----------------------------------------------------------
    # Layer 5: date gate
    # ----------------------------------------------------------
    if claim_has_date and date_signals:
        if date_signals["exact_date"]:
            return True, {"reason": "Exact claimed event date present."}

        if has_fact_check_signal(_text_of(result)):
            return True, {"reason": "Fact-check / debunking coverage."}

        if date_signals["different_date"]:
            return False, {"reason": "Candidate describes a different event date."}

        strong_location = bool(signals["location_match"] or signals["term_match"])
        strong_event = signals["keyword_count"] >= 2 or bool(signals["phrase_hits"])

        # Official records and portals are only admissible when they actually
        # carry the claimed date or several claim-specific details; otherwise
        # they are just pages that mention the right city.
        if tier in (TIER_OFFICIAL, TIER_AGGREGATOR):
            if not (
                (strong_location and strong_event and signals["keyword_count"] >= 3)
                or signals["phrase_hits"]
            ):
                return False, {
                    "reason": f"{tier} source without the claimed event date."
                }

        if strong_location and strong_event:
            return True, {"reason": "Same period, location and event match."}

        if strong_location or strong_event:
            return True, {"reason": "Partial location/event match; ranked lower."}

        return False, {"reason": "No usable date, location or event alignment."}

    return True, {"reason": "Accepted for undated claim."}


# ============================================================
# TIERED PRIORITY SCORING
# ============================================================

def score_candidate_relevance(candidate, profile, date_signals):
    """
    Retrieval priority for one candidate.

    For dated claims the Tavily score is capped at 0.50 so a high-scoring
    current article can never outrank an exact-date report, and source
    reliability is a small tie-breaker in favour of reputable publishers.
    """
    signals = _match_signals(candidate, profile)
    profile = profile or {}

    text = _text_of(candidate)
    lowered = text.lower()
    tavily_score = candidate.get("score", 0)
    is_factcheck = has_fact_check_signal(text)
    loc_hit = bool(signals["location_match"] or signals["term_match"])
    kw_hits = signals["keyword_count"]
    phrase_hit = bool(signals["phrase_hits"])
    claim_has_date = bool(profile.get("has_date"))

    profile_source = get_source_profile(
        candidate.get("url", ""), "scraped_article"
    )
    reliability = profile_source["reliability"]

    debug = {
        "tavily_score": round(tavily_score, 4),
        "keyword_hits": kw_hits,
        "keyword_ratio": signals["keyword_ratio"],
        "location_match": loc_hit,
        "phrase_match": phrase_hit,
        "factcheck_signal": is_factcheck,
        "exact_date": False,
        "same_year": False,
        "different_date": False,
        "reliability": reliability,
        "tier": profile_source["tier"],
        "tier_reason": profile_source["tier_reason"],
        "independence_group": profile_source["independence_group"],
        "tier_label": "unknown",
    }

    if not claim_has_date:
        priority = tavily_score
        if loc_hit:
            priority += 0.25
        if kw_hits >= 2:
            priority += 0.20
        elif kw_hits >= 1:
            priority += 0.10
        if phrase_hit:
            priority += 0.10
        priority += (reliability - 0.55) * 0.20
        debug["tier_label"] = "undated"
        return round(priority, 4), debug

    base = min(tavily_score, 0.50)
    date_signals = date_signals or {}
    exact_date = date_signals.get("exact_date", False)
    same_year = date_signals.get("same_year", False)
    different_date = date_signals.get("different_date", False)

    debug["exact_date"] = exact_date
    debug["same_year"] = same_year
    debug["different_date"] = different_date

    if is_factcheck and (exact_date or same_year):
        tier, priority = "T5_factcheck+date", base + 0.40 + 0.10 + (0.15 if loc_hit else 0)
    elif is_factcheck:
        tier, priority = "T5b_factcheck", base + 0.20
    elif exact_date and loc_hit and kw_hits >= 2:
        tier, priority = "T1_exact+loc+event", base + 0.60 + 0.25 + 0.20
    elif exact_date and kw_hits >= 2:
        tier, priority = "T2_exact+event", base + 0.60 + 0.20
    elif exact_date and loc_hit:
        tier, priority = "T3_exact+loc", base + 0.60 + 0.25
    elif exact_date:
        tier, priority = "T4_exact_date", base + 0.60
    elif same_year and loc_hit and kw_hits >= 2:
        tier, priority = "T6_year+loc+event", base + 0.25 + 0.20
    elif same_year and (loc_hit or kw_hits >= 1):
        tier, priority = "T7_year+partial", base + 0.15
    elif same_year:
        tier, priority = "T8_same_year", base + 0.05
    elif not different_date and loc_hit and kw_hits >= 2:
        tier, priority = "T9_context+loc+event", base + 0.20
    elif not different_date and (loc_hit or kw_hits >= 1):
        tier, priority = "T10_context+partial", base + 0.10
    elif different_date:
        tier, priority = "T12_diff_event_date", base - 0.80
    else:
        tier, priority = "T13_other", base - 0.10

    debug["tier_label"] = tier

    if loc_hit and "loc" not in tier:
        priority += 0.10
    if phrase_hit:
        priority += 0.05
    if kw_hits >= 3:
        priority += 0.05
    if is_factcheck and "factcheck" not in tier:
        priority += 0.10

    # Small, bounded preference for reputable publishers.
    priority += max(-0.06, min(0.08, (reliability - 0.55) * 0.20))

    # Aggregator and blog tiers are ranked below genuine reporting.
    if profile_source["tier"] in (TIER_AGGREGATOR, TIER_WEAK):
        priority -= 0.15

    # A rolling liveblog covers days the claim did not mention, so a day inside
    # it is weaker evidence than a single-event report from the same outlet.
    if is_rolling_page(candidate.get("url", ""), candidate.get("title", "")):
        priority -= 0.25
        debug["rolling_page"] = True

    return round(priority, 4), debug


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_candidates(candidates):
    """
    Remove duplicate URLs and syndicated copies of the same story.

    Syndicated copies are dropped rather than merged: two domains carrying the
    same wire text are one voice, and keeping both would inflate the appearance
    of independent corroboration.
    """
    by_url = {}
    for candidate in candidates:
        key = candidate["normalized_url"]
        existing = by_url.get(key)
        if existing is None:
            by_url[key] = candidate
        else:
            for label in candidate.get("found_via", []):
                if label not in existing["found_via"]:
                    existing["found_via"].append(label)

    unique = list(by_url.values())
    unique.sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    kept = []
    fingerprints = []

    for candidate in unique:
        body = f"{candidate.get('title', '')} {candidate.get('content', '')}"
        if len(body.split()) < 40:
            # Too short to judge: keep only if the URL is new.
            kept.append(candidate)
            continue

        duplicate_of = None
        for kept_candidate, fingerprint in fingerprints:
            if kept_candidate["normalized_url"] == candidate["normalized_url"]:
                duplicate_of = kept_candidate["url"]
                break
            if is_near_duplicate(body, fingerprint["body"], SYNDICATION_JACCARD):
                duplicate_of = kept_candidate["url"]
                break

        if duplicate_of:
            candidate["duplicate_of"] = duplicate_of
            continue

        fingerprints.append((candidate, {"body": body}))
        kept.append(candidate)

    return kept


# ============================================================
# CANDIDATE SELECTION
# ============================================================

def retrieval_class(candidate):
    """
    How a candidate may be used downstream.

        exact_date  carries the claimed event date -> primary evidence
        factcheck   explicitly examines the claim    -> primary evidence
        wrong_date  reports the claim's date as false -> primary evidence
        context     same topic, unverified date      -> context only
    """
    debug = candidate.get("_debug", {})

    if debug.get("factcheck_signal"):
        return "factcheck"
    if debug.get("exact_date"):
        return "exact_date"
    if debug.get("different_date"):
        return "wrong_date"
    return "context"


def select_candidates(candidates, limit=MAX_CANDIDATES):
    """
    Diversity-aware selection: best candidate per independence group first,
    then extra candidates from the strongest groups only.

    Grouping uses the registrable domain, so 'ndtv.com' and
    'ndtv.com/business' never count as two voices.
    """
    if not candidates:
        return []

    groups = {}
    for candidate in candidates:
        group = candidate.get("independence_group") or extract_domain(candidate["url"])
        groups.setdefault(group, []).append(candidate)

    for group in groups:
        groups[group].sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: item[1][0].get("priority_score", 0),
        reverse=True,
    )

    selected = []
    for _, group_candidates in ordered_groups:
        if len(selected) >= limit:
            break
        selected.append(group_candidates[0])

    if len(selected) < limit:
        for _, group_candidates in ordered_groups:
            if len(selected) >= limit:
                break
            for extra in group_candidates[1:2]:
                if extra not in selected:
                    selected.append(extra)

    selected.sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    return selected[:limit]


# ============================================================
# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(claim, max_candidates=MAX_CANDIDATES):
    """
    Run every planned search for a claim and return the surviving candidates.

    Returns a dict:
        candidates     selected candidate dicts (filtered, scored,
                       de-duplicated and ordered)
        plans          the searches that were actually issued
        search_errors  queries that failed
    """
    claim = str(claim or "").strip().strip('"').strip("'")
    if not claim:
        return {
            "candidates": [],
            "plans": [],
            "search_errors": ["Empty claim."],
        }

    profile = build_claim_profile(claim)
    claim_date = profile.get("primary_date")
    plans = plan_queries(claim, max_queries=MAX_QUERIES)

    if not plans:
        return {
            "candidates": [],
            "plans": [],
            "search_errors": ["No search queries could be generated."],
        }

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------
    candidates = []
    seen_urls = set()
    search_errors = []

    # News queries are pinned to the claimed event window so the result set is
    # about that event and not about the same city today.
    window_start, window_end = _date_window(claim_date, 2, 3)
    follow_start, follow_end = _date_window(claim_date, 0, 45)

    for plan in plans:
        search_kwargs = {
            "query": plan["query"],
            "max_results": RESULTS_PER_QUERY,
            "search_depth": "advanced",
            "topic": plan["topic"],
            "exclude_domains": EXCLUDED_DOMAINS,
            # Full page text: the difference between having real article text
            # when a site blocks our scraper and having nothing.
            "include_raw_content": True,
        }

        if claim_date and plan["topic"] == "news":
            # Constrain the news index to the claimed event window. Without this
            # a query about a July 2024 event returns September 2026 articles
            # about the same city.
            search_kwargs["start_date"] = window_start
            search_kwargs["end_date"] = window_end

        try:
            response = tavily.search(**search_kwargs)
        except Exception as exc:
            search_errors.append(f"{plan['label']}: {exc}")
            continue

        for result in response.get("results", []) or []:
            raw_url = result.get("url", "")
            if not raw_url:
                continue

            norm_url = normalize_url(raw_url)
            if norm_url in seen_urls:
                for existing in candidates:
                    if existing["normalized_url"] == norm_url:
                        if plan["label"] not in existing["found_via"]:
                            existing["found_via"].append(plan["label"])
                continue

            published_date = (
                result.get("published_date")
                or result.get("publication_date")
                or None
            )

            search_content = (result.get("content") or "").strip()
            raw_content = (result.get("raw_content") or "").strip()
            body = raw_content if len(raw_content) > len(search_content) else search_content

            candidate = {
                "title": (result.get("title") or "").strip(),
                "url": raw_url,
                "normalized_url": norm_url,
                "domain": extract_domain(raw_url),
                "content": body,
                "content_kind": (
                    "full_text" if len(body) >= MIN_FULL_TEXT_CHARS
                    else "snippet"
                ),
                "score": round(float(result.get("score", 0) or 0), 4),
                "published_date": published_date,
                "found_via": [plan["label"]],
                "query_intent": plan["intent"],
                "query": plan["query"],
            }

            date_signals = check_text_date_match(
                f"{candidate['title']} {candidate['content']}", claim_date
            )
            candidate["date_signals"] = date_signals

            relevant, reason = is_result_relevant(
                candidate, profile, date_signals, claim_date
            )
            candidate["relevance_reason"] = reason["reason"]

            if not relevant:
                continue

            source_profile = get_source_profile(raw_url, "scraped_article")
            candidate["source_tier"] = source_profile["tier"]
            candidate["source_tier_reason"] = source_profile["tier_reason"]
            candidate["source_reliability"] = source_profile["reliability"]
            candidate["independence_group"] = source_profile["independence_group"]

            seen_urls.add(norm_url)
            candidates.append(candidate)


    # ----------------------------------------------------
    # FOLLOW-UP COVERAGE
    #
    # Reporting a few weeks later ("what we know so far") is a second, cheap
    # source of same-event articles that the tight window can miss.
    # ----------------------------------------------------
    if claim_date:
        locations = profile.get("locations") or []
        keywords = profile.get("keywords") or []

        followup_query = " ".join(
            part for part in (
                _quote(locations[0]) if locations else "",
                " ".join(keywords[:2]),
                f'"{date_variants_for(claim_date)[0]}"' if True else "",
            ) if part
        ).strip()

        if followup_query:
            try:
                response = tavily.search(
                    query=followup_query,
                    max_results=RESULTS_PER_QUERY,
                    search_depth="advanced",
                    topic="news",
                    start_date=follow_start,
                    end_date=follow_end,
                    exclude_domains=EXCLUDED_DOMAINS,
                    include_raw_content=True,
                )
            except Exception as exc:
                search_errors.append(f"followup: {exc}")
                response = {"results": []}

            for result in response.get("results", []) or []:
                raw_url = result.get("url", "")
                if not raw_url or normalize_url(raw_url) in seen_urls:
                    continue

                search_content = (result.get("content") or "").strip()
                raw_content = (result.get("raw_content") or "").strip()
                body = (
                    raw_content
                    if len(raw_content) > len(search_content)
                    else search_content
                )

                followup_candidate = {
                    "title": (result.get("title") or "").strip(),
                    "url": raw_url,
                    "normalized_url": normalize_url(raw_url),
                    "domain": extract_domain(raw_url),
                    "content": body,
                    "content_kind": (
                        "full_text" if len(body) >= MIN_FULL_TEXT_CHARS
                        else "snippet"
                    ),
                    "score": round(float(result.get("score", 0) or 0), 4),
                    "published_date": (
                        result.get("published_date")
                        or result.get("publication_date")
                        or None
                    ),
                    "found_via": ["followup_window"],
                    "query_intent": "date",
                    "query": followup_query,
                }

                followup_signals = check_text_date_match(
                    f"{followup_candidate['title']} {followup_candidate['content']}",
                    claim_date,
                )
                followup_candidate["date_signals"] = followup_signals

                relevant, reason = is_result_relevant(
                    followup_candidate, profile, followup_signals, claim_date
                )
                followup_candidate["relevance_reason"] = reason["reason"]

                if not relevant:
                    continue

                source_profile = get_source_profile(
                    raw_url, "scraped_article"
                )
                followup_candidate["source_tier"] = source_profile["tier"]
                followup_candidate["source_tier_reason"] = source_profile["tier_reason"]
                followup_candidate["source_reliability"] = source_profile["reliability"]
                followup_candidate["independence_group"] = source_profile[
                    "independence_group"
                ]

                priority, debug = score_candidate_relevance(
                    followup_candidate, profile, followup_signals
                )
                followup_candidate["priority_score"] = priority
                followup_candidate["_debug"] = debug
                followup_candidate["retrieval_class"] = retrieval_class(
                    followup_candidate
                )
                followup_candidate.setdefault("source_tier", debug.get("tier"))
                followup_candidate.setdefault(
                    "source_reliability", debug.get("reliability")
                )
                followup_candidate.setdefault(
                    "independence_group", debug.get("independence_group")
                )

                seen_urls.add(followup_candidate["normalized_url"])
                candidates.append(followup_candidate)

    # ----------------------------------------------------
    # YEAR-LEVEL RECALL
    #
    # A quiet period can leave the date window nearly empty. Before settling for
    # almost no sources, look across the whole claimed year: this cannot create
    # support (the evidence layer still needs an explicit date match), but it
    # finds the context and any fact-check coverage a narrow window misses.
    # ----------------------------------------------------
    if claim_date and len(candidates) < 3:
        recall_query = " ".join(
            part for part in (
                _quote(profile["locations"][0]) if profile.get("locations") else "",
                " ".join((profile.get("keywords") or [])[:3]),
                str(claim_date["year"]),
            ) if part
        ).strip()

        if recall_query:
            try:
                response = tavily.search(
                    query=recall_query,
                    max_results=RESULTS_PER_QUERY,
                    search_depth="advanced",
                    topic="news",
                    exclude_domains=EXCLUDED_DOMAINS,
                    include_raw_content=True,
                )
            except Exception as exc:
                search_errors.append(f"year_recall: {exc}")
                response = {"results": []}

            for result in response.get("results", []) or []:
                raw_url = result.get("url", "")
                if not raw_url or normalize_url(raw_url) in seen_urls:
                    continue

                search_content = (result.get("content") or "").strip()
                raw_content = (result.get("raw_content") or "").strip()
                body = (
                    raw_content
                    if len(raw_content) > len(search_content)
                    else search_content
                )

                recall_candidate = {
                    "title": (result.get("title") or "").strip(),
                    "url": raw_url,
                    "normalized_url": normalize_url(raw_url),
                    "domain": extract_domain(raw_url),
                    "content": body,
                    "content_kind": (
                        "full_text" if len(body) >= MIN_FULL_TEXT_CHARS
                        else "snippet"
                    ),
                    "score": round(float(result.get("score", 0) or 0), 4),
                    "published_date": (
                        result.get("published_date")
                        or result.get("publication_date")
                        or None
                    ),
                    "found_via": ["year_recall"],
                    "query_intent": "broad",
                    "query": recall_query,
                }

                recall_signals = check_text_date_match(
                    f"{recall_candidate['title']} {recall_candidate['content']}",
                    claim_date,
                )
                recall_candidate["date_signals"] = recall_signals

                relevant, reason = is_result_relevant(
                    recall_candidate, profile, recall_signals, claim_date
                )
                recall_candidate["relevance_reason"] = reason["reason"]

                if not relevant:
                    continue

                source_profile = get_source_profile(
                    raw_url, "scraped_article"
                )
                recall_candidate["source_tier"] = source_profile["tier"]
                recall_candidate["source_tier_reason"] = source_profile["tier_reason"]
                recall_candidate["source_reliability"] = source_profile["reliability"]
                recall_candidate["independence_group"] = source_profile[
                    "independence_group"
                ]

                priority, debug = score_candidate_relevance(
                    recall_candidate, profile, recall_signals
                )
                recall_candidate["priority_score"] = priority
                recall_candidate["_debug"] = debug
                recall_candidate["retrieval_class"] = retrieval_class(
                    recall_candidate
                )

                seen_urls.add(recall_candidate["normalized_url"])
                candidates.append(recall_candidate)

    # Safe ordering: recall candidates are scored in the pass below.
    candidates.sort(key=lambda c: c.get("priority_score", 0), reverse=True)

    # --------------------------------------------------------
    # TIERED PRIORITY SCORING
    # --------------------------------------------------------
    for candidate in candidates:
        priority, debug = score_candidate_relevance(
            candidate, profile, candidate.get("date_signals")
        )
        candidate["priority_score"] = priority
        candidate["_debug"] = debug
        candidate.setdefault("source_tier", debug.get("tier"))
        candidate.setdefault("source_reliability", debug.get("reliability"))
        candidate.setdefault("independence_group", debug.get("independence_group"))
        candidate["retrieval_class"] = retrieval_class(candidate)

    candidates.sort(key=lambda c: c["priority_score"], reverse=True)

    # Aggressive filtering: anything that lost the date/location gates is
    # dropped unless it is the only thing we found.
    strong = [c for c in candidates if c["priority_score"] > 0.35]
    weak = [c for c in candidates if c["priority_score"] <= 0.35]

    if len(strong) < MIN_POOL_BEFORE_WIDENING:
        # Recall rescue. Every extra candidate still has to pass the full
        # evidence layer, so widening the pool cannot turn weak material into
        # evidence — it can only give a real corroborating source a chance to
        # appear.
        widened = strong + weak
        candidates = widened[:WIDENED_POOL_SIZE]
    else:
        candidates = strong

    # Syndicated / duplicate copies are removed before selection.
    candidates = deduplicate_candidates(candidates)

    # --------------------------------------------------------
    # PRECISION RETRY
    #
    # If nothing that actually carries the claimed date survived, one exact
    # phrase search is issued before giving up. Search-engine result sets are
    # unstable between runs, and losing the only exact-date source is the single
    # biggest cause of an under-confident verdict.
    # --------------------------------------------------------
    primary_class_candidates = [
        c for c in candidates
        if retrieval_class(c) in PRIMARY_RETRIEVAL_CLASSES
    ]

    if profile.get("primary_date") and len(primary_class_candidates) < 3:
        variants = profile.get("date_variants") or []
        locations = profile.get("locations") or []
        phrases = profile.get("phrases") or []

        retry_query = " ".join(
            part for part in (
                f'"{variants[0]}"' if variants else "",
                f'"{locations[0]}"' if locations else "",
                f'"{phrases[0]}"' if phrases else "",
            ) if part
        ).strip()

        if retry_query:
            try:
                response = tavily.search(
                    query=retry_query,
                    max_results=RESULTS_PER_QUERY,
                    search_depth="advanced",
                    topic="news",
                    exclude_domains=EXCLUDED_DOMAINS,
                    include_raw_content=True,
                )
            except Exception as exc:
                search_errors.append(f"precision_retry: {exc}")
                response = {"results": []}

            for result in response.get("results", []) or []:
                raw_url = result.get("url", "")
                if not raw_url or normalize_url(raw_url) in seen_urls:
                    continue

                search_content = (result.get("content") or "").strip()
                raw_content = (result.get("raw_content") or "").strip()
                body = (
                    raw_content
                    if len(raw_content) > len(search_content)
                    else search_content
                )

                retry_candidate = {
                    "title": (result.get("title") or "").strip(),
                    "url": raw_url,
                    "normalized_url": normalize_url(raw_url),
                    "domain": extract_domain(raw_url),
                    "content": body,
                    "content_kind": (
                        "full_text" if len(body) >= MIN_FULL_TEXT_CHARS
                        else "snippet"
                    ),
                    "score": round(float(result.get("score", 0) or 0), 4),
                    "published_date": (
                        result.get("published_date")
                        or result.get("publication_date")
                        or None
                    ),
                    "found_via": ["precision_retry"],
                    "query_intent": "date",
                    "query": retry_query,
                }

                date_signals = check_text_date_match(
                    f"{retry_candidate['title']} {retry_candidate['content']}",
                    claim_date,
                )
                retry_candidate["date_signals"] = date_signals

                relevant, reason = is_result_relevant(
                    retry_candidate, profile, date_signals, claim_date
                )
                retry_candidate["relevance_reason"] = reason["reason"]

                if not relevant:
                    continue

                source_profile = get_source_profile(raw_url, "scraped_article")
                retry_candidate["source_tier"] = source_profile["tier"]
                retry_candidate["source_tier_reason"] = source_profile["tier_reason"]
                retry_candidate["source_reliability"] = source_profile["reliability"]
                retry_candidate["independence_group"] = source_profile[
                    "independence_group"
                ]

                priority, debug = score_candidate_relevance(
                    retry_candidate, profile, date_signals
                )
                retry_candidate["priority_score"] = priority
                retry_candidate["_debug"] = debug
                retry_candidate["retrieval_class"] = retrieval_class(retry_candidate)

                seen_urls.add(retry_candidate["normalized_url"])
                retry_candidate.setdefault("source_tier", debug.get("tier"))
                retry_candidate.setdefault("source_reliability", debug.get("reliability"))
                retry_candidate.setdefault(
                    "independence_group", debug.get("independence_group")
                )

                candidates.append(retry_candidate)
                plans.append({
                    "label": "precision_retry",
                    "query": retry_query,
                    "intent": "date",
                    "topic": "news",
                })

    # Precision-retry candidates are appended after the main filter, so the
    # same date/location gate and de-duplication are applied to them here.
    candidates = deduplicate_candidates(candidates)

    results = (
        select_candidates(candidates, max_candidates)
        if candidates else []
    )

    return {
        "candidates": results,
        "plans": plans,
        "search_errors": search_errors,
    }


@tool
def web_search(query: str) -> str:
    """
    Retrieve candidate sources for a claim using multiple targeted searches.

    One claim produces several searches (exact wording, exact date in different
    formats, location + event, alternative wording, fact-check angle, official
    records), so the system never depends on a single query.

    Search output is candidate discovery only. Actual evidence comes from
    successfully scraped pages; snippets are weaker fallback evidence.

    For dated claims:
    - Retrieval is date-first: exact-date sources rank highest.
    - A high search score alone CANNOT override a date mismatch.
    - Fact-check/debunking articles are retained regardless of their own date.
    - Aggregator, blog and user-generated sources are ranked down, not trusted.
    """
    claim = str(query or "").strip().strip('"').strip("'")
    if not claim:
        return "NO CLAIM PROVIDED."

    outcome = retrieve(claim)

    plans = outcome["plans"]
    search_errors = outcome["search_errors"]
    results = outcome["candidates"]

    if not results:
        if search_errors:
            return (
                "No sufficiently relevant candidate web sources were found "
                f"(search errors: {'; '.join(search_errors[:3])})."
            )
        return "No sufficiently relevant candidate web sources were found."


    # --------------------------------------------------------
    # FORMAT OUTPUT
    # --------------------------------------------------------
    output = []

    for index, result in enumerate(results, 1):
        snippet = result.get("content", "")
        published = result.get("published_date")
        if published:
            snippet = f"[Published Date: {published}]\n{snippet}"

        debug = result.get("_debug", {})

        output.append(
            f"SOURCE {index}\n"
            f"Title:\n{result.get('title', '')}\n\n"
            f"URL:\n{result.get('url', '')}\n\n"
            f"Domain:\n{result.get('domain', '')}\n\n"
            f"Published Date:\n{published or 'Not stated'}\n\n"
            f"Found Via:\n{', '.join(result.get('found_via', [])) or 'n/a'}\n\n"
            f"Source Tier:\n{result.get('source_tier', 'unknown')} "
            f"({result.get('source_tier_reason', '')})\n\n"
            f"Reliability:\n{result.get('source_reliability', 0)}\n\n"
            f"Match Signals:\n"
            f"date={debug.get('exact_date')} / {debug.get('tier_label')} | "
            f"location={debug.get('location_match')} | "
            f"keywords={debug.get('keyword_hits')} | "
            f"phrase={debug.get('phrase_match')} | "
            f"factcheck={debug.get('factcheck_signal')}\n\n"
            f"Retrieval Class:\n{result.get('retrieval_class', 'context')}\n\n"
            f"Content Kind:\n{result.get('content_kind', 'snippet')} "
            f"({len(result.get('content', ''))} chars)\n\n"
            f"Search Score:\n{result.get('score', 0)}\n\n"
            f"Search Snippet:\n{snippet}\n"
        )

    return "\n--------------------\n".join(output)



# ============================================================
# SCRAPE FALLBACKS
# ============================================================

def _jsonld_article_bodies(soup):
    """
    Pull article bodies out of JSON-LD blocks.

    Many large publishers (Reuters, Hindustan Times, Times of India, AP) render
    the story inside a <script type="application/ld+json"> payload. Selecting
    <p> tags finds nothing there, which is why those pages used to come back as
    "could not extract content".
    """
    bodies = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "{}")
        except Exception:
            continue

        stack = data if isinstance(data, list) else [data]

        while stack:
            obj = stack.pop(0)
            if isinstance(obj, list):
                stack = list(obj) + stack
                continue
            if not isinstance(obj, dict):
                continue

            body = obj.get("articleBody")
            if isinstance(body, str) and len(body) > 200:
                bodies.append(re.sub(r"\s+", " ", body).strip())

            for key in ("@graph", "mainEntity", "mainEntityOfPage", "itemListElement"):
                value = obj.get(key)
                if isinstance(value, (dict, list)):
                    stack = [value] + stack

    return bodies


def _alternate_urls(url):
    """
    Alternative forms of the same page, tried when the main request fails.

    Publishers frequently serve a readable page to a mobile or AMP request but
    block the desktop one, and vice versa.
    """
    variants = []

    try:
        parsed = urlparse(url)
        params = [
            (key, value) for key, value in parse_qsl(parsed.query)
            if key not in ("amp", "output")
        ]
        rebuilt = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path, parsed.params,
            urlencode(params), "",
        ))
        if rebuilt != url:
            variants.append(rebuilt)
    except Exception:
        pass

    if re.search(r"^https://m\.", url):
        variants.append("https://" + url[len("https://m."):])
    elif url.startswith("https://www."):
        variants.append("https://" + url[len("https://www."):])
    elif url.startswith("https://"):
        variants.append("https://www." + url[len("https://"):])

    return [v for v in dict.fromkeys(variants) if v and v != url]


# ============================================================
# WEB SCRAPER (WITH RETRY)
# ============================================================

@tool
def scrape_url(url: str) -> str:
    """
    Extract clean article evidence from a webpage.

    Failed scraping returns a clearly marked error string:
    SCRAPE_FAILED: <reason>

    Hardening over the original version:
    - retries on transient HTTP errors and on 401/403 (with alternate forms of
      the same URL, which many publishers serve more readably)
    - JSON-LD `articleBody` recovery for publishers that render the story
      inside a script payload
    - denser paragraph harvesting (25 characters instead of 40)
    """
    if not url or not url.startswith("http"):
        return "SCRAPE_FAILED: Invalid URL provided."

    first_error = None
    attempts = [url] + _alternate_urls(url)

    for target in attempts[:3]:
        result = _scrape_once(target)
        if not result.startswith("SCRAPE_FAILED:"):
            return result
        if first_error is None:
            first_error = result
        print(f"  [SCRAPE] {target} -> {result[:90]}")

    return first_error or "SCRAPE_FAILED: No response received."


def _scrape_once(url):
    """Single fetch + extraction attempt."""

    # Richer browser-like headers
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    # Retry on transient server-side failures and on bot-blocking codes
    RETRYABLE_CODES = {401, 403, 429, 500, 502, 503, 504}
    MAX_ATTEMPTS = 2
    response = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                timeout=18,
                headers=headers,
                allow_redirects=True
            )

            if response.status_code in RETRYABLE_CODES:
                if attempt < MAX_ATTEMPTS:
                    wait = 2 * attempt
                    print(f"  [SCRAPE] HTTP {response.status_code} — retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})")
                    time.sleep(wait)
                    continue
                return f"SCRAPE_FAILED: HTTP {response.status_code} after {MAX_ATTEMPTS} attempts."

            if response.status_code == 404:
                return "SCRAPE_FAILED: HTTP 404 Not Found"
            if response.status_code >= 400:
                return f"SCRAPE_FAILED: HTTP {response.status_code} Error"

            break  # success

        except requests.exceptions.Timeout:
            if attempt < MAX_ATTEMPTS:
                print(f"  [SCRAPE] Timeout — retrying (attempt {attempt}/{MAX_ATTEMPTS})")
                time.sleep(2)
                continue
            return "SCRAPE_FAILED: Request timed out."
        except requests.exceptions.RequestException as e:
            if attempt < MAX_ATTEMPTS:
                time.sleep(2)
                continue
            return f"SCRAPE_FAILED: {str(e)}"
        except Exception as e:
            return f"SCRAPE_FAILED: {str(e)}"

    if response is None:
        return "SCRAPE_FAILED: No response received."

    try:
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unwanted script, style, navigation, footer, ad tags
        unwanted_tags = [
            "script", "style", "nav", "footer", "header", "aside", "form",
            "noscript", "iframe", "svg", "button", "input", "select",
            "textarea", "menu", "dialog"
        ]
        for tag in soup(unwanted_tags):
            tag.decompose()

        # ====================================================
        # TITLE EXTRACTION
        # ====================================================
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.get_text(" ", strip=True)

        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()

        # ====================================================
        # PUBLICATION DATE EXTRACTION
        # ====================================================
        publication_date = ""
        date_selectors = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "article:published_time"}),
            ("meta", {"itemprop": "datePublished"}),
            ("meta", {"property": "og:published_time"}),
            ("meta", {"name": "date"}),
            ("meta", {"name": "publish-date"}),
            ("meta", {"name": "publication-date"}),
            ("meta", {"name": "DC.date"}),
            ("time", {"datetime": True})
        ]

        for tag_name, attrs in date_selectors:
            tag = soup.find(tag_name, attrs=attrs)
            if tag:
                content_val = tag.get("content") or tag.get("datetime")
                if content_val:
                    publication_date = content_val.strip()
                    break

        # JSON-LD Date Fallback
        if not publication_date:
            json_scripts = soup.find_all("script", type="application/ld+json")
            for script in json_scripts:
                try:
                    data = json.loads(script.string or script.get_text())
                    objects = data if isinstance(data, list) else [data]
                    for obj in objects:
                        if isinstance(obj, dict) and obj.get("datePublished"):
                            publication_date = str(obj["datePublished"]).strip()
                            break
                    if publication_date:
                        break
                except Exception:
                    continue

        # ====================================================
        # MAIN ARTICLE CONTENT EXTRACTION (expanded selectors)
        # ====================================================
        article = None
        selectors = [
            "article",
            '[itemprop="articleBody"]',
            '[role="main"]',
            "main",
            ".article-content",
            ".article-body",
            ".article__body",
            ".story-content",
            ".story-body",
            ".article-body-content",
            ".content-body",
            ".post-content",
            ".entry-content",
            "#article-body",
            ".td-post-content",
            ".post-body",
            ".single-content",
            ".news-body",
            ".text-body",
        ]

        for selector in selectors:
            candidates = soup.select(selector)
            for candidate in candidates:
                text = candidate.get_text(separator=" ", strip=True)
                if len(text) > 300:
                    article = candidate
                    break
            if article:
                break

        if article is None:
            article = soup.body

        # Extract paragraphs
        paragraphs = []
        if article is not None:
            for p in article.find_all(["p", "h2", "h3"]):
                text = p.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) >= 25:
                    paragraphs.append(text)

        # Fallback text if paragraph selection yielded nothing
        if not paragraphs and article is not None:
            text = article.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 200:
                paragraphs = [text]

        # Last structural fallback: the article body embedded in JSON-LD.
        if not paragraphs:
            for body in _jsonld_article_bodies(soup):
                sentences = re.split(r"(?<=[.!?])\s+", body)
                current = []
                for sentence in sentences:
                    current.append(sentence)
                    if len(" ".join(current)) > 900:
                        paragraphs.append(" ".join(current))
                        current = []
                if current:
                    paragraphs.append(" ".join(current))
            if paragraphs:
                print("  [SCRAPE] Recovered article body from JSON-LD payload.")

        # Clean noise patterns
        noise_patterns = [
            r"Skip to main content",
            r"Sign Up",
            r"Log In",
            r"Subscribe",
            r"Follow us",
            r"Read more",
            r"Related Articles",
            r"Most Popular",
            r"Latest News",
            r"Recommended",
            r"Advertisement",
            r"Trending",
            r"Copyright ©.*",
            r"Terms of Use",
            r"Privacy Policy"
        ]

        cleaned_paragraphs = []
        for paragraph in paragraphs:
            if not any(re.search(pat, paragraph, re.IGNORECASE) for pat in noise_patterns):
                cleaned_paragraphs.append(paragraph)

        # Build final article text
        output = []
        if title:
            output.append(f"TITLE: {title}")
        if publication_date:
            output.append(f"PUBLICATION DATE: {publication_date}")
        if cleaned_paragraphs:
            output.append("\n".join(cleaned_paragraphs))

        final_text = "\n\n".join(output).strip()

        # Quality check
        if len(final_text) < 180:
            return (
                "SCRAPE_FAILED: Could not extract enough meaningful article "
                "content from this webpage."
            )

        return final_text[:12000]

    except Exception as e:
        return f"SCRAPE_FAILED: {str(e)}"
