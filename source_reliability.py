"""
source_reliability.py
=====================
Source identity, quality tiering and independence tracking.

Three responsibilities, all of which the evidence layer depends on:

1. RELIABILITY  — a defensible trust score per domain, with explicit tiers so
   that unknown domains can never be treated like a Reuters report.
2. INDEPENDENCE — a registrable-domain group id, so several URLs from one
   publisher (or one mirror family) count as ONE independent voice.
3. DUPLICATION — content fingerprints, so syndicated copies of one wire story
   are collapsed instead of counting as separate corroboration.

Weak sources (blogs, social posts, archive/aggregator pages, forums, wikis)
are never silently promoted: they are labelled, down-weighted, and marked as
non-primary so that higher layers can refuse to build a verdict on them.
"""

import re
from urllib.parse import urlparse


# ============================================================
# RELIABILITY TIERS
# ============================================================

TIER_OFFICIAL = "official"
TIER_WIRE = "established_wire"
TIER_MAJOR = "major_outlet"
TIER_REGIONAL = "regional/general_news"
TIER_AGGREGATOR = "aggregator_portal"
TIER_WEAK = "weak/blog"
TIER_UGC = "user_generated"
TIER_UNKNOWN = "unknown"

TIER_SCORES = {
    TIER_OFFICIAL: 0.95,
    TIER_WIRE: 0.92,
    TIER_MAJOR: 0.82,
    TIER_REGIONAL: 0.68,
    TIER_AGGREGATOR: 0.45,
    TIER_WEAK: 0.32,
    TIER_UGC: 0.15,
    TIER_UNKNOWN: 0.55,
}

# Tiers that may, on their own, establish or refute a claim.
PRIMARY_TIERS = {TIER_OFFICIAL, TIER_WIRE, TIER_MAJOR, TIER_REGIONAL}

# Tiers that can only ever contribute context.
CONTEXT_ONLY_TIERS = {TIER_AGGREGATOR, TIER_WEAK, TIER_UGC, TIER_UNKNOWN}


# ============================================================
# DOMAIN LISTS
# ============================================================

OFFICIAL_SUFFIXES = (".gov", ".gov.in", ".gov.uk", ".mil", ".int", ".police")

OFFICIAL_DOMAINS = {
    "who.int", "un.org", "unesco.org", "imf.org", "worldbank.org",
    "europa.eu", "nasa.gov", "fema.gov", "noaa.gov", "cdc.gov",
    "ndrf.gov.in", "pib.gov.in", "mha.gov.in", "indiaculture.gov.in",
    "ndma.gov.in", "imd.gov.in", "supremecourt.gov.in", "sebi.gov.in",
    "ecb.europa.eu", "unhcr.org", "amnesty.org", "wikileaks.org",
    "who.int", "icrc.org", "wto.org", "oecd.org", "iata.org",
}

WIRE_DOMAINS = {
    "reuters.com", "apnews.com", "afp.com", "bloomberg.com",
    "ptv7.com.pk", "pti.com.pk", "anadoluagency.com", "upi.com",
    "prnewswire.com", "businesswire.com",
}

MAJOR_OUTLETS = {
    "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "wsj.com", "cnn.com", "nbcnews.com",
    "abcnews.go.com", "cbsnews.com", "npr.org", "aljazeera.com",
    "dw.com", "france24.com", "euronews.com", "apnews.com",
    "ndtv.com", "indiatoday.in", "indianexpress.com",
    "hindustantimes.com", "timesofindia.indiatimes.com",
    "thehindu.com", "scroll.in", "theprint.in", "firstpost.com",
    "deccanherald.com", "newindianexpress.com", "business-standard.com",
    "moneycontrol.com", "livemint.com", "reuters.com",
    "nypost.com", "usatoday.com", "latimes.com", "forbes.com",
    "politico.com", "thehill.com", "axios.com", "vox.com",
    "semafor.com", "propublica.org", "reuters.co.uk", "thetimes.co.uk",
    "telegraph.co.uk", "independent.co.uk", "dailymail.co.uk",
    "news18.com", "pbc.gov.in", "zeenews.india.com",
}

# Portals that republish other people's reporting.
AGGREGATOR_DOMAINS = {
    "news.google.com", "news.yahoo.com", "msn.com", "flipkart.com",
    "wikipedia.org", "wikinews.org", "wikiwand.com", "dbpedia.org",
    "archive.org", "webcache.googleusercontent.com", "web.archive.org",
    "archive.ph", "r.jina.ai", "newsbreak.com", "news18.com/rss",
    "headlines24.com", "oneindia.com", "ibnlive.in", "news24tv.com",
    "bolnews.com", "khaleejtimes.com",
}

# Blogs, personal sites, content farms, marketing pages.
WEAK_DOMAIN_HINTS = (
    "blog", "blogs.", "wordpress.com", "blogspot.com", "wixsite.com",
    "weebly.com", "medium.com", "substack.com", "quora.com",
    "hubpages.com", "wikihow.com", "answers.com", "top10", "listicle",
    "clickbait", "viral", "news-", "prweb", "ezoic", "adsterra",
    "guestpost", "sponsored", "newsfeed", "newstrends", "fakenews",
    "wordpress", ".site", ".info", ".top", ".xyz", ".buzz", ".live",
    "mypulse", "pixiedust",
    # Directories, classifieds and listing sites: they mention every topic.
    "listing", "listings", "classified", "directory", "yellowpages",
    "matrimony", "realty", "property", "rentals", "deals", "coupon",
)

# Social / user-generated platforms.
UGC_HOSTS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "linkedin.com", "threads.net",
    "quora.com", "medium.com", "substack.com", "t.me", "telegram.me",
    "wa.me", "pinterest.com", "tumblr.com", "flickr.com", "vk.com",
    "weibo.com", "threads.com", "nextdoor.com", "mastodon.social",
    "bsky.app", "news.ycombinator.com", "tiktok.com",
}

UGC_PATH_HINTS = (
    "reddit.com/r/", "/r/", "facebook.com/groups", "facebook.com/groups/",
    "twitter.com/", "x.com/", "youtube.com/watch", "tiktok.com/",
)


# ============================================================
# URL / DOMAIN UTILITIES
# ============================================================

# Multipart public suffixes handled without a dependency.
_MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "net.in", "org.in",
    "gov.in", "ac.in", "edu.in", "com.au", "net.au", "org.au",
    "co.nz", "co.jp", "co.kr", "com.br", "com.mx", "com.tr",
    "com.sg", "com.hk", "co.za", "com.ng", "co.il", "com.ar",
}

_TRACKING_PARAMS = re.compile(
    r"(utm_[a-z]+|^ref$|^fbclid$|^gclid$|^cmpid$|^smid$|^mc_cid$|^mc_eid$)",
    re.IGNORECASE,
)

# On these hosts the subdomain IS the publisher, so two different blogs are
# two different voices and must not be merged into one independence group.
_SUBDOMAIN_IS_PUBLISHER = {
    "wordpress.com", "blogspot.com", "wixsite.com", "weebly.com",
    "medium.com", "substack.com", "tumblr.com", "notion.site",
    "github.io", "netlify.app", "vercel.app", "pages.dev", "blogger.com",
    "typepad.com", "ghost.io", "glideapp.io", "squarespace.com",
}


def extract_host(url):
    """Return lowercase host (with port stripped) or ''."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[-1]
        return netloc.split(":")[0]
    except Exception:
        return ""


def registrable_domain(url_or_host):
    """
    Registrable domain ('timesofindia.indiatimes.com' -> 'indiatimes.com').

    This is the unit of independence: two articles on the same registrable
    domain are one source, no matter how different the URLs look.
    """
    host = url_or_host
    if "://" in str(url_or_host) or str(url_or_host).startswith("//"):
        host = extract_host(url_or_host)
    host = str(host or "").lower().strip(".")
    if not host:
        return ""

    if ":" in host:
        host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    elif host.startswith("m."):
        host = host[2:]

    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host

    base = ".".join(parts[-2:])
    if base in _SUBDOMAIN_IS_PUBLISHER:
        # The subdomain identifies the individual publisher.
        return host

    if ".".join(parts[-2:]) in _MULTI_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def _is_official(host, domain):
    if any(host.endswith(suffix) for suffix in OFFICIAL_SUFFIXES):
        return True
    return any(
        host == d or host.endswith("." + d)
        for d in OFFICIAL_DOMAINS
    )


def classify_tier(url):
    """
    Return (tier, reason). Deterministic and domain-driven — no heuristics
    that depend on how persuasive a page sounds.
    """
    host = extract_host(url)
    if not host:
        return TIER_UNKNOWN, "Unparseable source URL."

    bare = host[4:] if host.startswith("www.") else host
    domain = registrable_domain(host)

    if any(bare == h or bare.endswith("." + h) for h in UGC_HOSTS):
        return TIER_UGC, f"User-generated platform ({bare})."

    if any(hint in bare for hint in ("/r/", "/groups/")):
        return TIER_UGC, "Forum/group discussion content."

    if domain in AGGREGATOR_DOMAINS or bare in AGGREGATOR_DOMAINS:
        return TIER_AGGREGATOR, f"Aggregator/portal/archive domain ({bare})."

    if any(domain.endswith(d) or bare == d for d in WIRE_DOMAINS):
        return TIER_WIRE, f"Established wire service ({bare})."

    if _is_official(host, domain):
        return TIER_OFFICIAL, f"Official or intergovernmental domain ({bare})."

    if any(domain.endswith(d) or bare == d for d in MAJOR_OUTLETS):
        return TIER_MAJOR, f"Established news outlet ({bare})."

    if any(hint in bare for hint in WEAK_DOMAIN_HINTS):
        return TIER_WEAK, f"Blog/content-farm style domain ({bare})."

    return TIER_REGIONAL, f"Uncategorised news-style domain ({bare})."


def get_source_reliability(url, evidence_type="scraped_article"):
    """
    Numeric trust score.

    Three retrieval modes, in descending trust:
        scraped_article    we retrieved and extracted the page ourselves
        search_extract     the search provider returned the page's article text
                           (real body copy, but not verified by us)
        search_snippet     only a short search-result excerpt

    A snippet is capped below every extracted article, so fallback evidence can
    never receive the same trust as article text.
    """
    tier, _ = classify_tier(url)
    score = TIER_SCORES.get(tier, TIER_SCORES[TIER_UNKNOWN])

    if evidence_type == "search_extract":
        score = min(score * 0.80, 0.62)
    elif evidence_type == "search_snippet":
        score = min(score * 0.65, 0.40)

    if tier in (TIER_WEAK, TIER_UGC):
        score = min(score, 0.32 if evidence_type == "scraped_article" else 0.22)

    return round(score, 4)


def get_source_profile(url, evidence_type="scraped_article"):
    """
    Full source profile used across the evidence layer.

    Returns:
        reliability        float 0..1 trust score
        tier               str   reliability tier label
        tier_reason        str   why this tier was assigned
        domain             str   host
        independence_group str   registrable domain (one voice)
        is_primary         bool  may this source alone establish/refute?
        is_search_snippet  bool
        is_extracted_text  bool
    """
    tier, reason = classify_tier(url)
    reliability = get_source_reliability(url, evidence_type)
    is_snippet = evidence_type == "search_snippet"
    is_extract = evidence_type == "search_extract"

    # Extracted article text may corroborate; a snippet never may.
    is_primary = tier in PRIMARY_TIERS and not is_snippet

    return {
        "reliability": reliability,
        "tier": tier,
        "tier_reason": reason,
        "domain": extract_host(url),
        "independence_group": registrable_domain(url) or extract_host(url) or "unknown",
        "is_primary": is_primary,
        "is_search_snippet": is_snippet,
        "is_extracted_text": is_extract,
    }


# ============================================================
# CONTENT FINGERPRINTING (syndication / near-duplicate detection)
# ============================================================

_WORD_RE = re.compile(r"[a-z0-9]+")

# Boilerplate that appears across unrelated pages and must not make two
# different articles look like the same story.
_BOILERPLATE = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "as", "by", "from", "is", "was", "were", "are", "has", "have",
    "had", "it", "its", "this", "that", "these", "those", "be", "been",
    "but", "not", "we", "he", "she", "they", "his", "her", "their", "our",
    "you", "your", "said", "says", "after", "before", "over", "under",
    "about", "into", "more", "than", "also", "who", "what", "when", "where",
    "which", "will", "would", "could", "has", "may", "can", "all", "one",
    "two", "new", "old", "up", "out", "so", "if", "as", "at", "do", "does",
}

# Syndication markers: these strings identify republished wire copy far more
# reliably than a similarity score alone.
_SYNDICATION_MARKERS = (
    "reuters", "associated press", "ap news", "afp", "pti", "agency",
    "firstpost", "anadolu", "xinhua", "afp.com", "reuters.com",
    "copyright 20", "all rights reserved", "read more:",
)


def _shingles(text, size=5):
    words = [w for w in _WORD_RE.findall((text or "").lower()) if w not in _BOILERPLATE]
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i:i + size])
        for i in range(len(words) - size + 1)
    }


def shingle_set(text, shingle_size=5):
    """Public shingle set for near-duplicate comparison."""
    return _shingles(text, shingle_size)


def content_fingerprint(text, shingle_size=5, limit=4000):
    """
    Cheap deterministic fingerprint for duplicate/syndication detection.

    Returns a dict with the shingle sample (for Jaccard comparison) and a
    stable hash (for exact-match bucketing).
    """
    sample = (text or "")[:limit]
    sh = _shingles(sample, shingle_size)
    stable = hash(frozenset(sh)) if sh else 0
    return {
        "shingles": sh,
        "hash": stable,
        "shingle_count": len(sh),
        "word_count": len(sample.split()),
    }


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def is_near_duplicate(
    text_a,
    text_b,
    threshold=0.62,
    containment_threshold=0.70,
    min_overlap_shingles=8,
):
    """
    Detect syndicated copies of one story.

    Three signals, because a single similarity threshold is unstable on short
    articles (dropping a few trailing words can move 5-gram Jaccard either way):

      1. an identical opening block
      2. a stable fingerprint hash
      3. either high Jaccard overlap, or one text being almost entirely
         contained in the other (a truncated or partial copy)

    Two independently written reports of the same event stay well below every
    one of these thresholds.
    """
    if not text_a or not text_b:
        return False

    a = (text_a or "").lower()
    b = (text_b or "").lower()

    if len(a) > 400 and len(b) > 400 and a[:400] == b[:400]:
        return True

    fa = content_fingerprint(a)
    fb = content_fingerprint(b)

    if not fa["shingles"] or not fb["shingles"]:
        return False

    if fa["hash"] == fb["hash"]:
        return True

    overlap = len(fa["shingles"] & fb["shingles"])

    if jaccard(fa["shingles"], fb["shingles"]) >= threshold:
        return True

    smaller = min(fa["shingle_count"], fb["shingle_count"])
    if (
        overlap >= min_overlap_shingles
        and smaller > 0
        and (overlap / smaller) >= containment_threshold
    ):
        return True

    return False


def has_syndication_marker(text):
    """True when the text carries an explicit wire/agency republishing marker."""
    lowered = (text or "")[:6000].lower()
    return any(marker in lowered for marker in _SYNDICATION_MARKERS)


def url_path_key(url):
    """Path-level key used to catch the same article hosted on two domains."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").lower()
        if not path or path in ("", "/"):
            return ""
        return path
    except Exception:
        return ""


if __name__ == "__main__":

    samples = [
        "https://www.reuters.com/world/india/fire-new-delhi-2024-08-12/",
        "https://en.wikipedia.org/wiki/Fire",
        "https://www.someblog.wordpress.com/fire-in-delhi",
        "https://www.facebook.com/somepage/posts/12345",
        "https://randomnews-site.xyz/2024/08/12/story",
    ]

    for sample in samples:
        profile = get_source_profile(sample)
        print(
            f"{sample}\n  tier={profile['tier']} "
            f"reliability={profile['reliability']} "
            f"group={profile['independence_group']} "
            f"primary={profile['is_primary']}"
        )

    a = "Fire officials said the blaze broke out on the top floor of a commercial building in New Delhi on 12 August 2024. Fire engines were stationed."
    b = "Fire officials said the blaze broke out on the top floor of a commercial building in New Delhi on 12 August 2024. Fire engines were stationed outside."
    c = "Heavy monsoon rains flooded several parts of the city, disrupting traffic and forcing schools shut."

    print("same story ->", is_near_duplicate(a, b))
    print("other story ->", is_near_duplicate(a, c))
