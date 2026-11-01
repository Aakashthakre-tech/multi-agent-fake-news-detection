"""
claim_ranker.py
===============
TruthLensAI Claim Extraction, Filtering, and Verifiability Ranking.

Specialized module for:
1. Distinguishing factual, verifiable propositions from greetings, apologies,
   conversational filler, chyrons, promotional slogans, and OCR noise.
2. Handling multimodal sources: Spoken speech (Whisper), On-Screen text (OCR),
   and reinforced multimodal assertions (both).
3. Scoring candidates deterministically on factuality, specificity, entity
   richness, temporal anchors, numerical data, and assertion completeness.
4. Deduplicating repeated captions and near-duplicate speech/OCR statements.
5. Providing full transparency with explicit selection/rejection reasons.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# LINGUISTIC & DOMAIN PATTERNS FOR FILTERING
# ============================================================

# Conversational apologies
_APOLOGY_PATTERNS = [
    re.compile(r"^(?:i'?m|im|i\s+am)?\s*sorry\b[.!?,]?$", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|im|i\s+am)?\s*sorry\s+(?:about|for|to)\s+(?:that|this|you|it)\b", re.IGNORECASE),
    re.compile(r"^(?:my\s+)?apolog(?:y|ies)\b[.!?,]?$", re.IGNORECASE),
    re.compile(r"\bpardon\s*(?:me)?\b[.!?,]?$", re.IGNORECASE),
    re.compile(r"\bexcuse\s+me\b[.!?,]?$", re.IGNORECASE),
]

# Conversational greetings and farewells
_GREETING_PATTERNS = [
    re.compile(
        r"^(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|"
        r"welcome(?:\s+back)?|greetings|howdy|what'?s\s+up|how\s+are\s+you)\b[.!?,]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:goodbye|bye|bye\s+bye|see\s+you(?:\s+later)?|take\s+care|"
        r"until\s+next\s+time|have\s+a\s+(?:good|great)\s+day)\b[.!?,]?",
        re.IGNORECASE,
    ),
]

# Politeness filler and acknowledgments
_FILLER_PATTERNS = [
    re.compile(
        r"^(?:thank\s+you(?:\s+very\s+much)?|thanks(?:\s+a\s+lot)?|"
        r"thanks\s+for\s+watching|thank\s+you\s+all)\b[.!?,]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:you'?re\s+welcome|no\s+problem|my\s+pleasure|don'?t\s+mention\s+it)\b[.!?,]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:ok|okay|alright|all\s+right|sure|yes|yeah|yep|yup|nope|nah|no|"
        r"uh|um|hmm|well|so|you\s+know|i\s+mean|right)\b[.!?,]?$",
        re.IGNORECASE,
    ),
]

# Promotional slogans, subscription prompts, social media calls-to-action
_PROMOTIONAL_PATTERNS = [
    re.compile(
        r"\b(?:subscribe(?:\s+to\s+our\s+channel)?|like\s+and\s+subscribe|"
        r"hit\s+the\s+bell\s+icon|follow\s+(?:for\s+more|us)|link\s+in\s+bio|"
        r"link\s+in\s+description|share\s+this\s+video|leave\s+a\s+comment|"
        r"visit\s+our\s+website|check\s+out\s+the\s+link|swipe\s+up|"
        r"watch\s+(?:until|till)\s+the\s+end|wait\s+for\s+(?:it|the\s+end)|"
        r"part\s+\d+\s+in\s+(?:bio|comments)|tag\s+your\s+friends|"
        r"don'?t\s+forget\s+to\s+like|sound\s+credit|credit\s+to\s+owner)\b",
        re.IGNORECASE,
    ),
]

# Isolated broadcast stamps / chyrons / noise without a predicate
_CHYRON_STAMP_PATTERNS = [
    re.compile(
        r"^(?:breaking\s+news|breaking|live|exclusive|special\s+report|"
        r"developing\s+story|news\s+alert|top\s+story|press\s+conference|"
        r"reuters|ap\s+news|bbc\s+news|cnn|ndtv|ani|pti|hindustan\s+times|"
        r"times\s+now|republic|page\s+\d+|hd|4k|cctv(?:\s+footage)?|"
        r"shot\s+on\s+iphone|kinemaster|capcut|tiktok\s*@\w+|"
        r"all\s+rights\s+reserved|copyright\s+disclaimer)$",
        re.IGNORECASE,
    ),
]

# Devanagari / Multilingual Incident, Governance, and Assertive Predicates
_DEVANAGARI_INCIDENT_VERBS = {
    "अपघात", "मृत्यू", "मृत्यु", "हल्ला", "आग", "धक्कादायक", "जखमी",
    "स्फोट", "भूकंप", "पूर", "पडला", "पडले", "ठार", "मरण", "हताहत",
}

_DEVANAGARI_ACTION_POLICY_VERBS = {
    "घोषणा", "सरकार", "सरकारने", "निर्णय", "बैन", "अटक", "निलंबित",
    "राजीनामा", "आदेश", "कोर्ट", "न्यायालय", "पोलिस", "पोलीस", "निवडणूक",
}

_DEVANAGARI_PREDICATES = {
    "आहे", "होता", "होती", "होते", "झाला", "झाली", "झाले",
    "केला", "केली", "केले", "आहेत",
}

ALL_DEVANAGARI_VERBS = _DEVANAGARI_INCIDENT_VERBS | _DEVANAGARI_ACTION_POLICY_VERBS | _DEVANAGARI_PREDICATES

# Event & Incident Verbs/Predicates indicating checkable real-world actions
_INCIDENT_VERBS = {
    "struck", "strike", "strikes", "hit", "hits", "killed", "kills", "killing",
    "injured", "injures", "injuring", "collapsed", "collapses", "collapsing",
    "erupted", "erupts", "exploded", "explodes", "crashed", "crashes",
    "destroyed", "destroys", "damaged", "damages", "burned", "burns", "burning",
    "flooded", "floods", "flooding", "derailed", "derails", "derailment",
    "died", "dies", "dead", "fatalities", "casualties", "displaced",
    "gutted", "engulfed", "capsized", "detonated", "sank", "sunk",
}

# Governance, Policy, Official, Legal & Public Statement Verbs
_ACTION_POLICY_VERBS = {
    "announced", "announces", "announcing", "banned", "bans", "banning",
    "declared", "declares", "declaring", "passed", "passes", "passing",
    "approved", "approves", "approving", "resigned", "resigns", "resignation",
    "arrested", "arrests", "arresting", "charged", "charges", "charging",
    "sentenced", "sentences", "convicted", "convicts", "acquitted", "acquits",
    "filed", "files", "filing", "launched", "launches", "launching",
    "signed", "signs", "signing", "confirmed", "confirms", "confirming",
    "denied", "denies", "denying", "reported", "reports", "reporting",
    "stated", "states", "stating", "said", "says", "saying",
    "decided", "decides", "deciding", "voted", "votes", "voting",
    "suspended", "suspends", "suspending", "elected", "elects", "rejects",
    "introduced", "introduces", "warned", "warns", "ordered", "orders",
    "debate", "debates", "debating", "held", "concluded", "concludes",
    "attended", "meets", "convened", "adjourned", "addressed", "addresses",
}

# General assertive and state predicates
_PREDICATE_VERBS = {
    "is", "was", "are", "were", "has", "have", "had", "been",
    "became", "becomes", "will", "would", "can", "could",
    "occurred", "happened", "took", "held", "recorded", "reached",
    "increased", "decreased", "rose", "fell", "dropped", "surpassed",
}

ALL_VERIFY_VERBS = _INCIDENT_VERBS | _ACTION_POLICY_VERBS | _PREDICATE_VERBS

# Known geographical hints (countries, major cities, regions)
_GEO_ENTITIES = {
    "india", "pakistan", "bangladesh", "nepal", "sri lanka", "china",
    "japan", "korea", "brazil", "mexico", "canada", "australia", "france",
    "germany", "uk", "united kingdom", "united states", "america", "usa",
    "russia", "ukraine", "egypt", "kenya", "nigeria", "indonesia", "thailand",
    "vietnam", "dubai", "delhi", "new delhi", "mumbai", "kolkata", "chennai",
    "bengaluru", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "assam",
    "odisha", "maharashtra", "gujarat", "kerala", "tamil nadu", "bihar",
    "tokyo", "beijing", "seoul", "london", "paris", "berlin", "madrid",
    "rome", "sydney", "washington", "new york", "california", "texas",
    "florida", "chicago", "morbi", "manila", "jakarta", "kyiv", "gaza",
}

# Key institutions and public titles
_INSTITUTION_KEYWORDS = {
    "government", "govt", "parliament", "congress", "senate", "court",
    "supreme court", "high court", "police", "ministry", "minister",
    "prime minister", "pm", "president", "governor", "mayor", "army",
    "military", "navy", "air force", "isro", "nasa", "who", "un", "unesco",
    "ed", "cbi", "rbi", "bcci", "assembly", "cabinet", "administration",
}

# A Devanagari word (Marathi / Hindi). Used to recognise entities in scripts
# that have no capitalisation, where the Title-Case rule cannot apply.
_DEVANAGARI_WORD_RE = re.compile(r"^[\u0900-\u097F][\u0900-\u097F\u200C\u200D]+$")

# Very common Marathi/Hindi function words. They carry no entity information
# and would otherwise be counted as named entities.
_DEVANAGARI_FUNCTION_WORDS = {
    "आहे", "आहेत", "आणि", "मध्ये", "वर", "चे", "चा", "ची", "चे", "ला", "ने",
    "पासून", "साठी", "पर्यंत", "नंतर", "आधी", "कधी", "कुठे", "कोण", "काय",
    "म्हणून", "पण", "हा", "ही", "हे", "या", "त्या", "ज्या", "काही", "सर्व",
    "आज", "काल", "मला", "तुम्ही", "आम्ही", "त्यांनी", "आपण", "मी",
    "भारत", "देश", "वर्ष", "दिवस", "केंद्र", "सरकार", "शहर",
}


# ============================================================
# LINGUISTIC & FACTUALITY ANALYZERS
# ============================================================

def _first_matching_verb(category: Iterable[str], verbs_found: Iterable[str]) -> str:
    """
    Return the verb that actually belongs to ``category``.

    ``verbs_found`` is in token order, so its first element is not necessarily
    the verb that matched the category. Reporting that one made the rejection
    / selection reason name a verb that had nothing to do with the category.
    """
    category_set = set(category)
    for verb in verbs_found:
        if verb in category_set:
            return verb
    return ""


def clean_claim_text(text: str) -> str:
    """Normalize whitespace and strip stray surrounding quotes/brackets."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t.strip("\"'`“”‘’").strip()


def extract_numbers_and_stats(text: str) -> List[str]:
    """Identify numerical figures, percentages, currencies, or magnitudes."""
    pattern = (
        r"\b(?:\$|€|£|₹|rs\.?\s*)?\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:%|percent|"
        r"crore|lakh|million|billion|trillion|thousand|dead|injured|killed|"
        r"people|magnitude|magnitude\s+of|km|mph|celsius|°c|°f))?\b"
    )
    matches = re.findall(pattern, text, re.IGNORECASE)
    cleaned = []
    for m in matches:
        m_str = m.strip()
        if m_str and any(ch.isdigit() for ch in m_str):
            cleaned.append(m_str)
    return cleaned


def extract_potential_entities(text: str) -> List[str]:
    """
    Extract proper nouns, capitalized sequences, recognized institutions,
    and geographical locations.
    """
    entities = []
    text_lower = text.lower()

    # Check known geo entities
    for geo in _GEO_ENTITIES:
        if re.search(rf"\b{re.escape(geo)}\b", text_lower):
            entities.append(geo.title())

    # Check institutions
    for inst in _INSTITUTION_KEYWORDS:
        if re.search(rf"\b{re.escape(inst)}\b", text_lower):
            entities.append(inst.title())

    # Capitalized word sequences not at the very start of a sentence
    # (avoiding classifying sentence starters as proper nouns)
    words = text.split()
    for i, word in enumerate(words):
        clean_word = word.strip(".,:;!?\"'()[]{}")
        if i > 0 and clean_word.istitle() and len(clean_word) >= 3:
            if clean_word.lower() not in {"the", "this", "that", "there", "they", "today", "yesterday"}:
                if clean_word not in entities:
                    entities.append(clean_word)

    # Devanagari (Marathi/Hindi) words have no case, so the Title-Case rule
    # above never matches them and a perfectly good Marathi statement ended up
    # with no entities at all. Recognise Devanagari words directly, skipping
    # the very common function words that carry no entity information.
    for word in words:
        clean_word = word.strip(".,:;!?\"'()[]{}।")
        if len(clean_word) < 3 or not _DEVANAGARI_WORD_RE.match(clean_word):
            continue
        if clean_word in _DEVANAGARI_FUNCTION_WORDS:
            continue
        if clean_word not in entities:
            entities.append(clean_word)

    return list(dict.fromkeys(entities))


def detect_predicate_and_verbs(text: str) -> Tuple[bool, List[str]]:
    """Determine whether the candidate has a verifiable predicate/verb, supporting multilingual & Devanagari text."""
    # Tokenize preserving non-Latin words and combining vowel marks/matras
    words = [w.strip("\"'`“”‘’").lower() for w in re.findall(r"[^\s.,!?;:\"\'()\[\]{}—–]+", text)]
    found_verbs = [w for w in words if w in ALL_VERIFY_VERBS or w in ALL_DEVANAGARI_VERBS]

    # Handle past tense -ed forms or passive participles (e.g. 'derailed', 'arrested')
    for w in words:
        if len(w) > 4 and (w.endswith("ed") or w.endswith("ing")) and w not in found_verbs:
            # Common verb heuristic
            found_verbs.append(w)

    has_predicate = bool(found_verbs)
    return has_predicate, found_verbs


def is_non_factual_or_filler(text: str) -> Tuple[bool, str]:
    """
    Check if a candidate string is conversational filler, greeting, apology,
    promotional CTA, broadcast stamp, or OCR noise.
    """
    clean = clean_claim_text(text)
    if not clean:
        return True, "Empty text"

    # 1. OCR Noise & Low Quality
    # Filter isolated digital clock/timestamp (e.g., "12:45:00", "09:30 AM")
    if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:am|pm|ist|gmt|utc))?$", clean.lower()):
        return True, "OCR noise: isolated digital clock / timestamp without assertion"

    # Letters AND combining marks. Devanagari vowel signs (and the matras of
    # other Indic scripts) are Unicode category Mc/Mn, so `str.isalpha()` is
    # False for them. Counting only `isalpha()` characters made valid
    # Marathi/Hindi sentences fall below the ratio and be rejected as OCR
    # noise, which silently discarded real factual claims.
    alpha_chars = sum(
        1 for ch in clean
        if ch.isalpha() or unicodedata.category(ch)[0] == "M"
    )
    total_len = len(clean)
    if total_len > 0 and (alpha_chars / total_len) < 0.50:
        return True, "OCR noise: low alphabetic character ratio or garbled symbols"

    # Single or very few repeated characters
    if len(set(clean.lower().replace(" ", ""))) <= 2 and total_len > 3:
        return True, "Repetitive character sequence / noise"

    # 2. Pure Apology Check
    for pat in _APOLOGY_PATTERNS:
        if pat.search(clean):
            # Check if this is an isolated apology vs embedded reporting
            # e.g. "I'm sorry." is filler; "PM said 'I'm sorry' in parliament" is a factual report
            words = clean.split()
            if len(words) <= 5 and not any(geo in clean.lower() for geo in _GEO_ENTITIES):
                return True, "Conversational apology lacking verifiable factual proposition"

    # 3. Pure Greeting Check
    for pat in _GREETING_PATTERNS:
        if pat.search(clean):
            words = clean.split()
            if len(words) <= 5 and not any(w in clean.lower() for w in _INCIDENT_VERBS):
                return True, "Conversational greeting/farewell lacking verifiable factual proposition"

    # 4. Pure Politeness & Conversational Filler
    for pat in _FILLER_PATTERNS:
        if pat.search(clean):
            words = clean.split()
            if len(words) <= 5:
                return True, "Conversational filler or politeness phrase"

    # 5. Promotional CTA
    for pat in _PROMOTIONAL_PATTERNS:
        if pat.search(clean):
            return True, "Promotional slogan or social media call-to-action"

    # 6. Isolated Broadcast Stamp / Chyron Header
    for pat in _CHYRON_STAMP_PATTERNS:
        if pat.search(clean):
            return True, "Broadcast chyron stamp or channel label lacking a proposition"

    # 7. Incomplete Dangling Clauses
    dangling_starts = ("and then", "because of", "in order to", "with respect to", "as well as", "due to the")
    if clean.lower().startswith(dangling_starts) and len(clean.split()) < 6:
        return True, "Incomplete syntactic clause fragment"

    return False, ""


# ============================================================
# CANDIDATE CLAIM EVALUATION & SCORING
# ============================================================

def evaluate_candidate_claim(
    text: str,
    source: str = "whisper",
    timestamps: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Score and classify an individual candidate claim.

    Args:
        text: Raw or clean claim text.
        source: Modality source ('whisper', 'ocr', 'both').
        timestamps: Optional start/end time.
        context: Optional dictionary with dates, video info, or OCR context.

    Returns:
        Structured evaluation dictionary containing priority_score, status
        ('selected' or 'rejected'), selection_reason, is_factual, is_verifiable,
        and extracted entities/dates.
    """
    clean_text = clean_claim_text(text)
    is_filler, filler_reason = is_non_factual_or_filler(clean_text)

    # Date references in text or context
    from video_module.video_pipeline import extract_dates_from_text
    dates_found = extract_dates_from_text(clean_text)
    if not dates_found and context:
        if context.get("date_reference"):
            dates_found = [context["date_reference"]]
        elif context.get("on_screen_context"):
            dates_found = extract_dates_from_text(context["on_screen_context"])

    entities = extract_potential_entities(clean_text)
    numbers = extract_numbers_and_stats(clean_text)
    has_predicate, verbs_found = detect_predicate_and_verbs(clean_text)

    word_count = len(clean_text.split())

    # --------------------------------------------------------
    # REJECTION RULES
    # --------------------------------------------------------
    if is_filler:
        return {
            "claim_text": clean_text,
            "source_provenance": source,
            "source_modality": "speech" if source == "whisper" else source,
            "priority_score": 0.05,
            "is_factual": False,
            "is_verifiable": False,
            "selection_status": "rejected",
            "selection_reason": f"Rejected: {filler_reason}.",
            "entities": entities,
            "dates": dates_found,
            "numbers": numbers,
            "has_predicate": has_predicate,
        }

    # Minimum check: a claim must have an assertion/predicate to be verifiable
    if not has_predicate:
        # Check if it has dates/entities that might make it a chyron summary
        if not (entities and (dates_found or numbers)):
            return {
                "claim_text": clean_text,
                "source_provenance": source,
                "source_modality": "speech" if source == "whisper" else source,
                "priority_score": 0.15,
                "is_factual": False,
                "is_verifiable": False,
                "selection_status": "rejected",
                "selection_reason": "Rejected: Incomplete statement lacking a verifiable predicate or assertion.",
                "entities": entities,
                "dates": dates_found,
                "numbers": numbers,
                "has_predicate": False,
            }

    # Isolated words check (length < 3 words without specific event predicate)
    if word_count < 3 and not (entities and any(v in _INCIDENT_VERBS for v in verbs_found)):
        return {
            "claim_text": clean_text,
            "source_provenance": source,
            "source_modality": "speech" if source == "whisper" else source,
            "priority_score": 0.20,
            "is_factual": False,
            "is_verifiable": False,
            "selection_status": "rejected",
            "selection_reason": "Rejected: Too brief / isolated words without verifiable factual context.",
            "entities": entities,
            "dates": dates_found,
            "numbers": numbers,
            "has_predicate": has_predicate,
        }

    # --------------------------------------------------------
    # PRIORITY SCORING FOR FACTUAL CLAIMS
    # --------------------------------------------------------
    score = 0.20

    # 1. Action / Incident / Policy Predicate Strength
    if any(v in _INCIDENT_VERBS for v in verbs_found):
        score += 0.25
    elif any(v in _ACTION_POLICY_VERBS for v in verbs_found):
        score += 0.22
    elif has_predicate:
        score += 0.12

    # 2. Named Entities / Locations / Organizations
    if entities:
        score += min(len(entities) * 0.10, 0.25)

    # 3. Temporal Anchors (Dates, Years)
    if dates_found:
        score += 0.15

    # 4. Numerical Data / Statistics / Quantities
    if numbers:
        score += min(len(numbers) * 0.08, 0.15)

    # 5. Multimodal Corroboration Boost
    if source == "both":
        score += 0.15

    # 6. Appropriate Statement Length / Completeness
    if 4 <= word_count <= 35:
        score += 0.08

    priority_score = round(min(max(score, 0.0), 1.0), 3)

    # Criteria for selecting
    is_verifiable = bool(has_predicate and (entities or dates_found or numbers or any(v in _INCIDENT_VERBS for v in verbs_found)))
    selection_status = "selected" if (is_verifiable and priority_score >= 0.40) else "rejected"

    if selection_status == "selected":
        reasons = []
        if any(v in _INCIDENT_VERBS for v in verbs_found):
            reasons.append(f"event action ('{_first_matching_verb(_INCIDENT_VERBS, verbs_found)}')")
        elif any(v in _ACTION_POLICY_VERBS for v in verbs_found):
            reasons.append(f"policy/public action ('{_first_matching_verb(_ACTION_POLICY_VERBS, verbs_found)}')")
        elif has_predicate:
            reasons.append("assertive predicate")

        if entities:
            reasons.append(f"named entities ({', '.join(entities[:2])})")
        if dates_found:
            reasons.append(f"temporal anchor ('{dates_found[0]}')")
        if numbers:
            reasons.append(f"numerical data ({', '.join(numbers[:2])})")
        if source == "both":
            reasons.append("multimodal speech/OCR reinforcement")

        reason_str = f"Selected: Verifiable assertion containing {', '.join(reasons)}."
    else:
        if not has_predicate:
            # The claim carried anchors (entities, dates or numbers) but no
            # assertion verb, so nothing in it can actually be checked. The
            # reason must say that, rather than claiming it lacked the
            # entities and dates it in fact had.
            reason_str = (
                "Rejected: No verifiable predicate detected - the statement carries "
                "anchors but no assertion verb, so nothing in it can be checked."
            )
        else:
            reason_str = (
                "Rejected: Insufficient factual anchors (lacks specific entities, "
                "dates, or verifiable event detail)."
            )

    return {
        "claim_text": clean_text,
        "source_provenance": source,
        "source_modality": "speech" if source == "whisper" else source,
        "priority_score": priority_score,
        "is_factual": is_verifiable,
        "is_verifiable": is_verifiable,
        "selection_status": selection_status,
        "selection_reason": reason_str,
        "entities": entities,
        "dates": dates_found,
        "numbers": numbers,
        "has_predicate": has_predicate,
    }


# ============================================================
# CANDIDATE DEDUPLICATION & MULTI-MODAL CONSOLIDATION
# ============================================================

def deduplicate_candidate_claims(
    candidates: List[Dict[str, Any]],
    similarity_threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Deduplicate repeated captions and cross-modal duplicates (speech vs OCR).
    Combines provenance into 'both' when a claim appears in both speech and OCR.
    Preserves frame references and expands time spans.
    """
    if not candidates:
        return []

    deduped: List[Dict[str, Any]] = []

    for cand in candidates:
        cand_text = cand.get("claim_text", "")
        cand_words = set(re.findall(r"\b\w+\b", cand_text.lower()))

        matched_idx = -1
        best_sim = 0.0

        for idx, existing in enumerate(deduped):
            exist_text = existing.get("claim_text", "")
            exist_words = set(re.findall(r"\b\w+\b", exist_text.lower()))

            # Jaccard word similarity
            if cand_words and exist_words:
                jaccard = len(cand_words & exist_words) / len(cand_words | exist_words)
            else:
                jaccard = 0.0

            seq_ratio = difflib.SequenceMatcher(None, cand_text.lower(), exist_text.lower()).ratio()
            combined_sim = max(jaccard, seq_ratio)

            if combined_sim >= similarity_threshold and combined_sim > best_sim:
                best_sim = combined_sim
                matched_idx = idx

        if matched_idx >= 0:
            # Merge with existing candidate
            target = deduped[matched_idx]

            # Merge sources: speech + ocr -> both
            s1 = target.get("source_provenance", "unknown")
            s2 = cand.get("source_provenance", "unknown")
            if s1 != s2 and {"whisper", "speech"} & {s1, s2} and "ocr" in {s1, s2}:
                target["source_provenance"] = "both"
                target["source_modality"] = "both"
                target["priority_score"] = round(min(target.get("priority_score", 0.5) + 0.15, 1.0), 3)
                target["selection_reason"] = (
                    target.get("selection_reason", "")
                    + " [Reinforced across speech and on-screen OCR]"
                ).strip()

            # Merge timestamps
            t_curr = target.get("timestamps", {})
            t_cand = cand.get("timestamps", {})
            target["timestamps"] = {
                "start_sec": min(t_curr.get("start_sec", 0.0), t_cand.get("start_sec", 0.0)),
                "end_sec": max(t_curr.get("end_sec", 0.0), t_cand.get("end_sec", 0.0)),
            }

            # Merge frame references
            all_frames = set(target.get("frame_references", [])) | set(cand.get("frame_references", []))
            target["frame_references"] = sorted(list(all_frames))

            # Keep more detailed text
            if len(cand_text) > len(target.get("claim_text", "")):
                target["claim_text"] = cand_text
                target["original_text"] = cand.get("original_text", cand_text)

        else:
            deduped.append(cand.copy())

    return deduped


# ============================================================
# COMPREHENSIVE CLAIM PIPELINE RANKER
# ============================================================

def process_and_rank_candidate_claims(
    raw_candidates: List[Dict[str, Any]],
    video_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Filter, score, deduplicate, and order all candidate claims.

    Returns:
        Structured dictionary with ranked claims, statistics, and top verifiable claims.
    """
    evaluated_list: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_candidates, 1):
        text = item.get("text") or item.get("claim_text") or ""
        source = item.get("source") or item.get("source_provenance") or "unknown"
        timestamps = {
            "start_sec": item.get("start_sec", 0.0),
            "end_sec": item.get("end_sec", 0.0),
        }
        frames = item.get("frame_references", [])
        context = item.get("context", {})

        eval_res = evaluate_candidate_claim(
            text=text,
            source=source,
            timestamps=timestamps,
            context=context,
        )

        candidate_obj = {
            "claim_id": f"claim_{idx:03d}",
            "claim_text": eval_res["claim_text"],
            "original_text": text,
            "source_provenance": eval_res["source_provenance"],
            "source_modality": eval_res["source_modality"],
            "timestamps": timestamps,
            "frame_references": frames,
            "context": {
                "date_reference": (
                    eval_res["dates"][0]
                    if eval_res["dates"]
                    else (context.get("date_reference") if context else None)
                ),
                "on_screen_context": item.get("on_screen_context"),
                "video_file": video_metadata.get("file_name") if video_metadata else None,
                "entities": eval_res["entities"],
                "numbers": eval_res["numbers"],
            },
            "status": "candidate",
            "selection_status": eval_res["selection_status"],
            "priority_score": eval_res["priority_score"],
            "selection_reason": eval_res["selection_reason"],
            "is_factual": eval_res["is_factual"],
            "is_verifiable": eval_res["is_verifiable"],
            "disagreements": list(item.get("disagreements", [])),
            "review_flags": list(item.get("review_flags", [])),
            "needs_manual_review": bool(item.get("review_flags", []) or item.get("disagreements", [])),
        }
        evaluated_list.append(candidate_obj)

    # Deduplicate claims
    deduped = deduplicate_candidate_claims(evaluated_list)

    # Sort claims:
    # 1. 'selected' before 'rejected'
    # 2. Higher priority_score first
    # 3. Earlier start_sec
    deduped.sort(
        key=lambda c: (
            1 if c.get("selection_status") == "selected" else 0,
            c.get("priority_score", 0.0),
            -c.get("timestamps", {}).get("start_sec", 0.0),
        ),
        reverse=True,
    )

    # Renumber claim IDs sequentially
    for i, c in enumerate(deduped, 1):
        c["claim_id"] = f"claim_{i:03d}"

    selected_count = sum(1 for c in deduped if c.get("selection_status") == "selected")
    rejected_count = sum(1 for c in deduped if c.get("selection_status") == "rejected")

    status = "success" if selected_count > 0 else ("insufficient_claims" if deduped else "empty")

    return {
        "status": status,
        "candidate_claims_count": len(deduped),
        "selected_claims_count": selected_count,
        "rejected_claims_count": rejected_count,
        "claims_flagged_for_review_count": sum(1 for c in deduped if c["needs_manual_review"]),
        "candidate_claims": deduped,
    }
