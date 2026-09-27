"""
test_evidence_engine.py
=======================
Offline regression tests for the evidence engine.

No network, no LLM calls. Model-backed helpers (NLI, similarity) are exercised
only through their pure logic, and every structural guarantee of the pipeline is
asserted here:

  * source tiering and independence groups
  * syndicated-copy detection
  * multi-angle query planning
  * retrieval filters (listing pages, wrong dates, incident claims)
  * date handling (event date vs publication date)
  * event matching (location conflicts, claim-detail coverage)
  * classification gates (similarity is never sufficient)
  * scoring (structure beats similarity)
  * source-level aggregation and independent corroboration
  * conservative verdict gating

Run:  python test_evidence_engine.py
"""

import sys
import unittest

from claim_profile import build_claim_profile, strip_publication_boilerplate
from query_planner import plan_queries
from source_reliability import (
    get_source_profile,
    get_source_reliability,
    registrable_domain,
    is_near_duplicate,
)
from tools import (
    check_text_date_match,
    is_result_relevant,
    is_listing_or_index_page,
    score_candidate_relevance,
    select_candidates,
    retrieve,
    retrieval_class,
    SYNDICATION_JACCARD,
)
from evidence_analyzer import (
    check_date_compatibility,
    evaluate_event_match,
    extract_claim_attributes,
    classify_evidence,
    _dedupe_chunks,
    _resolve_chunk_budget,
)
from evidence_scorer import calculate_evidence_score, apply_class_ceiling
from evidence_fusion import calculate_source_agreement, calculate_fusion_score
from final_verdict import (
    generate_final_verdict,
    extract_verdict,
    extract_confidence,
    deterministic_gate,
)

FIRE_CLAIM = (
    "A major fire broke out at a commercial building in New Delhi, "
    "India on 12 Aug 2024."
)

PRIMARY = {"is_primary": True, "tier": "established_wire", "reliability": 0.92}
WEAK = {"is_primary": False, "tier": "weak/blog", "reliability": 0.32}


def candidate(title, content, url, score=0.7, published=None):
    return {
        "title": title,
        "content": content,
        "url": url,
        "score": score,
        "published_date": published,
        "domain": registrable_domain(url),
    }


def chunk(source_id, url, evidence_class, score, evidence_type="scraped_article",
          group=None):
    return {
        "source_id": source_id,
        "url": url,
        "domain": url.split("/")[2],
        "title": "title",
        "evidence_type": evidence_type,
        "chunk": "evidence passage",
        "similarity": 0.80,
        "nli_label": (
            "entailment" if evidence_class == "SUPPORT"
            else "contradiction" if evidence_class == "CONTRADICTION"
            else "neutral"
        ),
        "nli_score": 0.9,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.9,
        "source_reliability": 0.9,
        "source_tier": "established_wire",
        "independence_group": group or url.split("/")[2].replace("www.", ""),
        "is_primary_source": True,
        "evidence_score": score,
        "evidence_class": evidence_class,
        "date_status": "match",
        "event_match": "strong",
        "location_match": True,
        "detail_coverage": 0.9,
        "claim_has_location": True,
        "nli_status": "ok",
        "debunking": False,
    }


def signals(**overrides):
    base = {
        "date_status": "match",
        "event_match": "strong",
        "similarity": 0.75,
        "raw_nli_label": "entailment",
        "raw_nli_score": 0.92,
        "nli_status": "ok",
        "location_match": True,
        "claim_has_location": True,
        "detail_coverage": 0.9,
        "debunking": False,
    }
    base.update(overrides)
    return base


# ============================================================
# SOURCE QUALITY AND INDEPENDENCE
# ============================================================

class TestSourceQuality(unittest.TestCase):

    def test_tiers(self):
        self.assertEqual(
            get_source_profile("https://www.reuters.com/world/india/x")["tier"],
            "established_wire",
        )
        self.assertEqual(
            get_source_profile("https://mha.gov.in/press")["tier"],
            "official",
        )
        self.assertEqual(
            get_source_profile("https://en.wikipedia.org/wiki/Fire")["tier"],
            "aggregator_portal",
        )
        self.assertEqual(
            get_source_profile("https://x.wordpress.com/post")["tier"],
            "weak/blog",
        )
        self.assertEqual(
            get_source_profile("https://www.facebook.com/page/posts/1")["tier"],
            "user_generated",
        )

    def test_snippets_never_equal_articles(self):
        article = get_source_reliability("https://www.reuters.com/a")
        snippet = get_source_reliability("https://www.reuters.com/a", "search_snippet")
        self.assertLess(snippet, article)
        self.assertLessEqual(snippet, 0.40)
        self.assertFalse(
            get_source_profile("https://www.reuters.com/a", "search_snippet")["is_primary"]
        )

    def test_weak_sources_are_not_primary(self):
        for url in (
            "https://blog.example.com/fire",
            "https://en.wikipedia.org/wiki/Delhi",
            "https://www.reddit.com/r/news/",
        ):
            self.assertFalse(get_source_profile(url)["is_primary"], url)

    def test_independence_groups(self):
        self.assertEqual(
            registrable_domain("https://www.ndtv.com/x"), "ndtv.com"
        )
        self.assertEqual(
            registrable_domain("https://m.timesofindia.indiatimes.com/x"),
            "indiatimes.com",
        )
        self.assertEqual(
            registrable_domain("https://foo.bar.maharashtra.gov.in/x"),
            "maharashtra.gov.in",
        )
        # Separate blogs are separate voices.
        self.assertNotEqual(
            registrable_domain("https://a.wordpress.com/1"),
            registrable_domain("https://b.wordpress.com/2"),
        )

    def test_extracted_article_text_outranks_snippet(self):
        """
        Provider-extracted article text is real reporting and may corroborate a
        claim; a short search snippet may not. Losing this distinction was the
        main reason a true claim fell back to UNVERIFIED.
        """
        article = get_source_reliability("https://www.reuters.com/a")
        extract = get_source_reliability("https://www.reuters.com/a", "search_extract")
        snippet = get_source_reliability("https://www.reuters.com/a", "search_snippet")

        self.assertLess(extract, article)
        self.assertGreater(extract, snippet)
        self.assertLessEqual(extract, 0.62)
        self.assertLessEqual(snippet, 0.40)

        self.assertTrue(
            get_source_profile("https://www.reuters.com/a", "search_extract")["is_primary"]
        )
        self.assertFalse(
            get_source_profile("https://www.reuters.com/a", "search_snippet")["is_primary"]
        )

    def test_extract_ceiling_is_below_article(self):
        self.assertLessEqual(
            apply_class_ceiling(0.99, "SUPPORT", "search_extract"), 0.80
        )
        self.assertLessEqual(
            apply_class_ceiling(0.99, "SUPPORT", "scraped_article"), 0.95
        )

    def test_syndication_detection(self):
        wire = (
            "Fire officials said the blaze broke out on the top floor of a "
            "commercial building in New Delhi on 12 August 2024. Fire engines "
            "were stationed outside the building while crews searched the premises."
        )
        syndicated = wire.replace("outside the building while", "outside while")
        reworded = wire.replace("Fire officials said", "Officials said").replace(
            "the blaze", "a fire"
        )
        independent = (
            "The fire service confirmed that four engines were dispatched to "
            "the New Delhi incident, where the upper floors were gutted."
        )
        unrelated = (
            "Heavy monsoon rains flooded several parts of the city, disrupting "
            "traffic and forcing schools to shut for the day."
        )
        self.assertTrue(is_near_duplicate(wire, syndicated, SYNDICATION_JACCARD))
        self.assertTrue(is_near_duplicate(wire, reworded, SYNDICATION_JACCARD))
        self.assertFalse(is_near_duplicate(wire, independent, SYNDICATION_JACCARD))
        self.assertFalse(is_near_duplicate(wire, unrelated, SYNDICATION_JACCARD))


# ============================================================
# QUERY PLANNING
# ============================================================

class TestQueryPlanning(unittest.TestCase):

    def test_multiple_angles_for_dated_claim(self):
        plans = plan_queries(FIRE_CLAIM)
        intents = {p["intent"] for p in plans}
        self.assertGreaterEqual(len(plans), 6)
        for required in ("exact", "date", "location", "wording", "factcheck", "official"):
            self.assertIn(required, intents, f"missing {required}")

    def test_date_formats_differ(self):
        plans = plan_queries(FIRE_CLAIM)
        date_queries = [p["query"] for p in plans if p["intent"] == "date"]
        self.assertGreaterEqual(len(date_queries), 1)
        joined = " ".join(date_queries)
        self.assertTrue("12 August 2024" in joined or "August 12, 2024" in joined)

    def test_official_query_uses_general_topic(self):
        plans = plan_queries(FIRE_CLAIM)
        official = [p for p in plans if p["intent"] == "official"]
        self.assertTrue(official)
        self.assertEqual(official[0]["topic"], "general")

    def test_alternative_wording(self):
        plans = plan_queries(FIRE_CLAIM)
        wording = " ".join(
            p["query"] for p in plans if p["intent"] == "wording"
        ).lower()
        self.assertTrue("blaze" in wording or "fire" in wording)

    def test_undated_claim_still_plans(self):
        plans = plan_queries("The Taj Mahal is located in Mumbai.")
        self.assertGreaterEqual(len(plans), 3)


# ============================================================
# RETRIEVAL
# ============================================================

class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.profile = build_claim_profile(FIRE_CLAIM)
        self.date = self.profile["primary_date"]

    def test_listing_pages_rejected(self):
        listing = candidate(
            "Industrial Building for Sale in New Delhi",
            "Buy now. 5000 sq ft. EMI available. Plot area 2000. "
            "Contact seller for price on request.",
            "https://magicbricks.com/property/x",
        )
        self.assertTrue(is_listing_or_index_page(
            listing["title"] + " " + listing["content"], listing["url"]
        ))
        self.assertFalse(
            is_result_relevant(listing, self.profile, None, self.date)[0]
        )

    def test_exact_date_article_admitted_and_ranked_top(self):
        good = candidate(
            "Fire engulfs commercial building in New Delhi",
            "A fire broke out at a commercial building in New Delhi on "
            "12 August 2024, fire officials said. Several fire engines were "
            "dispatched and no casualties were reported.",
            "https://www.reuters.com/india/x",
        )
        signals = check_text_date_match(good["title"] + " " + good["content"], self.date)
        self.assertTrue(signals["exact_date"])
        self.assertTrue(is_result_relevant(good, self.profile, signals, self.date)[0])

        score, debug = score_candidate_relevance(good, self.profile, signals)
        self.assertGreater(score, 1.0)
        self.assertEqual(debug["tier_label"], "T1_exact+loc+event")
        self.assertEqual(retrieval_class({"_debug": debug}), "exact_date")

    def test_different_event_date_rejected(self):
        other = candidate(
            "Fire erupts in timber factory in West Delhi",
            "A blaze broke out at a factory on 3 March 2019 in West Delhi, "
            "officials said four people died.",
            "https://economictimes.indiatimes.com/x",
        )
        signals = check_text_date_match(other["title"] + " " + other["content"], self.date)
        self.assertTrue(signals["different_date"])
        self.assertFalse(is_result_relevant(other, self.profile, signals, self.date)[0])

    def test_social_platform_never_evidence(self):
        social = candidate(
            "Fire in New Delhi",
            "A fire broke out in New Delhi on 12 August 2024, say posts.",
            "https://www.facebook.com/somepage/posts/1",
        )
        self.assertFalse(
            is_result_relevant(social, self.profile, None, self.date)[0]
        )

    def test_no_candidate_dependence_on_one_query(self):
        # The planner is the single source of searches; retrieval must not
        # collapse to a single query.
        self.assertGreaterEqual(len(plan_queries(FIRE_CLAIM)), 6)

    def test_selection_prefers_independent_groups(self):
        profile = build_claim_profile(FIRE_CLAIM)
        cands = []
        for index, (url, score) in enumerate(
            [
                ("https://www.ndtv.com/a", 0.9),
                ("https://m.ndtv.com/b", 0.85),
                ("https://www.bbc.com/c", 0.8),
            ],
            1,
        ):
            cands.append({
                "url": url,
                "domain": registrable_domain(url),
                "independence_group": registrable_domain(url),
                "priority_score": score,
            })
        selected = select_candidates(cands, limit=2)
        groups = [c["independence_group"] for c in selected]
        self.assertEqual(len(groups), len(set(groups)))


# ============================================================
# DATE HANDLING
# ============================================================

class TestDateHandling(unittest.TestCase):

    def setUp(self):
        self.attrs = extract_claim_attributes(FIRE_CLAIM)

    def test_publication_date_is_extracted_from_page_metadata(self):
        from claim_profile import extract_publication_date

        parsed = extract_publication_date(
            "TITLE: Delhi fire\nPUBLICATION DATE: 2024-08-12T08:01:45+05:30\nBody."
        )
        self.assertEqual(
            (parsed["day"], parsed["month"], parsed["year"]), (12, 8, 2024)
        )

    def test_same_day_publication_is_its_own_status(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "Fire officials in New Delhi said crews were working on the scene. "
            "A 2024 study cited earlier work.",
            self.attrs,
            publication_date={"day": 12, "month": 8, "year": 2024},
        )
        self.assertEqual(status["status"], "same_day_published")

    def test_historical_year_does_not_create_a_conflict(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "The IMD predicted more rain. The city last saw such flooding in "
            "1916, the worst on record.",
            self.attrs,
            source_years={2024},
        )
        self.assertNotEqual(status["status"], "different")

    def test_source_from_another_year_is_a_conflict(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "A fire broke out at a commercial building in New Delhi on "
            "3 March 2019.",
            self.attrs,
        )
        self.assertEqual(status["status"], "different")

    def test_aggregate_reporting_is_demoted(self):
        level, _ = evaluate_event_match(
            FIRE_CLAIM,
            "Delhi recorded 214 building fires in the first half of 2024, data "
            "released on 12 August 2024 showed. Most were small, officials said.",
            self.attrs,
            "match",
            0.6,
        )
        self.assertEqual(level, "weak")

    def test_publication_metadata_removed(self):
        text = (
            "Published Date: 2026-03-04\n"
            "A fire broke out in New Delhi on 12 August 2024.\n"
            "Copyright 2026"
        )
        cleaned = strip_publication_boilerplate(text)
        self.assertNotIn("Copyright", cleaned)
        self.assertIn("12 August 2024", cleaned)

    def test_dateline_removed_sentence_kept(self):
        cleaned = strip_publication_boilerplate(
            "NEW DELHI: A fire broke out in a commercial building here."
        )
        self.assertNotIn("NEW DELHI", cleaned)
        self.assertIn("commercial building", cleaned)

    def test_modern_article_about_old_event(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "NEW DELI: A fire broke out at a commercial building in New Delhi "
            "on 12 August 2024, officials said. Published Date: 2026-03-04.",
            self.attrs,
        )
        self.assertEqual(status["status"], "match")

    def test_other_year_reported_as_different(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "A fire broke out at a commercial building in New Delhi on "
            "3 March 2019, officials said four people died.",
            self.attrs,
        )
        self.assertEqual(status["status"], "different")

    def test_same_year_only(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "Delhi recorded 214 commercial building fires in 2024, according "
            "to data released by the state.",
            self.attrs,
        )
        self.assertEqual(status["status"], "same_year")

    def test_explicit_other_day_of_same_year_is_a_conflict(self):
        status = check_date_compatibility(
            FIRE_CLAIM,
            "Delhi recorded 214 commercial building fires in 2024, data "
            "released on 14 May 2024 showed.",
            self.attrs,
        )
        self.assertEqual(status["status"], "different")


# ============================================================
# EVENT MATCHING
# ============================================================

class TestEventMatching(unittest.TestCase):

    def setUp(self):
        self.attrs = extract_claim_attributes(FIRE_CLAIM)

    def match(self, text, similarity=0.6):
        status = check_date_compatibility(FIRE_CLAIM, text, self.attrs)["status"]
        return evaluate_event_match(
            FIRE_CLAIM, text, self.attrs, status, similarity
        )

    def test_matching_event_is_strong(self):
        level, _ = self.match(
            "A fire broke out at a commercial building in New Delhi on "
            "12 August 2024, fire officials said. Fire engines were dispatched."
        )
        self.assertEqual(level, "strong")

    def test_other_city_is_a_conflict(self):
        level, sig = self.match(
            "A fire broke out at a commercial building in Agra on "
            "12 August 2024, fire officials said. Fire engines were dispatched."
        )
        self.assertEqual(level, "none")
        self.assertTrue(sig["location_conflict"])

    def test_stray_capitalised_word_is_not_a_conflict(self):
        # The old heuristic rejected this passage because of the words
        # "Police", "Control" and "Delhi" appearing outside the claim.
        level, sig = self.match(
            "Fire officials in New Delhi said the Delhi Fire Service and the "
            "Control Room responded with eight engines on 12 August 2024. The "
            "commercial building was evacuated."
        )
        self.assertFalse(sig["location_conflict"])
        self.assertIn(level, ("moderate", "strong"))

    def test_policy_text_is_not_evidence(self):
        level, _ = self.match(
            "Safety regulations for commercial buildings in New Delhi were "
            "updated on 12 August 2024, guidelines issued by the state."
        )
        self.assertEqual(level, "none")

    def test_related_statistics_are_weak(self):
        level, _ = self.match(
            "Delhi recorded 214 building fires in the first half of 2024, data "
            "released on 12 August 2024 showed. Most were small."
        )
        self.assertEqual(level, "weak")

    def test_chunk_budget_shrinks_for_long_claims(self):
        short = _resolve_chunk_budget(FIRE_CLAIM)
        long = _resolve_chunk_budget("word " * 400)
        self.assertLessEqual(long, short)
        self.assertGreaterEqual(long, 48)

    def test_duplicate_chunks_collapse(self):
        text = "Fire officials confirmed the blaze started on the top floor. " * 8
        self.assertEqual(len(_dedupe_chunks([text] * 6)), 1)


# ============================================================
# CLASSIFICATION GATES
# ============================================================

class TestClassification(unittest.TestCase):

    def test_extract_mode_can_support_but_snippet_cannot(self):
        extract_profile = {"is_primary": True, "tier": "established_wire",
                           "reliability": 0.62}
        self.assertEqual(
            classify_evidence(signals(), "search_extract", extract_profile), "SUPPORT"
        )
        self.assertNotEqual(
            classify_evidence(signals(), "search_snippet", PRIMARY), "SUPPORT"
        )

    def test_support_requires_exact_date_and_primary_source(self):
        self.assertEqual(
            classify_evidence(signals(), "scraped_article", PRIMARY), "SUPPORT"
        )
        self.assertNotEqual(
            classify_evidence(signals(), "search_snippet", PRIMARY), "SUPPORT"
        )
        self.assertNotEqual(
            classify_evidence(signals(), "scraped_article", WEAK), "SUPPORT"
        )
        self.assertNotEqual(
            classify_evidence(
                signals(date_status="same_year"), "scraped_article", PRIMARY
            ),
            "SUPPORT",
        )

    def test_similarity_alone_never_classifies(self):
        self.assertEqual(
            classify_evidence(
                signals(
                    event_match="weak", detail_coverage=0.05,
                    location_match=False, similarity=0.99,
                    raw_nli_label="neutral", raw_nli_score=0.5,
                ),
                "scraped_article", PRIMARY,
            ),
            "NON_EVIDENCE",
        )

    def test_nli_failure_never_classifies(self):
        self.assertNotEqual(
            classify_evidence(
                signals(nli_status="error"), "scraped_article", PRIMARY
            ),
            "SUPPORT",
        )
        self.assertNotEqual(
            classify_evidence(
                signals(nli_status="unavailable",
                        raw_nli_label="entailment", raw_nli_score=0.99),
                "scraped_article", PRIMARY,
            ),
            "SUPPORT",
        )

    def test_contradiction_needs_date_or_correction(self):
        self.assertEqual(
            classify_evidence(
                signals(raw_nli_label="contradiction", raw_nli_score=0.96),
                "scraped_article", PRIMARY,
            ),
            "CONTRADICTION",
        )
        self.assertNotEqual(
            classify_evidence(
                signals(date_status="different",
                        raw_nli_label="contradiction", raw_nli_score=0.96),
                "scraped_article", PRIMARY,
            ),
            "CONTRADICTION",
        )
        self.assertEqual(
            classify_evidence(
                signals(date_status="different", debunking=True,
                        raw_nli_label="contradiction", raw_nli_score=0.96),
                "scraped_article", PRIMARY,
            ),
            "CONTRADICTION",
        )


# ============================================================
# SCORING
# ============================================================

class TestScoring(unittest.TestCase):

    perfect = dict(
        similarity=0.95, nli_label="entailment", nli_score=0.95,
        source_reliability=0.92, date_status="match", event_match="strong",
        detail_coverage=1.0, location_match=True, claim_has_location=True,
    )

    def test_structure_beats_similarity(self):
        structured = calculate_evidence_score(**self.perfect)
        surface_only = calculate_evidence_score(
            **dict(self.perfect, date_status="different", event_match="none",
                   detail_coverage=0.0, location_match=False)
        )
        self.assertGreater(structured, surface_only * 2)

    def test_neutral_nli_contributes_nothing(self):
        with_neutral = calculate_evidence_score(
            **dict(self.perfect, nli_label="neutral", nli_score=0.95)
        )
        with_entailment = calculate_evidence_score(**self.perfect)
        self.assertLess(with_neutral, with_entailment)

    def test_class_ceilings(self):
        self.assertLessEqual(
            apply_class_ceiling(0.99, "SUPPORT", "scraped_article"), 0.95
        )
        self.assertLessEqual(
            apply_class_ceiling(0.99, "RELATED_CONTEXT", "scraped_article"), 0.25
        )
        self.assertLessEqual(
            apply_class_ceiling(0.99, "SUPPORT", "search_snippet"), 0.45
        )
        self.assertLessEqual(
            apply_class_ceiling(0.99, "NON_EVIDENCE", "scraped_article"), 0.02
        )


# ============================================================
# CORROBORATION
# ============================================================

class TestCorroboration(unittest.TestCase):

    def test_chunks_do_not_multiply_into_sources(self):
        results = [chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9 - i * 0.03)
                   for i in range(6)]
        agreement = calculate_source_agreement(results)
        self.assertEqual(agreement["total_sources"], 1)
        self.assertEqual(agreement["independent_supporting"], 1)
        self.assertEqual(agreement["evidence_sufficiency"], "single_source")

    def test_domains_are_not_publishers(self):
        results = [
            chunk(1, "https://www.ndtv.com/a", "SUPPORT", 0.9),
            chunk(2, "https://m.ndtv.com/b", "SUPPORT", 0.9, group="ndtv.com"),
            chunk(3, "https://www.bbc.com/c", "SUPPORT", 0.9),
        ]
        agreement = calculate_source_agreement(results)
        self.assertEqual(agreement["independence_groups"], 2)
        self.assertEqual(agreement["independent_supporting"], 2)

    def test_independent_support_raises_fusion(self):
        one = [chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9)]
        many = [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
            chunk(3, "https://www.ndtv.com/c", "SUPPORT", 0.85),
        ]
        single = calculate_fusion_score(
            one, calculate_source_agreement(one)
        )
        triple = calculate_fusion_score(
            many, calculate_source_agreement(many)
        )
        self.assertGreater(triple, single)
        self.assertLessEqual(single, 0.62)

    def test_conflict_caps_fusion(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
            chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.9),
            chunk(4, "https://apnews.com/d", "CONTRADICTION", 0.88),
        ]
        agreement = calculate_source_agreement(results)
        self.assertLessEqual(calculate_fusion_score(results, agreement), 0.50)

    def test_no_evidence(self):
        agreement = calculate_source_agreement([])
        self.assertEqual(agreement["evidence_sufficiency"], "none")
        self.assertEqual(calculate_fusion_score([], agreement), 0.0)


# ============================================================
# FINAL VERDICT
# ============================================================

REAL_AGENT = (
    "Verdict:\nREAL\n\nConfidence:\n94%\n\nReasoning:\nConfirmed by sources."
)
FAKE_AGENT = (
    "Verdict:\nFAKE\n\nConfidence:\n88%\n\nReasoning:\nNever happened."
)


class TestFinalVerdict(unittest.TestCase):

    def decide(self, results, verification=REAL_AGENT, critic=REAL_AGENT):
        agreement = calculate_source_agreement(results)
        return generate_final_verdict(
            results, agreement, calculate_fusion_score(results, agreement),
            verification, critic,
        )

    def test_single_source_cannot_be_real(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.reuters.com/b", "SUPPORT", 0.88,
                  group="reuters.com"),
        ]
        verdict = self.decide(results)
        self.assertEqual(verdict["verdict"], "UNVERIFIED")
        self.assertTrue(verdict["verdict_overridden_by_evidence"])

    def test_two_independent_primary_sources_can_be_real(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
        ]
        verdict = self.decide(results)
        self.assertEqual(verdict["verdict"], "REAL")
        self.assertGreater(verdict["confidence"], 50)

    def test_context_only_cannot_be_fake(self):
        results = [
            chunk(1, "https://www.bbc.com/a", "RELATED_CONTEXT", 0.2),
            chunk(2, "https://www.ndtv.com/b", "RELATED_CONTEXT", 0.2),
        ]
        verdict = self.decide(results, FAKE_AGENT, FAKE_AGENT)
        self.assertEqual(verdict["verdict"], "UNVERIFIED")

    def test_contested_evidence_is_unverified(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
            chunk(2, "https://www.bbc.com/b", "SUPPORT", 0.88),
            chunk(3, "https://www.ndtv.com/c", "CONTRADICTION", 0.9),
            chunk(4, "https://apnews.com/d", "CONTRADICTION", 0.88),
        ]
        verdict = self.decide(results)
        self.assertEqual(verdict["verdict"], "UNVERIFIED")

    def test_snippet_only_support_is_unverified(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "RELATED_CONTEXT", 0.3,
                  evidence_type="search_snippet"),
            chunk(2, "https://www.bbc.com/b", "RELATED_CONTEXT", 0.3,
                  evidence_type="search_snippet"),
        ]
        verdict = self.decide(results)
        self.assertEqual(verdict["verdict"], "UNVERIFIED")

    def test_empty_evidence_is_unverified(self):
        verdict = generate_final_verdict(
            [], calculate_source_agreement([]), 0.0, REAL_AGENT, REAL_AGENT
        )
        self.assertEqual(verdict["verdict"], "UNVERIFIED")

    def test_independent_contradiction_can_be_fake(self):
        results = [
            chunk(1, "https://www.reuters.com/a", "CONTRADICTION", 0.9),
            chunk(2, "https://www.bbc.com/b", "CONTRADICTION", 0.88),
        ]
        verdict = self.decide(results, FAKE_AGENT, FAKE_AGENT)
        self.assertEqual(verdict["verdict"], "FAKE")

    def test_gate_is_permissive_only_narrowing(self):
        gate = deterministic_gate(
            {
                "independent_supporting": 3,
                "independent_contradicting": 0,
                "primary_supporting_groups": 2,
            },
            [],
        )
        self.assertEqual(gate["permitted_verdicts"],
                         {"REAL", "MISLEADING", "UNVERIFIED"})

    def test_parsers(self):
        self.assertEqual(
            extract_verdict("Recommended Verdict: UNVERIFIED"), "UNVERIFIED"
        )
        self.assertEqual(
            extract_confidence("Recommended Confidence: 62%"), 62.0
        )
        self.assertIsNone(extract_verdict(""))
        self.assertIsNone(extract_confidence(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
