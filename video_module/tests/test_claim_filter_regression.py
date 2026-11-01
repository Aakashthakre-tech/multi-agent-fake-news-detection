"""
video_module/tests/test_claim_filter_regression.py
==================================================
Regression tests for the video factual-claim filter.

Background
----------
A previous real run (saved artifacts in
``video_module/outputs/tmpx191xwhp``) produced 117 extracted frames, 36
Whisper segments, OCR text in 11 of 15 sampled frames, 32 candidate claims
and ZERO claims that passed factual-claim filtering, so web research was
skipped and the verdict was INSUFFICIENT_VERIFIABLE_CLAIMS.

Replaying the current code against those saved artifacts reproduces the same
outcome exactly. The rejection was CORRECT: the Whisper ``base`` model
hallucinated multilingual gibberish on that audio track
("рок melodies", "아무래 %", "Vakrila Boz清楚"), so there were no sentences
and therefore no factual claims to extract. The only on-screen text was a
date lower-third ("25 March 2023") which has no predicate and is not a
verifiable assertion.

These tests lock in that correct behaviour, and simultaneously prove the
filter still recovers genuine factual claims from speech and from OCR
chyrons - so a future change cannot silently break either direction.
"""

import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_module.claim_ranker import (  # noqa: E402
    detect_predicate_and_verbs,
    evaluate_candidate_claim,
    is_non_factual_or_filler,
)
from video_module.video_pipeline import (  # noqa: E402
    fuse_whisper_and_ocr,
    prepare_candidate_claims,
)

REAL_RUN = PROJECT_ROOT / "video_module" / "outputs" / "tmpx191xwhp"


class TestClaimFilterRegression(unittest.TestCase):

    # -----------------------------------------------------------------
    # The real failure pattern
    # -----------------------------------------------------------------

    def test_01_real_garbled_run_still_yields_zero_verifiable_claims(self):
        """
        The saved run must keep producing zero verifiable claims.

        This is the regression guard for the original investigation: the
        rejection was justified, and it must stay justified.
        """
        if not REAL_RUN.exists():
            self.skipTest("saved video artifacts not present")

        transcript = json.loads((REAL_RUN / "transcript.json").read_text(encoding="utf-8"))
        ocr = json.loads((REAL_RUN / "ocr.json").read_text(encoding="utf-8"))
        if isinstance(transcript, list):
            transcript = {"segments": transcript}
        if isinstance(ocr, list):
            ocr = {"observations": ocr}

        fusion = fuse_whisper_and_ocr(transcript, ocr)
        ranked = prepare_candidate_claims(
            fusion, video_metadata={"file_name": "tmpx191xwhp.mp4", "duration_sec": 213.24}
        )

        claims = ranked.get("candidate_claims", [])
        self.assertEqual(len(claims), 32, "the saved run produced 32 candidate claims")
        self.assertEqual(
            ranked.get("selected_claims_count", 0),
            0,
            "the garbled transcript contains no verifiable claim, so none may be accepted",
        )

        # Every rejection must carry a reason, so the report can explain it.
        for claim in claims:
            self.assertTrue(claim.get("selection_reason"))
            self.assertEqual(claim.get("selection_status"), "rejected")

        # The saved artifact must agree with the replayed result.
        saved = json.loads(
            (REAL_RUN / "candidate_claims.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved.get("candidate_claims_count"), len(claims))
        self.assertEqual(saved.get("selected_claims_count"), 0)

    def test_02_garbled_multilingual_segments_are_rejected(self):
        """
        Hallucinated multilingual output must never be treated as a claim.
        """
        garbled = [
            "рок melodies",
            "아무래 %",
            "Vakrila Boz清楚",
            "seiner Brushpflicht",
            "Norðiði þriðja þjóð",
        ]
        for text in garbled:
            with self.subTest(text=text):
                self.assertTrue(
                    is_non_factual_or_filler(text),
                    f"gibberish must be classified as non-factual: {text!r}",
                )

    def test_03_date_only_chyron_is_not_a_claim(self):
        """
        The run's only real on-screen text was a date lower-third.

        A bare date has no predicate, so it is not a verifiable assertion and
        must not become a claim.
        """
        for chyron in ("25 March 2023", "Suhana 25 March 2023", "abp Hf"):
            with self.subTest(chyron=chyron):
                has_predicate, verbs = detect_predicate_and_verbs(chyron)
                self.assertFalse(
                    has_predicate,
                    f"a date-only chyron must not expose a predicate: {chyron!r}",
                )

    # -----------------------------------------------------------------
    # The filter must still recover genuine claims
    # -----------------------------------------------------------------

    def test_04_real_ocr_chyron_with_a_predicate_becomes_a_claim(self):
        """
        A chyron that states something verifiable MUST be recovered.

        This is the case the later BREAKING run exercised: the lower third
        "BREAKING R VIDEO Hespoke about Asim Munir and his absurd speech to his
        soldiers | BREAKING V18/10/25 R" carries a predicate, so it must
        survive the filter.
        """
        chyron = (
            "BREAKING R VIDEO Hespoke about Asim Munir and his absurd speech "
            "to his soldiers | BREAKING V18/10/25 R"
        )
        result = evaluate_candidate_claim(chyron, source="ocr")
        self.assertTrue(result["is_factual"], result["selection_reason"])
        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["source_provenance"], "ocr")

    def test_05_real_sentence_from_speech_becomes_a_claim(self):
        """A normal factual sentence in speech must be accepted."""
        sentence = (
            "The Prime Minister announced the new policy on 12 March 2024 "
            "in Mumbai."
        )
        result = evaluate_candidate_claim(sentence, source="whisper")
        self.assertTrue(result["is_factual"], result["selection_reason"])
        self.assertEqual(result["selection_status"], "selected")
        self.assertTrue(result["has_predicate"])

    def test_06_marathi_factual_sentence_becomes_a_claim(self):
        """Devanagari claims must be recognised, not filtered out as noise."""
        sentence = "राज्य सरकार ने 25 मार्च 2023 को नई नीति की घोषणा की।"
        result = evaluate_candidate_claim(sentence, source="whisper")
        self.assertTrue(
            result["has_predicate"],
            "Devanagari predicate verbs must be detected",
        )
        self.assertTrue(result["is_factual"], result["selection_reason"])

    def test_07_filler_greetings_and_apologies_are_still_rejected(self):
        """Filler, greetings and apologies must never become claims."""
        noise = [
            "Hello everyone, welcome to the channel.",
            "Sorry, my microphone was not working.",
            "Thank you very much, and we will see you in the next video.",
            "um, uh, you know, like, I mean...",
            "Please subscribe to my channel and hit the bell icon.",
        ]
        for text in noise:
            with self.subTest(text=text):
                result = evaluate_candidate_claim(text, source="whisper")
                self.assertFalse(
                    result["is_factual"],
                    f"filler must not become a claim: {text!r} -> {result['selection_reason']}",
                )
                self.assertEqual(result["selection_status"], "rejected")

    def test_08_ocr_claim_recovered_when_speech_is_irrelevant(self):
        """
        OCR evidence must not be discarded when the speech is irrelevant.

        Whisper returns noise for the whole track; the chyron carries the only
        verifiable statement. The fused transcript must surface it and the
        ranker must turn it into a candidate claim.
        """
        transcript = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 4.0, "text": "um uh you know"},
                {"id": 1, "start": 4.0, "end": 8.0, "text": "yeah right so anyway"},
            ]
        }
        ocr = {
            "observations": [
                {
                    "frame_index": 3,
                    "timestamp_sec": 5.0,
                    "text": "PM Modi announced the new policy on 12 March 2024",
                    "confidence": 0.91,
                }
            ]
        }

        fusion = fuse_whisper_and_ocr(transcript, ocr)
        ranked = prepare_candidate_claims(
            fusion, video_metadata={"file_name": "noise.mp4", "duration_sec": 8.0}
        )

        claims = ranked.get("candidate_claims", [])
        self.assertTrue(claims, "OCR evidence must still produce candidates")
        self.assertTrue(
            any(c.get("source_provenance") == "ocr" for c in claims),
            "the OCR-derived claim must not be dropped because the speech was filler",
        )
        self.assertGreaterEqual(ranked.get("selected_claims_count", 0), 1)

    # -----------------------------------------------------------------
    # Reason accuracy
    # -----------------------------------------------------------------

    def test_09_rejection_reason_names_the_real_blocker(self):
        """
        A claim with entities and a date but no predicate must not be rejected
        with a reason claiming it lacked entities and dates.

        The old reason ("lacks specific entities, dates, or verifiable event
        detail") misdescribed the evidence: the statement did have entities
        and dates; what it lacked was an assertion verb.
        """
        result = evaluate_candidate_claim("Suhana 25 March 2023 abp Hf", source="ocr")

        self.assertEqual(result["selection_status"], "rejected")
        self.assertNotIn(
            "lacks specific entities",
            result["selection_reason"],
            "the rejection reason must not claim the claim lacked anchors it had",
        )
        self.assertIn(
            "predicate",
            result["selection_reason"].lower(),
            "the rejection reason must name the predicate that was missing",
        )

    def test_10_selection_reason_names_a_verb_from_the_reported_category(self):
        """
        The reported verb must be the one that matched the category.

        ``verbs_found`` is in token order, so its first element was not
        necessarily the verb that produced the "+0.25 incident verb" score.
        """
        result = evaluate_candidate_claim(
            "Police reported that two people were injured during the protest in Delhi.",
            source="whisper",
        )
        self.assertEqual(result["selection_status"], "selected")
        self.assertIn("event action", result["selection_reason"])

        from video_module.claim_ranker import _INCIDENT_VERBS

        match = re.search(r"event action \('([^']+)'\)", result["selection_reason"])
        self.assertIsNotNone(match)
        self.assertIn(match.group(1), _INCIDENT_VERBS)

    # -----------------------------------------------------------------
    # Script fairness
    # -----------------------------------------------------------------

    def test_11_devanagari_entities_are_recognised(self):
        """
        Devanagari has no capitalisation, so the Title-Case entity rule never
        matched a Marathi or Hindi statement and it ended up with no entities
        at all.
        """
        from video_module.claim_ranker import extract_potential_entities

        entities = extract_potential_entities(
            "राज्य सरकार ने 25 मार्च 2023 को नई नीति की घोषणा की।"
        )
        self.assertTrue(entities, "Devanagari words must be recognised as entities")
        self.assertIn("राज्य", entities)

    def test_12_marathi_factual_claim_is_selected(self):
        """A real Marathi statement must survive the filter."""
        result = evaluate_candidate_claim(
            "राज्य सरकार ने 25 मार्च 2023 को नई नीति की घोषणा की।",
            source="whisper",
        )
        self.assertTrue(result["is_factual"], result["selection_reason"])
        self.assertEqual(result["selection_status"], "selected")

    def test_13_devanagari_filler_is_still_rejected(self):
        """
        Recognising Devanagari entities must not turn Devanagari noise into
        claims.
        """
        for text in (
            "नमस्ते दोस्तों, चैनल पर स्वागत है",
            "माफ कीजिए, मेरा माइक्रोफोन काम नहीं कर रहा था",
            "धन्यवाद, अगले वीडियो में मिलेंगे",
        ):
            with self.subTest(text=text):
                result = evaluate_candidate_claim(text, source="whisper")
                self.assertNotEqual(
                    result["selection_status"],
                    "selected",
                    f"Devanagari filler must not become a claim: {text!r}",
                )


if __name__ == "__main__":
    unittest.main()
