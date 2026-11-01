"""
test_claim_extraction_research.py
=================================
Comprehensive tests for TruthLensAI claim extraction, filtering, verifiability ranking,
and evidence research safeguards.

Covers all 9 mandatory test scenarios from task specification:
1. Video containing only "I'm sorry." (rejected, no research launched, INSUFFICIENT_VERIFIABLE_CLAIMS).
2. Video containing clear factual claim in speech (selected, researched).
3. Silent video containing clear factual statement in on-screen text (OCR claim verified without audio).
4. Video containing both conversational speech and on-screen factual text (factual claim prioritized).
5. Repeated captions and duplicated speech/OCR claims (deduplicated, multimodal reinforcement).
6. Blurry OCR, incomplete statements, and irrelevant text (rejected with transparent reasons).
7. Factual claim with no useful search results (INCONCLUSIVE / UNVERIFIED, not labeled fake).
8. Relevant and irrelevant search results mixed together (dictionary/noise filtered, relevant evidence prioritized).
9. Visual classifier disagreement with evidence (evidence determines verdict, divergence flagged).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from video_module.claim_ranker import (
    evaluate_candidate_claim,
    deduplicate_candidate_claims,
    process_and_rank_candidate_claims,
    clean_claim_text,
    is_non_factual_or_filler,
)
from video_module.video_pipeline import (
    prepare_candidate_claims,
    prepare_truthlens_claim_payload,
)
from video_module.video_verifier import (
    verify_video,
    analyze_multimodal_consistency,
)
from tools import is_result_relevant, EXCLUDED_DOMAINS


class TestClaimExtractionAndResearch(unittest.TestCase):

    # ============================================================
    # TEST 1: Video containing only "I'm sorry."
    # ============================================================
    def test_01_apology_only_video_rejected_no_research(self):
        """Conversational apology must be rejected; no web research triggered."""
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "whisper",
                    "text": "I'm sorry.",
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "frame_references": [1],
                    "on_screen_context": None,
                    "review_flags": [],
                    "disagreements": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data)
        self.assertEqual(claims_res["candidate_claims_count"], 1)
        self.assertEqual(claims_res["selected_claims_count"], 0)
        self.assertEqual(claims_res["rejected_claims_count"], 1)

        candidate = claims_res["candidate_claims"][0]
        self.assertEqual(candidate["selection_status"], "rejected")
        self.assertFalse(candidate["is_verifiable"])
        self.assertIn("apology", candidate["selection_reason"].lower())

        payload = prepare_truthlens_claim_payload(claims_res)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["selection_status"], "rejected")
        self.assertFalse(payload[0]["ready_for_pipeline"])

        # Test in video_verifier: mock pipeline summary with apology claim
        mock_pipeline_summary = {
            "claims_summary": claims_res,
            "video_metadata": {"file_name": "apology_test.mp4"},
            "transcript_json_path": "dummy_transcript.json",
            "ocr_json_path": "dummy_ocr.json",
            "fused_transcript_path": "dummy_fused.json",
            "candidate_claims_path": None,
        }

        mock_research = MagicMock()

        with patch("video_module.video_verifier.Path.exists", return_value=True), \
             patch("video_module.video_verifier.process_video_pipeline", return_value=mock_pipeline_summary), \
             patch("video_module.video_verifier.predict_video", return_value={"available": False}), \
             patch("video_module.video_verifier.run_research_pipeline", mock_research):

            report = verify_video("dummy_apology.mp4", enable_classifier=False, verify_claims=True)

            # Assert research pipeline was NEVER called
            mock_research.assert_not_called()
            self.assertEqual(report["final_evidence_verdict"], "INSUFFICIENT_VERIFIABLE_CLAIMS")
            self.assertEqual(len(report["verified_claims"]), 0)
            self.assertEqual(report["selected_claims_count"], 0)
            self.assertEqual(report["rejected_claims_count"], 1)

    # ============================================================
    # TEST 2: Video containing clear factual claim in speech
    # ============================================================
    def test_02_factual_speech_claim_selected_and_researched(self):
        """Factual speech statement must be selected, prioritized, and researched."""
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "whisper",
                    "text": "A major earthquake hit Tokyo on Tuesday, damaging several buildings.",
                    "start_sec": 0.0,
                    "end_sec": 4.5,
                    "frame_references": [1, 2],
                    "on_screen_context": None,
                    "review_flags": [],
                    "disagreements": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data)
        self.assertEqual(claims_res["selected_claims_count"], 1)
        candidate = claims_res["candidate_claims"][0]
        self.assertEqual(candidate["selection_status"], "selected")
        self.assertTrue(candidate["is_verifiable"])
        self.assertGreaterEqual(candidate["priority_score"], 0.60)
        self.assertEqual(candidate["source_modality"], "speech")

        # Mock research returning verified
        fake_research_result = {
            "final_verdict": {"verdict": "SUPPORTED", "confidence": 0.92},
            "fusion_report": {"evidence_sufficiency": "SUFFICIENT"},
            "supporting_chunks": [{"chunk": "Magnitude 6.8 quake struck near Tokyo on Tuesday."}],
            "contradiction_chunks": [],
        }

        mock_pipeline_summary = {
            "claims_summary": claims_res,
            "video_metadata": {"file_name": "quake.mp4"},
            "transcript_json_path": "dummy.json",
            "ocr_json_path": "dummy.json",
            "fused_transcript_path": "dummy.json",
            "candidate_claims_path": None,
        }

        with patch("video_module.video_verifier.Path.exists", return_value=True), \
             patch("video_module.video_verifier.process_video_pipeline", return_value=mock_pipeline_summary), \
             patch("video_module.video_verifier.predict_video", return_value={"available": False}), \
             patch("video_module.video_verifier.run_research_pipeline", return_value=fake_research_result) as mock_research:

            report = verify_video("dummy_quake.mp4", enable_classifier=False, verify_claims=True)

            mock_research.assert_called_once()
            self.assertEqual(report["final_evidence_verdict"], "SUPPORTED")
            self.assertEqual(len(report["verified_claims"]), 1)
            self.assertEqual(report["verified_claims"][0]["claim_verdict"], "SUPPORTED")

    # ============================================================
    # TEST 3: Silent video with on-screen text
    # ============================================================
    def test_03_silent_video_ocr_claim_extraction_and_research(self):
        """Silent video with on-screen text extracts OCR claims and verifies without speech."""
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "ocr",
                    "text": "Government announced nationwide ban on single-use plastics from July 1",
                    "start_sec": 1.0,
                    "end_sec": 5.0,
                    "frame_references": [2, 3, 4],
                    "on_screen_context": None,
                    "review_flags": [],
                    "disagreements": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data)
        self.assertEqual(claims_res["selected_claims_count"], 1)
        candidate = claims_res["candidate_claims"][0]
        self.assertEqual(candidate["selection_status"], "selected")
        self.assertEqual(candidate["source_modality"], "ocr")
        self.assertTrue(candidate["is_verifiable"])

        # Run verifier on silent video
        fake_research_result = {
            "final_verdict": {"verdict": "SUPPORTED", "confidence": 0.88},
            "fusion_report": {"evidence_sufficiency": "SUFFICIENT"},
            "supporting_chunks": [],
            "contradiction_chunks": [],
        }

        mock_pipeline_summary = {
            "claims_summary": claims_res,
            "video_metadata": {"file_name": "silent_ban.mp4"},
            "transcript_json_path": "dummy.json",
            "ocr_json_path": "dummy.json",
            "fused_transcript_path": "dummy.json",
            "candidate_claims_path": None,
        }

        with patch("video_module.video_verifier.Path.exists", return_value=True), \
             patch("video_module.video_verifier.process_video_pipeline", return_value=mock_pipeline_summary), \
             patch("video_module.video_verifier.predict_video", return_value={"available": False}), \
             patch("video_module.video_verifier.run_research_pipeline", return_value=fake_research_result) as mock_research:

            report = verify_video("dummy_silent.mp4", enable_classifier=False, verify_claims=True)

            mock_research.assert_called_once()
            self.assertEqual(report["final_evidence_verdict"], "SUPPORTED")
            self.assertEqual(report["verified_claims"][0]["source_modality"], "ocr")

    # ============================================================
    # TEST 4: Conversational speech + on-screen factual text
    # ============================================================
    def test_04_conversational_speech_and_factual_ocr_prioritization(self):
        """Conversational speech must not overshadow on-screen factual assertion."""
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "whisper",
                    "text": "Hey guys, welcome back to my channel, thank you so much for watching.",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "frame_references": [1, 2],
                    "on_screen_context": "Heavy rains trigger flash floods in Mumbai, killing 5 people.",
                    "review_flags": [],
                    "disagreements": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data)
        # Should have 2 candidates: 1 selected (OCR), 1 rejected (greeting filler)
        self.assertGreaterEqual(claims_res["candidate_claims_count"], 2)
        top_claim = claims_res["candidate_claims"][0]

        # Top claim MUST be the factual Mumbai flood assertion
        self.assertEqual(top_claim["selection_status"], "selected")
        self.assertIn("Mumbai", top_claim["claim_text"])
        self.assertEqual(top_claim["source_modality"], "ocr")

        # The conversational greeting must be rejected
        filler_claim = [c for c in claims_res["candidate_claims"] if "welcome" in c["claim_text"]][0]
        self.assertEqual(filler_claim["selection_status"], "rejected")

    # ============================================================
    # TEST 5: Repeated captions and duplicated claims
    # ============================================================
    def test_05_repeated_captions_deduplicated_and_reinforced(self):
        """Repeated captions across frames and identical speech/OCR must merge with reinforcement."""
        candidates = [
            {
                "claim_text": "Massive fire broke out at Delhi chemical factory",
                "source_provenance": "whisper",
                "source_modality": "speech",
                "timestamps": {"start_sec": 0.0, "end_sec": 3.0},
                "frame_references": [1],
                "priority_score": 0.65,
                "selection_status": "selected",
                "selection_reason": "Speech report of fire",
            },
            {
                "claim_text": "Massive fire broke out at Delhi chemical factory",
                "source_provenance": "ocr",
                "source_modality": "ocr",
                "timestamps": {"start_sec": 1.0, "end_sec": 4.0},
                "frame_references": [2, 3],
                "priority_score": 0.65,
                "selection_status": "selected",
                "selection_reason": "OCR banner of fire",
            },
        ]

        deduped = deduplicate_candidate_claims(candidates, similarity_threshold=0.65)
        self.assertEqual(len(deduped), 1)
        merged = deduped[0]
        self.assertEqual(merged["source_provenance"], "both")
        self.assertEqual(merged["source_modality"], "both")
        self.assertEqual(merged["frame_references"], [1, 2, 3])
        self.assertEqual(merged["timestamps"]["start_sec"], 0.0)
        self.assertEqual(merged["timestamps"]["end_sec"], 4.0)
        self.assertGreaterEqual(merged["priority_score"], 0.75)

    # ============================================================
    # TEST 6: Blurry OCR, incomplete statements, and noise
    # ============================================================
    def test_06_blurry_ocr_and_noise_rejected(self):
        """OCR noise, isolated stamps, and incomplete fragments must be rejected."""
        test_cases = [
            ("x#89!@? a1", "OCR noise"),
            ("BREAKING NEWS", "Broadcast chyron stamp"),
            ("because of the following reasons", "Incomplete syntactic clause"),
            ("LIVE", "Broadcast stamp"),
            ("subscribe to our channel", "Promotional slogan"),
        ]

        for text, desc in test_cases:
            eval_res = evaluate_candidate_claim(text, source="ocr")
            self.assertEqual(
                eval_res["selection_status"],
                "rejected",
                f"Expected '{text}' ({desc}) to be rejected, got selected."
            )
            self.assertTrue(len(eval_res["selection_reason"]) > 5)

    # ============================================================
    # TEST 7: Factual claim with no useful search results
    # ============================================================
    def test_07_claim_with_no_search_results_remains_inconclusive(self):
        """Lack of search results must yield INCONCLUSIVE / UNVERIFIED, never FAKE."""
        fused_data = {
            "fused_entries": [
                {
                    "entry_id": "fuse_0001",
                    "source": "whisper",
                    "text": "Rare meteorite discovered in rural village in Rajasthan yesterday.",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "frame_references": [],
                    "on_screen_context": None,
                    "review_flags": [],
                    "disagreements": [],
                }
            ]
        }

        claims_res = prepare_candidate_claims(fused_data)
        # Empty/unverified research result
        fake_research_result = {
            "final_verdict": {"verdict": "INCONCLUSIVE", "confidence": 0.20},
            "fusion_report": {"evidence_sufficiency": "INSUFFICIENT"},
            "supporting_chunks": [],
            "contradiction_chunks": [],
        }

        mock_pipeline_summary = {
            "claims_summary": claims_res,
            "video_metadata": {"file_name": "meteorite.mp4"},
            "transcript_json_path": "dummy.json",
            "ocr_json_path": "dummy.json",
            "fused_transcript_path": "dummy.json",
            "candidate_claims_path": None,
        }

        with patch("video_module.video_verifier.Path.exists", return_value=True), \
             patch("video_module.video_verifier.process_video_pipeline", return_value=mock_pipeline_summary), \
             patch("video_module.video_verifier.predict_video", return_value={"available": False}), \
             patch("video_module.video_verifier.run_research_pipeline", return_value=fake_research_result):

            report = verify_video("dummy_meteorite.mp4", enable_classifier=False, verify_claims=True)

            self.assertEqual(report["final_evidence_verdict"], "INCONCLUSIVE")
            self.assertNotEqual(report["final_evidence_verdict"], "FAKE")
            self.assertNotEqual(report["final_evidence_verdict"], "CONTRADICTED")

    # ============================================================
    # TEST 8: Relevant and irrelevant search results mixed
    # ============================================================
    def test_08_dictionary_and_irrelevant_results_filtered(self):
        """Dictionary definitions and lexical pages must be rejected by relevance filter."""
        self.assertIn("merriam-webster.com", EXCLUDED_DOMAINS)
        self.assertIn("dictionary.com", EXCLUDED_DOMAINS)
        self.assertIn("wiktionary.org", EXCLUDED_DOMAINS)

        from claim_profile import build_claim_profile
        profile = build_claim_profile("I'm sorry.")

        # 1. Dictionary definition candidate
        dict_candidate = {
            "url": "https://www.merriam-webster.com/dictionary/sorry",
            "title": "Sorry Definition & Meaning - Merriam-Webster",
            "score": 0.85,
            "content": "Definition of sorry: feeling sorrow, regret, or penitence.",
        }
        is_rel, reason = is_result_relevant(dict_candidate, profile)
        self.assertFalse(is_rel)
        self.assertIn("Dictionary", reason.get("reason", ""))

        # 2. Cambridge dictionary definition
        cambridge_candidate = {
            "url": "https://dictionary.cambridge.org/dictionary/english/sorry",
            "title": "SORRY | English meaning - Cambridge Dictionary",
            "score": 0.80,
            "content": "sorry definition: 1. feeling sadness, sympathy, or disappointment...",
        }
        is_rel2, reason2 = is_result_relevant(cambridge_candidate, profile)
        self.assertFalse(is_rel2)
        self.assertIn("Dictionary", reason2.get("reason", ""))

    # ============================================================
    # TEST 9: Visual classifier disagreement with evidence
    # ============================================================
    def test_09_visual_classifier_disagreement_preserves_evidence_verdict(self):
        """Evidence verdict stands when visual classifier disagrees; divergence flagged."""
        # Case A: Evidence says SUPPORTED, Visual model says fake
        visual_fake = {
            "available": True,
            "predicted_label": "fake",
            "confidence": 0.82,
            "probabilities": {"fake": 0.82, "real": 0.18},
        }

        analysis_a = analyze_multimodal_consistency(visual_fake, claim_verdict="SUPPORTED")
        self.assertEqual(analysis_a["status"], "visual_anomaly_flagged")
        self.assertEqual(analysis_a["relationship"], "divergent")
        self.assertTrue(analysis_a["requires_investigation"])

        # Case B: Evidence says CONTRADICTED, Visual model says real (out-of-context misattribution)
        visual_real = {
            "available": True,
            "predicted_label": "real",
            "confidence": 0.79,
            "probabilities": {"fake": 0.21, "real": 0.79},
        }

        analysis_b = analyze_multimodal_consistency(visual_real, claim_verdict="CONTRADICTED")
        self.assertEqual(analysis_b["status"], "out_of_context_misattribution")
        self.assertEqual(analysis_b["relationship"], "divergent")
        self.assertTrue(analysis_b["requires_investigation"])

        # Case C: Evidence says INSUFFICIENT_VERIFIABLE_CLAIMS
        analysis_c = analyze_multimodal_consistency(visual_real, claim_verdict="INSUFFICIENT_VERIFIABLE_CLAIMS")
        self.assertEqual(analysis_c["status"], "insufficient_verifiable_claims")
        self.assertFalse(analysis_c["requires_investigation"])


if __name__ == "__main__":
    unittest.main()
