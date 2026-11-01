from image_module.ocr import extract_text
from image_module.visual_analysis import analyze_image
from image_module.ocr_cleaner import clean_ocr_results
from image_module.claim_extractor import extract_image_claim
from image_module.claim_normalizer import normalize_image_claim

from pipeline import run_research_pipeline

from ai_image_signal import (
    get_ai_generation_signal,
    describe_signal_consistency,
    print_ai_generation_signal
)


def run_image_pipeline(image_path):

    print("\n======================================")
    print("IMAGE VERIFICATION PIPELINE")
    print("======================================")

    # ==================================
    # STEP 1 - OCR
    # ==================================

    print("\nSTEP 1 - Extracting text from image...")

    ocr_results = extract_text(image_path)

    cleaned_results = clean_ocr_results(
        ocr_results
    )

    print("OCR extraction complete.")


    # ==================================
    # STEP 2 - VISUAL ANALYSIS
    # ==================================

    print("\nSTEP 2 - Analyzing image...")

    detected_objects = analyze_image(
        image_path
    )

    print("Detected objects:")

    for obj in detected_objects:
        print(
            obj["object"],
            "->",
            obj["confidence"]
        )


    # ==================================
    # STEP 3 - IMAGE CLAIM EXTRACTION
    # ==================================

    print("\nSTEP 3 - Extracting image claim...")

    image_claim = extract_image_claim(
        cleaned_results,
        detected_objects
    )

    print("Image claim extracted.")


    # ==================================
    # STEP 4 - CLAIM NORMALIZATION
    # ==================================

    print("\nSTEP 4 - Normalizing claim...")

    normalized_claim = normalize_image_claim(
        image_claim["ocr_text"],
        image_claim["detected_objects"],
        image_claim.get("headline_text")
    )

    claim = normalized_claim.get("claim")

    if not claim:
        print("\nNo verifiable factual claim or headline extracted from image.")
        print("Factual research skipped (no verifiable claim to research).")
        verification_result = {
            "final_verdict": {
                "verdict": "INSUFFICIENT_VERIFIABLE_CLAIMS",
                "confidence": None,
                "evidence_sufficiency": "INSUFFICIENT",
                "reasoning": (
                    "No verifiable headline, on-screen text, or factual claim could be extracted "
                    "from the image. Visual authenticity is evaluated independently via forensic models."
                ),
                "total_sources": 0,
                "supporting_sources": 0,
                "contradicting_sources": 0,
                "independent_supporting": 0,
                "independent_contradicting": 0,
            },
            "evidence_package": {},
            "evidence_brief": "",
            "verification_agent_analysis": "",
            "critic_agent_analysis": "",
        }
    else:
        print("\nNORMALIZED CLAIM:")
        print(claim)

        # ==================================
        # STEP 5 - EXISTING RESEARCH PIPELINE
        # ==================================

        print("\nSTEP 5 - Sending claim to Evidence Engine...")

        verification_result = run_research_pipeline(
            claim,
            extracted_text=_build_image_extracted_text(
                cleaned_results, image_claim, normalized_claim
            ),
        )


    # ==================================
    # STEP 6 - AI IMAGE ORIGIN SIGNAL
    # ==================================
    # Separate signal. It is not passed to the evidence engine, the
    # verification agent, the critic, or the final verdict, and it cannot
    # change the verdict produced above.

    print("\nSTEP 6 - Checking image origin...")

    ai_generation = get_ai_generation_signal(
        image_path
    )

    ai_generation_note = describe_signal_consistency(
        ai_generation,
        {"verification": verification_result}
    )

    print_ai_generation_signal(
        ai_generation,
        ai_generation_note
    )

    return {
        "image_path": image_path,
        "ocr_results": ocr_results,
        "detected_objects": detected_objects,
        "image_claim": image_claim,
        "normalized_claim": normalized_claim,
        "verification": verification_result,
        "ai_generation": ai_generation,
        "ai_generation_note": ai_generation_note
    }


def _build_image_extracted_text(
    ocr_results: list,
    image_claim: dict,
    normalized_claim: dict,
) -> str:
    """
    Build a concise extracted-text representation for an image:
    OCR text, the extracted image claim, and the normalized claim.
    No OCR is re-run here; only already-collected outputs are reused.
    """
    parts = []

    ocr_texts = []
    for result in (ocr_results or []):
        text = (result or {}).get("text")
        if text:
            ocr_texts.append(str(text).strip())
    if ocr_texts:
        parts.append("OCR text: " + " | ".join(ocr_texts))

    image_claim_text = (image_claim or {}).get("claim") or (image_claim or {}).get("ocr_text")
    if image_claim_text:
        parts.append("Extracted claim: " + str(image_claim_text).strip())

    normalized_claim_text = (normalized_claim or {}).get("claim")
    if normalized_claim_text and normalized_claim_text != image_claim_text:
        parts.append("Normalized claim: " + str(normalized_claim_text).strip())

    return " ".join(parts)


if __name__ == "__main__":

    image_path = "image_module/mumbai.png"

    result = run_image_pipeline(
        image_path
    )

    print("\n======================================")
    print("IMAGE VERIFICATION COMPLETE")
    print("======================================")

    print("\nFinal Verdict:")

    print(
        result["verification"]["final_verdict"]
    )

    print("\nAI Image Origin Signal (separate):")

    print(
        result["ai_generation"].get("verdict", "UNAVAILABLE"),
        "-",
        result["ai_generation"].get("ai_probability", "n/a"),
        "% AI"
    )