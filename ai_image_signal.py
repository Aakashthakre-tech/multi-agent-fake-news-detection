"""
ai_image_signal.py
==================
Additive layer that runs the already-tested AI image detection module
(ai_image_detection/detector.py) as a SEPARATE signal next to the existing
image verification pipeline.

Design constraints honoured here:

  * No existing file is modified. image_pipeline.run_image_pipeline is
    imported and called unchanged, and its result is forwarded verbatim.
  * The AI-generation finding is a separate signal. It is NOT passed into the
    evidence analyzer, evidence scorer, evidence fusion, verification agent,
    critic agent, evidence brief, or the deterministic final verdict, and it
    does not change any threshold, prompt, or existing verdict.
  * The existing verdict is never overridden. Agreement or conflict between
    the two signals is reported descriptively, for a human to read.
  * A detector failure (missing weights, unreadable file, torch error) can
    never take down the existing pipeline: the signal reports itself as
    unavailable and the pipeline result is still returned.
"""

from ai_image_detection.detector import (
    AIImageDetector,
    MODEL_NAME,
    TEMPERATURE,
    THRESHOLD,
    UNCERTAIN_BAND,
)


def _operating_point() -> dict:
    """The detector's frozen operating point, as actually configured."""
    return {
        "threshold": THRESHOLD,
        "uncertain_band": UNCERTAIN_BAND,
        "temperature": TEMPERATURE,
        "uncertain_low": round(THRESHOLD - UNCERTAIN_BAND, 4),
        "uncertain_high": round(THRESHOLD + UNCERTAIN_BAND, 4),
    }


# The detector loads a transformer model, which is expensive. It is created
# once, on first use, and reused for every later image in the same process.
_DETECTOR = None
_DETECTOR_LOAD_FAILED = False


def get_detector():
    """Return the shared AIImageDetector, or None if it cannot be loaded."""
    global _DETECTOR, _DETECTOR_LOAD_FAILED

    if _DETECTOR is not None:
        return _DETECTOR

    if _DETECTOR_LOAD_FAILED:
        return None

    try:
        _DETECTOR = AIImageDetector()
    except Exception as error:
        _DETECTOR_LOAD_FAILED = True
        print(f"\nAI image detection unavailable: {error}")
        return None

    return _DETECTOR


def get_ai_generation_signal(image_path):
    """
    Run the tested detector on one image and return its finding as a
    standalone signal. Never raises.
    """
    detector = get_detector()

    if detector is None:
        return {
            "available": False,
            "model": MODEL_NAME,
            "operating_point": _operating_point(),
            "error": "Detector could not be loaded."
        }

    try:
        result = detector.predict(image_path)
    except Exception as error:
        return {
            "available": False,
            "model": MODEL_NAME,
            "operating_point": _operating_point(),
            "error": str(error)
        }

    result = dict(result)
    result["operating_point"] = _operating_point()
    return result


def describe_signal_consistency(ai_signal, verification_result):
    """
    Compare the two independent signals for reporting only.

    Returns a short descriptive note. It carries no verdict and cannot change
    the verdict the existing pipeline already produced.
    """
    if not ai_signal.get("available"):
        return "AI-origin signal unavailable; verdict rests on evidence alone."

    image_verdict = ai_signal.get("verdict")
    claim_verdict = (
        (verification_result.get("verification") or {})
        .get("final_verdict", {})
        .get("verdict")
    )

    if image_verdict == "UNCERTAIN":
        return (
            "Image origin is UNCERTAIN (ai_probability "
            f"{ai_signal.get('ai_probability')}); this is a genuine "
            "limitation of the detector, not a contradiction. Claim verdict "
            f"{claim_verdict or 'UNKNOWN'} is reported separately."
        )

    if image_verdict == "AI-GENERATED" and claim_verdict == "REAL":
        return (
            "Conflict: image appears AI-generated while the claim evidence "
            "supports REAL. Both signals are reported unchanged."
        )

    if image_verdict == "AI-GENERATED" and claim_verdict == "FAKE":
        return (
            "Consistent: AI-generated image, and the claim evidence "
            "contradicts it."
        )

    if image_verdict == "AI-GENERATED":
        return (
            f"AI-generated image, claim verdict {claim_verdict or 'UNKNOWN'}. "
            "Image origin is reported separately from claim truth."
        )

    return (
        f"Image appears real, claim verdict {claim_verdict or 'UNKNOWN'}. "
        "Both signals agree."
    )


def print_ai_generation_signal(ai_signal, consistency_note=""):
    """Print the AI-origin signal in the style of the existing pipeline."""
    print("\n" + "=" * 60)
    print("AI IMAGE ORIGIN SIGNAL (SEPARATE SIGNAL)")
    print("=" * 60)

    if not ai_signal.get("available"):
        print("\nStatus  : UNAVAILABLE")
        print("Reason  :", ai_signal.get("error", "unknown"))
        return

    print("\nOrigin Verdict  :", ai_signal.get("verdict"))
    print("AI Probability  :", ai_signal.get("ai_probability"))
    print("Real Probability:", ai_signal.get("real_probability"))
    print("Confidence      :", ai_signal.get("confidence"))
    print("Model           :", ai_signal.get("model"))

    if consistency_note:
        print("\nConsistency     :", consistency_note)


def run_image_pipeline_with_ai_detection(image_path):
    """
    Run the existing image verification pipeline unchanged, then attach the
    AI-origin signal as a separate, additional key.

    The existing pipeline runs first and any exception it raises propagates
    exactly as before, so error behaviour is unchanged.
    """
    # Imported here, not at module level, so image_pipeline.py can import
    # this module without a circular import.
    from image_pipeline import run_image_pipeline

    pipeline_result = run_image_pipeline(image_path)

    # Reuse ai_generation if already computed by image_pipeline, avoiding redundant inference
    ai_signal = pipeline_result.get("ai_generation") or get_ai_generation_signal(image_path)

    consistency_note = pipeline_result.get("ai_generation_note") or describe_signal_consistency(
        ai_signal,
        pipeline_result
    )

    combined_result = dict(pipeline_result)
    combined_result["ai_generation"] = ai_signal
    combined_result["ai_generation_note"] = consistency_note

    return combined_result


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    image_path = "image_module/mumbai.png"

    result = run_image_pipeline_with_ai_detection(
        image_path
    )

    print("\n" + "=" * 60)
    print("IMAGE VERIFICATION COMPLETE")
    print("=" * 60)

    print("\nClaim Verdict (existing pipeline, unchanged):")
    print(
        result["verification"]["final_verdict"]["verdict"]
    )

    print("\nImage Origin Signal (separate):")
    print(
        result["ai_generation"].get("verdict", "UNAVAILABLE")
    )
