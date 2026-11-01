"""
Quick integration test to verify new synthetic_data fields are correctly returned.
"""
import json
import sys
import time

sys.path.insert(0, '.')

from multimodal_router import verify_image

IMG = "image_module/mumbai.png"

print("Testing image:", IMG)
t = time.time()
result = verify_image(IMG)
elapsed = time.time() - t

print("\n--- RESULT ---")
print("Status:", result["status"])
print("Modality:", result["modality"])
print("Primary Claim:", result.get("primary_claim", "N/A"))
print("Factual Verdict:", result["factual_verification"]["verdict"])

synth = result.get("synthetic_media_detection") or {}
print("\n--- SYNTHETIC MEDIA DETECTION ---")
print("Available:", synth.get("available"))
print("Signal:", synth.get("signal"))
print("Confidence:", synth.get("confidence"))
print("Model:", synth.get("model"))

probs = synth.get("probabilities") or {}
print("  ai_probability:", probs.get("ai_probability"))
print("  real_probability:", probs.get("real_probability"))
print("  manipulation_score:", probs.get("manipulation_score"))

print("manipulation_status:", synth.get("manipulation_status"))
print("manipulation_note:", synth.get("manipulation_note"))

print("\nElapsed: {:.2f}s".format(elapsed))
print("\n[PASS] All new fields present and correctly populated." if synth.get("manipulation_status") else "[FAIL] manipulation_status missing!")
