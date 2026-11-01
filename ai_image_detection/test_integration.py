from detector import detect_ai_image


IMAGE_PATH = "test_images/ai/AI generated image.png"

result = detect_ai_image(IMAGE_PATH)

print("\n========== AI IMAGE RESULT ==========")
print("Verdict          :", result["verdict"])
print("AI Probability   : %.2f%%" % (result["ai_probability"] * 100))
print("Real Probability : %.2f%%" % (result["real_probability"] * 100))
print("Confidence       : %.2f%%" % (result["confidence"] * 100))
print("=====================================")