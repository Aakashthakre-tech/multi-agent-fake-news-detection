from .image_input import load_image
from .ocr import extract_text
from .visual_analysis import analyze_image
from .ocr_cleaner import clean_ocr_results
from .ocr_phrase_grouping import group_ocr_phrases
from .image_text_matcher import calculate_region_text_similarity
from .image_text_scorer import calculate_consistency_score
from .claim_extractor import extract_image_claim
from .claim_normalizer import normalize_image_claim


image_path = "image_module/mumbai.png"

print("\n==============================")
print("IMAGE MODULE TEST")
print("==============================")


# Image loading
image = load_image(image_path)


# OCR
print("\nExtracting text...\n")

ocr_results = extract_text(image_path)

print("RAW OCR RESULTS:")

for result in ocr_results:
    print(
        result["text"],
        "->",
        result["confidence"]
    )


# OCR cleaning
cleaned_results = clean_ocr_results(ocr_results)


# Phrase grouping
phrases = group_ocr_phrases(cleaned_results)

print("\nOCR PHRASES:")

for phrase in phrases:
    print(
        phrase["text"],
        "->",
        phrase["confidence"]
    )


# Visual analysis
print("\nDetecting objects...\n")

objects = analyze_image(image_path)

print("DETECTED OBJECTS:")

for obj in objects:
    print(
        obj["object"],
        "->",
        obj["confidence"]
    )

# Image claim extraction
print("\nExtracting image claim...\n")

image_claim = extract_image_claim(
    cleaned_results,
    objects
)

print("IMAGE CLAIM DATA:")

print("\nOCR TEXT:")
print(image_claim["ocr_text"])

print("\nDETECTED OBJECTS:")
print(image_claim["detected_objects"])

# Claim normalization
print("\nNormalizing image claim...\n")

normalized_claim = normalize_image_claim(
    image_claim["ocr_text"],
    image_claim["detected_objects"],
    image_claim.get("headline_text")
)

print("NORMALIZED CLAIM:")
print(normalized_claim["claim"])

print("\nCLEANED OCR:")
print(normalized_claim["cleaned_ocr"])


# Image-text relationship
print("\nChecking region-level image-text relationship...\n")

region_scores = []

for phrase in phrases:

    # Calculate CLIP similarity
    similarity = calculate_region_text_similarity(
        image_path,
        phrase["box"],
        phrase["text"]
    )

    # Calculate consistency score
    consistency = calculate_consistency_score(
        similarity
    )

    # Store results
    region_scores.append({
        "text": phrase["text"],
        "ocr_confidence": phrase["confidence"],
        "similarity": similarity,
        "consistency": consistency
    })

    print(
        phrase["text"],
        "| OCR:", phrase["confidence"],
        "| CLIP:", similarity,
        "| Consistency:", consistency
    )


# Overall consistency
if region_scores:

    average_consistency = sum(
        item["consistency"]
        for item in region_scores
    ) / len(region_scores)

    print(
        "\nOverall Image-Text Consistency:",
        round(average_consistency, 2),
        "/ 100"
    )


print("\n==============================")
print("TEST COMPLETE")
print("==============================")