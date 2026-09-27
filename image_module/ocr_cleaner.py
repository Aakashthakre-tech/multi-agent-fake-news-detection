def clean_ocr_results(ocr_results, min_confidence=0.40):

    cleaned = []

    for result in ocr_results:

        text = result["text"].strip()
        confidence = result["confidence"]
        box = result["box"]

        if confidence < min_confidence:
            continue

        if len(text) < 2:
            continue

        if len(text.split()) == 1 and confidence < 0.60:
            continue

        alphanumeric = sum(c.isalnum() for c in text)

        if alphanumeric < 2:
            continue

        cleaned.append({
            "text": text,
            "confidence": confidence,
            "box": box
        })

    return cleaned