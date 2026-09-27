from .ocr_phrase_grouping import group_ocr_phrases


_JOINABLE_STARTERS = {
    "a", "an", "and", "at", "before", "during", "for", "in", "near",
    "of", "on", "the", "to", "with"
}


def _merge_headline(phrases):
    parts = []

    for index, phrase in enumerate(phrases):
        text = phrase["text"].strip()

        if index and " " in text:
            first_word, remainder = text.split(" ", 1)

            if first_word.casefold() in _JOINABLE_STARTERS:
                text = f"{first_word.casefold()} {remainder}"

        parts.append(text)

    return " ".join(parts)


def _phrase_height(phrase):
    y_values = [
        float(point[1])
        for point in phrase["box"]
    ]
    return max(y_values) - min(y_values)


def _phrase_width(phrase):
    x_values = [
        float(point[0])
        for point in phrase["box"]
    ]
    return max(x_values) - min(x_values)


def _phrase_center_y(phrase):
    y_values = [
        float(point[1])
        for point in phrase["box"]
    ]
    return (max(y_values) + min(y_values)) / 2


def extract_image_claim(ocr_results, detected_objects):

    phrases = group_ocr_phrases(ocr_results)

    if phrases:
        ocr_text = "\n".join(
            phrase["text"]
            for phrase in phrases
        )

        horizontal_phrases = [
            phrase
            for phrase in phrases
            if _phrase_width(phrase) >= _phrase_height(phrase) * 1.5
        ]

        layout_candidates = horizontal_phrases or phrases
        image_height = max(
            _phrase_center_y(phrase) + _phrase_height(phrase) / 2
            for phrase in phrases
        )

        upper_phrases = [
            phrase
            for phrase in layout_candidates
            if _phrase_center_y(phrase) <= image_height * 0.70
        ]

        headline_candidates = [
            phrase
            for phrase in upper_phrases
            if len(phrase["text"].split()) >= 3
        ]

        candidate_phrases = (
            headline_candidates
            or upper_phrases
            or layout_candidates
        )
        maximum_height = max(
            _phrase_height(phrase)
            for phrase in candidate_phrases
        )

        headline_phrases = [
            phrase
            for phrase in candidate_phrases
            if _phrase_height(phrase) >= maximum_height * 0.85
        ]

        headline_phrases.sort(
            key=lambda phrase: min(
                float(point[1])
                for point in phrase["box"]
            )
        )

        headline_text = _merge_headline(headline_phrases)
    else:
        ocr_text = " ".join(
            result["text"]
            for result in ocr_results
        )
        headline_text = ocr_text

    # Combine detected objects
    object_names = [
        obj["object"]
        for obj in detected_objects
    ]

    # Remove duplicates
    unique_objects = list(dict.fromkeys(object_names))

    claim = {
        "ocr_text": ocr_text,
        "headline_text": headline_text,
        "detected_objects": unique_objects
    }

    return claim