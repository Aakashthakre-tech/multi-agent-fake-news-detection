from .image_text_matcher import calculate_region_text_similarity
from .image_text_scorer import calculate_consistency_score
from PIL import Image


test_cases = [

    # =========================
    # FIRE IMAGE - POSITIVE
    # =========================

    {
        "name": "Fire + Correct",
        "image": "image_module/fire.png",
        "text": "A major fire broke out at a commercial building.",
        "expected": 1
    },

    {
        "name": "Fire + Firefighters",
        "image": "image_module/fire.png",
        "text": "Firefighters are responding to a building fire.",
        "expected": 1
    },

    {
        "name": "Fire + Building Fire",
        "image": "image_module/fire.png",
        "text": "A commercial building is on fire.",
        "expected": 1
    },

    {
        "name": "Fire + Emergency Response",
        "image": "image_module/fire.png",
        "text": "Emergency crews are responding to a fire.",
        "expected": 1
    },


    # =========================
    # FIRE IMAGE - NEGATIVE
    # =========================

    {
        "name": "Fire + Taj Mahal",
        "image": "image_module/fire.png",
        "text": "The Taj Mahal is located in Agra.",
        "expected": 0
    },

    {
        "name": "Fire + Cricket",
        "image": "image_module/fire.png",
        "text": "India won a cricket match against Australia.",
        "expected": 0
    },

    {
        "name": "Fire + Beach",
        "image": "image_module/fire.png",
        "text": "A beautiful beach is located near Mumbai.",
        "expected": 0
    },

    {
        "name": "Fire + Football",
        "image": "image_module/fire.png",
        "text": "A football team won the championship.",
        "expected": 0
    },

    {
        "name": "Fire + Airport",
        "image": "image_module/fire.png",
        "text": "A passenger plane landed safely at the airport.",
        "expected": 0
    },

    {
        "name": "Fire + Mountain",
        "image": "image_module/fire.png",
        "text": "Tourists are hiking through a mountain region.",
        "expected": 0
    },


    # =========================
    # TAJ MAHAL - POSITIVE
    # =========================

    {
        "name": "Taj + Correct",
        "image": "image_module/taj.png",
        "text": "The Taj Mahal is located in Agra, India.",
        "expected": 1
    },

    {
        "name": "Taj + Monument",
        "image": "image_module/taj.png",
        "text": "The image shows a famous white marble monument.",
        "expected": 1
    },

    {
        "name": "Taj + White Marble",
        "image": "image_module/taj.png",
        "text": "A large white marble monument is shown.",
        "expected": 1
    },

    {
        "name": "Taj + Famous Landmark",
        "image": "image_module/taj.png",
        "text": "The image shows a famous Indian landmark.",
        "expected": 1
    },


    # =========================
    # TAJ MAHAL - NEGATIVE
    # =========================

    {
        "name": "Taj + Fire",
        "image": "image_module/taj.png",
        "text": "A major fire broke out at a commercial building.",
        "expected": 0
    },

    {
        "name": "Taj + Cricket",
        "image": "image_module/taj.png",
        "text": "India won a cricket match against Australia.",
        "expected": 0
    },

    {
        "name": "Taj + Beach",
        "image": "image_module/taj.png",
        "text": "A beautiful beach is located near Mumbai.",
        "expected": 0
    },

    {
        "name": "Taj + Football",
        "image": "image_module/taj.png",
        "text": "A football team won the championship.",
        "expected": 0
    },

    {
        "name": "Taj + Airport",
        "image": "image_module/taj.png",
        "text": "A passenger plane landed safely at the airport.",
        "expected": 0
    },

    {
        "name": "Taj + Mountain",
        "image": "image_module/taj.png",
        "text": "Tourists are hiking through a mountain region.",
        "expected": 0
    }
]


# ==========================================
# CALCULATE SCORES
# ==========================================

print("\n======================================")
print("IMAGE-TEXT CALIBRATION")
print("======================================")

results = []


for case in test_cases:

    image = Image.open(case["image"])

    width, height = image.size

    full_image_box = [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]

    similarity = calculate_region_text_similarity(
        case["image"],
        full_image_box,
        case["text"],
        padding=0
    )

    score = calculate_consistency_score(
        similarity
    )

    result = {
        "name": case["name"],
        "expected": case["expected"],
        "score": score
    }

    results.append(result)

    print(
        case["name"],
        "| Expected:",
        "POSITIVE" if case["expected"] == 1 else "NEGATIVE",
        "| Score:",
        score
    )


# ==========================================
# THRESHOLD EVALUATION
# ==========================================

def evaluate_threshold(results, threshold):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:

        predicted = 1 if result["score"] >= threshold else 0
        actual = result["expected"]

        if predicted == 1 and actual == 1:
            tp += 1

        elif predicted == 0 and actual == 0:
            tn += 1

        elif predicted == 1 and actual == 0:
            fp += 1

        elif predicted == 0 and actual == 1:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2)
    }


# ==========================================
# TEST MULTIPLE THRESHOLDS
# ==========================================

print("\n======================================")
print("THRESHOLD EVALUATION")
print("======================================")

evaluations = []

for threshold in range(50, 71):

    evaluation = evaluate_threshold(
        results,
        threshold
    )

    evaluations.append(evaluation)

    print(
        f'Threshold: {threshold} | '
        f'Accuracy: {evaluation["accuracy"]}% | '
        f'Precision: {evaluation["precision"]}% | '
        f'Recall: {evaluation["recall"]}% | '
        f'F1: {evaluation["f1"]}%'
    )


# ==========================================
# BEST THRESHOLD BY F1
# ==========================================

best = max(
    evaluations,
    key=lambda x: x["f1"]
)

print("\n======================================")
print("BEST CALIBRATION RESULT")
print("======================================")

print("Threshold:", best["threshold"])
print("Accuracy:", best["accuracy"], "%")
print("Precision:", best["precision"], "%")
print("Recall:", best["recall"], "%")
print("F1:", best["f1"], "%")

print("\nConfusion Matrix:")
print("TP:", best["tp"])
print("TN:", best["tn"])
print("FP:", best["fp"])
print("FN:", best["fn"])

print("\n======================================")
print("TEST COMPLETE")
print("======================================")