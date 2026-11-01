import os
from detector import AIImageDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


REAL_DIR = "benchmark/real"
AI_DIR = "benchmark/ai"

detector = AIImageDetector()

y_true = []
scores = []


def get_images(folder):
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


# REAL = 0
for path in get_images(REAL_DIR):
    result = detector.predict(path)

    y_true.append(0)
    scores.append(result["ai_probability"] / 100)


# AI = 1
for path in get_images(AI_DIR):
    result = detector.predict(path)

    y_true.append(1)
    scores.append(result["ai_probability"] / 100)


print("\n==============================================")
print("          THRESHOLD CALIBRATION")
print("==============================================")

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


best_threshold = None
best_f1 = 0


for threshold in [x / 100 for x in range(30, 71, 5)]:

    predictions = [
        1 if score >= threshold else 0
        for score in scores
    ]

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    print(
        f"{threshold:.2f}        "
        f"{accuracy * 100:.2f}%       "
        f"{precision * 100:.2f}%       "
        f"{recall * 100:.2f}%       "
        f"{f1 * 100:.2f}%"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n==============================================")
print("Best threshold by F1:", best_threshold)
print("Best F1             :", round(best_f1 * 100, 2), "%")
print("==============================================")