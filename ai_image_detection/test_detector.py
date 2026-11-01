import os

from detector import AIImageDetector

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


REAL_DIR = "benchmark/real"
AI_DIR = "benchmark/ai"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


detector = AIImageDetector()

y_true = []
y_pred = []


def get_images(folder):
    return [
        os.path.join(folder, file)
        for file in sorted(os.listdir(folder))
        if file.lower().endswith(VALID_EXTENSIONS)
    ]


# =========================
# REAL IMAGES
# =========================

n_uncertain = 0

for image_path in get_images(REAL_DIR):

    result = detector.predict(image_path)

    actual = 0       # REAL
    # UNCERTAIN is neither class. Counting it as REAL would quietly inflate
    # real recall and hide a third of the set, so it is tallied separately
    # and reported as UNSURE rather than silently scored.
    if result["verdict"] == "UNCERTAIN":
        n_uncertain += 1
        predicted = -1
    else:
        predicted = 1 if result["verdict"] == "AI-GENERATED" else 0

    y_true.append(actual)
    y_pred.append(predicted)

    print(
        f"{os.path.basename(image_path):15}"
        f" Actual: REAL"
        f"  Predicted: {result['verdict']}"
        f"  AI: {result['ai_probability'] * 100:.2f}%"
    )


# =========================
# AI IMAGES
# =========================

for image_path in get_images(AI_DIR):

    result = detector.predict(image_path)

    actual = 1       # AI
    if result["verdict"] == "UNCERTAIN":
        n_uncertain += 1
        predicted = -1
    else:
        predicted = 1 if result["verdict"] == "AI-GENERATED" else 0

    y_true.append(actual)
    y_pred.append(predicted)

    print(
        f"{os.path.basename(image_path):15}"
        f" Actual: AI"
        f"  Predicted: {result['verdict']}"
        f"  AI: {result['ai_probability'] * 100:.2f}%"
    )


# =========================
# METRICS
# =========================

# Binary metrics are computed over decided images only, so UNCERTAIN never
# inflates either class. The UNCERTAIN count is reported next to them.
decided = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
decided_true = [t for t, _ in decided]
decided_pred = [p for _, p in decided]

accuracy = accuracy_score(decided_true, decided_pred)

cm = confusion_matrix(
    decided_true,
    decided_pred,
    labels=[0, 1]
)

print("\n======================================")
print("       AI IMAGE DETECTOR RESULTS")
print("======================================")

print(f"Total Images      : {len(y_true)}")
print(f"Uncertain         : {n_uncertain}")
print(f"Decided           : {len(decided_true)}")
print(f"Accuracy (decided): {accuracy * 100:.2f}%")

print("\nClassification Report (decided images only):")
print(
    classification_report(
        decided_true,
        decided_pred,
        labels=[0, 1],
        target_names=["REAL", "AI-GENERATED"],
        digits=4,
        zero_division=0
    )
)

print("Confusion Matrix:")
print("                 Predicted")
print("                 REAL    AI")
print(f"Actual REAL      {cm[0][0]:3}     {cm[0][1]:3}")
print(f"Actual AI        {cm[1][0]:3}     {cm[1][1]:3}")

tn, fp, fn, tp = cm.ravel()

print("\nDetailed:")
print("True Negative  :", tn)
print("False Positive :", fp)
print("False Negative :", fn)
print("True Positive  :", tp)

fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0

print(f"False Positive Rate : {fpr * 100:.2f}%")
print(f"False Negative Rate : {fnr * 100:.2f}%")

print("======================================")