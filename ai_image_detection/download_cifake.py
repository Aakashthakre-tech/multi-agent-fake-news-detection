from datasets import load_dataset
import os


REAL_DIR = "benchmark/real"
AI_DIR = "benchmark/ai"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)


print("Loading CIFAKE test split...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="test",
    streaming=True
)

real_count = 0
ai_count = 0

for item in dataset:

    image = item["image"]
    label = item["label"]

    # CIFAKE:
    # 0 = FAKE
    # 1 = REAL

    if label == 1 and real_count < 20:

        image.convert("RGB").save(
            f"{REAL_DIR}/real_{real_count + 1}.jpg"
        )

        real_count += 1
        print(f"REAL: {real_count}/20")

    elif label == 0 and ai_count < 20:

        image.convert("RGB").save(
            f"{AI_DIR}/ai_{ai_count + 1}.jpg"
        )

        ai_count += 1
        print(f"AI: {ai_count}/20")

    if real_count == 20 and ai_count == 20:
        break


print("\n==============================")
print("CIFAKE SAMPLE READY")
print("==============================")
print("REAL:", real_count)
print("AI  :", ai_count)
print("==============================")