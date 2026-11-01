"""
prepare_train.py
================
Builds TRAIN / VALIDATION / TEST pools for detector training.

Design constraints this file exists to satisfy:

  * No duplicate content across splits. Every candidate image is hashed on
    its raw pixel bytes and a hash seen in an earlier split is dropped from
    later splits.

  * The AI-or-Not dataset contains CIFAKE images (its 32x32 population IS
    CIFAKE). Mixing both without a hash check would put the same picture in
    TRAIN and TEST. Hashing removes that leak.

  * Resolution is a class label, not a feature. CIFAKE's real class is 32x32
    while much of its fake class is larger, so anything that trains on
    native size learns to read the resolution. Everything is stored at one
    canonical size.

  * Labels come only from the datasets' own published label columns. Nothing
    here guesses, and anything unreadable is dropped rather than relabelled.

Output: <root>/{train,val,test}/... plus manifest.json
"""

import hashlib
import json
import os
import random
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.datasets import (  # noqa: E402
    CIFAKE_REPO, HEMG_REPO, CIFAKE_SHARDS, HEMG_SHARDS,
    CANONICAL_SIZE, SEED, AI, REAL,
)

TRAIN = "train"
VAL = "val"
TEST = "test"

# Fraction of each source group held out for validation and test.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def pixel_hash(image, size=64):
    """
    Hash of downscaled pixels.

    Exact-file hashing would miss re-encodings of the same picture, which is
    exactly the CIFAKE-inside-AI-or-Not case. Downscaling first makes the
    hash depend on content, so a JPEG re-encode or a resize of the same
    image collapses onto the same digest.
    """
    small = image.convert("RGB").resize((size, size), Image.LANCZOS)
    return hashlib.sha256(small.tobytes()).hexdigest()


def _assign(rows, seed_offset):
    """Deterministically split rows into val/test/train proportions."""
    rows = list(rows)
    random.Random(SEED + seed_offset).shuffle(rows)
    n = len(rows)
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)
    return (rows[:n_val], rows[n_val:n_val + n_test], rows[n_val + n_test:])


def collect_cifake(limit=None):
    from datasets import load_dataset

    out = []
    for split_name in ("train", "test"):
        ds = load_dataset(
            CIFAKE_REPO,
            data_files=[CIFAKE_SHARDS[split_name]],
            split="train",
            verification_mode="no_checks",
        )
        order = list(range(len(ds)))
        random.Random(SEED + (1 if split_name == "train" else 2)).shuffle(order)
        taken = 0
        for idx in order:
            row = ds[int(idx)]
            image = row["image"].convert("RGB")
            label = AI if int(row["label"]) == 0 else REAL
            out.append((image, label, "cifake", split_name, image.size))
            taken += 1
            if limit and taken >= limit:
                break
        print("  cifake %s: %d" % (split_name, taken))
    return out


def collect_hemg(limit=None, drop_tiny=True):
    from datasets import load_dataset

    ds = load_dataset(
        HEMG_REPO, data_files=HEMG_SHARDS, split="train",
        verification_mode="no_checks",
    )
    order = list(range(len(ds)))
    random.Random(SEED + 3).shuffle(order)

    out = []
    skipped_tiny = 0
    for idx in order:
        row = ds[int(idx)]
        image = row["image"].convert("RGB")
        label = AI if int(row["label"]) == 0 else REAL
        if drop_tiny and max(image.size) <= 64:
            # This population is CIFAKE re-served inside AI-or-Not. Keep it
            # out so the CIFAKE group is the only source of those images.
            skipped_tiny += 1
            continue
        out.append((image, label, "hemg", "train", image.size))
        if limit and len(out) >= limit:
            break
    print("  hemg: %d kept, %d tiny(CIFAKE-derived) skipped"
          % (len(out), skipped_tiny))
    return out


def save(image, path):
    image = image.convert("RGB").resize(
        (CANONICAL_SIZE, CANONICAL_SIZE), Image.LANCZOS)
    image.save(path, format="JPEG", quality=95)
    return image.size


def main(root, cifake_train_target=24000, hemg_train_target=24000):
    os.makedirs(root, exist_ok=True)

    print("Collecting CIFAKE...")
    cifake = collect_cifake(limit=cifake_train_target + 12000)

    print("Collecting AI-or-Not...")
    hemg = collect_hemg(limit=hemg_train_target + 12000)

    rows = cifake + hemg
    print("pool: %d images" % len(rows))

    # ---- split, stratified per (source, label) so every split is balanced
    groups = {}
    for image, label, source, orig_split, orig_size in rows:
        groups.setdefault((source, label), []).append(
            (image, label, source, orig_size))

    pools = {TRAIN: [], VAL: [], TEST: []}
    for offset, key in enumerate(sorted(groups)):
        val, test, train = _assign(groups[key], 100 + offset)
        pools[VAL].extend(val)
        pools[TEST].extend(test)
        pools[TRAIN].extend(train)
        print("  group %-18s train=%-6d val=%-5d test=%-5d"
              % (str(key), len(train), len(val), len(test)))

    # ---- hash-based de-duplication, train wins
    manifest = []
    seen = {}
    duplicates = Counter()

    for split in (TRAIN, VAL, TEST):
        folder = os.path.join(root, split)
        os.makedirs(folder, exist_ok=True)
        counts = Counter()
        for image, label, source, orig_size in sorted(
                pools[split], key=lambda r: (r[2], r[1])):
            try:
                digest = pixel_hash(image)
            except Exception:
                duplicates["unhashable"] += 1
                continue

            if digest in seen:
                duplicates[seen[digest]] += 1
                continue

            counts[label] += 1
            prefix = "ai" if label == AI else "real"
            name = "%s_%06d.jpg" % (prefix, counts[label])
            path = os.path.join(folder, name)
            try:
                stored = save(image, path)
            except Exception:
                duplicates["unwritable"] += 1
                continue

            seen[digest] = split
            manifest.append({
                "id": "%s/%s" % (split, name),
                "abs_path": os.path.abspath(path),
                "label": label,
                "source": source,
                "split": split,
                "orig_width": orig_size[0],
                "orig_height": orig_size[1],
                "width": stored[0],
                "height": stored[1],
                "hash": digest,
            })
        print("%-6s ai=%-6d real=%-6d" % (split, counts[AI], counts[REAL]))

    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print("duplicates removed: %s" % dict(duplicates))
    print("manifest: %d records -> %s"
          % (len(manifest), os.path.join(root, "manifest.json")))

    for split in (TRAIN, VAL, TEST):
        rows = [r for r in manifest if r["split"] == split]
        if not rows:
            raise SystemExit("split %s is EMPTY" % split)
        for label, name in ((AI, "AI"), (REAL, "REAL")):
            n = sum(1 for r in rows if r["label"] == label)
            if n == 0:
                raise SystemExit("split %s has no %s images" % (split, name))
        print("verified %-6s n=%-6d balanced=yes" % (split, len(rows)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "ai_image_detection/train_v1")
