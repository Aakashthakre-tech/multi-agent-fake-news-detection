"""
datasets.py
===========
Builds labelled benchmark splits from public datasets with published ground
truth. Nothing here is hand-labelled and no image is selected by hand.

Two sources are used, and they cover deliberately different domains:

  CIFAKE  (dragonintelligence/CIFAKE-image-dataset)
      Label column present: 0 = FAKE, 1 = REAL.
      Images are 32x32. This is the same source as the project's original
      40-image benchmark, so results here are directly comparable to the
      previously reported 82.50% accuracy, and it supplies a train/test split
      that the original benchmark did not have.

  AI-or-Not (Hemg/AI-Generated-vs-Real-Images-Datasets)
      Label column present: 0 = AiArtData, 1 = RealArt.
      Contains genuine full-resolution photographs as the real class, which
      is much closer to the kind of image the TruthLensAI pipeline receives.

Output is a JSON manifest plus JPEG files on disk, so every later experiment
reads exactly the same pixels and never re-samples.
"""

import json
import os
import random
from collections import Counter

from PIL import Image

# Project label convention.
AI = 1
REAL = 0

CIFAKE_REPO = "dragonintelligence/CIFAKE-image-dataset"
HEMG_REPO = "Hemg/AI-Generated-vs-Real-Images-Datasets"

# The CIFAKE repository publishes exactly two parquet files, one per split.
# Verified against the repository file listing.
CIFAKE_SHARDS = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

HEMG_SHARDS = [
    "data/train-00000-of-00006-336b26d54a26e17a.parquet",
    "data/train-00001-of-00006-8ad2d550254dea81.parquet",
    "data/train-00002-of-00006-ac8970f21c0418c1.parquet",
    "data/train-00003-of-00006-f635132ef309a732.parquet",
    "data/train-00004-of-00006-1101eaf5152e1c5f.parquet",
    "data/train-00005-of-00006-4bd152a5ab76dba7.parquet",
]

SEED = 20260927

# Every benchmark image is stored at this size.
#
# This is deliberate and necessary. In the source datasets the real class and
# the AI class have systematically different resolutions (in the AI-or-Not
# shards the real images are almost entirely 32x32 while a large share of the
# AI images are 256x256 or larger). A detector handed images at their native
# size could separate the classes from resolution alone and score well while
# detecting nothing. Storing everything at one canonical size removes that
# shortcut. It also matches production, where every model resizes its input to
# a fixed tensor size anyway.
CANONICAL_SIZE = 256


def _shard_for(split):
    if split not in CIFAKE_SHARDS:
        raise KeyError(
            "CIFAKE has no split %r; available: %s"
            % (split, sorted(CIFAKE_SHARDS))
        )
    return CIFAKE_SHARDS[split]


def collect_cifake(split, per_class):
    """CIFAKE rows. Source label 1 = REAL, 0 = FAKE -> project AI/REAL."""
    from datasets import load_dataset

    ds = load_dataset(
        CIFAKE_REPO,
        data_files=[_shard_for(split)],
        split="train",
        verification_mode="no_checks",
    )

    picked = {REAL: [], AI: []}
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)

    for idx in order:
        row = ds[idx]
        src_label = int(row["label"])
        label = AI if src_label == 0 else REAL
        if len(picked[label]) >= per_class:
            if all(len(v) >= per_class for v in picked.values()):
                break
            continue
        picked[label].append((row["image"].convert("RGB"), label))
        if all(len(v) >= per_class for v in picked.values()):
            break

    out = []
    for label in (AI, REAL):
        for image, _ in picked[label]:
            out.append((image, label, "cifake", split, None))
    return out


def collect_hemg(per_class):
    """AI-or-Not rows. Source label 0 = AiArtData (AI), 1 = RealArt (REAL)."""
    from datasets import load_dataset

    ds = load_dataset(
        HEMG_REPO,
        data_files=HEMG_SHARDS,
        split="train",
        verification_mode="no_checks",
    )

    picked = {REAL: [], AI: []}
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)

    for idx in order:
        row = ds[idx]
        src_label = int(row["label"])
        label = AI if src_label == 0 else REAL
        if len(picked[label]) >= per_class:
            if all(len(v) >= per_class for v in picked.values()):
                break
            continue
        picked[label].append((row["image"].convert("RGB"), label))
        if all(len(v) >= per_class for v in picked.values()):
            break

    out = []
    for label in (AI, REAL):
        for image, _ in picked[label]:
            out.append((image, label, "hemg", None, None))
    return out


def _save(image, path, size=CANONICAL_SIZE):
    """
    Store the image at one canonical size and return both the original
    dimensions and the stored dimensions, so a resolution-sensitivity
    experiment can use the original size later without re-downloading.
    """
    image = image.convert("RGB")
    original = image.size
    image = image.resize((size, size), Image.LANCZOS)
    image.save(path, format="JPEG", quality=95)
    return original, image.size


EXPECTED_SPLITS = ("cifake_val", "cifake_test", "hemg_val", "hemg_test")


def verify_manifest(manifest, root, min_per_class=None):
    """
    Assert that every split exists, is populated, and is class balanced.

    The benchmark must never be allowed to run on a partially built dataset,
    because a missing split silently changes which images a threshold is
    tuned on. Returns a per-split summary and raises on any shortfall.
    """
    summary = {}
    problems = []

    for split in EXPECTED_SPLITS:
        rows = [r for r in manifest if r["split"] == split]
        ai = sum(1 for r in rows if r["label"] == AI)
        real = sum(1 for r in rows if r["label"] == REAL)

        missing = [r["abs_path"] for r in rows
                   if not os.path.exists(r["abs_path"])]

        summary[split] = {
            "total": len(rows),
            "ai": ai,
            "real": real,
            "files_missing_on_disk": len(missing),
            "ai_sources": sorted({r["source"] for r in rows
                                  if r["label"] == AI}),
            "real_sources": sorted({r["source"] for r in rows
                                    if r["label"] == REAL}),
        }

        if len(rows) == 0:
            problems.append("%s is EMPTY" % split)
        if ai != real:
            problems.append(
                "%s is class unbalanced (ai=%d real=%d)" % (split, ai, real))
        if ai == 0 or real == 0:
            problems.append("%s has only one class" % split)
        if missing:
            problems.append(
                "%s has %d missing files on disk" % (split, len(missing)))
        if min_per_class is not None and ai < min_per_class:
            problems.append(
                "%s has only %d images per class (floor %d)"
                % (split, ai, min_per_class))

    # VAL and TEST must not overlap, or the protocol is meaningless.
    val_ids = {r["id"] for r in manifest if r["split"] == "cifake_val"}
    test_ids = {r["id"] for r in manifest if r["split"] == "cifake_test"}
    hemg_val_ids = {r["id"] for r in manifest if r["split"] == "hemg_val"}
    hemg_test_ids = {r["id"] for r in manifest if r["split"] == "hemg_test"}

    if val_ids & test_ids:
        problems.append("cifake_val and cifake_test overlap")
    if hemg_val_ids & hemg_test_ids:
        problems.append("hemg_val and hemg_test overlap")

    return summary, problems


def print_verification(summary, problems):
    print()
    print("=" * 72)
    print("BENCHMARK VERIFICATION")
    print("=" * 72)
    print("%-14s %8s %8s %8s %10s  %s"
          % ("split", "total", "ai", "real", "files_ok", "sources"))
    print("-" * 72)
    for split, s in summary.items():
        sources = ",".join(s["ai_sources"] + s["real_sources"])
        print("%-14s %8d %8d %8d %10s  %s" % (
            split, s["total"], s["ai"], s["real"],
            "yes" if s["files_missing_on_disk"] == 0 else "NO(%d)"
            % s["files_missing_on_disk"],
            sources))
    print("-" * 72)

    if problems:
        print("FAILED:")
        for p in problems:
            print("   - %s" % p)
    else:
        print("PASSED: all four splits populated, class balanced, on disk.")
    print("=" * 72)
    return not problems


def build(root, cifake_per_class=600, hemg_per_class=400):
    """
    Build every split and write a manifest. Returns the manifest list.
    """
    from datasets import load_dataset  # noqa: F401  (import cost is here)

    os.makedirs(root, exist_ok=True)
    manifest = []

    def emit(rows, split_name):
        folder = os.path.join(root, split_name)
        os.makedirs(folder, exist_ok=True)
        counts = {AI: 0, REAL: 0}
        orig_sizes = Counter()
        for image, label, source, _orig_split, category in rows:
            counts[label] += 1
            name = "%s_%05d.jpg" % (
                "ai" if label == AI else "real", counts[label])
            path = os.path.join(folder, name)
            original, stored = _save(image, path)
            orig_sizes[original] += 1
            manifest.append({
                "id": "%s/%s" % (split_name, name),
                "path": os.path.relpath(path, os.path.dirname(root)),
                "abs_path": os.path.abspath(path),
                "label": label,
                "source": source,
                "split": split_name,
                "orig_width": original[0],
                "orig_height": original[1],
                "width": stored[0],
                "height": stored[1],
                "category": category,
            })
        print("  %-14s ai=%d real=%d" % (
            split_name, counts[AI], counts[REAL]))
        print("      source resolutions: %s" % (
            ", ".join("%dx%d x%d" % (s[0], s[1], c)
                      for s, c in orig_sizes.most_common(4))))

    print("Collecting CIFAKE...")
    emit(collect_cifake("train", cifake_per_class), "cifake_val")
    emit(collect_cifake("test", cifake_per_class), "cifake_test")

    print("Collecting AI-or-Not...")
    hemg_rows = collect_hemg(hemg_per_class)

    # Split per class, not on the pooled list. Splitting a shuffled pool would
    # leave each half slightly class unbalanced, which biases every rate
    # computed on it.
    by_label = {AI: [], REAL: []}
    for image, label, source, _orig, category in hemg_rows:
        by_label[label].append((image, label, source, None, category))

    val_rows, test_rows = [], []
    for label in (AI, REAL):
        group = by_label[label]
        random.Random(SEED + 1 + label).shuffle(group)
        cut = len(group) // 2
        val_rows.extend(group[:cut])
        test_rows.extend(group[cut:])

    emit(val_rows, "hemg_val")
    emit(test_rows, "hemg_test")

    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print("Manifest: %s (%d records)" % (manifest_path, len(manifest)))

    summary, problems = verify_manifest(
        manifest, root, min_per_class=200)
    print_verification(summary, problems)

    if problems:
        raise SystemExit(
            "Benchmark preparation FAILED; refusing to continue:\n  - "
            + "\n  - ".join(problems))

    return manifest


def load_manifest(root):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)
