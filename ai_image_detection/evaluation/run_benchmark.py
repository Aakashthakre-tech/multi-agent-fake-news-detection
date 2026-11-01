"""
run_benchmark.py
================
Runs every candidate detector over the labelled benchmark and stores the raw
AI probability for each image.

The expensive step is inference, and it happens exactly once per
(model, image) pair. Everything downstream - threshold sweeps, calibration,
uncertainty bands, ensembles, error analysis - reads the cached scores, so no
later experiment re-runs a model or re-samples the data.

Usage:
    python -m ai_image_detection.evaluation.run_benchmark --root <dir>
"""

import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.datasets import load_manifest  # noqa: E402
from ai_image_detection import models as model_registry  # noqa: E402

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _model_disk_bytes(name):
    """Best-effort on-disk size of a cached model, for the runtime table."""
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    folder = "models--" + name.replace("/", "--")
    base = os.path.join(hub, folder)
    total = 0
    for root, _dirs, files in os.walk(base):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def score_manifest(manifest, model_key, root, progress_every=50):
    """
    Return {image_id: ai_probability} for one model over the manifest.

    Resumable: already-computed ids are loaded from the cache file and
    skipped, so an interrupted run continues instead of restarting.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_path = os.path.join(RESULTS_DIR, "scores_%s.json" % model_key)

    scores = {}
    meta = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        scores = payload.get("scores", {})
        meta = payload.get("meta", {})

    todo = [r for r in manifest if r["id"] not in scores]
    if not todo:
        print("[%s] all %d images already scored" % (model_key, len(scores)))
        return scores, meta

    print("[%s] scoring %d images (%d cached)"
          % (model_key, len(todo), len(scores)))

    adapter = model_registry.build_adapter(model_key)
    described = adapter.describe()

    t_start = time.time()
    for i, record in enumerate(todo, 1):
        path = record["abs_path"]
        try:
            with Image.open(path) as im:
                prob = adapter.predict_probability(im)
        except Exception as error:
            print("   FAILED %s: %s" % (record["id"], str(error)[:120]))
            continue

        scores[record["id"]] = float(prob)

        if i % progress_every == 0 or i == len(todo):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed else 0
            remaining = (len(todo) - i) / rate if rate else 0
            print("   %d/%d  %.2fs/img  eta %.0fs"
                  % (i, len(todo), elapsed / i, remaining))

    meta = {
        "model": described,
        "model_key": model_key,
        "seconds": time.time() - t_start,
        "seconds_per_image": (time.time() - t_start) / max(1, len(todo)),
        "n_images": len(todo),
        "disk_bytes": _model_disk_bytes(described["name"]),
    }

    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "scores": scores}, fh, indent=1)

    return scores, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="benchmark root containing manifest.json")
    parser.add_argument("--models", nargs="*", default=None,
                        help="candidate keys; default is the whole registry")
    args = parser.parse_args()

    manifest = load_manifest(args.root)
    print("manifest: %d records" % len(manifest))

    keys = args.models or list(model_registry.CANDIDATES)

    for key in keys:
        print("=" * 60)
        print("MODEL:", key)
        try:
            score_manifest(manifest, key, args.root)
        except Exception as error:
            print("MODEL %s FAILED: %s" % (key, str(error)[:300]))

    print("=" * 60)
    print("Scoring complete. Cached in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
