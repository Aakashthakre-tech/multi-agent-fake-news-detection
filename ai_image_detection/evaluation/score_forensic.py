"""
score_forensic.py
=================
Runs the trained forensic CNN over the validation and test splits and stores
per-image AI probabilities in the same cache format the Vision Transformer
candidates use.

Storing scores rather than re-running inference keeps every later experiment
- threshold sweeps, uncertainty bands, calibration, ensembles, error slices -
a pure function of this one file, so none of them can accidentally disagree
with another.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.forensic import (  # noqa: E402
    ForensicCNN, compute_residual, residual_to_tensor, count_parameters,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")


@torch.no_grad()
def score_rows(net, rows, progress_every=1000):
    net.eval()
    scores = {}
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        try:
            with Image.open(row["abs_path"]) as image:
                residual = compute_residual(image.convert("RGB"))
        except Exception:
            skipped += 1
            continue
        logits = net(residual_to_tensor(residual))
        scores[row["id"]] = float(torch.sigmoid(logits).item())
        if progress_every and i % progress_every == 0:
            rate = i / (time.time() - t0)
            print("   %d/%d  %.0f img/s  eta %.0fs"
                  % (i, len(rows), rate, (len(rows) - i) / rate), flush=True)
    return scores, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ai_image_detection/train_v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--key", default="forensic_cnn")
    args = parser.parse_args()

    model_path = args.model or os.path.join(RESULTS_DIR,
                                            "forensic_model.pt")
    with open(os.path.join(args.root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    rows = [r for r in manifest if r["split"] in ("val", "test")]
    print("scoring %d images" % len(rows))

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    net = ForensicCNN()
    net.load_state_dict(payload["state_dict"])

    scores, skipped = score_rows(net, rows)
    print("scored %d, skipped %d" % (len(scores), skipped))

    size_bytes = os.path.getsize(model_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "scores_%s.json" % args.key)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "model": {
                    "name": "forensic_cnn_residual_%d" % count_parameters(net),
                    "type": "trained_residual_cnn",
                    "val_acc_at_selection": payload.get("val_acc"),
                    "epoch": payload.get("epoch"),
                },
                "seconds_per_image": 0.0,
                "disk_bytes": size_bytes,
                "n_images": len(scores),
            },
            "scores": scores,
        }, fh, indent=1)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
