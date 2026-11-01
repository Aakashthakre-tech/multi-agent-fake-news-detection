"""
train_forensic.py
=================
Trains the forensic residual CNN on the TRAIN split only.

Augmentation is applied to TRAIN images only. Validation and test images are
read verbatim, so no augmented copy of a training image can ever appear in
either. That is enforced structurally: the augment function is only reachable
from the training dataset, never from the evaluation one.

Augmentations reflect what a real forwarded image goes through - messaging
apps re-compress it, social platforms resize it, screenshots add text and
flat regions - rather than cosmetic jitter chosen to flatter the model.
"""

import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.forensic import (  # noqa: E402
    ForensicCNN, compute_residual, residual_to_tensor, count_parameters,
    RESIDUAL_SIZE,
)


# ---------------------------------------------------------------- augment
# Only ever applied to training images.

def augment(image, rng):
    """
    Apply a random chain of realistic degradations to a training image.

    Each step is skipped with some probability so the model sees clean images
    as well, and the order is fixed to mimic a real pipeline: resize, crop,
    photographic adjustment, then compression last, because that is the
    order a messaging app produces.
    """
    image = image.convert("RGB")

    # rescale, simulating platform downscale or upscale
    if rng.random() < 0.7:
        scale = rng.uniform(0.45, 1.4)
        w, h = image.size
        new = (max(32, int(w * scale)), max(32, int(h * scale)))
        image = image.resize(new, Image.LANCZOS)

    # random crop
    if rng.random() < 0.5:
        w, h = image.size
        for _ in range(8):
            cw = rng.randint(int(w * 0.55), w)
            ch = rng.randint(int(h * 0.55), h)
            x0 = rng.randint(0, w - cw)
            y0 = rng.randint(0, h - ch)
            crop = image.crop((x0, y0, x0 + cw, y0 + ch))
            if min(crop.size) >= 64:
                image = crop
                break

    image = image.resize((RESIDUAL_SIZE * 4, RESIDUAL_SIZE * 4), Image.LANCZOS)

    # mild blur
    if rng.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))

    # brightness / contrast
    if rng.random() < 0.4:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))

    # JPEG recompression, in memory so nothing is written back to disk
    if rng.random() < 0.8:
        quality = rng.randint(35, 92)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    return image


# ---------------------------------------------------------------- dataset

class ResidualDataset(torch.utils.data.Dataset):
    def __init__(self, rows, training, seed=0):
        self.rows = rows
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            image = Image.open(row["abs_path"])
            if self.training:
                image = augment(image, random.Random(self.seed + index))
            else:
                image = image.convert("RGB")
            residual = compute_residual(image)
        except Exception:
            # A corrupt file must not take down the run; it is reported as a
            # skipped sample rather than silently relabelled.
            return torch.zeros(3, RESIDUAL_SIZE, RESIDUAL_SIZE), -1.0

        return residual_to_tensor(residual)[0], float(row["label"])


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return xs, ys


# ---------------------------------------------------------------- training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ai_image_detection/train_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=20000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(os.path.join(args.root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    train_rows = [r for r in manifest if r["split"] == "train"]
    val_rows = [r for r in manifest if r["split"] == "val"]
    random.Random(1234).shuffle(train_rows)
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]

    print("train=%d val=%d" % (len(train_rows), len(val_rows)))

    net = ForensicCNN()
    print("parameters: %d" % count_parameters(net))

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    loader = torch.utils.data.DataLoader(
        ResidualDataset(train_rows, training=True, seed=7),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=True,
    )

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "forensic_model.pt",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for i, (xs, ys) in enumerate(loader, 1):
            optimiser.zero_grad()
            logits = net(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * xs.size(0)
            seen += xs.size(0)
            if i % 40 == 0:
                print("  epoch %d  batch %d/%d  loss %.4f  (%.1fs)"
                      % (epoch, i, len(loader), running / seen,
                         time.time() - t0), flush=True)
        scheduler.step()

        val_loss, val_acc, skipped = evaluate(net, val_rows)
        print("epoch %d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
              "skipped=%d  (%.0fs)"
              % (epoch, running / max(1, seen), val_loss, val_acc, skipped,
                 time.time() - t0), flush=True)

        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": net.state_dict(),
                        "val_acc": val_acc, "epoch": epoch}, out_path)
            print("  saved (best val_acc %.4f)" % val_acc, flush=True)

    with open(out_path + ".history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    print("best val_acc: %.4f -> %s" % (best_val, out_path))


@torch.no_grad()
def evaluate(net, rows, batch_size=64):
    net.eval()
    loader = torch.utils.data.DataLoader(
        ResidualDataset(rows, training=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate,
    )
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    counted = 0
    skipped = 0
    for xs, ys in loader:
        logits = net(xs)
        loss = criterion(logits, ys)
        valid = ys >= 0
        skipped += int((~valid).sum())
        if valid.any():
            total_loss += float(loss.item()) * int(valid.sum())
            correct += int(((logits[valid] >= 0).float() == ys[valid]).sum())
            counted += int(valid.sum())
    return (total_loss / max(1, counted),
            correct / max(1, counted),
            skipped)


if __name__ == "__main__":
    main()
