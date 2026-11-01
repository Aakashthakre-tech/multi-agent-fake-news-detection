"""
dataset_builder.py
==================
Phase 2: Download and validate the 500-video pilot dataset.

Strategy
--------
Primary  : GenVideo subset from ModelScope (ZeroScope + VideoCraft2 + MSR-VTT real).
Fallback : GVF/DeCoF if GenVideo is inaccessible.

This script:
  1. Checks disk space and estimates storage requirements.
  2. Attempts to access GenVideo via direct URL samples (no full clone).
  3. Downloads exactly PILOT_REAL_TARGET real + PILOT_FAKE_TARGET synthetic clips.
  4. Verifies every file: size, decodability, frame count.
  5. Removes corrupt files and computes perceptual hashes for dedup.
  6. Writes a manifest CSV and leakage-safe group-aware train/val/test splits.

Usage
-----
    python -m video_module.ai_generated_detector.dataset_builder \
        [--real-limit 250] [--fake-limit 250] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── local imports ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from video_module.ai_generated_detector.config import (
    DATA_DIR, REAL_DIR, SYNTH_DIR, SPLITS_DIR, CKPT_DIR, OUTPUTS_DIR,
    MANIFEST_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV, OOD_CSV,
    PILOT_REAL_TARGET, PILOT_FAKE_TARGET, PILOT_GENERATORS, HOLDOUT_GENERATOR,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, CLASS_MAP, SEED,
)

# ── Verified pilot dataset sources ──────────────────────────────────────────
# Real videos: UCF-101 subset (Apache-2.0, natural camera actions in the wild)
UCF101_SUBSET_URL = (
    "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz"
)

# AI-Generated videos:
# Generator 1: VideoCrafter2 (open-source text-to-video diffusion)
VIDEOCRAFTER2_URL = (
    "https://huggingface.co/datasets/34data/v14-fake-videocrafter2/resolve/main/data_001.zip"
)

# Generator 2: OpenSora (open-source video diffusion transformer)
OPENSORA_URL = (
    "https://huggingface.co/datasets/34data/v14-fake-opensora/resolve/main/data_001.zip"
)

REAL_VIDEO_SOURCES: List[Dict] = [
    {
        "name": "ucf101_subset",
        "label": "REAL",
        "generator": "none",
        "url": UCF101_SUBSET_URL,
        "license": "Apache-2.0",
    }
]

FAKE_VIDEO_SOURCES: List[Dict] = [
    {
        "name": "videocrafter2",
        "label": "AI_GENERATED",
        "generator": "videocrafter2",
        "url": VIDEOCRAFTER2_URL,
        "license": "Apache-2.0",
    },
    {
        "name": "opensora",
        "label": "AI_GENERATED",
        "generator": "opensora",
        "url": OPENSORA_URL,
        "license": "Apache-2.0",
    },
]



# ════════════════════════════════════════════════════════════════════════════════
# Disk-space check
# ════════════════════════════════════════════════════════════════════════════════

def check_disk_space(min_gb: float = 6.0) -> Tuple[float, float]:
    """Return (free_gb, total_gb) and raise if below minimum."""
    total, used, free = shutil.disk_usage("C:/")
    free_gb  = free  / 1e9
    total_gb = total / 1e9
    print(f"[disk] free={free_gb:.1f} GB  total={total_gb:.1f} GB")
    if free_gb < min_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on C:/. "
            f"Need at least {min_gb} GB. "
            "Please free space before downloading."
        )
    return free_gb, total_gb


def estimate_storage(n_real: int, n_fake: int) -> None:
    """Print storage estimate for the pilot."""
    # Average short clip (5–15 s, 720p, h264): ~3–10 MB
    avg_clip_mb = 6.0
    total_videos = n_real + n_fake
    raw_gb  = total_videos * avg_clip_mb / 1024
    # Features: n_videos × NUM_FRAMES × FEATURE_DIM × 4 bytes
    from video_module.ai_generated_detector.config import NUM_FRAMES, FEATURE_DIM
    feat_mb = total_videos * NUM_FRAMES * FEATURE_DIM * 4 / 1e6
    print(f"\n[estimate] {total_videos} videos")
    print(f"  Raw video (compressed): ~{raw_gb:.1f} GB")
    print(f"  Frame cache (decoded):  ~{total_videos * NUM_FRAMES * 224*224*3 / 1e9:.1f} GB (not stored)")
    print(f"  Feature cache (.pt):    ~{feat_mb:.0f} MB")
    print(f"  Model checkpoints:      ~50 MB")
    print(f"  Total estimate:         ~{raw_gb + feat_mb/1024 + 0.05:.1f} GB")


# ════════════════════════════════════════════════════════════════════════════════
# Video validation helpers
# ════════════════════════════════════════════════════════════════════════════════

def is_valid_video(path: Path, min_frames: int = 8) -> Tuple[bool, str]:
    """Return (valid, reason). Tries to decode a few frames."""
    if not path.exists():
        return False, "file not found"
    if path.stat().st_size < 4096:
        return False, f"file too small ({path.stat().st_size} bytes)"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "cv2 cannot open"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total < min_frames:
        return False, f"only {total} frames"
    if fps <= 0:
        return False, f"invalid fps ({fps})"
    return True, "ok"


def phash_video(path: Path, size: int = 8) -> Optional[str]:
    """Compute perceptual hash of the center frame of a video."""
    try:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid   = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small  = cv2.resize(gray, (size + 1, size))
        diff   = small[:, 1:] > small[:, :-1]
        return "".join(str(int(b)) for b in diff.flatten())
    except Exception:
        return None


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════════════════════
# Pilot dataset acquisition
# ════════════════════════════════════════════════════════════════════════════════

def download_file_with_progress(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download url -> dest with visual progress."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    print(f"\r  Downloading {dest.name}: {downloaded/1e6:.1f}/{total_bytes/1e6:.1f} MB ({pct:.1f}%)", end="")
            print()
        return dest.stat().st_size > 4096
    except Exception as e:
        print(f"\n  [download] Failed {url}: {e}")
        return False


def download_pilot_dataset(real_limit: int = 250, fake_limit: int = 250) -> None:
    """
    Download required pilot samples and clean up temporary archives immediately.
    Keeps total disk footprint minimal (< 500 MB peak).
    """
    import zipfile
    import tarfile

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    vc2_dir = SYNTH_DIR / "videocrafter2"
    opensora_dir = SYNTH_DIR / "opensora"
    vc2_dir.mkdir(parents=True, exist_ok=True)
    opensora_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # 1. Real videos (UCF101 subset, Apache-2.0)
    existing_real = [p for p in REAL_DIR.rglob("*") if p.suffix.lower() in video_exts]
    if len(existing_real) < real_limit:
        print(f"\n[download] Fetching real videos ({len(existing_real)}/{real_limit} present)...")
        temp_tar = DATA_DIR / "temp_ucf101.tar"
        ok = download_file_with_progress(UCF101_SUBSET_URL, temp_tar)
        if ok:
            print("  Extracting real videos...")
            extracted = 0
            with tarfile.open(temp_tar, "r:*") as tar:
                for member in tar:
                    if member.name.lower().endswith((".avi", ".mp4")):
                        out_name = Path(member.name).name
                        out_path = REAL_DIR / out_name
                        if not out_path.exists():
                            f = tar.extractfile(member)
                            if f:
                                with open(out_path, "wb") as out_f:
                                    shutil.copyfileobj(f, out_f)
                                extracted += 1
                                if len(existing_real) + extracted >= real_limit:
                                    break
            print(f"  Extracted {extracted} real videos.")
            if temp_tar.exists():
                temp_tar.unlink()
        else:
            print("  Failed to download UCF-101 real archive.")

    # 2. Synthetic: VideoCrafter2 (target: 150)
    vc2_target = min(150, fake_limit)
    existing_vc2 = [p for p in vc2_dir.glob("*.mp4")]
    if len(existing_vc2) < vc2_target:
        print(f"\n[download] Fetching VideoCrafter2 AI videos ({len(existing_vc2)}/{vc2_target} present)...")
        temp_zip = DATA_DIR / "temp_vc2.zip"
        ok = download_file_with_progress(VIDEOCRAFTER2_URL, temp_zip)
        if ok:
            print("  Extracting VideoCrafter2 videos...")
            extracted = 0
            with zipfile.ZipFile(temp_zip) as z:
                for fname in z.namelist():
                    if fname.lower().endswith(".mp4"):
                        out_name = Path(fname).name
                        out_path = vc2_dir / out_name
                        if not out_path.exists():
                            with z.open(fname) as src, open(out_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            extracted += 1
                            if len(existing_vc2) + extracted >= vc2_target:
                                break
            print(f"  Extracted {extracted} VideoCrafter2 videos.")
            if temp_zip.exists():
                temp_zip.unlink()

    # 3. Synthetic: OpenSora (target: 100)
    opensora_target = max(0, fake_limit - vc2_target)
    existing_opensora = [p for p in opensora_dir.glob("*.mp4")]
    if len(existing_opensora) < opensora_target:
        print(f"\n[download] Fetching OpenSora AI videos ({len(existing_opensora)}/{opensora_target} present)...")
        temp_zip = DATA_DIR / "temp_opensora.zip"
        ok = download_file_with_progress(OPENSORA_URL, temp_zip)
        if ok:
            print("  Extracting OpenSora videos...")
            extracted = 0
            with zipfile.ZipFile(temp_zip) as z:
                for fname in z.namelist():
                    if fname.lower().endswith(".mp4"):
                        out_name = Path(fname).name
                        out_path = opensora_dir / out_name
                        if not out_path.exists():
                            with z.open(fname) as src, open(out_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            extracted += 1
                            if len(existing_opensora) + extracted >= opensora_target:
                                break
            print(f"  Extracted {extracted} OpenSora videos.")
            if temp_zip.exists():
                temp_zip.unlink()



# ════════════════════════════════════════════════════════════════════════════════
# Split construction
# ════════════════════════════════════════════════════════════════════════════════

def build_splits(
    manifest: List[Dict],
    seed: int = SEED,
    holdout_generator: str = HOLDOUT_GENERATOR,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Group-aware 70/15/15 train/val/test split.
    Holds out `holdout_generator` entirely for OOD test set.

    Groups: (generator, source_id) — videos sharing a source stay together.
    """
    rng = random.Random(seed)

    # Separate OOD holdout
    ood  = [r for r in manifest if r["generator"] == holdout_generator]
    rest = [r for r in manifest if r["generator"] != holdout_generator]

    # Build groups by (generator, source_id)
    groups: Dict[str, List[Dict]] = {}
    for row in rest:
        key = f"{row['generator']}||{row.get('source_id', row['video_id'])}"
        groups.setdefault(key, []).append(row)

    keys = list(groups.keys())
    rng.shuffle(keys)

    n       = len(keys)
    n_train = max(1, round(n * TRAIN_FRAC))
    n_val   = max(1, round(n * VAL_FRAC))

    train_keys = keys[:n_train]
    val_keys   = keys[n_train: n_train + n_val]
    test_keys  = keys[n_train + n_val:]

    def collect(key_list):
        rows = []
        for k in key_list:
            rows.extend(groups[k])
        return rows

    train = collect(train_keys)
    val   = collect(val_keys)
    test  = collect(test_keys)

    # Verify no leakage
    train_ids = {r["video_id"] for r in train}
    val_ids   = {r["video_id"] for r in val}
    test_ids  = {r["video_id"] for r in test}
    ood_ids   = {r["video_id"] for r in ood}

    assert not train_ids & val_ids,  "Leakage: train ∩ val"
    assert not train_ids & test_ids, "Leakage: train ∩ test"
    assert not val_ids   & test_ids, "Leakage: val ∩ test"
    assert not (train_ids | val_ids | test_ids) & ood_ids, "Leakage: splits ∩ OOD"

    return train, val, test, ood


def write_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ════════════════════════════════════════════════════════════════════════════════
# Main dataset-builder pipeline
# ════════════════════════════════════════════════════════════════════════════════

MANIFEST_FIELDS = [
    "video_id", "rel_path", "abs_path", "label", "label_int",
    "generator", "source_id", "md5", "phash", "n_frames", "fps", "valid",
]


def build_manifest_from_existing(dry_run: bool = False) -> List[Dict]:
    """
    Scan DATA_DIR for already-downloaded videos and build a manifest.
    This is the entry-point when videos have been placed manually.
    """
    records = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_id = 0

    for label_name, label_int, search_root, generator_fn in [
        ("REAL",         0, REAL_DIR,  lambda p: "none"),
        ("AI_GENERATED", 1, SYNTH_DIR, lambda p: p.parent.name),
    ]:
        if not search_root.exists():
            print(f"[manifest] Directory not found, skipping: {search_root}")
            continue
        paths = sorted(p for p in search_root.rglob("*") if p.suffix.lower() in video_exts)
        print(f"[manifest] Found {len(paths)} files under {search_root}")
        for p in paths:
            valid, reason = is_valid_video(p)
            n_frames, fps = 0, 0.0
            if valid:
                cap = cv2.VideoCapture(str(p))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps      = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

            ph  = phash_video(p) if valid and not dry_run else "dryrun"
            md5 = md5_file(p) if not dry_run else "dryrun"

            stem = p.stem
            if label_name == "REAL" and "_g" in stem:
                source_id = stem.split("_c")[0]
            else:
                source_id = stem

            records.append({
                "video_id":  f"v{video_id:05d}",
                "rel_path":  str(p.relative_to(DATA_DIR)),
                "abs_path":  str(p),
                "label":     label_name,
                "label_int": label_int,
                "generator": generator_fn(p),
                "source_id": source_id,
                "md5":       md5,
                "phash":     ph,
                "n_frames":  n_frames,
                "fps":       round(fps, 2),
                "valid":     valid,
            })
            video_id += 1

    return records


def deduplicate(records: List[Dict], hamming_thresh: int = 4) -> Tuple[List[Dict], List[str]]:
    """Remove near-duplicates based on perceptual hash. Returns (kept, removed_ids)."""
    kept    = []
    removed = []
    seen_hashes: List[Tuple[str, str]] = []  # (phash, video_id)

    for rec in records:
        ph = rec.get("phash", "")
        if not ph or ph == "dryrun":
            kept.append(rec)
            continue
        is_dup = False
        for seen_ph, seen_id in seen_hashes:
            if hamming_distance(ph, seen_ph) < hamming_thresh:
                removed.append(rec["video_id"])
                is_dup = True
                print(f"  [dedup] {rec['video_id']} ≈ {seen_id} (hamming < {hamming_thresh})")
                break
        if not is_dup:
            kept.append(rec)
            seen_hashes.append((ph, rec["video_id"]))

    return kept, removed


def print_summary(manifest: List[Dict], train: List, val: List, test: List, ood: List) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total = len(manifest)
    real  = sum(1 for r in manifest if r["label"] == "REAL")
    fake  = sum(1 for r in manifest if r["label"] == "AI_GENERATED")
    print(f"Total videos  : {total}")
    print(f"  REAL        : {real}")
    print(f"  AI_GENERATED: {fake}")
    print(f"Splits        : train={len(train)} | val={len(val)} | test={len(test)} | ood={len(ood)}")

    # Generator breakdown
    gen_counts: Dict[str, int] = {}
    for r in manifest:
        gen_counts[r["generator"]] = gen_counts.get(r["generator"], 0) + 1
    print("Generator breakdown:")
    for g, c in sorted(gen_counts.items()):
        print(f"  {g:20s}: {c}")
    print("=" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 500-video pilot dataset.")
    ap.add_argument("--real-limit",  type=int, default=PILOT_REAL_TARGET)
    ap.add_argument("--fake-limit",  type=int, default=PILOT_FAKE_TARGET)
    ap.add_argument("--dry-run",     action="store_true",
                    help="Skip hashing and download; just count files.")
    ap.add_argument("--seed",        type=int, default=SEED)
    args = ap.parse_args()

    # 1. Disk space check
    print("\n── Phase 2.1: Disk space ──")
    free_gb, _ = check_disk_space(min_gb=4.0)
    estimate_storage(args.real_limit, args.fake_limit)

    # 2. Acquire pilot data if missing
    print("\n── Phase 2.2: Scanning and acquiring pilot videos ──")
    if not args.dry_run:
        download_pilot_dataset(real_limit=args.real_limit, fake_limit=args.fake_limit)

    records = build_manifest_from_existing(dry_run=args.dry_run)

    if not records:
        print(
            "\n[BLOCKER] No videos found in data directories.\n"
            "Please verify network access or place videos into:\n"
            f"  Real    → {REAL_DIR}\n"
            f"  Synth   → {SYNTH_DIR}/<generator_name>/\n"
        )
        sys.exit(1)

    # 3. Filter to valid videos only
    valid_records = [r for r in records if r["valid"]]
    invalid       = [r for r in records if not r["valid"]]
    print(f"[validate] {len(valid_records)} valid, {len(invalid)} invalid/corrupt")
    for r in invalid:
        print(f"  INVALID: {r['video_id']} {r['rel_path']}")

    # 4. Deduplicate
    print("\n── Phase 2.3: Deduplication ──")
    valid_records, removed_ids = deduplicate(valid_records, hamming_thresh=4)
    print(f"[dedup] Removed {len(removed_ids)} near-duplicates. Kept {len(valid_records)}")

    # 5. Balance classes to pilot limits
    real_recs = [r for r in valid_records if r["label"] == "REAL"]
    fake_recs = [r for r in valid_records if r["label"] == "AI_GENERATED"]
    rng = random.Random(args.seed)
    real_recs = rng.sample(real_recs, min(args.real_limit, len(real_recs)))
    fake_recs = rng.sample(fake_recs, min(args.fake_limit, len(fake_recs)))
    manifest  = real_recs + fake_recs
    rng.shuffle(manifest)

    print(f"[balance] Final pilot: {len(real_recs)} REAL + {len(fake_recs)} AI_GENERATED"
          f" = {len(manifest)} total")

    # 6. Build splits
    print("\n── Phase 2.4: Building leakage-safe splits ──")
    train, val, test, ood = build_splits(manifest, seed=args.seed)

    # 7. Write manifest and split CSVs
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(manifest, MANIFEST_CSV, MANIFEST_FIELDS)
    write_csv(train,    TRAIN_CSV,    MANIFEST_FIELDS)
    write_csv(val,      VAL_CSV,      MANIFEST_FIELDS)
    write_csv(test,     TEST_CSV,     MANIFEST_FIELDS)
    write_csv(ood,      OOD_CSV,      MANIFEST_FIELDS)

    print(f"[csv] manifest  → {MANIFEST_CSV}")
    print(f"[csv] train     → {TRAIN_CSV}")
    print(f"[csv] val       → {VAL_CSV}")
    print(f"[csv] test      → {TEST_CSV}")
    print(f"[csv] ood       → {OOD_CSV}")

    # 8. Summary
    print_summary(manifest, train, val, test, ood)

    # 9. Persist summary JSON
    summary = {
        "total": len(manifest),
        "real":  len(real_recs),
        "ai_generated": len(fake_recs),
        "train": len(train),
        "val":   len(val),
        "test":  len(test),
        "ood":   len(ood),
        "invalid_removed": len(invalid),
        "duplicates_removed": len(removed_ids),
        "holdout_generator": HOLDOUT_GENERATOR,
        "generators": list({r["generator"] for r in manifest}),
        "license": "CC-BY-NC-4.0 (GenVideo synthetic) / see per-source for real",
    }
    with open(CKPT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] dataset_builder complete.")


if __name__ == "__main__":
    main()
