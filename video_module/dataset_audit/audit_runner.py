import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System")
OUTPUT_DIR = PROJECT_ROOT / "video_module" / "fvc_downloads"
AUDIT_DIR = PROJECT_ROOT / "video_module" / "dataset_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.csv"
REPORT_PATH = OUTPUT_DIR / "download_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "download_summary.json"
FVC_PATH = PROJECT_ROOT / "video_module" / "fake-video-corpus-master" / "FVC.csv"
DUP_PATH = PROJECT_ROOT / "video_module" / "fake-video-corpus-master" / "fvc_downloader" / "FVC_dup.csv"
QUERIES_PATH = PROJECT_ROOT / "video_module" / "fake-video-corpus-master" / "FVC_text_queries.csv"

def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_audit():
    print("=== STARTING TRUTHLENSAI FVC DATASET AUDIT ===")

    # 1. Load Manifest
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"[Manifest] Total records in dataset_manifest.csv: {len(manifest_rows)}")

    # Check physical files
    missing_files = []
    fake_on_disk = list((OUTPUT_DIR / "fake").glob("*.mp4"))
    real_on_disk = list((OUTPUT_DIR / "real").glob("*.mp4"))
    all_disk_files = {p.name: p for p in fake_on_disk + real_on_disk}
    
    print(f"[Disk] Fake folder count: {len(fake_on_disk)}")
    print(f"[Disk] Real folder count: {len(real_on_disk)}")
    print(f"[Disk] Total MP4s on disk: {len(all_disk_files)}")

    manifest_cids = set()
    manifest_by_cid = {}
    label_mismatches = []
    file_hashes = {}
    hash_to_cids = defaultdict(list)

    durations_fake = []
    durations_real = []
    sizes_fake = []
    sizes_real = []
    resolutions = defaultdict(int)

    for r in manifest_rows:
        cid = r["cascade_id"]
        manifest_cids.add(cid)
        manifest_by_cid[cid] = r
        expected_label = r["label"]
        file_path = Path(r["file_path"])
        
        if not file_path.exists():
            missing_files.append((cid, str(file_path)))
            continue
            
        actual_parent = file_path.parent.name
        if actual_parent != expected_label:
            label_mismatches.append((cid, expected_label, actual_parent))

        # Hashes
        f_hash = compute_file_hash(file_path)
        file_hashes[cid] = f_hash
        hash_to_cids[f_hash].append((cid, expected_label, r["file_name"]))

        dur = float(r.get("duration_sec", 0))
        sz_mb = float(r.get("file_size_bytes", 0)) / (1024 * 1024)
        w = int(r.get("width", 0))
        h = int(r.get("height", 0))
        res_str = f"{w}x{h}"
        resolutions[res_str] += 1

        if expected_label == "fake":
            durations_fake.append(dur)
            sizes_fake.append(sz_mb)
        else:
            durations_real.append(dur)
            sizes_real.append(sz_mb)

    # Exact duplicate analysis
    exact_duplicates = {h: items for h, items in hash_to_cids.items() if len(items) > 1}
    print(f"[Audit] Content-Hash Exact Duplicate Groups: {len(exact_duplicates)}")
    suspicious_hash_conflicts = []
    for h, items in exact_duplicates.items():
        labels = {item[1] for item in items}
        cids = [item[0] for item in items]
        print(f"   Hash {h[:12]}... shared by {cids} | labels={labels}")
        if len(labels) > 1:
            suspicious_hash_conflicts.append((h, items))

    # 2. Check FVC original and fallback mappings
    fvc_rows = {}
    with open(FVC_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fvc_rows[r["cascade_id"]] = r

    # Check FVC text queries for event titles
    event_by_cid = {}
    event_groups = defaultdict(list)
    if QUERIES_PATH.exists():
        with open(QUERIES_PATH, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 4:
                    cid = row[0].strip()
                    title = row[3].strip()
                    if title:
                        event_by_cid[cid] = title
                        event_groups[title.lower()].append(cid)

    # Check near-duplicate URL overlap from FVC_dup.csv
    dup_url_to_cids = defaultdict(list)
    cid_to_dup_urls = defaultdict(set)
    if DUP_PATH.exists():
        with open(DUP_PATH, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = r.get("cascade_id", "").strip()
                u = r.get("video_url", "").strip()
                if c and u:
                    dup_url_to_cids[u].append(c)
                    cid_to_dup_urls[c].add(u)

    # 3. Construct Graph / Disjoint Set of Related Videos for Leakage Prevention
    # A group connects videos that share:
    # (a) Identical file content hash
    # (b) Same event_title
    # (c) Overlapping duplicate URLs in FVC_dup.csv
    # (d) Same actual_url downloaded

    adj = defaultdict(set)
    
    # Connect by exact content hash
    for h, items in exact_duplicates.items():
        cids = [item[0] for item in items]
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                adj[cids[i]].add(cids[j])
                adj[cids[j]].add(cids[i])

    # Connect by same event_title (only for downloaded cids)
    for title, cids in event_groups.items():
        present_cids = [c for c in cids if c in manifest_cids]
        if len(present_cids) > 1:
            for i in range(len(present_cids)):
                for j in range(i + 1, len(present_cids)):
                    adj[present_cids[i]].add(present_cids[j])
                    adj[present_cids[j]].add(present_cids[i])

    # Connect by actual_url downloaded
    url_to_manifest_cids = defaultdict(list)
    for r in manifest_rows:
        act_url = r.get("actual_url", "").strip()
        if act_url:
            url_to_manifest_cids[act_url].append(r["cascade_id"])
    for u, cids in url_to_manifest_cids.items():
        if len(cids) > 1:
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    adj[cids[i]].add(cids[j])
                    adj[cids[j]].add(cids[i])

    # Connect by shared near-duplicate URLs in FVC_dup
    for u, cids in dup_url_to_cids.items():
        present_cids = [c for c in set(cids) if c in manifest_cids]
        if len(present_cids) > 1:
            for i in range(len(present_cids)):
                for j in range(i + 1, len(present_cids)):
                    adj[present_cids[i]].add(present_cids[j])
                    adj[present_cids[j]].add(present_cids[i])

    # Connected Components (BFS)
    visited = set()
    clusters = []
    all_manifest_cids_sorted = sorted(list(manifest_cids))

    for c in all_manifest_cids_sorted:
        if c not in visited:
            comp = []
            queue = [c]
            visited.add(c)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(comp))

    print(f"\n[Leakage Grouping] Found {len(clusters)} independent clusters across {len(manifest_cids)} videos.")
    multi_video_clusters = [cl for cl in clusters if len(cl) > 1]
    print(f"[Leakage Grouping] Multi-video clusters: {len(multi_video_clusters)}")
    for i, cl in enumerate(multi_video_clusters[:10], 1):
        cl_labels = [manifest_by_cid[x]['label'] for x in cl]
        print(f"   Cluster #{i} ({len(cl)} videos): {cl} -> {cl_labels}")

    # 4. Propose Leakage-Safe Stratified Train/Val/Test Split (70% / 15% / 15%)
    # Use StratifiedGroupKFold or deterministic stratified binning over clusters
    from sklearn.model_selection import StratifiedGroupKFold

    # Prepare sample-level arrays for StratifiedGroupKFold
    sample_cids = []
    sample_labels = []
    sample_groups = []

    for group_id, cl in enumerate(clusters):
        for c in cl:
            sample_cids.append(c)
            sample_labels.append(1 if manifest_by_cid[c]["label"] == "fake" else 0)
            sample_groups.append(group_id)

    sample_cids = np.array(sample_cids)
    sample_labels = np.array(sample_labels)
    sample_groups = np.array(sample_groups)

    # First split: 70% train vs 30% temp (val + test)
    # Using 10 folds: folds 0..6 (7 folds ~ 70%) for train, fold 7,8 for val, fold 9,10 for test
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(sgkf.split(sample_cids, sample_labels, sample_groups))

    # Aggregate folds: 7 folds for train (70%), 1.5 folds for val (15%), 1.5 folds for test (15%)
    # Folds 0..6: train
    # Fold 7 and first half of fold 8: val
    # Fold 9 and second half of fold 8: test
    train_indices = []
    for f in range(7):
        train_indices.extend(splits[f][1])

    val_indices = list(splits[7][1])
    test_indices = list(splits[8][1])
    # Distribute fold 9 between val and test by groups
    fold_9_groups = defaultdict(list)
    for idx in splits[9][1]:
        fold_9_groups[sample_groups[idx]].append(idx)
    
    for g_id, g_indices in fold_9_groups.items():
        if len(val_indices) <= len(test_indices):
            val_indices.extend(g_indices)
        else:
            test_indices.extend(g_indices)

    train_cids = [sample_cids[i] for i in train_indices]
    val_cids = [sample_cids[i] for i in val_indices]
    test_cids = [sample_cids[i] for i in test_indices]

    train_counts = {"fake": sum(1 for c in train_cids if manifest_by_cid[c]["label"] == "fake"),
                    "real": sum(1 for c in train_cids if manifest_by_cid[c]["label"] == "real")}
    val_counts = {"fake": sum(1 for c in val_cids if manifest_by_cid[c]["label"] == "fake"),
                  "real": sum(1 for c in val_cids if manifest_by_cid[c]["label"] == "real")}
    test_counts = {"fake": sum(1 for c in test_cids if manifest_by_cid[c]["label"] == "fake"),
                   "real": sum(1 for c in test_cids if manifest_by_cid[c]["label"] == "real")}

    print("\nActual Split Results (Zero Leakage):")
    print(f"   Train: {len(train_cids)} videos (fake: {train_counts['fake']}, real: {train_counts['real']})")
    print(f"   Val:   {len(val_cids)} videos (fake: {val_counts['fake']}, real: {val_counts['real']})")
    print(f"   Test:  {len(test_cids)} videos (fake: {test_counts['fake']}, real: {test_counts['real']})")
    print(f"   Sum:   {len(train_cids) + len(val_cids) + len(test_cids)} == {len(manifest_rows)}")

    # Verify no overlap between splits
    assert len(set(train_cids) & set(val_cids)) == 0, "Leakage between train and val!"
    assert len(set(train_cids) & set(test_cids)) == 0, "Leakage between train and test!"
    assert len(set(val_cids) & set(test_cids)) == 0, "Leakage between val and test!"

    # Save split manifests
    def write_split_csv(split_name, cids_list):
        out_p = AUDIT_DIR / f"{split_name}_manifest.csv"
        rows_to_write = [manifest_by_cid[c] for c in sorted(cids_list)]
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows_to_write)
        print(f"   Saved {split_name} manifest: {out_p} ({len(rows_to_write)} rows)")

    write_split_csv("train", train_cids)
    write_split_csv("val", val_cids)
    write_split_csv("test", test_cids)

    # Save clusters
    with open(AUDIT_DIR / "leakage_groups.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_clusters": len(clusters),
            "multi_video_clusters_count": len(multi_video_clusters),
            "clusters": clusters
        }, f, indent=2)

    # 5. Compute Statistics
    total_fake = sum(1 for r in manifest_rows if r["label"] == "fake")
    total_real = sum(1 for r in manifest_rows if r["label"] == "real")

    audit_summary = {
        "dataset_name": "TruthLensAI Fake Video Corpus (FVC)",
        "audit_timestamp": "2026-10-17",
        "total_manifest_records": len(manifest_rows),
        "total_files_on_disk": len(all_disk_files),
        "missing_files_count": len(missing_files),
        "label_mismatches_count": len(label_mismatches),
        "class_counts": {
            "fake": total_fake,
            "real": total_real,
            "fake_ratio": round(total_fake / len(manifest_rows), 4),
            "real_ratio": round(total_real / len(manifest_rows), 4),
        },
        "exact_content_duplicate_groups": len(exact_duplicates),
        "exact_duplicates_details": [
            {"hash": h[:16], "cascades": [item[0] for item in items], "labels": list({item[1] for item in items})}
            for h, items in exact_duplicates.items()
        ],
        "suspicious_conflicting_labels_on_identical_video": [
            {"hash": h[:16], "cascades": [item[0] for item in items], "labels": list({item[1] for item in items})}
            for h, items in suspicious_hash_conflicts
        ],
        "duration_seconds": {
            "fake": {
                "min": round(float(np.min(durations_fake)), 2),
                "max": round(float(np.max(durations_fake)), 2),
                "mean": round(float(np.mean(durations_fake)), 2),
                "median": round(float(np.median(durations_fake)), 2),
                "std": round(float(np.std(durations_fake)), 2),
            },
            "real": {
                "min": round(float(np.min(durations_real)), 2),
                "max": round(float(np.max(durations_real)), 2),
                "mean": round(float(np.mean(durations_real)), 2),
                "median": round(float(np.median(durations_real)), 2),
                "std": round(float(np.std(durations_real)), 2),
            },
            "all": {
                "min": round(float(np.min(durations_fake + durations_real)), 2),
                "max": round(float(np.max(durations_fake + durations_real)), 2),
                "mean": round(float(np.mean(durations_fake + durations_real)), 2),
                "median": round(float(np.median(durations_fake + durations_real)), 2),
                "std": round(float(np.std(durations_fake + durations_real)), 2),
            }
        },
        "file_size_mb": {
            "fake": {
                "min": round(float(np.min(sizes_fake)), 2),
                "max": round(float(np.max(sizes_fake)), 2),
                "mean": round(float(np.mean(sizes_fake)), 2),
                "median": round(float(np.median(sizes_fake)), 2),
                "std": round(float(np.std(sizes_fake)), 2),
            },
            "real": {
                "min": round(float(np.min(sizes_real)), 2),
                "max": round(float(np.max(sizes_real)), 2),
                "mean": round(float(np.mean(sizes_real)), 2),
                "median": round(float(np.median(sizes_real)), 2),
                "std": round(float(np.std(sizes_real)), 2),
            },
            "total_size_gb": round(sum(sizes_fake + sizes_real) / 1024, 2),
        },
        "resolution_distribution": dict(sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:15]),
        "proposed_split": {
            "split_strategy": "GroupKFold / Connected Components on (exact hash, event_title, fallback duplicate clusters)",
            "leakage_safe": True,
            "train": {"total": len(train_cids), "fake": train_counts["fake"], "real": train_counts["real"]},
            "val": {"total": len(val_cids), "fake": val_counts["fake"], "real": val_counts["real"]},
            "test": {"total": len(test_cids), "fake": test_counts["fake"], "real": test_counts["real"]},
        }
    }

    with open(AUDIT_DIR / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n[Audit] Saved full audit report to: {AUDIT_DIR / 'audit_report.json'}")

if __name__ == "__main__":
    run_audit()
