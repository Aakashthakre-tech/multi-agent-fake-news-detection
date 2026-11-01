"""
report.py
=========
Produces the full experimental report from cached scores.

Stages, in order:
  1. per-model threshold chosen on VAL, reported on TEST
  2. threshold sensitivity sweep
  3. generalisation slices on TEST
  4. UNCERTAIN band chosen on VAL
  5. ensemble search, adopted only if it beats the best single model on TEST
  6. confidence calibration

Writes results/report.json and results/report.txt.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.datasets import load_manifest  # noqa: E402
from ai_image_detection.evaluation import analyze as A  # noqa: E402
from ai_image_detection.evaluation.metrics import (  # noqa: E402
    evaluate_scores, threshold_sweep, reliability_bins,
)

VAL_SPLITS = ["cifake_val", "hemg_val"]
TEST_SPLITS = ["cifake_test", "hemg_test"]
RESULTS_DIR = A.RESULTS_DIR


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float) and np.isnan(x):
        return "-"
    return ("%%.%df" % nd) % x


def print_metric_table(title, rows):
    if not rows:
        print("\n%s\n(no data)" % title)
        return
    cols = ["model", "n", "accuracy", "precision", "recall", "f1",
            "macro_f1", "ai_recall", "real_recall", "fpr", "fnr"]
    print()
    print(title)
    print("-" * 118)
    print("%-26s %5s %8s %9s %8s %8s %9s %9s %10s %7s %7s" % tuple(cols))
    print("-" * 118)
    for r in rows:
        print("%-26s %5s %8s %9s %8s %8s %9s %9s %10s %7s %7s" % (
            str(r.get("model"))[:26],
            r.get("n", "-"),
            _fmt(r.get("accuracy")), _fmt(r.get("precision")),
            _fmt(r.get("recall")), _fmt(r.get("f1")), _fmt(r.get("macro_f1")),
            _fmt(r.get("ai_recall")), _fmt(r.get("real_recall")),
            _fmt(r.get("fpr")), _fmt(r.get("fnr")),
        ))
    print("-" * 118)


def resolution_bucket(w, h):
    long_side = max(w, h)
    if long_side <= 64:
        return "tiny (<=64px)"
    if long_side <= 256:
        return "small (<=256px)"
    if long_side <= 512:
        return "medium (<=512px)"
    return "large (>512px)"


def main(root):
    manifest = load_manifest(root)
    models = A.available_models()

    report = {"benchmark_root": os.path.abspath(root),
              "n_manifest": len(manifest), "models": {}}

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 118)
    emit("TRUTHLENS AI - AI IMAGE DETECTOR EVALUATION")
    emit("=" * 118)
    emit("benchmark : %s" % os.path.abspath(root))
    emit("images    : %d manifest records" % len(manifest))
    emit("val splits: %s" % ", ".join(VAL_SPLITS))
    emit("test splits: %s" % ", ".join(TEST_SPLITS))
    emit("models    : %s" % ", ".join(models))

    # ---------------------------------------------------------- stage 1
    emit()
    emit("#" * 118)
    emit("# STAGE 1 - THRESHOLD CHOSEN ON VAL, REPORTED ON TEST")
    emit("#" * 118)

    main_rows = []
    for key in models:
        res = A.analyse_model(manifest, key, VAL_SPLITS, TEST_SPLITS)
        if res is None:
            emit("  %s: no validation scores" % key)
            continue
        report["models"][key] = res

        meta = res.get("meta", {})
        model_name = meta.get("model", {}).get("name", key)
        row = dict(res["test"])
        row["model"] = key
        row["n"] = res["test"].get("n_total")
        main_rows.append(row)

        emit()
        emit("model %s (%s)" % (key, model_name))
        emit("  label mapping : %s" % meta.get("model", {}).get("label_mapping_source"))
        emit("  id2label       : %s" % meta.get("model", {}).get("id2label"))
        emit("  threshold (VAL-selected) : %s" % _fmt(res["threshold"]))
        emit("  VAL  f1=%s acc=%s fpr=%s fnr=%s" % (
            _fmt(res["val"]["f1"]), _fmt(res["val"]["accuracy"]),
            _fmt(res["val"]["fpr"]), _fmt(res["val"]["fnr"])))
        for sp, m in res["test_by_split"].items():
            emit("  TEST %-12s n=%-5s acc=%s f1=%s macro_f1=%s ai_rec=%s "
                 "real_rec=%s fpr=%s fnr=%s" % (
                     sp, m["n_total"], _fmt(m["accuracy"]), _fmt(m["f1"]),
                     _fmt(m["macro_f1"]), _fmt(m["ai_recall"]),
                     _fmt(m["real_recall"]), _fmt(m["fpr"]), _fmt(m["fnr"])))

    print_metric_table("TEST RESULTS - ALL MODELS (threshold from VAL)", main_rows)

    # runtime / size
    emit()
    emit("INFERENCE COST (CPU)")
    emit("%-26s %14s %14s %14s" % ("model", "sec/image", "disk MB", "n images"))
    for key in models:
        meta = report["models"].get(key, {}).get("meta", {})
        if not meta:
            continue
        emit("%-26s %14.3f %14.1f %14d" % (
            key, meta.get("seconds_per_image", float("nan")),
            meta.get("disk_bytes", 0) / 1e6, meta.get("n_images", 0)))

    # ---------------------------------------------------------- stage 2
    emit()
    emit("#" * 118)
    emit("# STAGE 2 - THRESHOLD SENSITIVITY (measured on TEST, for understanding only)")
    emit("#" * 118)

    for key in models:
        res = report["models"].get(key)
        if not res:
            continue
        scores, _ = A.load_scores(key)
        ys, ps = [], []
        for sp in TEST_SPLITS:
            y, p, _ = A.split_arrays(manifest, scores, sp)
            if len(y):
                ys.append(y)
                ps.append(p)
        if not ys:
            continue
        y, p = np.concatenate(ys), np.concatenate(ps)

        emit()
        emit("model %s" % key)
        emit("%-9s %9s %9s %9s %9s %9s %9s %9s %9s" % (
            "thresh", "acc", "prec", "recall", "f1", "macroF1", "ai_rec",
            "fpr", "fnr"))
        for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
            m = evaluate_scores(y, p, t)
            emit("%-9.2f %9s %9s %9s %9s %9s %9s %9s %9s" % (
                t, _fmt(m["accuracy"]), _fmt(m["precision"]), _fmt(m["recall"]),
                _fmt(m["f1"]), _fmt(m["macro_f1"]), _fmt(m["ai_recall"]),
                _fmt(m["fpr"]), _fmt(m["fnr"])))

    # ---------------------------------------------------------- stage 3
    emit()
    emit("#" * 118)
    emit("# STAGE 3 - GENERALISATION SLICES (TEST)")
    emit("#" * 118)

    for key in models:
        res = report["models"].get(key)
        if not res:
            continue
        scores, _ = A.load_scores(key)
        thr = res["threshold"]
        by_id = {r["id"]: r for r in manifest}

        slices = {}
        for sp in TEST_SPLITS:
            y, p, ids = A.split_arrays(manifest, scores, sp)
            if not len(y):
                continue
            buckets = {}
            for i, ident in enumerate(ids):
                rec = by_id[ident]
                # Slice on the ORIGINAL source resolution, not the canonical
                # 256x256 the benchmark stores. Every stored image is 256x256,
                # so using the stored size would collapse all slices into one.
                b = resolution_bucket(
                    rec.get("orig_width", rec["width"]),
                    rec.get("orig_height", rec["height"]))
                buckets.setdefault(b, ([], []))
                buckets[b][0].append(y[i])
                buckets[b][1].append(p[i])
            for b, (yy, pp) in buckets.items():
                slices[b] = evaluate_scores(np.array(yy), np.array(pp), thr)

        if slices:
            rows = [{"model": b, "n": slices[b]["n_total"],
                     **{k: slices[b][k] for k in
                        ("accuracy", "precision", "recall", "ai_recall",
                         "real_recall", "fpr", "fnr", "f1")}}
                    for b in sorted(slices)]
            print_metric_table("resolution slices - %s" % key, rows)
            report["models"][key]["resolution_slices"] = slices

    # ---------------------------------------------------------- stage 4
    emit()
    emit("#" * 118)
    emit("# STAGE 4 - UNCERTAIN BAND (chosen on VAL)")
    emit("#" * 118)

    for key in models:
        res = report["models"].get(key)
        if not res:
            continue
        width, curve = A.uncertainty_band(
            manifest, key, res["threshold"], VAL_SPLITS)
        res["uncertain_width"] = width
        res["uncertain_curve"] = curve

        emit()
        emit("model %s -> band +/-%s around %s" % (
            key, width, _fmt(res["threshold"])))
        emit("%-9s %9s %9s %9s %9s %9s %9s" % (
            "width", "unc_rate", "acc_dec", "prec", "ai_rec", "fpr", "n_unc"))
        for c in curve:
            if c["width"] > 0.30:
                continue
            emit("%-9.2f %9s %9s %9s %9s %9s %9d" % (
                c["width"], _fmt(c["uncertain_rate"]), _fmt(c["accuracy"]),
                _fmt(c["precision"]), _fmt(c["ai_recall"]), _fmt(c["fpr"]),
                c["n_uncertain"]))

        # effect of the chosen band on TEST
        scores, _ = A.load_scores(key)
        ys, ps = [], []
        for sp in TEST_SPLITS:
            y, p, _ = A.split_arrays(manifest, scores, sp)
            if len(y):
                ys.append(y)
                ps.append(p)
        if ys and width:
            m = evaluate_scores(
                np.concatenate(ys), np.concatenate(ps), res["threshold"],
                uncertain_band=(res["threshold"] - width, res["threshold"] + width))
            res["test_with_uncertain"] = m
            emit("  TEST with band: decided=%d uncertain=%d acc_on_decided=%s "
                 "prec=%s ai_rec=%s fpr=%s" % (
                     m["n_decided"], m["n_uncertain"], _fmt(m["accuracy"]),
                     _fmt(m["precision"]), _fmt(m["ai_recall"]), _fmt(m["fpr"])))

    # ---------------------------------------------------------- stage 5
    emit()
    emit("#" * 118)
    emit("# STAGE 5 - ENSEMBLE (only adopted if it beats the best single model)")
    emit("#" * 118)

    best_single = None
    if main_rows:
        best_single = max(main_rows, key=lambda r: r["macro_f1"])

    combos = []
    if len(models) >= 2:
        from itertools import combinations
        for size in (2, 3):
            for combo in combinations(models, size):
                for method in ("mean", "agreement"):
                    combos.append((combo, method))

    ens_rows = []
    for combo, method in combos:
        vy, vp = [], []
        for sp in VAL_SPLITS:
            y, p = A.ensemble_scores(manifest, combo, sp, method)
            if y is not None and len(y):
                vy.append(y)
                vp.append(p)
        if not vy:
            continue
        vy, vp = np.concatenate(vy), np.concatenate(vp)
        from ai_image_detection.evaluation.metrics import best_threshold
        thr, _ = best_threshold(vy, vp, metric="macro_f1")

        ty, tp = [], []
        for sp in TEST_SPLITS:
            y, p = A.ensemble_scores(manifest, combo, sp, method)
            if y is not None and len(y):
                ty.append(y)
                tp.append(p)
        if not ty:
            continue
        m = evaluate_scores(np.concatenate(ty), np.concatenate(tp), thr)
        row = dict(m)
        row["model"] = "+".join(combo) + " [%s]" % method
        ens_rows.append(row)

    print_metric_table("ENSEMBLE RESULTS (TEST)", ens_rows)

    report["ensembles"] = ens_rows
    if best_single and ens_rows:
        best_ens = max(ens_rows, key=lambda r: r["macro_f1"])
        improved = best_ens["macro_f1"] > best_single["macro_f1"]
        emit()
        emit("comparison metric: MACRO F1 (plain F1 is degenerate on a "
             "balanced split)")
        emit("best single   : %-46s macroF1=%s acc=%s"
             % (best_single["model"], _fmt(best_single["macro_f1"]),
                _fmt(best_single["accuracy"])))
        emit("best ensemble : %-46s macroF1=%s acc=%s"
             % (best_ens["model"], _fmt(best_ens["macro_f1"]),
                _fmt(best_ens["accuracy"])))
        emit("VERDICT: ensemble %s" % (
            "IMPROVES - candidate" if improved else
            "does NOT improve - keep best single model"))
        report["ensemble_improves"] = bool(improved)

    # ---------------------------------------------------------- stage 6
    emit()
    emit("#" * 118)
    emit("# STAGE 6 - CONFIDENCE CALIBRATION")
    emit("#" * 118)

    emit()
    emit("%-26s %10s %10s %10s %10s" % (
        "model", "val_ECE", "val_ECE_T", "test_ECE", "test_ECE_T"))
    emit("%-26s %10s %10s %10s %10s" % (
        "", "raw", "temp=%s" % "T", "raw", "scaled"))
    cal_rows = []
    for key in models:
        res = report["models"].get(key)
        if not res:
            continue
        emit("%-26s %10s %10s %10s %10s   T=%s" % (
            key, _fmt(res.get("val_ece")), _fmt(res.get("val_ece_calibrated")),
            _fmt(res.get("test_ece")), _fmt(res.get("test_ece_calibrated")),
            _fmt(res.get("temperature"), 2)))
        cal_rows.append({
            "model": key,
            "val_ece": res.get("val_ece"),
            "val_ece_calibrated": res.get("val_ece_calibrated"),
            "test_ece": res.get("test_ece"),
            "test_ece_calibrated": res.get("test_ece_calibrated"),
            "temperature": res.get("temperature"),
        })

    # reliability curve for the leading model
    if main_rows:
        lead = max(main_rows, key=lambda r: r["macro_f1"])["model"]
        res = report["models"].get(lead)
        if res:
            scores, _ = A.load_scores(lead)
            ty, tp = [], []
            for sp in TEST_SPLITS:
                y, p, _ = A.split_arrays(manifest, scores, sp)
                if len(y):
                    ty.append(y)
                    tp.append(p)
            if ty:
                emit()
                emit("reliability bins - %s (TEST, raw scores)" % lead)
                emit("%-16s %8s %14s %16s" % (
                    "bin", "count", "mean_predicted", "observed_ai_rate"))
                for row in reliability_bins(np.concatenate(ty),
                                           np.concatenate(tp)):
                    if row["count"] == 0:
                        continue
                    emit("%-16s %8d %14s %16s" % (
                        "%.1f-%.1f" % (row["bin_low"], row["bin_high"]),
                        row["count"], _fmt(row["mean_predicted"]),
                        _fmt(row["observed_ai_rate"])))

    report["calibration"] = cal_rows

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print()
    print("report written to %s" % os.path.join(RESULTS_DIR, "report.txt"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ai_image_detection/benchmark_v2")
