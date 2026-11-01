"""
analyze.py
===========
Turns cached detector scores into the numbers that decide the final design.

Everything here is read-only with respect to the benchmark: it consumes the
score files produced by run_benchmark.py and never re-runs a model.

Protocol followed throughout:

    threshold / uncertainty band / temperature  -> chosen on VAL
    reported accuracy / F1 / FPR / FNR         -> measured on TEST

The test split is never used to pick anything.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.metrics import (  # noqa: E402
    evaluate_scores,
    best_threshold,
    expected_calibration_error,
    reliability_bins,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

METRIC_FIELDS = [
    "accuracy", "precision", "recall", "f1", "macro_f1",
    "ai_recall", "real_recall", "fpr", "fnr",
]


# ----------------------------------------------------------------- loading

def load_scores(model_key):
    path = os.path.join(RESULTS_DIR, "scores_%s.json" % model_key)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["scores"], payload.get("meta", {})


def available_models():
    if not os.path.isdir(RESULTS_DIR):
        return []
    out = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("scores_") and f.endswith(".json"):
            out.append(f[len("scores_"):-len(".json")])
    return out


def split_arrays(manifest, scores, split):
    """(y_true, p_ai) for one split, skipping images with no cached score."""
    ys, ps, ids = [], [], []
    for record in manifest:
        if record["split"] != split:
            continue
        if record["id"] not in scores:
            continue
        ys.append(record["label"])
        ps.append(scores[record["id"]])
        ids.append(record["id"])
    return np.array(ys), np.array(ps), ids


# ------------------------------------------------------------- calibration

def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y_true, probs, grid=None):
    """
    Fit a single temperature on validation data by minimising NLL.

    This is the standard scalar calibration for a binary classifier: it does
    not change the ranking of images, only how the probability is read, so it
    cannot fix a bad detector and is never used as if it could.
    """
    if grid is None:
        grid = np.arange(0.25, 5.01, 0.05)
    z = _logit(probs)
    best_t, best_nll = None, None
    for t in grid:
        p = _sigmoid(z / t)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -np.mean(
            y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        if best_nll is None or nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


# --------------------------------------------------------------- reporting

def _row(label, m, extra=None):
    row = {"row": label}
    for f in METRIC_FIELDS:
        row[f] = m.get(f)
    row["threshold"] = m.get("threshold")
    row["n_decided"] = m.get("n_decided")
    row["n_uncertain"] = m.get("n_uncertain")
    if extra:
        row.update(extra)
    return row


def print_table(title, rows):
    if not rows:
        return
    cols = ["row", "accuracy", "precision", "recall", "f1", "macro_f1",
            "ai_recall", "real_recall", "fpr", "fnr"]
    header = ("%-34s" % cols[0]) + "".join(
        "%12s" % c[:11] for c in cols[1:])
    print()
    print(title)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "%-34s" % str(r["row"])[:34]
        for c in cols[1:]:
            v = r.get(c)
            line += "%12s" % ("-" if v is None or
                              (isinstance(v, float) and np.isnan(v))
                              else "%.4f" % v)
        print(line)


# ---------------------------------------------------------------- analysis

def analyse_model(manifest, model_key, val_splits, test_splits):
    """
    Full per-model analysis: threshold chosen on VAL, reported on TEST.
    """
    scores, meta = load_scores(model_key)

    val_y, val_p = [], []
    for sp in val_splits:
        y, p, _ = split_arrays(manifest, scores, sp)
        if len(y):
            val_y.append(y)
            val_p.append(p)
    val_y = np.concatenate(val_y) if val_y else np.array([])
    val_p = np.concatenate(val_p) if val_p else np.array([])

    if len(val_y) == 0:
        return None

    thr, val_best = best_threshold(val_y, val_p, metric="macro_f1")
    # Accuracy-optimal threshold kept for comparison only; macro_f1 is the
    # criterion that avoids the degenerate all-AI operating point.
    thr_acc, val_best_acc = best_threshold(val_y, val_p, metric="accuracy")

    result = {
        "model_key": model_key,
        "meta": meta,
        "threshold": thr,
        "threshold_accuracy_optimal": thr_acc,
        "val": val_best,
        "val_accuracy_optimal": val_best_acc,
        "test": {},
        "test_by_split": {},
    }

    for sp in test_splits:
        y, p, _ = split_arrays(manifest, scores, sp)
        if len(y) == 0:
            continue
        m = evaluate_scores(y, p, thr)
        result["test_by_split"][sp] = m

    all_y, all_p = [], []
    for sp in test_splits:
        y, p, _ = split_arrays(manifest, scores, sp)
        if len(y):
            all_y.append(y)
            all_p.append(p)
    if all_y:
        result["test"] = evaluate_scores(
            np.concatenate(all_y), np.concatenate(all_p), thr)

    # Calibration
    result["val_ece"] = expected_calibration_error(val_y, val_p)
    temperature, _ = fit_temperature(val_y, val_p)
    result["temperature"] = temperature
    result["val_ece_calibrated"] = expected_calibration_error(
        val_y, _sigmoid(_logit(val_p) / temperature))
    if all_y:
        ty, tp_ = np.concatenate(all_y), np.concatenate(all_p)
        result["test_ece"] = expected_calibration_error(ty, tp_)
        result["test_ece_calibrated"] = expected_calibration_error(
            ty, _sigmoid(_logit(tp_) / temperature))
        result["test_calibrated"] = evaluate_scores(
            ty, _sigmoid(_logit(tp_) / temperature), thr)

    return result


def uncertainty_band(manifest, model_key, threshold, val_splits,
                     max_uncertain_rate=0.30, width=0.02):
    """
    Choose an UNCERTAIN band around the threshold using validation data only.

    Scores strictly inside (threshold - w, threshold + w) are reported as
    UNCERTAIN instead of being forced into a binary answer. w is the smallest
    width that maximises accuracy on the decided images while keeping the
    UNCERTAIN rate under the cap. Smaller is preferred, so the detector stays
    decisive by default.
    """
    scores, _ = load_scores(model_key)
    ys, ps = [], []
    for sp in val_splits:
        y, p, _ = split_arrays(manifest, scores, sp)
        if len(y):
            ys.append(y)
            ps.append(p)
    if not ys:
        return None, []
    y, p = np.concatenate(ys), np.concatenate(ps)

    curve = []
    w = 0.0
    while w <= 0.60 + 1e-9:
        m = evaluate_scores(y, p, threshold, uncertain_band=(threshold - w, threshold + w))
        curve.append({"width": round(w, 3), **{k: m[k] for k in
                      ("accuracy", "precision", "recall", "ai_recall",
                       "real_recall", "fpr", "fnr", "uncertain_rate",
                       "n_decided", "n_uncertain")}})
        w += width

    usable = [c for c in curve
              if c["uncertain_rate"] <= max_uncertain_rate
              and not np.isnan(c["accuracy"])]
    if not usable:
        return 0.0, curve

    best_acc = max(c["accuracy"] for c in usable)
    # smallest width achieving the best decided-set accuracy
    chosen = min((c for c in usable if c["accuracy"] >= best_acc - 1e-12),
                 key=lambda c: c["width"])
    return chosen["width"], curve


def ensemble_scores(manifest, model_keys, split, method="mean"):
    """
    Combine cached scores from several models for one split.

    Only the simple, defensible combinations are implemented: plain averaging
    and agreement. Anything more elaborate would need its own validation and
    there is no evidence yet that it is warranted.
    """
    if method not in ("mean", "agreement"):
        raise ValueError("unknown ensemble method %r" % method)

    per_model = []
    base_ids = None
    for key in model_keys:
        scores, _ = load_scores(key)
        y, p, ids = split_arrays(manifest, scores, split)
        if len(y) == 0:
            continue
        if base_ids is None:
            base_ids = ids
        elif ids != base_ids:
            common = [i for i in ids if i in set(base_ids)]
            keep = set(common)
            y = np.array([v for v, i in zip(y, ids) if i in keep])
            p = np.array([v for v, i in zip(p, ids) if i in keep])
            ids = common
        per_model.append((y, p))

    if not per_model:
        return None, None

    y0 = per_model[0][0]
    stack = np.vstack([p for _, p in per_model])

    if method == "mean":
        return y0, stack.mean(axis=0)

    # agreement: how many models call it AI, mapped to a probability
    votes = (stack >= 0.5).sum(axis=0).astype(float)
    return y0, votes / stack.shape[0]
