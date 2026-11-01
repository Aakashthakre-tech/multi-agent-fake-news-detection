"""
metrics.py
==========
Metric definitions for the AI-image detector.

The positive class is AI-GENERATED. Every rate is reported explicitly so a
number is never ambiguous:

    ai_recall  = of all AI images, how many were caught        (1 - FNR)
    real_recall= of all real images, how many were left alone  (1 - FPR)
    fpr        = real images wrongly called AI
    fnr        = AI images wrongly called real

Nothing here is tuned to a dataset; these are plain definitions.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Positive class = AI-GENERATED.
POSITIVE_LABEL = 1


def _as_arrays(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    return y_true, y_pred


def confusion_counts(y_true, scores, threshold):
    """Raw confusion counts. Returns tp, fp, tn, fn."""
    y_true, y_pred = _as_arrays(y_true, scores, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(tn), int(fn)


def evaluate_scores(y_true, scores, threshold=0.5, uncertain_band=None):
    """
    Full metric bundle for one threshold.

    uncertain_band: optional (low, high). Scores strictly inside the band
    become UNCERTAIN and are excluded from the binary counts, so the reported
    precision/recall describe only the decided images. When None, behaviour is
    a plain binary decision at `threshold`.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if uncertain_band is None:
        mask = np.ones(len(scores), dtype=bool)
        y_true, y_pred = _as_arrays(y_true, scores, threshold)
    else:
        low, high = uncertain_band
        uncertain = (scores > low) & (scores < high)
        mask = ~uncertain
        y_true = y_true[mask]
        y_pred = (scores[mask] >= threshold).astype(int)

    n_total = int(len(scores))
    n_uncertain = int(n_total - mask.sum())
    n_decided = int(mask.sum())

    if n_decided == 0:
        return {
            "threshold": float(threshold),
            "n_total": n_total,
            "n_decided": 0,
            "n_uncertain": n_uncertain,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "macro_f1": float("nan"),
            "ai_recall": float("nan"),
            "real_recall": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "uncertain_rate": n_uncertain / n_total if n_total else float("nan"),
        }

    tp, fp, tn, fn = confusion_counts(y_true, y_pred, threshold)

    n_ai = tp + fn
    n_real = tn + fp
    n_pred_ai = tp + fp
    n_pred_real = tn + fn

    result = {
        "threshold": float(threshold),
        "n_total": n_total,
        "n_decided": n_decided,
        "n_uncertain": n_uncertain,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # precision / recall / f1 are for the AI class (the positive class)
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "ai_precision": (tp / n_pred_ai) if n_pred_ai else float("nan"),
        "ai_recall": (tp / n_ai) if n_ai else float("nan"),
        "real_precision": (tn / n_pred_real) if n_pred_real else float("nan"),
        "real_recall": (tn / n_real) if n_real else float("nan"),
        "fpr": (fp / n_real) if n_real else float("nan"),
        "fnr": (fn / n_ai) if n_ai else float("nan"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "uncertain_rate": n_uncertain / n_total if n_total else float("nan"),
    }
    return result


def threshold_sweep(y_true, scores, step=0.01):
    """Evaluate every threshold on a grid. Returns a list of metric bundles."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    grid = np.arange(0.0, 1.0 + step / 2, step)
    return [evaluate_scores(y_true, scores, float(t)) for t in grid]


def best_threshold(y_true, scores, metric="macro_f1", step=0.01):
    """
    Grid-search the threshold that maximises `metric` on the supplied data.

    Default is macro_f1, and deliberately not plain f1.

    On a class-balanced evaluation set, a degenerate rule that calls every
    image AI scores precision 0.5, recall 1.0 and therefore F1 0.667. Plain
    F1 will happily select that rule, producing a 100% false positive rate.
    Macro F1 gives it 0.333 instead, so a genuinely balanced operating point
    is only chosen when it is actually better.

    This must only ever be called on a validation split. The returned
    threshold is then applied unchanged to the test split.
    """
    sweep = threshold_sweep(y_true, scores, step=step)
    usable = [r for r in sweep if not np.isnan(r[metric])]
    if not usable:
        return None, None
    best = max(usable, key=lambda r: r[metric])
    return best["threshold"], best


def reliability_bins(y_true, scores, n_bins=10):
    """
    Bin predicted AI-probability and measure the observed AI rate per bin.

    This is the evidence for whether a reported probability can be read as a
    real-world frequency. A well calibrated model has observed frequency close
    to the mean predicted probability in every bin.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            sel = (scores >= lo) & (scores <= hi)
        else:
            sel = (scores > lo) & (scores <= hi)
        n = int(sel.sum())
        rows.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "count": n,
            "mean_predicted": float(scores[sel].mean()) if n else float("nan"),
            "observed_ai_rate": float(y_true[sel].mean()) if n else float("nan"),
        })
    return rows


def expected_calibration_error(y_true, scores, n_bins=10):
    """Standard ECE: mean gap between confidence and observed frequency."""
    rows = reliability_bins(y_true, scores, n_bins=n_bins)
    total = sum(r["count"] for r in rows)
    if total == 0:
        return float("nan")
    ece = 0.0
    for r in rows:
        if r["count"] == 0:
            continue
        gap = abs(r["observed_ai_rate"] - r["mean_predicted"])
        ece += (r["count"] / total) * gap
    return float(ece)
