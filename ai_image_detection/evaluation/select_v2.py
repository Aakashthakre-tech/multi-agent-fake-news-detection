"""
select_v2.py
============
Chooses the final configuration under the train/val/test protocol.

Everything is decided on VALIDATION, then applied once to TEST:

    model choice      -> VAL macro F1
    ensemble choice   -> VAL macro F1, and only if it beats the best single
    temperature       -> fitted on VAL by NLL
    threshold         -> selected on VAL in the calibrated space
    uncertainty band  -> selected on VAL around that threshold

Scores are read from the caches produced by run_benchmark.py (Vision
Transformer candidates) and score_forensic.py (trained CNN), so no model is
re-run here.
"""

import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.metrics import (  # noqa: E402
    evaluate_scores, best_threshold, expected_calibration_error,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

MAX_UNCERTAIN_RATE = 0.35
AI, REAL = 1, 0

CANDIDATES = ["steganalysis_current", "forensic_cnn"]


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y, p, grid=None):
    if grid is None:
        grid = np.arange(0.25, 6.01, 0.05)
    z = _logit(p)
    best_t, best_nll = None, None
    for t in grid:
        q = np.clip(_sigmoid(z / t), 1e-12, 1 - 1e-12)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if best_nll is None or nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def load(key):
    path = os.path.join(RESULTS_DIR, "scores_%s.json" % key)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["scores"], payload.get("meta", {})


def pool(manifest, key, split):
    scores, _ = load(key)
    ys, ps = [], []
    for row in manifest:
        if row["split"] != split:
            continue
        if row["id"] not in scores:
            continue
        ys.append(row["label"])
        ps.append(scores[row["id"]])
    return np.array(ys), np.array(ps)


def fit_band(y, p, threshold):
    best_w, best_acc = 0.0, -1
    w = 0.0
    while w <= 0.60 + 1e-9:
        m = evaluate_scores(y, p, threshold,
                            uncertain_band=(threshold - w, threshold + w))
        if m["uncertain_rate"] <= MAX_UNCERTAIN_RATE and not np.isnan(m["accuracy"]):
            if m["accuracy"] > best_acc + 1e-12:
                best_acc, best_w = m["accuracy"], w
        w += 0.01
    return best_w


def full_config(manifest, sources, split="val", method="mean"):
    """
    Fit temperature, threshold and band for one combination of models on the
    given split. sources is a list of (key, weight) or a single key.
    """
    if isinstance(sources, str):
        sources = [(sources, 1.0)]
    else:
        # Accept a bare list of model keys as well as (key, weight) pairs.
        sources = [
            (s, 1.0) if isinstance(s, str) else (s[0], s[1])
            for s in sources
        ]

    ys = None
    stack = []
    for key, weight in sources:
        y, p = pool(manifest, key, split)
        if ys is None:
            ys = y
        elif len(y) != len(ys):
            # restrict to ids present in both
            continue
        stack.append(p * weight)

    if not stack or ys is None or len(ys) == 0:
        return None

    if method == "mean":
        combined = np.sum(stack, axis=0) / sum(w for _, w in sources)
    else:
        votes = np.sum(
            [(p >= 0.5).astype(float) * w for p, (_, w) in zip(stack, sources)],
            axis=0)
        combined = votes / sum(w for _, w in sources)

    temperature = fit_temperature(ys, combined)
    cal = _sigmoid(_logit(combined) / temperature)
    threshold, _ = best_threshold(ys, cal, metric="macro_f1")
    band = fit_band(ys, cal, threshold)

    m = evaluate_scores(ys, cal, threshold,
                        uncertain_band=(threshold - band, threshold + band))
    return {
        "sources": [k for k, _ in sources],
        "method": method,
        "temperature": temperature,
        "threshold": threshold,
        "uncertain_band": band,
        "val_metrics": m,
        "val_ece_raw": expected_calibration_error(ys, combined),
        "val_ece_calibrated": expected_calibration_error(ys, cal),
    }


def apply_config(manifest, cfg, split):
    ys = None
    stack = []
    weights = []
    for key in cfg["sources"]:
        y, p = pool(manifest, key, split)
        if ys is None:
            ys = y
        elif len(y) != len(ys):
            return None
        stack.append(p)
        weights.append(1.0)
    if not stack:
        return None
    combined = np.mean(stack, axis=0)
    cal = _sigmoid(_logit(combined) / cfg["temperature"])
    return ys, evaluate_scores(ys, cal, cfg["threshold"],
                               uncertain_band=(cfg["threshold"] - cfg["uncertain_band"],
                                               cfg["threshold"] + cfg["uncertain_band"]))


def main(root):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    n_val = sum(1 for r in manifest if r["split"] == "val")
    n_test = sum(1 for r in manifest if r["split"] == "test")
    print("manifest: val=%d test=%d" % (n_val, n_test))

    # ------------------------------------------------ VAL: pick configuration
    print()
    print("=" * 104)
    print("VALIDATION - every choice made here")
    print("=" * 104)

    singles = {}
    for key in CANDIDATES:
        cfg = full_config(manifest, key, "val")
        if cfg:
            singles[key] = cfg
            m = cfg["val_metrics"]
            print("%-24s T=%5.2f thr=%.3f band=%.2f  acc=%.4f macroF1=%.4f "
                  "ai_rec=%.4f real_rec=%.4f fpr=%.4f fnr=%.4f unc=%.3f"
                  % (key, cfg["temperature"], cfg["threshold"],
                     cfg["uncertain_band"], m["accuracy"], m["macro_f1"],
                     m["ai_recall"], m["real_recall"], m["fpr"], m["fnr"],
                     m["uncertain_rate"]))

    combos = {}
    available = [k for k in CANDIDATES if k in singles]
    for size in (2, 3):
        for combo in combinations(available, size):
            for method in ("mean", "agreement"):
                cfg = full_config(manifest, list(combo), "val", method)
                if cfg:
                    combos["+".join(combo) + " [%s]" % method] = cfg

    if combos:
        print()
        print("ensembles (VAL):")
        for name, cfg in sorted(
                combos.items(),
                key=lambda kv: -kv[1]["val_metrics"]["macro_f1"]):
            m = cfg["val_metrics"]
            print("%-52s macroF1=%.4f acc=%.4f fpr=%.4f fnr=%.4f"
                  % (name, m["macro_f1"], m["accuracy"], m["fpr"], m["fnr"]))

    best_single = max(singles, key=lambda k: singles[k]["val_metrics"]["macro_f1"])
    print()
    print("best single on VAL: %s (macroF1=%.4f)"
          % (best_single, singles[best_single]["val_metrics"]["macro_f1"]))

    best_ens_name, best_ens = None, None
    if combos:
        best_ens_name = max(combos, key=lambda k: combos[k]["val_metrics"]["macro_f1"])
        best_ens = combos[best_ens_name]
        print("best ensemble on VAL: %s (macroF1=%.4f)"
              % (best_ens_name, best_ens["val_metrics"]["macro_f1"]))

    chosen = dict(singles[best_single])
    ensemble_improves = bool(
        best_ens is not None
        and best_ens["val_metrics"]["macro_f1"]
        > singles[best_single]["val_metrics"]["macro_f1"] + 1e-9)

    if ensemble_improves:
        chosen = dict(best_ens)
        print("-> ensemble ADOPTED (improves on VAL)")
    else:
        print("-> ensemble REJECTED (no VAL improvement); keep %s" % best_single)

    # ------------------------------------------------ LOCK, single TEST pass
    print()
    print("=" * 104)
    print("LOCKED: %s" % ("+".join(chosen["sources"]) + " [%s]" % chosen["method"]
                          if len(chosen["sources"]) > 1 else chosen["sources"][0]))
    print("=" * 104)

    out = {
        "selected_sources": chosen["sources"],
        "method": chosen["method"],
        "temperature": chosen["temperature"],
        "threshold": chosen["threshold"],
        "uncertain_band": chosen["uncertain_band"],
        "ensemble_improves_on_val": ensemble_improves,
        "val": chosen["val_metrics"],
        "candidates": {k: v["val_metrics"] for k, v in singles.items()},
        "test": {},
    }

    hdr = ("%-14s %6s %7s %7s %9s %9s %9s %9s %9s %8s %8s"
           % ("split", "n", "dec", "unc", "acc", "ai_prec", "ai_rec",
              "real_prec", "real_rec", "fpr", "fnr"))
    print()
    print(hdr)
    print("-" * len(hdr))
    for split in ("val", "test"):
        got = apply_config(manifest, chosen, split)
        if not got:
            continue
        _, m = got
        out["test" if split == "test" else "val_recheck"] = m
        print("%-14s %6d %7d %7d %9.4f %9.4f %9.4f %9.4f %9.4f %8.4f %8.4f"
              % (split, m["n_total"], m["n_decided"], m["n_uncertain"],
                 m["accuracy"], m["ai_precision"], m["ai_recall"],
                 m["real_precision"], m["real_recall"], m["fpr"], m["fnr"]))

    # every candidate on TEST, for the before/after table
    print()
    print("all single candidates on TEST (each with its own VAL-fitted config):")
    print(hdr)
    print("-" * len(hdr))
    out["test_all_candidates"] = {}
    for key, cfg in singles.items():
        got = apply_config(manifest, cfg, "test")
        if not got:
            continue
        _, m = got
        out["test_all_candidates"][key] = m
        print("%-14s %6d %7d %7d %9.4f %9.4f %9.4f %9.4f %9.4f %8.4f %8.4f"
              % (key, m["n_total"], m["n_decided"], m["n_uncertain"],
                 m["accuracy"], m["ai_precision"], m["ai_recall"],
                 m["real_precision"], m["real_recall"], m["fpr"], m["fnr"]))

    if ensemble_improves and best_ens is not None:
        got = apply_config(manifest, best_ens, "test")
        if got:
            _, m = got
            out["test_ensemble"] = m
            print("%-14s %6d %7d %7d %9.4f %9.4f %9.4f %9.4f %9.4f %8.4f %8.4f"
                  % ("ENSEMBLE", m["n_total"], m["n_decided"], m["n_uncertain"],
                     m["accuracy"], m["ai_precision"], m["ai_recall"],
                     m["real_precision"], m["real_recall"], m["fpr"], m["fnr"]))

    m = out["test"]
    if m:
        print()
        print("confusion matrix (TEST, decided only):")
        print("                    predicted REAL   predicted AI")
        print("   actual REAL   %15d %15d" % (m["tn"], m["fp"]))
        print("   actual AI     %15d %15d" % (m["fn"], m["tp"]))

    with open(os.path.join(RESULTS_DIR, "final_config_v2.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print()
    print("written: %s" % os.path.join(RESULTS_DIR, "final_config_v2.json"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ai_image_detection/train_v1")
