"""
select_final.py
===============
Produces the frozen production configuration.

The whole configuration is fitted on VALIDATION data in the correct order,
because the steps are dependent:

    1. temperature            fitted on VAL (calibration of probabilities)
    2. decision threshold     selected on VAL in the CALIBRATED space
    3. UNCERTAIN band         selected on VAL around that threshold
    4. LOCK
    5. evaluate once on TEST

Re-selecting a threshold on raw scores after temperature scaling would be
wrong: scaling moves every probability, so the operating point has to be
re-derived in the space the detector will actually run in.

The script also runs a paired bootstrap on TEST to test whether an ensemble
beats the best single model by more than sampling noise. A difference that is
not significant is not a reason to accept three times the inference cost.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ai_image_detection.evaluation.analyze import (  # noqa: E402
    load_scores, split_arrays, fit_temperature, uncertainty_band,
)
from ai_image_detection.evaluation.metrics import (  # noqa: E402
    evaluate_scores, best_threshold, expected_calibration_error,
)

VAL_SPLITS = ["cifake_val", "hemg_val"]
TEST_SPLITS = ["cifake_test", "hemg_test"]
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

MAX_UNCERTAIN_RATE = 0.35


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _pool(manifest, scores, splits):
    ys, ps, ids = [], [], []
    for sp in splits:
        y, p, i = split_arrays(manifest, scores, sp)
        if len(y):
            ys.append(y)
            ps.append(p)
            ids.extend(i)
    if not ys:
        return None, None, None
    return np.concatenate(ys), np.concatenate(ps), ids


def fit_on_val(manifest, key):
    """Fit temperature, threshold and uncertainty band on VAL only."""
    scores, meta = load_scores(key)
    vy, vp, _ = _pool(manifest, scores, VAL_SPLITS)
    if vy is None:
        return None

    temperature, _ = fit_temperature(vy, vp)
    vcal = _sigmoid(_logit(vp) / temperature)

    threshold, val_metrics = best_threshold(vy, vcal, metric="macro_f1")

    # uncertainty band fitted in the calibrated space
    band = 0.0
    curve = []
    w = 0.0
    while w <= 0.60 + 1e-9:
        m = evaluate_scores(vy, vcal, threshold,
                            uncertain_band=(threshold - w, threshold + w))
        curve.append({"width": round(w, 3), "uncertain_rate": m["uncertain_rate"],
                      "accuracy_decided": m["accuracy"],
                      "ai_recall": m["ai_recall"], "real_recall": m["real_recall"],
                      "fpr": m["fpr"], "fnr": m["fnr"]})
        w += 0.01

    usable = [c for c in curve
              if c["uncertain_rate"] <= MAX_UNCERTAIN_RATE
              and not np.isnan(c["accuracy_decided"])]
    if usable:
        best = max(c["accuracy_decided"] for c in usable)
        chosen = min((c for c in usable
                      if c["accuracy_decided"] >= best - 1e-12),
                     key=lambda c: c["width"])
        band = chosen["width"]
    val_final = evaluate_scores(vy, vcal, threshold,
                                uncertain_band=(threshold - band,
                                                threshold + band))

    return {
        "model_key": key,
        "model": meta.get("model", {}),
        "temperature": temperature,
        "threshold": threshold,
        "uncertain_band": band,
        "val": val_final,
        "val_no_uncertain": evaluate_scores(vy, vcal, threshold),
        "val_ece_raw": expected_calibration_error(vy, vp),
        "val_ece_calibrated": expected_calibration_error(vy, vcal),
        "band_curve": curve,
    }


def apply_locked(manifest, key, temperature, threshold, band, splits):
    scores, _ = load_scores(key)
    y, p, _ = _pool(manifest, scores, splits)
    if y is None:
        return None
    cal = _sigmoid(_logit(p) / temperature)
    return evaluate_scores(y, cal, threshold,
                           uncertain_band=(threshold - band, threshold + band))


def ensemble_pool(manifest, keys, splits, method="mean"):
    """Average calibrated scores across models for one split group."""
    ys, ps = [], []
    base = None
    for k in keys:
        scores, _ = load_scores(k)
        y, p, ids = split_arrays(manifest, scores, splits[0])
        # gather the same ids across all splits
        yy, pp = [], []
        for sp in splits:
            y2, p2, i2 = split_arrays(manifest, scores, sp)
            if len(y2):
                yy.append(y2)
                pp.append(p2)
        if not yy:
            continue
        yy = np.concatenate(yy)
        pp = np.concatenate(pp)
        if base is None:
            base = yy
        ys.append(pp)
    if not ys:
        return None, None
    stack = np.vstack(ys)
    if method == "mean":
        return base, stack.mean(axis=0)
    votes = (stack >= 0.5).sum(axis=0).astype(float)
    return base, votes / stack.shape[0]


def paired_bootstrap(y, pred_a_correct, pred_b_correct, n_boot=2000, seed=7):
    """
    Paired bootstrap over images. Returns the probability that model B beats
    model A, so a near-0.5 result means the difference is sampling noise.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(pred_b_correct[idx].mean() - pred_a_correct[idx].mean())
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "p_b_better_than_a": float((diffs > 0).mean()),
        "ci95": [float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))],
    }


def main(root):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    from ai_image_detection.evaluation.analyze import available_models
    keys = available_models()

    fitted = {}
    for k in keys:
        f = fit_on_val(manifest, k)
        if f:
            fitted[k] = f

    print("=" * 100)
    print("CONFIGURATION FITTED ON VALIDATION (TEST NOT TOUCHED YET)")
    print("=" * 100)
    print("%-24s %10s %10s %10s %10s %10s"
          % ("model", "T", "thresh", "band", "val_acc", "val_unc%"))
    for k, f in fitted.items():
        print("%-24s %10.2f %10.3f %10.3f %10.4f %10.2f"
              % (k, f["temperature"], f["threshold"], f["uncertain_band"],
                 f["val"]["accuracy"], 100 * f["val"]["uncertain_rate"]))

    # ---- lock the best single model on VAL macro F1
    best_key = max(fitted, key=lambda k: fitted[k]["val"]["macro_f1"]
                   if not np.isnan(fitted[k]["val"]["macro_f1"]) else -1)
    best = fitted[best_key]
    print()
    print("best single on VAL: %s" % best_key)

    # ---- ensemble candidates, also fitted on VAL
    ens_candidates = {}
    for k in keys:
        if k == best_key:
            continue
        combo = tuple(sorted([best_key, k]))
        vy, vp = ensemble_pool(manifest, combo, VAL_SPLITS)
        if vy is None:
            continue
        t, _ = fit_temperature(vy, vp)
        vcal = _sigmoid(_logit(vp) / t)
        thr, vm = best_threshold(vy, vcal, metric="macro_f1")
        w = 0.0
        while w <= 0.60 + 1e-9:
            m = evaluate_scores(vy, vcal, thr,
                                uncertain_band=(thr - w, thr + w))
            w += 0.01
            if m["uncertain_rate"] > MAX_UNCERTAIN_RATE or np.isnan(m["accuracy"]):
                continue
            if w - 0.01 == 0.0:
                band, best_acc = 0.0, m["accuracy"]
            elif m["accuracy"] >= best_acc - 1e-12:
                band, best_acc = w - 0.01, m["accuracy"]
        ens_candidates["+".join(combo)] = {
            "temperature": t, "threshold": thr, "band": band,
            "val": evaluate_scores(vy, vcal, thr,
                                   uncertain_band=(thr - band, thr + band)),
            "combo": combo,
        }

    # ---- lock: single vs ensemble, judged on VAL first
    ens_best_key = None
    if ens_candidates:
        ens_best_key = max(ens_candidates,
                           key=lambda k: ens_candidates[k]["val"]["macro_f1"])
        ev = ens_candidates[ens_best_key]["val"]["macro_f1"]
        if not (ev > best["val"]["macro_f1"]):
            ens_best_key = None

    print("ensemble accepted on VAL: %s"
          % (ens_best_key if ens_best_key else "NONE (no VAL gain)"))

    # =================================================================
    # LOCKED. Single pass over TEST below.
    # =================================================================
    print()
    print("=" * 100)
    print("LOCKED CONFIGURATION - SINGLE PASS OVER TEST")
    print("=" * 100)

    final = {
        "selected_model": best_key,
        "temperature": best["temperature"],
        "threshold": best["threshold"],
        "uncertain_band": best["uncertain_band"],
        "val_splits": VAL_SPLITS,
        "test_splits": TEST_SPLITS,
        "max_uncertain_rate_target": MAX_UNCERTAIN_RATE,
    }

    for sp in TEST_SPLITS:
        m = apply_locked(manifest, best_key, best["temperature"],
                         best["threshold"], best["uncertain_band"], [sp])
        final.setdefault("test_by_split", {})[sp] = m
    m_all = apply_locked(manifest, best_key, best["temperature"],
                         best["threshold"], best["uncertain_band"], TEST_SPLITS)
    final["test_combined"] = m_all
    m_all_noband = apply_locked_no_band = None
    scores, _ = load_scores(best_key)
    ty, tp, _ = _pool(manifest, scores, TEST_SPLITS)
    tcal = _sigmoid(_logit(tp) / best["temperature"])
    final["test_combined_no_uncertain"] = evaluate_scores(
        ty, tcal, best["threshold"])
    final["test_ece_raw"] = expected_calibration_error(ty, tp)
    final["test_ece_calibrated"] = expected_calibration_error(ty, tcal)

    hdr = ("%-16s %6s %8s %8s %8s %9s %9s %9s %9s %8s %8s %8s"
           % ("split", "n", "dec", "unc", "unc%", "acc_dec", "ai_prec",
              "ai_rec", "real_prec", "real_rec", "fpr", "fnr"))
    print()
    print(hdr)
    print("-" * len(hdr))
    for sp, m in list(final["test_by_split"].items()) + [("COMBINED", m_all)]:
        print("%-16s %6d %8d %8d %8.1f %9.4f %9.4f %9.4f %9.4f %8.4f %8.4f %8.4f"
              % (sp, m["n_total"], m["n_decided"], m["n_uncertain"],
                 100 * m["uncertain_rate"], m["accuracy"], m["ai_precision"],
                 m["ai_recall"], m["real_precision"], m["real_recall"],
                 m["fpr"], m["fnr"]))

    nb = final["test_combined_no_uncertain"]
    print()
    print("same model, NO uncertainty band: acc=%.4f macroF1=%.4f "
          "fpr=%.4f fnr=%.4f"
          % (nb["accuracy"], nb["macro_f1"], nb["fpr"], nb["fnr"]))

    # confusion matrices
    print()
    print("confusion matrix (TEST, decided images only):")
    print("                     predicted REAL   predicted AI")
    print("   actual REAL   %15d %15d" % (m_all["tn"], m_all["fp"]))
    print("   actual AI     %15d %15d" % (m_all["fn"], m_all["tp"]))

    # ---- ensemble significance on TEST (reported, not re-selected)
    if ens_best_key:
        cfg = ens_candidates[ens_best_key]
        tey, tep = ensemble_pool(manifest, cfg["combo"], TEST_SPLITS)
        ecal = _sigmoid(_logit(tep) / cfg["temperature"])
        em = evaluate_scores(tey, ecal, cfg["threshold"],
                             uncertain_band=(cfg["threshold"] - cfg["band"],
                                             cfg["threshold"] + cfg["band"]))
        # same mask of decided images for a fair paired comparison
        lo, hi = best["threshold"] - best["uncertain_band"], best["threshold"] + best["uncertain_band"]
        keep = ~((tcal > lo) & (tcal < hi))
        elo, ehi = (cfg["threshold"] - cfg["band"]), (cfg["threshold"] + cfg["band"])
        keep_e = ~((ecal > elo) & (ecal < ehi))
        common = keep & keep_e
        a_correct = ((tcal[common] >= best["threshold"]).astype(int)
                     == ty[common])
        b_correct = ((ecal[common] >= cfg["threshold"]).astype(int)
                     == ty[common])
        boot = paired_bootstrap(ty[common], a_correct, b_correct)
        print()
        print("ENSEMBLE vs SINGLE on TEST (paired, same decided images):")
        print("  single   acc=%.4f  n=%d" % (a_correct.mean(), len(a_correct)))
        print("  ensemble acc=%.4f" % b_correct.mean())
        print("  mean diff=%.4f  95%% CI [%.4f, %.4f]  P(ensemble better)=%.3f"
              % (boot["mean_diff"], boot["ci95"][0], boot["ci95"][1],
                 boot["p_b_better_than_a"]))
        final["ensemble_test"] = {
            "members": list(cfg["combo"]),
            "ensemble_accuracy": float(b_correct.mean()),
            "single_accuracy": float(a_correct.mean()),
            "bootstrap": boot,
            "accepted": False,
        }
        significant = (boot["ci95"][0] > 0)
        final["ensemble_test"]["significant"] = bool(significant)
        if significant:
            final["ensemble_test"]["accepted"] = True
            final["selected_model"] = "+".join(cfg["combo"])
            final["temperature"] = cfg["temperature"]
            final["threshold"] = cfg["threshold"]
            final["uncertain_band"] = cfg["band"]
        print("  -> %s" % ("ACCEPTED (significant)" if significant
                            else "REJECTED (difference is sampling noise)"))

    out = os.path.join(RESULTS_DIR, "final_config.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=1, default=float)
    print()
    print("frozen config -> %s" % out)
    return final


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ai_image_detection/benchmark_v2")
