"""End-to-end nuclei analysis: segment -> compare methods -> morphometry -> grade.

Run with:  python -m src.pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             classification_report)
from skimage import segmentation as skseg

from . import classical, metrics, ml_pixel, morphometry, synth

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLT_STYLE = {"figure.dpi": 125, "savefig.dpi": 125, "font.size": 9,
             "axes.grid": False, "axes.spines.top": False,
             "axes.spines.right": False}


# ---------------------------------------------------------------------------
def train_pixel_classifier(fields, n_per_class=2500, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for f in fields:
        X, y = ml_pixel.sample_training_pixels(f.image, f.labels,
                                               n_per_class=n_per_class, rng=rng)
        Xs.append(X)
        ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    model = ml_pixel.build_model(seed)
    model.fit(X, y)
    return model, X.shape


def tune_on_training_fields(model, train_fields, n_tune: int = 4) -> dict:
    """Select hyper-parameters for BOTH methods using training fields only.

    Tuning on the test set is a subtle but fatal form of leakage: a method
    whose threshold was chosen by looking at test performance is no longer
    being evaluated on unseen data.  Both pipelines get the same tuning budget
    so the comparison is fair, and the chosen values are reported alongside
    the results.
    """
    fields = train_fields[:n_tune]

    # Classical: background-correction strength and watershed seed separation.
    # `None` (no correction at all) is included so the step has to earn itself.
    best_c, best_c_score = (25.0, 7), -1.0
    for bg_sigma in (None, 15.0, 25.0, 40.0):
        for min_distance in (5, 7, 9, 11):
            scores = [metrics.evaluate_segmentation(
                f.labels, classical.segment(f.image, background_sigma=bg_sigma,
                                            min_distance=min_distance))["aji"]
                for f in fields]
            if np.mean(scores) > best_c_score:
                best_c_score, best_c = float(np.mean(scores)), (bg_sigma, min_distance)

    # Learned: interior-probability threshold used to seed the watershed
    probs = [(f, ml_pixel.predict_probabilities(model, f.image)) for f in fields]
    best_m, best_m_score = 0.8, -1.0
    for thr in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90):
        scores = [metrics.evaluate_segmentation(
            f.labels, ml_pixel.segment_from_probabilities(pr, interior_thr=thr))["aji"]
            for f, pr in probs]
        if np.mean(scores) > best_m_score:
            best_m_score, best_m = float(np.mean(scores)), thr

    return {"classical_background_sigma": best_c[0],
            "classical_min_distance": best_c[1], "classical_train_aji": best_c_score,
            "ml_interior_thr": best_m, "ml_train_aji": best_m_score}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_overview(field, seg_classical, seg_ml, proba, path):
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.4))
    ax[0, 0].imshow(field.image, cmap="gray")
    ax[0, 0].set_title(f"{field.name}: simulated field\n"
                       f"({field.n_nuclei} nuclei, SNR {field.meta['snr']:.0f})")
    ax[0, 1].imshow(field.labels, cmap="nipy_spectral")
    ax[0, 1].set_title("Ground-truth instances")
    ax[0, 2].imshow(classical.correct_illumination(field.image), cmap="gray")
    ax[0, 2].set_title("After white-top-hat illumination correction")

    ax[1, 0].imshow(proba[..., ml_pixel.INTERIOR], cmap="magma", vmin=0, vmax=1)
    ax[1, 0].set_title("Learned P(nuclear interior)")
    ax[1, 1].imshow(skseg.mark_boundaries(
        _norm(field.image), seg_classical, color=(1, 0.3, 0)))
    ax[1, 1].set_title(f"Classical Otsu + watershed  ({seg_classical.max()} objects)")
    ax[1, 2].imshow(skseg.mark_boundaries(
        _norm(field.image), seg_ml, color=(0, 1, 0.4)))
    ax[1, 2].set_title(f"RF pixel classifier + watershed  ({seg_ml.max()} objects)")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _norm(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def fig_error_map(field, seg, path, title="Segmentation errors"):
    """Green = correctly matched, red = missed GT, blue = spurious prediction."""
    iou, gt_ids, pred_ids = metrics._iou_matrix(field.labels, seg)
    matched_gt, matched_pred = set(), set()
    m = iou.copy()
    while m.size and m.max() >= 0.5:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        matched_gt.add(int(gt_ids[i]))
        matched_pred.add(int(pred_ids[j]))
        m[i, :] = 0
        m[:, j] = 0

    rgb = np.dstack([_norm(field.image)] * 3) * 0.55
    miss = np.isin(field.labels, [i for i in gt_ids if i not in matched_gt]) & (field.labels > 0)
    spur = np.isin(seg, [j for j in pred_ids if j not in matched_pred]) & (seg > 0)
    hit = np.isin(seg, list(matched_pred)) & (seg > 0)
    rgb[hit] = [0.1, 0.85, 0.3]
    rgb[miss] = [0.95, 0.15, 0.15]
    rgb[spur] = [0.2, 0.45, 1.0]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title(f"{title}\ngreen = matched (IoU≥0.5)   red = missed   blue = spurious")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_metric_comparison(df, path):
    keys = ["dice", "iou", "aji", "f1_at_50", "mean_f1"]
    labels = ["Dice", "IoU", "AJI", "Object F1\n@IoU 0.5", "Mean object F1\n@IoU 0.5-0.9"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.38
    x = np.arange(len(keys))
    for k, (method, colour) in enumerate([("classical", "#e76f51"), ("ml", "#2a9d8f")]):
        sub = df[df.method == method]
        means = [sub[c].mean() for c in keys]
        errs = [sub[c].std() for c in keys]
        ax.bar(x + (k - 0.5) * w, means, w, yerr=errs, capsize=3, color=colour,
               label={"classical": "Otsu + distance watershed",
                      "ml": "RF pixel classifier + watershed"}[method])
        for xi, v in zip(x + (k - 0.5) * w, means):
            ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (mean ± SD over held-out fields)")
    ax.set_title("Segmentation performance, held-out fields")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_morphometry(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    colours = {"benign": "#2a9d8f", "malignant": "#e63946"}
    for cls, sub in df.groupby("true_class"):
        if cls == "unmatched":
            continue
        ax[0].scatter(sub.area, sub.circularity, s=9, alpha=0.5,
                      c=colours[cls], label=cls)
        ax[1].scatter(sub.aspect_ratio, sub.solidity, s=9, alpha=0.5, c=colours[cls])
        ax[2].scatter(sub.mean_intensity, sub.glcm_contrast, s=9, alpha=0.5, c=colours[cls])
    ax[0].set(xlabel="area (px)", ylabel="circularity  4πA/P²",
              title="Size vs. shape regularity")
    ax[1].set(xlabel="aspect ratio", ylabel="solidity (boundary indentation)",
              title="Elongation vs. contour irregularity")
    ax[2].set(xlabel="mean intensity (hyperchromasia)", ylabel="GLCM contrast",
              title="Chromatin texture")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("Nuclear morphometry of segmented nuclei, coloured by true class")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_classification(y_true, y_score, y_pred, importances, names, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax[0].plot(fpr, tpr, lw=2, color="#264653", label=f"AUROC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="1 − specificity", ylabel="sensitivity",
              title="Malignant-nucleus detection (held-out fields)")
    ax[0].legend()

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    im = ax[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax[1].set_xticks([0, 1], ["benign", "malignant"])
    ax[1].set_yticks([0, 1], ["benign", "malignant"])
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.1f}%", ha="center",
                       va="center", color="white" if cmn[i, j] > 0.55 else "black",
                       fontsize=8)
    ax[1].set(xlabel="predicted", ylabel="true", title="Confusion matrix")

    order = np.argsort(importances)[::-1][:10]
    ax[2].barh([names[i] for i in order][::-1], importances[order][::-1],
               color="#2a9d8f")
    ax[2].set(xlabel="Gini importance", title="Top morphometric features")
    ax[2].grid(axis="x", alpha=0.25)
    for a in ax[:1]:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(n_fields: int = 20, size: int = 512, seed: int = 7) -> dict:
    plt.rcParams.update(PLT_STYLE)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"[1/6] Generating {n_fields} annotated fields ({size}x{size}) ...")
    cohort = synth.synth_cohort(n_fields, (size, size), seed=seed)
    n_train = int(0.6 * n_fields)
    train, test = cohort[:n_train], cohort[n_train:]
    print(f"      {sum(f.n_nuclei for f in cohort)} nuclei total; "
          f"{n_train} train / {len(test)} test fields")

    print("[2/6] Training the random-forest pixel classifier ...")
    model, shape = train_pixel_classifier(train, seed=seed)
    print(f"      trained on {shape[0]:,} pixels x {shape[1]} filter-bank features")

    print("[3/6] Tuning both methods on training fields ...")
    tuned = tune_on_training_fields(model, train)
    print(f"      classical bg_sigma={tuned['classical_background_sigma']} "
          f"min_distance={tuned['classical_min_distance']} "
          f"(train AJI {tuned['classical_train_aji']:.3f});  "
          f"ml interior_thr={tuned['ml_interior_thr']} "
          f"(train AJI {tuned['ml_train_aji']:.3f})")

    print("      Segmenting held-out fields with both methods ...")
    rows, per_field = [], []
    for f in test:
        seg_c = classical.segment(
            f.image, background_sigma=tuned["classical_background_sigma"],
            min_distance=tuned["classical_min_distance"])
        proba = ml_pixel.predict_probabilities(model, f.image)
        seg_m = ml_pixel.segment_from_probabilities(
            proba, interior_thr=tuned["ml_interior_thr"])
        for name, seg in (("classical", seg_c), ("ml", seg_m)):
            m = metrics.evaluate_segmentation(f.labels, seg)
            rows.append({"field": f.name, "method": name,
                         **{k: v for k, v in m.items() if k != "by_threshold"}})
        per_field.append((f, seg_c, seg_m, proba))
        print(f"      {f.name}: GT={f.n_nuclei:3d}  classical={seg_c.max():3d}  "
              f"ml={seg_m.max():3d}")

    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(RESULTS / "segmentation_metrics_per_field.csv", index=False)
    summary_seg = seg_df.groupby("method")[
        ["dice", "iou", "aji", "f1_at_50", "mean_f1", "count_error_pct"]].agg(["mean", "std"])
    print("\n" + summary_seg.round(3).to_string() + "\n")

    print("[4/6] Extracting morphometric features ...")
    best_method = seg_df.groupby("method").aji.mean().idxmax()
    print(f"      using the better segmenter ('{best_method}') for every field")

    def segment_with_best(field):
        """Segment with the winning method.

        Both splits must go through the *same* segmenter.  Training the
        morphometry classifier on features from one segmentation method and
        testing it on another shifts the feature distribution between train
        and test - segmenters differ systematically in the areas and
        boundary smoothness they produce - and silently wrecks calibration.
        """
        if best_method == "ml":
            proba = ml_pixel.predict_probabilities(model, field.image)
            return ml_pixel.segment_from_probabilities(
                proba, interior_thr=tuned["ml_interior_thr"])
        return classical.segment(
            field.image, background_sigma=tuned["classical_background_sigma"],
            min_distance=tuned["classical_min_distance"])

    feat_frames = []
    test_names = {f.name for f in test}
    for f in cohort:
        if f.name in test_names:
            i = [t.name for t in test].index(f.name)
            seg = per_field[i][2] if best_method == "ml" else per_field[i][1]
            split = "test"
        else:
            seg = segment_with_best(f)
            split = "train"
        df = morphometry.measure_nuclei(seg, f.image, f.name)
        df = morphometry.assign_ground_truth_class(df, seg, f.labels, f.classes)
        df["split"] = split
        feat_frames.append(df)
    feats = pd.concat(feat_frames, ignore_index=True)
    feats.to_csv(RESULTS / "nuclear_morphometry.csv", index=False)
    matched = feats[feats.true_class != "unmatched"]
    print(f"      {len(matched)} nuclei matched to ground truth "
          f"({100*len(matched)/max(len(feats),1):.1f}% of segmented objects)")

    print("[5/6] Classifying nuclear pleomorphism ...")
    cols = morphometry.FEATURE_COLUMNS
    tr = matched[matched.split == "train"]
    te = matched[matched.split == "test"]
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=seed)
    clf.fit(tr[cols], tr.true_class)
    y_true = (te.true_class == "malignant").astype(int).to_numpy()
    y_score = clf.predict_proba(te[cols])[:, list(clf.classes_).index("malignant")]
    y_pred = clf.predict(te[cols])
    auc = float(roc_auc_score(y_true, y_score))
    rep = classification_report(te.true_class, y_pred, output_dict=True,
                                zero_division=0)
    print(f"      AUROC={auc:.3f}  accuracy={rep['accuracy']*100:.1f}%  "
          f"malignant recall={rep['malignant']['recall']*100:.1f}%")

    te = te.copy()
    te["pred_class"] = y_pred
    grades = pd.DataFrame([{"field": name, **morphometry.pleomorphism_score(g)}
                           for name, g in te.groupby("field")])
    truth = pd.DataFrame([{"field": f.name,
                           "true_malignant_fraction":
                               f.meta["n_malignant"] / max(f.n_nuclei, 1)}
                          for f in test])
    grades = grades.merge(truth, on="field")
    grades["abs_error"] = (grades.malignant_fraction
                           - grades.true_malignant_fraction).abs()
    grades.to_csv(RESULTS / "field_pleomorphism_scores.csv", index=False)
    r = float(np.corrcoef(grades.malignant_fraction,
                          grades.true_malignant_fraction)[0, 1])
    print(f"      field-level malignant fraction: r={r:.3f}, "
          f"mean abs error={grades.abs_error.mean():.3f}")

    print("[6/6] Writing figures ...")
    f0, seg_c0, seg_m0, proba0 = per_field[0]
    fig_overview(f0, seg_c0, seg_m0, proba0, RESULTS / "fig1_segmentation_overview.png")
    fig_error_map(f0, seg_c0, RESULTS / "fig2_errors_classical.png",
                  "Classical Otsu + distance watershed")
    fig_error_map(f0, seg_m0, RESULTS / "fig3_errors_ml.png",
                  "RF pixel classifier + watershed")
    fig_metric_comparison(seg_df, RESULTS / "fig4_metric_comparison.png")
    fig_morphometry(matched, RESULTS / "fig5_morphometry.png")
    fig_classification(y_true, y_score, (y_pred == "malignant").astype(int),
                       clf.feature_importances_, cols,
                       RESULTS / "fig6_pleomorphism_classification.png")

    out = {
        "cohort": {"n_fields": n_fields, "image_size": size,
                   "n_train_fields": n_train, "n_test_fields": len(test),
                   "total_nuclei": int(sum(f.n_nuclei for f in cohort))},
        "tuned_hyperparameters": tuned,
        "segmentation": {
            m: {k: float(seg_df[seg_df.method == m][k].mean())
                for k in ["dice", "iou", "aji", "f1_at_50", "mean_f1",
                          "count_error_pct"]}
            for m in ("classical", "ml")},
        "pleomorphism_classification": {
            "auroc": auc, "accuracy": float(rep["accuracy"]),
            "malignant_recall": float(rep["malignant"]["recall"]),
            "malignant_precision": float(rep["malignant"]["precision"]),
            "benign_recall": float(rep["benign"]["recall"]),
            "n_train_nuclei": int(len(tr)), "n_test_nuclei": int(len(te))},
        "field_grading": {"pearson_r": r,
                          "mean_abs_error": float(grades.abs_error.mean())},
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone in {out['runtime_s']}s. Results in {RESULTS}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Nuclei segmentation & morphometry")
    ap.add_argument("--n-fields", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n_fields, a.size, a.seed)
