"""Quantitative nuclear morphometry and pleomorphism classification.

Pathologists grade tumours partly on *nuclear pleomorphism* - variation in
nuclear size, shape and chromatin pattern (the "N" score of the Nottingham /
Bloom-Richardson system for breast carcinoma).  Scoring it by eye is
notoriously subjective, with reported inter-observer kappa around 0.5.

This module turns each segmented nucleus into an interpretable feature vector
covering the same properties a pathologist looks at:

  size        area, equivalent diameter, perimeter
  shape       eccentricity, solidity (a proxy for boundary indentation),
              circularity, aspect ratio, extent
  chromatin   mean/std/range of intensity, plus GLCM contrast, homogeneity,
              energy and correlation (Haralick texture)

A random forest then classifies each nucleus as benign or malignant, and the
per-field malignant fraction gives an objective, reproducible pleomorphism
score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")

SHAPE_FEATURES = ["area", "equivalent_diameter", "perimeter", "eccentricity",
                  "solidity", "extent", "circularity", "aspect_ratio",
                  "mean_intensity", "std_intensity", "intensity_range"]
FEATURE_COLUMNS = SHAPE_FEATURES + [f"glcm_{p}" for p in GLCM_PROPS]


def _glcm_features(crop: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict:
    """Haralick texture inside one nucleus, from a grey-level co-occurrence matrix."""
    vals = crop[mask]
    if vals.size < 8:
        return {f"glcm_{p}": 0.0 for p in GLCM_PROPS}
    lo, hi = float(vals.min()), float(vals.max())
    scale = (levels - 1) / (hi - lo) if hi > lo else 0.0
    q = np.zeros(crop.shape, dtype=np.uint8)
    q[mask] = np.clip((vals - lo) * scale, 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2,
                                                  3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    return {f"glcm_{p}": float(np.mean(graycoprops(glcm, p))) for p in GLCM_PROPS}


def measure_nuclei(label_image: np.ndarray, intensity_image: np.ndarray,
                   field_name: str = "") -> pd.DataFrame:
    """One row of morphometric features per segmented nucleus."""
    rows = []
    for r in measure.regionprops(label_image, intensity_image=intensity_image):
        if r.area < 10:
            continue
        perim = float(r.perimeter) if r.perimeter > 0 else 1.0
        minor = float(r.axis_minor_length) if r.axis_minor_length > 0 else 1e-6
        vals = r.image_intensity[r.image]
        row = {
            "field": field_name,
            "label": int(r.label),
            "centroid_y": float(r.centroid[0]),
            "centroid_x": float(r.centroid[1]),
            "area": float(r.area),
            "equivalent_diameter": float(r.equivalent_diameter_area),
            "perimeter": perim,
            "eccentricity": float(r.eccentricity),
            "solidity": float(r.solidity),
            "extent": float(r.extent),
            # 1.0 for a perfect circle; falls as the boundary becomes lobulated
            "circularity": float(4 * np.pi * r.area / (perim ** 2)),
            "aspect_ratio": float(r.axis_major_length / minor),
            "mean_intensity": float(np.mean(vals)),
            "std_intensity": float(np.std(vals)),
            "intensity_range": float(np.ptp(vals)) if vals.size else 0.0,
        }
        row.update(_glcm_features(r.image_intensity, r.image))
        rows.append(row)
    return pd.DataFrame(rows)


def assign_ground_truth_class(df: pd.DataFrame, pred_labels: np.ndarray,
                              gt_labels: np.ndarray, gt_classes: dict,
                              min_overlap: float = 0.5) -> pd.DataFrame:
    """Label each *segmented* nucleus using the GT instance it best overlaps.

    Segmented objects that do not cover at least `min_overlap` of any single
    GT nucleus are marked 'unmatched' (debris, merged clusters, fragments) and
    are excluded from classifier training - but they are still counted by the
    segmentation metrics.
    """
    out = df.copy()
    assigned = []
    for lab in out["label"].to_numpy():
        m = pred_labels == lab
        overlaps = gt_labels[m]
        overlaps = overlaps[overlaps > 0]
        if overlaps.size == 0:
            assigned.append("unmatched")
            continue
        vals, counts = np.unique(overlaps, return_counts=True)
        best = int(vals[np.argmax(counts)])
        frac = counts.max() / m.sum()
        assigned.append(gt_classes.get(best, "unmatched")
                        if frac >= min_overlap else "unmatched")
    out["true_class"] = assigned
    return out


def pleomorphism_score(df: pd.DataFrame, class_column: str = "pred_class") -> dict:
    """Field-level summary a pathologist would recognise.

    Beyond the malignant fraction, the *coefficient of variation of nuclear
    area* is reported: pleomorphism is fundamentally about variability, so a
    field of uniformly large nuclei is graded differently from one containing
    both tiny and giant nuclei.
    """
    if df.empty:
        return {"n_nuclei": 0, "malignant_fraction": float("nan"),
                "area_cv_pct": float("nan"), "mean_area": float("nan")}
    area = df["area"].to_numpy()
    frac = float(np.mean(df[class_column] == "malignant")) if class_column in df else float("nan")
    return {
        "n_nuclei": int(len(df)),
        "malignant_fraction": frac,
        "mean_area": float(np.mean(area)),
        "area_cv_pct": float(100.0 * np.std(area) / np.mean(area)),
        "mean_circularity": float(df["circularity"].mean()),
        "mean_solidity": float(df["solidity"].mean()),
    }
