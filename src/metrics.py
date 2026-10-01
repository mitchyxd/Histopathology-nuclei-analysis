"""Segmentation quality metrics.

Two levels are reported, because they answer different questions:

* **Semantic** (Dice / IoU on the binary foreground) - "did we find the
  nuclear material?"  It is blind to whether two touching nuclei were merged
  into one blob, so it is not sufficient on its own.
* **Instance** (object F1 at IoU thresholds, Aggregated Jaccard Index) -
  "did we find the right *number* of nuclei, each correctly delineated?"
  This is the metric that matters for counting and morphometry, and the one
  that punishes under-segmentation of touching nuclei.
"""

from __future__ import annotations

import numpy as np


def dice(gt_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    inter = np.logical_and(gt_binary, pred_binary).sum()
    total = gt_binary.sum() + pred_binary.sum()
    return float(2 * inter / total) if total else 1.0


def iou_binary(gt_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    inter = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    return float(inter / union) if union else 1.0


def _iou_matrix(gt: np.ndarray, pred: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise IoU between every GT and predicted instance.

    Built from a 2-D histogram of the label pair at each pixel, which is O(N)
    in image size rather than O(n_gt * n_pred) in mask comparisons.
    """
    gt_ids = np.unique(gt)[1:]
    pred_ids = np.unique(pred)[1:]
    if gt_ids.size == 0 or pred_ids.size == 0:
        return np.zeros((gt_ids.size, pred_ids.size)), gt_ids, pred_ids

    gt_map = {v: i for i, v in enumerate(gt_ids)}
    pr_map = {v: i for i, v in enumerate(pred_ids)}
    inter = np.zeros((gt_ids.size, pred_ids.size), dtype=np.int64)

    both = (gt > 0) & (pred > 0)
    g = np.vectorize(gt_map.get)(gt[both]) if both.any() else np.array([], int)
    p = np.vectorize(pr_map.get)(pred[both]) if both.any() else np.array([], int)
    if g.size:
        np.add.at(inter, (g.astype(int), p.astype(int)), 1)

    gt_area = np.array([(gt == v).sum() for v in gt_ids])[:, None]
    pr_area = np.array([(pred == v).sum() for v in pred_ids])[None, :]
    union = gt_area + pr_area - inter
    return inter / np.maximum(union, 1), gt_ids, pred_ids


def object_scores(gt: np.ndarray, pred: np.ndarray,
                  thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    """Greedy one-to-one matching at each IoU threshold."""
    iou, gt_ids, pred_ids = _iou_matrix(gt, pred)
    out = {"n_gt": int(gt_ids.size), "n_pred": int(pred_ids.size), "by_threshold": {}}
    f1s = []
    for t in thresholds:
        m = iou.copy()
        tp = 0
        while m.size and m.max() >= t:
            i, j = np.unravel_index(np.argmax(m), m.shape)
            tp += 1
            m[i, :] = 0
            m[:, j] = 0
        fp = pred_ids.size - tp
        fn = gt_ids.size - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        out["by_threshold"][f"{t:.1f}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    out["mean_f1"] = float(np.mean(f1s))
    out["f1_at_50"] = out["by_threshold"]["0.5"]["f1"]
    return out


def aggregated_jaccard_index(gt: np.ndarray, pred: np.ndarray) -> float:
    """AJI (Kumar et al., IEEE TMI 2017), the standard nuclei-segmentation score.

    Each GT nucleus is greedily paired with the predicted nucleus of highest
    IoU; intersections and unions are accumulated globally and every unmatched
    prediction is added to the denominator, so both over- and under-
    segmentation are penalised.
    """
    iou, gt_ids, pred_ids = _iou_matrix(gt, pred)
    if gt_ids.size == 0:
        return 0.0 if pred_ids.size else 1.0
    if pred_ids.size == 0:
        return 0.0

    gt_area = np.array([(gt == v).sum() for v in gt_ids])
    pr_area = np.array([(pred == v).sum() for v in pred_ids])

    used = np.zeros(pred_ids.size, dtype=bool)
    inter_sum = 0
    union_sum = 0
    for i in range(gt_ids.size):
        row = iou[i].copy()
        row[used] = -1
        j = int(np.argmax(row))
        if row[j] <= 0:
            union_sum += gt_area[i]
            continue
        used[j] = True
        inter = int(round(row[j] * (gt_area[i] + pr_area[j]) / (1 + row[j])))
        inter_sum += inter
        union_sum += gt_area[i] + pr_area[j] - inter
    union_sum += pr_area[~used].sum()
    return float(inter_sum / union_sum) if union_sum else 0.0


def evaluate_segmentation(gt: np.ndarray, pred: np.ndarray) -> dict:
    out = {
        "dice": dice(gt > 0, pred > 0),
        "iou": iou_binary(gt > 0, pred > 0),
        "aji": aggregated_jaccard_index(gt, pred),
    }
    out.update(object_scores(gt, pred))
    out["count_error_pct"] = float(
        100.0 * (out["n_pred"] - out["n_gt"]) / max(out["n_gt"], 1))
    return out
