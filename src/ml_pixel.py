"""Learned pixel classifier for nuclei segmentation (an ilastik-style approach).

Two ideas do the heavy lifting:

1. **A multiscale filter bank.** Every pixel is described by intensity,
   gradient magnitude, Laplacian, Hessian eigenvalues and difference-of-
   Gaussians computed at several scales.  A random forest on top of these
   features learns a texture-aware, illumination-robust foreground detector
   without needing a GPU or thousands of annotated images.

2. **Three classes, not two: background / nuclear interior / nuclear
   boundary.** Predicting boundaries explicitly is what lets touching nuclei
   be separated - the interior probability supplies clean, well-separated
   markers and the boundary probability supplies the watershed ridges.  This
   is the same formulation used by 3-class U-Nets (Ronneberger et al. 2015)
   and by DSB2018-era winning solutions, just with a random forest instead of
   a deep network.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import feature as skfeature, filters, morphology, segmentation
from sklearn.ensemble import RandomForestClassifier

from .classical import drop_small

SIGMAS = (1.0, 2.0, 4.0, 8.0)

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2


def feature_stack(img: np.ndarray, sigmas: tuple[float, ...] = SIGMAS
                  ) -> np.ndarray:
    """Per-pixel feature volume, shape (H, W, n_features)."""
    feats = [img]
    prev_gauss = None
    for s in sigmas:
        g = filters.gaussian(img, sigma=s, preserve_range=True)
        feats.append(g)
        feats.append(filters.sobel(g))                       # gradient magnitude
        feats.append(ndi.gaussian_laplace(img, sigma=s))     # blob response
        hxx, hxy, hyy = skfeature.hessian_matrix(
            img, sigma=s, order="rc", use_gaussian_derivatives=False)
        e1, e2 = skfeature.hessian_matrix_eigvals((hxx, hxy, hyy))
        feats.extend([e1, e2])                               # ridge / blob shape
        if prev_gauss is not None:
            feats.append(g - prev_gauss)                     # difference of Gaussians
        prev_gauss = g
    return np.stack([f.astype(np.float32) for f in feats], axis=-1)


def feature_names(sigmas: tuple[float, ...] = SIGMAS) -> list[str]:
    names = ["raw"]
    for i, s in enumerate(sigmas):
        names += [f"gauss_s{s:g}", f"sobel_s{s:g}", f"log_s{s:g}",
                  f"hess_e1_s{s:g}", f"hess_e2_s{s:g}"]
        if i > 0:
            names.append(f"dog_s{s:g}")
    return names


def pixel_labels(instance_labels: np.ndarray, boundary_width: int = 2
                 ) -> np.ndarray:
    """Convert an instance mask into background / interior / boundary classes."""
    fg = instance_labels > 0
    edges = segmentation.find_boundaries(instance_labels, mode="thick")
    if boundary_width > 1:
        edges = morphology.dilation(edges, morphology.disk(boundary_width - 1))
    out = np.full(instance_labels.shape, BACKGROUND, dtype=np.uint8)
    out[fg] = INTERIOR
    out[edges & fg] = BOUNDARY
    return out


def sample_training_pixels(img: np.ndarray, instance_labels: np.ndarray,
                           n_per_class: int = 3000,
                           rng: np.random.Generator | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Balanced pixel sample from one image."""
    rng = rng or np.random.default_rng(0)
    X = feature_stack(img).reshape(-1, len(feature_names()))
    y = pixel_labels(instance_labels).ravel()
    idx = []
    for c in (BACKGROUND, INTERIOR, BOUNDARY):
        pool = np.flatnonzero(y == c)
        if pool.size == 0:
            continue
        idx.append(rng.choice(pool, size=min(n_per_class, pool.size),
                              replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def build_model(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150, max_depth=18, min_samples_leaf=4,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=seed)


def predict_probabilities(model: RandomForestClassifier, img: np.ndarray
                          ) -> np.ndarray:
    """Per-pixel class probabilities, shape (H, W, 3)."""
    H, W = img.shape
    X = feature_stack(img).reshape(-1, len(feature_names()))
    proba = np.zeros((H * W, 3), dtype=np.float32)
    # Predict in chunks to keep peak memory low on modest machines
    step = 60000
    for i in range(0, X.shape[0], step):
        p = model.predict_proba(X[i:i + step])
        for j, c in enumerate(model.classes_):
            proba[i:i + step, c] = p[:, j]
    return proba.reshape(H, W, 3)


def segment_from_probabilities(proba: np.ndarray, min_area: int = 60,
                               interior_thr: float = 0.55,
                               fg_thr: float = 0.50) -> np.ndarray:
    """Marker-controlled watershed driven by the learned probability maps."""
    interior = proba[..., INTERIOR]
    foreground = proba[..., INTERIOR] + proba[..., BOUNDARY]

    markers_mask = interior > interior_thr
    markers_mask = drop_small(markers_mask, max(8, min_area // 4))
    markers = ndi.label(markers_mask)[0]

    mask = foreground > fg_thr
    mask = ndi.binary_fill_holes(mask)
    mask = drop_small(mask, min_area)

    # Flood the *negative interior probability*: watershed lines settle where
    # the network is most confident a boundary lies.
    labels = segmentation.watershed(-interior, markers, mask=mask)
    return drop_small(labels, min_area)
