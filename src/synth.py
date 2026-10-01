"""Synthetic nuclear-stain microscopy images with exact instance ground truth.

Real annotated histopathology (MoNuSeg, PanNuke, TNBC) requires download and
carries licence restrictions, and hand annotation is the bottleneck in the
field.  Here images are *generated*, so every nucleus has a pixel-perfect
instance label and a known biological class - which makes it possible to
measure segmentation quality exactly rather than against noisy human labels.

The generator reproduces the imaging physics that actually breaks naive
segmentation:

  * touching and partially overlapping nuclei (the reason plain thresholding
    fails and watershed is required)
  * chromatin texture inside each nucleus
  * a smooth multiplicative illumination field (vignetting / uneven
    excitation) that defeats a single global threshold
  * out-of-focus blur from the microscope point-spread function
  * Poisson shot noise plus Gaussian camera read noise
  * autofluorescent debris that mimics small nuclei

Two nuclear classes are produced, mirroring nuclear-pleomorphism grading
(the "N" component of the Nottingham system):

    benign     - small, round, smooth boundary, even chromatin
    malignant  - larger, elongated, irregular/indented boundary,
                 coarse hyperchromatic chromatin
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Field:
    """One synthetic field of view."""

    name: str
    image: np.ndarray          # float32, roughly 0-1, what the microscope sees
    labels: np.ndarray         # int32 instance mask, 0 = background
    classes: dict              # {instance_id: "benign" | "malignant"}
    meta: dict = field(default_factory=dict)

    @property
    def binary(self) -> np.ndarray:
        return self.labels > 0

    @property
    def n_nuclei(self) -> int:
        return len(self.classes)


def _irregular_nucleus(radius: float, elongation: float, irregularity: float,
                       rng: np.random.Generator, n_pts: int = 128
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Boundary of one nucleus as a radially-perturbed ellipse.

    Low-frequency sinusoidal perturbation of the radius produces the lobulated
    and indented contours seen in malignant nuclei; `irregularity` = 0 gives a
    clean ellipse.
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.ones(n_pts)
    for harmonic in (2, 3, 5, 7):
        r += (irregularity / harmonic) * np.sin(harmonic * theta
                                                + rng.uniform(0, 2 * np.pi))
    r = np.clip(r, 0.45, 1.9) * radius
    x = r * np.cos(theta) * elongation
    y = r * np.sin(theta) / elongation
    phi = rng.uniform(0, np.pi)
    xr = x * np.cos(phi) - y * np.sin(phi)
    yr = x * np.sin(phi) + y * np.cos(phi)
    return xr, yr


def _rasterise(xr: np.ndarray, yr: np.ndarray, cx: float, cy: float,
               shape: tuple[int, int]) -> np.ndarray:
    """Fill the closed polygon defined by the boundary points."""
    from skimage.draw import polygon
    rr, cc = polygon(yr + cy, xr + cx, shape=shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m


def _chromatin_fields(shape: tuple[int, int], rng: np.random.Generator
                      ) -> dict[bool, np.ndarray]:
    """Two pre-computed texture fields, sampled per nucleus by position.

    Building them once per image (rather than once per nucleus) keeps the
    generator fast while still giving every nucleus a different local pattern.
    Fine, even texture models benign chromatin; coarse, clumped texture models
    the hyperchromatic nuclei of high-grade tumours.
    """
    out = {}
    for coarse, sigma, amp in ((False, 1.2, 0.18), (True, 3.0, 0.42)):
        tex = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
        tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
        out[coarse] = 1.0 - amp + amp * 2 * tex
    return out


def synth_field(name: str = "field00", shape: tuple[int, int] = (512, 512),
                n_nuclei: int = 90, malignant_fraction: float = 0.3,
                crowding: float = 0.85, snr: float = 12.0,
                seed: int | None = 0) -> Field:
    """Generate one annotated field of view.

    `crowding` < 1 allows nuclei to be placed closer than the sum of their
    radii, forcing touching clusters that a distance-transform watershed has
    to split.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    intensity = np.zeros(shape, dtype=np.float64)
    labels = np.zeros(shape, dtype=np.int32)
    classes: dict[int, str] = {}

    chromatin = _chromatin_fields(shape, rng)
    centres: list[tuple[float, float, float]] = []
    next_id = 1
    attempts = 0
    while len(classes) < n_nuclei and attempts < n_nuclei * 60:
        attempts += 1
        malignant = rng.random() < malignant_fraction
        if malignant:
            radius = rng.uniform(11.0, 19.0)
            elong = rng.uniform(1.25, 1.95)
            irreg = rng.uniform(0.22, 0.45)
            base = rng.uniform(0.62, 0.92)      # hyperchromatic
        else:
            radius = rng.uniform(7.0, 11.5)
            elong = rng.uniform(1.0, 1.22)
            irreg = rng.uniform(0.0, 0.10)
            base = rng.uniform(0.45, 0.68)

        cx = rng.uniform(radius + 3, W - radius - 3)
        cy = rng.uniform(radius + 3, H - radius - 3)

        # Reject placements that overlap too much to remain distinguishable
        too_close = False
        for ox, oy, orad in centres:
            if np.hypot(cx - ox, cy - oy) < crowding * (radius + orad) * 0.60:
                too_close = True
                break
        if too_close:
            continue

        xr, yr = _irregular_nucleus(radius, elong, irreg, rng)
        mask = _rasterise(xr, yr, cx, cy, shape)
        if mask.sum() < 30:
            continue

        # Later nuclei occlude earlier ones, as in a real projection
        labels[mask] = next_id
        intensity[mask] = base * chromatin[malignant][mask]
        classes[next_id] = "malignant" if malignant else "benign"
        centres.append((cx, cy, radius))
        next_id += 1

    # Drop instances that got fully occluded by later nuclei
    present = set(np.unique(labels)) - {0}
    classes = {k: v for k, v in classes.items() if k in present}

    img = intensity.copy()

    # Autofluorescent debris: small bright specks that are NOT nuclei
    for _ in range(int(rng.integers(3, 12))):
        dy, dx = rng.integers(0, H), rng.integers(0, W)
        rad = rng.integers(2, 5)
        yy, xx = np.ogrid[:H, :W]
        img[(yy - dy) ** 2 + (xx - dx) ** 2 <= rad ** 2] += rng.uniform(0.2, 0.5)

    # Microscope PSF (out-of-focus blur)
    img = ndi.gaussian_filter(img, rng.uniform(1.2, 2.4))

    # Uneven illumination.  Widefield fluorescence routinely loses 40-60% of
    # intensity at the corners through vignetting and non-uniform excitation,
    # and sits on a raised, spatially varying autofluorescence background.  At
    # this severity a single global threshold cannot serve the whole field,
    # which is precisely what background correction exists to fix.
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    cy_i, cx_i = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    illum = (1.0
             - 1.35 * ((yy - cy_i) ** 2 + (xx - cx_i) ** 2)
             + 0.18 * np.sin(2 * np.pi * xx * rng.uniform(0.5, 1.5)))
    illum = np.clip(illum, 0.30, 1.20)
    background = 0.35 * illum + 0.20 * (1.0 - yy)   # sloped autofluorescence
    img = img * illum + background

    # Poisson shot noise then Gaussian read noise, scaled to the target SNR
    peak = max(snr ** 2, 1.0)
    img = rng.poisson(np.clip(img, 0, None) * peak) / peak
    img = img + rng.normal(0, 0.5 / snr, size=shape)

    img = np.clip(img, 0, None).astype(np.float32)
    return Field(name=name, image=img, labels=labels, classes=classes,
                 meta={"n_nuclei": len(classes), "snr": snr, "seed": seed,
                       "malignant_fraction": malignant_fraction,
                       "n_malignant": sum(v == "malignant" for v in classes.values())})


def synth_cohort(n_fields: int = 24, shape: tuple[int, int] = (512, 512),
                 seed: int = 7) -> list[Field]:
    """A cohort of fields spanning a range of density, grade and image quality."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_fields):
        out.append(synth_field(
            name=f"field{i:02d}",
            shape=shape,
            n_nuclei=int(rng.integers(70, 140)),
            malignant_fraction=float(rng.uniform(0.10, 0.60)),
            crowding=float(rng.uniform(0.70, 1.00)),
            snr=float(rng.uniform(6.0, 18.0)),
            seed=int(rng.integers(0, 10 ** 6)),
        ))
    return out
