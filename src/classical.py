"""Classical (non-learned) nuclei segmentation baseline.

Pipeline:  illumination correction -> denoise -> Otsu threshold ->
morphological cleanup -> distance-transform watershed to split touching nuclei.

This is the workflow implemented by CellProfiler and by most ImageJ macros,
and it remains the sensible baseline that any learned method must beat.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy import ndimage as ndi
from skimage import (filters, measure, morphology, restoration, segmentation)

# scikit-image renamed the size argument of remove_small_objects in 0.26 and
# changed the comparison from "<" to "<=".  Resolve it once at import time so
# the project runs unchanged on both old and new installations.
_RSO_PARAMS = inspect.signature(morphology.remove_small_objects).parameters
_RSO_USES_MAX_SIZE = "max_size" in _RSO_PARAMS


def drop_small(mask_or_labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than `min_area` pixels."""
    if _RSO_USES_MAX_SIZE:
        return morphology.remove_small_objects(mask_or_labels, max_size=min_area - 1)
    return morphology.remove_small_objects(mask_or_labels, min_size=min_area)


def correct_illumination(img: np.ndarray, background_sigma: float = 25.0,
                         presmooth: float = 1.5) -> np.ndarray:
    """Subtract a large-scale Gaussian estimate of the background.

    The background is the image blurred with a sigma far larger than any
    nucleus, so nuclei average out of it while vignetting and the sloped
    autofluorescence survive.  Subtracting it makes a single global threshold
    valid across the whole field.

    Two failure modes found while building this are worth recording, because
    both are silent:

    * **White top-hat** (``img - opening(img)``) is the textbook choice, but on
      shot-noise-limited fluorescence it does nothing at all.  Poisson noise
      puts near-zero pixels inside every structuring element, so the grey
      erosion - and hence the opening - collapses to zero and the "corrected"
      image is byte-identical to the input.  A unit test comparing background
      variance before and after is what caught it.
    * **Rolling-ball** (Sternberg 1983) is the other classic, but skimage's
      ``radius`` argument measures the ball in *intensity* units as well as
      spatial ones.  On a float image scaled 0-1 a spatially sensible radius
      is enormous in intensity, so the ball climbs over the nuclei and erases
      them - foreground recall fell to 0.40 here.  It works correctly only
      with an explicit ``ellipsoid_kernel`` separating the two scales, at
      roughly 4x the runtime, for slightly worse accuracy than this method.
    """
    smoothed = filters.gaussian(img, sigma=presmooth, preserve_range=True)
    background = ndi.gaussian_filter(smoothed, sigma=background_sigma)
    return np.clip(smoothed - background, 0, None)


def rolling_ball_illumination(img: np.ndarray, extent_px: int = 101,
                              intensity_height: float = 0.4) -> np.ndarray:
    """Classic rolling-ball correction, kept for comparison (see above)."""
    smoothed = filters.gaussian(img, sigma=1.5, preserve_range=True)
    kernel = restoration.ellipsoid_kernel((extent_px, extent_px), intensity_height)
    return np.clip(smoothed - restoration.rolling_ball(smoothed, kernel=kernel),
                   0, None)


def segment(img: np.ndarray, min_area: int = 60, sigma: float = 1.5,
            min_distance: int = 7, background_sigma: float | None = 25.0,
            split: bool = True) -> np.ndarray:
    """Return an int label image of segmented nuclei.

    `background_sigma=None` disables background correction, so that the tuner
    can decide empirically whether the step earns its place on a given dataset
    rather than it being assumed.
    """
    corrected = (img if background_sigma is None
                 else correct_illumination(img, background_sigma))
    smooth = filters.gaussian(corrected, sigma=sigma, preserve_range=True)

    thr = filters.threshold_otsu(smooth)
    mask = smooth > thr

    # Morphological cleanup: drop debris, close pinholes, smooth boundaries
    mask = drop_small(mask, min_area)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.opening(mask, morphology.disk(2))
    mask = drop_small(mask, min_area)

    if not split:
        return measure.label(mask)

    return watershed_split(mask, min_distance=min_distance, min_area=min_area)


def watershed_split(mask: np.ndarray, min_distance: int = 7,
                    min_area: int = 60) -> np.ndarray:
    """Split touching objects using the Euclidean distance transform.

    Each nucleus produces a local maximum of the distance-to-background map;
    those maxima seed a watershed flooding of the inverted distance map, and
    the resulting ridges fall in the narrow necks between touching nuclei.
    """
    from skimage.feature import peak_local_max

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance,
                            labels=mask, exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers = ndi.grey_dilation(markers, size=3)

    labels = segmentation.watershed(-distance, markers, mask=mask)
    return drop_small(labels, min_area)
