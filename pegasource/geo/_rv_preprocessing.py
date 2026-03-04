"""
Preprocessing utilities: thresholding and skeletonization of 2D density histograms.
"""

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, binary_dilation, disk


def binarize(histogram: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Convert a 2D density histogram to a binary road mask.

    Parameters
    ----------
    histogram : np.ndarray
        2D array of density values (non-negative).
    threshold : float or None
        Pixel values **above** this threshold are considered road.
        If ``None``, Otsu's method is used to determine the threshold
        automatically.

    Returns
    -------
    np.ndarray
        Boolean 2D array — ``True`` where roads are detected.
    """
    if histogram.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {histogram.ndim}D.")

    hist = histogram.astype(np.float64)

    if threshold is None:
        # Otsu needs at least two distinct values
        if hist.max() == hist.min():
            return np.zeros_like(hist, dtype=bool)
        threshold = threshold_otsu(hist)

    return hist > threshold


def skeletonize_map(binary_mask: np.ndarray, dilate_radius: int = 0) -> np.ndarray:
    """Thin a binary road mask to 1-pixel-wide centre-lines.

    Parameters
    ----------
    binary_mask : np.ndarray
        Boolean 2D array (e.g. output of :func:`binarize`).
    dilate_radius : int
        If > 0, dilate the mask before skeletonizing. Useful when the
        road regions are very thin and fragmented.

    Returns
    -------
    np.ndarray
        Boolean 2D skeleton array.
    """
    mask = binary_mask.astype(bool)

    if dilate_radius > 0:
        mask = binary_dilation(mask, disk(dilate_radius))

    return skeletonize(mask)
