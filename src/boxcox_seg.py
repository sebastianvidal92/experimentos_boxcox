"""Preprocessing for crack image segmentation.

Grayscale -> (optional contrast preprocessing) -> single-intensity feature
-> pixel-wise classifier -> segmentation metrics vs ground-truth mask.

Replicates the pipeline of Vallejos et al. (2025) Fig. 1 and box_cox_img.py, and
extends it with Histogram Equalization and CLAHE for the preprocessing comparison.
"""
import numpy as np
import cv2
from scipy import stats

# CLAHE defaults (OpenCV common values)
CLAHE_CLIP = 2.0
CLAHE_TILE = 8


def to_gray(rgb):
    """ITU-R 601 luma, as in the paper (0.299R+0.587G+0.114B)."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _to_uint8(gray):
    """Grayscale float (0..255 scale) -> uint8 for OpenCV ops."""
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _minmax01(x):
    """Stretch an array to [0,1] (returns zeros if constant)."""
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def boxcox_stretch(gray):
    """Box-Cox transform of a grayscale image with lambda by profile MLE,
    followed by histogram stretching to [0,1]. Returns (feature_image, lambda)."""
    x = gray.flatten().astype(np.float64)
    # min-max to (0,1] then small epsilon so all values are strictly positive
    rng = x.max() - x.min()
    if rng == 0:
        return np.zeros_like(gray), np.nan
    xn = (x - x.min()) / rng + 1e-5
    bc, lam = stats.boxcox(xn)
    brng = bc.max() - bc.min()
    bc = (bc - bc.min()) / brng if brng > 0 else np.zeros_like(bc)
    return bc.reshape(gray.shape), float(lam)


def histeq_stretch(gray):
    """Global histogram equalization (cv2.equalizeHist) of the grayscale image,
    rescaled to [0,1]. Returns (feature_image, nan)."""
    eq = cv2.equalizeHist(_to_uint8(gray))
    return eq.astype(np.float64) / 255.0, np.nan


def clahe_stretch(gray, clip=CLAHE_CLIP, tile=CLAHE_TILE):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) of the grayscale
    image, rescaled to [0,1]. Returns (feature_image, nan)."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(_to_uint8(gray))
    return out.astype(np.float64) / 255.0, np.nan


def load_pair(rgb_path, mask_path, size=256):
    """Load rgb image + mask, resize, return (rgb_uint8, binary_mask)."""
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)  # 1 = crack
    return rgb, mask


# Preprocessing methods available for the comparison. "raw" is the baseline
# (untransformed grayscale intensity); the rest are per-image contrast transforms.
PREPROC = {
    "raw": lambda gray: (gray / 255.0, np.nan),
    "boxcox": boxcox_stretch,
    "histeq": histeq_stretch,
    "clahe": clahe_stretch,
}


def make_feature_pp(rgb, method="raw"):
    """Single-intensity feature image in [0,1] for a given preprocessing method.

    method in {"raw", "boxcox", "histeq", "clahe"}. Returns (feature_image, lambda),
    where lambda is the Box-Cox MLE (nan for the other methods).
    """
    if method not in PREPROC:
        raise ValueError(f"unknown preprocessing method {method!r}; "
                         f"expected one of {sorted(PREPROC)}")
    return PREPROC[method](to_gray(rgb))


def make_feature(rgb, use_boxcox):
    """Backward-compatible wrapper: boolean Box-Cox toggle (see make_feature_pp)."""
    return make_feature_pp(rgb, "boxcox" if use_boxcox else "raw")
