"""Box-Cox prefiltering for crack image segmentation.

Grayscale -> (optional Box-Cox + histogram stretching) -> single-intensity feature
-> pixel-wise classifier -> segmentation metrics vs ground-truth mask.

Replicates the pipeline of Vallejos et al. (2025) Fig. 1 and box_cox_img.py.
"""
import numpy as np
import cv2
from scipy import stats


def to_gray(rgb):
    """ITU-R 601 luma, as in the paper (0.299R+0.587G+0.114B)."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


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


def load_pair(rgb_path, mask_path, size=256):
    """Load rgb image + mask, resize, return (rgb_uint8, binary_mask)."""
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)  # 1 = crack
    return rgb, mask


def make_feature(rgb, use_boxcox):
    """Return (feature_image in [0,1], lambda or nan)."""
    gray = to_gray(rgb)
    if use_boxcox:
        return boxcox_stretch(gray)
    g = gray / 255.0
    return g, np.nan
