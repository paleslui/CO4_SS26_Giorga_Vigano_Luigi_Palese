"""
dropletseg.py  -  Classic-CV segmentation of back-lit sessile droplets.

Pipeline (follows the multi-step hint, but uses only thresholding/edges/morphology):
  1. preprocess      : grayscale + edge-preserving denoise
  2. find_baseline   : locate the support-plate top edge (the contact baseline)
  3. _edge_barrier   : build a boundary map from the droplet rim
  4. flood background: fill the BACKGROUND from the top/sides; the droplet is the
                       region above the baseline the background could NOT reach
  5. refine          : close gaps, fill holes (removes specular highlights),
                       keep the largest blob on the baseline, clip at baseline

All thresholds come from each image's own statistics (median / percentile), so the
same function is meant to generalise across the whole dataset without hand-tuning.
"""
import cv2 as cv
import numpy as np


def preprocess(image):
    """BGR (or gray) uint8 -> denoised uint8 grayscale."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # bilateral: flattens the smooth background while preserving the droplet rim
    return cv.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def find_baseline(gray):
    """Row index of the plate's top edge = the contact baseline.

    The plate top is the strongest *long, horizontal* intensity step in the lower
    part of the frame. We average the vertical Sobel response across each row and
    pick the peak inside a plausible band (45%-97% of the height)."""
    h, w = gray.shape
    sob_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    row_strength = np.abs(sob_y).mean(axis=1)
    k = max(3, (h // 200) | 1)                       # odd smoothing window
    row_strength = cv.GaussianBlur(row_strength.reshape(-1, 1), (1, k), 0).ravel()
    lo, hi = int(0.45 * h), int(0.97 * h)
    return lo + int(np.argmax(row_strength[lo:hi]))


def _edge_barrier(gray):
    """Binary boundary map (auto-tuned) used as 'walls' for the flood fill."""
    v = float(np.median(gray))
    canny = cv.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = mag > np.percentile(mag, 92.0)
    barrier = ((canny > 0) | strong).astype(np.uint8) * 255
    return cv.dilate(barrier, np.ones((3, 3), np.uint8), iterations=1)


def _fill_holes(mask):
    """Fill interior holes of a 0/1 binary mask."""
    h, w = mask.shape
    ff = mask.copy()
    cv.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    out = mask.copy()
    out[ff == 0] = 1                                 # unreached pixels = holes
    return out


def _largest_on_baseline(mask, y_base, band=10):
    """Keep the largest connected component touching the baseline band."""
    n, lab, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    y0 = max(0, y_base - band)
    best, best_area = 0, 0
    for i in range(1, n):
        touches = (lab[y0:y_base + 1, :] == i).any()
        area = stats[i, cv.CC_STAT_AREA]
        if touches and area > best_area:
            best, best_area = i, area
    return (lab == best).astype(np.uint8) if best else np.zeros_like(mask)


def segment(image, return_debug=False):
    """Segment the droplet in a back-lit side-view image.

    Parameters
    ----------
    image : np.ndarray (H,W,3 BGR or H,W gray), uint8
    return_debug : if True also return a dict of intermediates for plotting

    Returns
    -------
    mask : np.ndarray (H,W) uint8 with values {0,255}
    """
    gray = preprocess(image)
    h, w = gray.shape
    y_base = find_baseline(gray)
    barrier = _edge_barrier(gray)

    # --- flood the BACKGROUND through the free (non-edge) space ---------------
    free = (barrier == 0).astype(np.uint8)
    free[y_base:, :] = 0                              # never cross into the plate
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(2, 2), (w - 3, 2), (w // 2, 2),
             (2, max(2, y_base // 3)), (w - 3, max(2, y_base // 3))]
    for sx, sy in seeds:
        if free[sy, sx] == 1:
            cv.floodFill(free, ffmask, (sx, sy), 2)
    background = free == 2

    # --- droplet = above baseline AND not reachable as background -------------
    above = np.zeros((h, w), np.uint8)
    above[:y_base, :] = 1
    cand = ((above == 1) & (~background)).astype(np.uint8)

    # --- refine ---------------------------------------------------------------
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cand = _fill_holes(cand)
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = _largest_on_baseline(cand, y_base)
    mask[y_base:, :] = 0
    mask = (mask * 255).astype(np.uint8)

    if return_debug:
        return mask, dict(gray=gray, y_base=y_base, barrier=barrier,
                          background=(background.astype(np.uint8) * 255))
    return mask
