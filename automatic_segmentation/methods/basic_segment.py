"""
basic_segment.py - From-scratch droplet segmentation.

Five steps, no helpers, no external preprocessing:
  1. Grayscale + mild median denoise.
  2. Find the baseline (plate-top): strongest long horizontal gradient in lower half.
  3. Per-column background subtraction: for each column, the very top is sky;
     subtract that column's sky value from every pixel below.  The droplet
     deviates regardless of whether it is darker or brighter than the sky.
  4. Threshold the deviation map (Otsu, with a small floor to ignore noise),
     restrict to above the baseline, close small gaps, fill enclosed holes.
  5. Keep the largest connected component that sits on the baseline.
"""
import cv2 as cv
import numpy as np


def detect_baseline(gray):
    h, w = gray.shape
    sob_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    row_energy = np.abs(sob_y).mean(axis=1)
    # smooth a bit so a single noisy row does not win
    row_energy = cv.GaussianBlur(row_energy.reshape(-1, 1), (1, 11), 0).ravel()
    lo, hi = int(0.40 * h), int(0.97 * h)
    return lo + int(np.argmax(row_energy[lo:hi]))


def per_column_deviation(gray, baseline):
    # the very top of each column is pure sky -> per-column reference value
    top_band = max(15, baseline // 10)
    sky = np.median(gray[:top_band, :], axis=0).astype(np.int32)
    dev = np.abs(gray.astype(np.int32) - sky).astype(np.uint8)
    return dev


def segment(image):
    """Returns (mask uint8 0/255, baseline_y)."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv.medianBlur(gray, 5)

    baseline = detect_baseline(gray)

    dev = per_column_deviation(gray, baseline)
    dev_above = dev[:baseline, :]
    # Otsu threshold on the deviation, with a floor of 8 grayscale levels
    # (so a flat noisy image cannot produce a tiny threshold and segment everything).
    otsu_thr, _ = cv.threshold(dev_above, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thr = max(8.0, float(otsu_thr))
    cand = (dev > thr).astype(np.uint8)
    cand[baseline:, :] = 0

    # bridge tiny gaps in the rim, fill the enclosed interior, drop slivers
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ff = cand.copy()
    cv.floodFill(ff, np.zeros((cand.shape[0] + 2, cand.shape[1] + 2), np.uint8), (0, 0), 1)
    cand[ff == 0] = 1
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # keep the largest blob actually sitting on the baseline
    n, lab, st, _ = cv.connectedComponentsWithStats(cand, 8)
    h, w = gray.shape
    band = 14
    best, best_area = 0, 0
    for i in range(1, n):
        if (lab[max(0, baseline - band):baseline + 1, :] == i).any():
            a = st[i, cv.CC_STAT_AREA]
            if a > best_area and a < 0.45 * h * w:
                best, best_area = i, a

    mask = np.zeros_like(gray)
    if best > 0:
        # Shape sanity: a real droplet rises meaningfully above the baseline.
        # A flat plate-top reflection does not -> reject it.
        ys, _ = np.where(lab == best)
        height_above = baseline - int(ys.min())
        if height_above >= max(25, int(0.025 * h)):
            mask[lab == best] = 255
    mask[baseline:, :] = 0
    return mask, baseline
