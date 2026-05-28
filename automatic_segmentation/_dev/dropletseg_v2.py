"""Automatic droplet segmentation (classic CV), v2.
Runs on images already cleaned by preprocessing.py and follows the 4-step
recipe: (1) region-grow the background, (2) treat below-baseline as plate,
(3) droplet = enclosed by background+plate, (4) refine."""
import cv2 as cv, numpy as np

def _fill_holes(m):
    h, w = m.shape; ff = m.copy()
    cv.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 1)
    o = m.copy(); o[ff == 0] = 1; return o

def _largest_on_baseline(m, yb, band=14):
    n, lab, st, _ = cv.connectedComponentsWithStats(m.astype(np.uint8), 8)
    y0 = max(0, yb-band); best, ba = 0, 0
    for i in range(1, n):
        if (lab[y0:yb+1, :] == i).any() and st[i, cv.CC_STAT_AREA] > ba:
            best, ba = i, st[i, cv.CC_STAT_AREA]
    return (lab == best).astype(np.uint8) if best else np.zeros_like(m)

def _refine(cand, yb):
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cand = _fill_holes(cand)
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return _largest_on_baseline(cand, yb)

def segment_core(bg_sub, prep, yb, is_blank=False, tol=30):
    """bg_sub: background-subtracted gray (uniform bright bg);
    prep: fully preprocessed gray; yb: baseline row. Returns 0/1 mask."""
    h, w = bg_sub.shape
    if is_blank:
        return np.zeros((h, w), np.uint8)
    above = np.zeros((h, w), np.uint8); above[:yb, :] = 1
    bg_level = float(np.median(bg_sub[:max(5, yb//6), :]))

    # --- barriers: a pixel blocks the background flood if it is either
    #     (a) clearly darker than the background, or (b) a clean edge ---
    edges = cv.Canny(bg_sub, int(max(0, 0.66*np.median(bg_sub))),
                     int(min(255, 1.33*np.median(bg_sub))))
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), 1)
    bright = bg_sub > (bg_level - tol)
    free = ((bright) & (edges == 0)).astype(np.uint8)
    free[yb:, :] = 0

    # --- step 1: region-grow background from top corners/edges ---
    ff = np.zeros((h+2, w+2), np.uint8); ff[yb+1:, :] = 1
    flags = 4 | cv.FLOODFILL_MASK_ONLY | (255 << 8)
    fl = free.copy()
    for sx, sy in [(2, 2), (w//2, 2), (w-3, 2), (2, max(2, yb//4)), (w-3, max(2, yb//4))]:
        if fl[sy, sx] == 1:
            cv.floodFill(fl, ff, (sx, sy), 2, 0, 0, flags)
    bg = ff[1:h+1, 1:w+1] == 255

    cand = ((above == 1) & (~bg)).astype(np.uint8)     # steps 2+3
    mask = _refine(cand, yb)                            # step 4

    # --- guardrail: if the flood was blocked (huge area), fall back to
    #     a pure threshold-and-fill of the dark rim/body ---
    if mask.sum() / max(1, above.sum()) > 0.45:
        dark = ((bg_sub < bg_level - tol) & (above == 1)).astype(np.uint8)
        dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        dark = _fill_holes(dark)
        mask = _refine(dark, yb)
    return mask

def segment(image, preprocess_fn):
    """Full entry point on an original BGR image. preprocess_fn(image)->dict
    with keys preprocessed, substrate_y, is_blank, and we also need bg_sub.
    Returns a full-size 0/255 mask aligned to the ORIGINAL image."""
    raise NotImplementedError  # wired up in the notebook
