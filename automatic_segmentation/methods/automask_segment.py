"""
automask_segment.py - FINAL automatic droplet segmentation.

Approach (proven on the dataset):
  - SAM (Segment Anything, Meta) automatic mask generation proposes every object
    in the image with no prompt.
  - We then select the proposal that looks like a sessile droplet, by SHAPE
    rather than by an exact baseline (which was the brittle part before):
        (a) does NOT touch the top edge          -> not the background
        (b) bounding box < 75% of the frame width -> not the plate/background band
        (c) solidity > 0.5                        -> compact / convex, i.e. blob-like
        (d) 0.3% <= area <= 25% of the frame      -> plausible droplet size
    A rough baseline (topmost sustained horizontal edge) is used only as a soft
    tie-breaker that nudges selection toward a blob resting near the surface.

This removes the single biggest point of failure in the earlier pipeline: a
hand-rolled baseline detector that often locked onto the wrong edge and caused
even obvious droplets to be discarded.
"""
import os, cv2 as cv, numpy as np, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

_CKPT = os.path.expanduser("~/Library/Caches/sam_models/sam_vit_b_01ec64.pth")
_GEN = None


def get_generator():
    global _GEN
    if _GEN is None:
        sam = sam_model_registry["vit_b"](checkpoint=_CKPT)
        sam.to("cpu")  # automatic generator needs float64 -> CPU (MPS unsupported)
        _GEN = SamAutomaticMaskGenerator(
            sam, points_per_side=16, pred_iou_thresh=0.86,
            stability_score_thresh=0.90, min_mask_region_area=400)
    return _GEN


def rough_baseline(gray):
    h, w = gray.shape
    g = cv.medianBlur(gray, 5)
    e = np.abs(cv.Sobel(g, cv.CV_32F, 0, 1, 3)).mean(1)
    e = cv.GaussianBlur(e.reshape(-1, 1), (1, 11), 0).ravel()
    lo, hi = int(0.45 * h), int(0.98 * h)
    seg = e[lo:hi]
    run = np.convolve((seg > 0.30 * seg.max()).astype(int), np.ones(8, int), "valid")
    return lo + int(np.argmax(run >= 5))


def _solidity(m):
    cnt, _ = cv.findContours(m.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnt:
        return 0.0
    c = max(cnt, key=cv.contourArea)
    hull = cv.contourArea(cv.convexHull(c))
    return cv.contourArea(c) / hull if hull > 0 else 0.0


def segment(image, generator=None):
    """Return a full-size uint8 mask (0/255) for the droplet, or zeros if none."""
    gen = generator or get_generator()
    h, w = image.shape[:2]
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    base = rough_baseline(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

    masks = gen.generate(rgb)
    best, best_score = None, -1.0
    for d in masks:
        m = d["segmentation"]
        area = m.sum() / (h * w)
        if area < 0.003 or area > 0.25:
            continue
        ys, xs = np.where(m)
        if m[:int(0.05 * h), :].any():        # touches top -> background
            continue
        if (xs.max() - xs.min()) > 0.75 * w:  # full width -> plate / bg band
            continue
        sol = _solidity(m)
        if sol < 0.5:
            continue
        base_bonus = 1.0 - min(1.0, abs(ys.max() - base) / 300.0)
        score = sol * 0.5 + min(area, 0.10) / 0.10 * 0.3 + base_bonus * 0.2
        if score > best_score:
            best_score, best = score, m

    if best is None:
        return np.zeros((h, w), np.uint8)
    return (best.astype(np.uint8) * 255)
