"""
ensemble_segment.py - per-image method selection (the principled "different
treatment for different pictures").

For each image we run two very different segmenters (classical basic_segment and
SAM), clean both outputs the same way, score both by mask quality, and keep the
better one.  The choice is driven only by measurable quality - never by knowing
which substrate the image shows - so it still generalises.

clean(): keep the largest connected component that touches the baseline and fill
its holes.  This alone removes the "scattered specks in the sky" and most
"smeared along the plate" artefacts, whatever method produced them.
"""
import cv2 as cv, numpy as np
from basic_segment import segment as basic_seg
from sam_segment_v2 import segment as sam_seg


def clean(mask, baseline, band=16):
    if mask.sum() == 0:
        return np.zeros_like(mask)
    n, lab, st, _ = cv.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    best, ba = 0, 0
    lo = max(0, baseline - band)
    for i in range(1, n):
        if (lab[lo:baseline + 1, :] == i).any() and st[i, cv.CC_STAT_AREA] > ba:
            best, ba = i, st[i, cv.CC_STAT_AREA]
    if best == 0:
        return np.zeros_like(mask)
    m = (lab == best).astype(np.uint8)
    ff = m.copy()
    cv.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    m[ff == 0] = 1
    m[baseline:, :] = 0
    return (m * 255).astype(np.uint8)


def quality(mask, shape):
    h, w = shape
    if mask.sum() == 0:
        return -1.0
    area = (mask > 0).sum() / float(h * w)
    if area < 0.003 or area > 0.35:
        return -1.0
    # whole-frame catch signature: large area AND spans almost the full width.
    # (a real big droplet has large area but a much narrower bounding box.)
    xs = np.where(mask > 0)[1]
    span = (xs.max() - xs.min() + 1) / float(w)
    if area > 0.18 and span > 0.80:
        return -1.0
    c, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(c, key=cv.contourArea)
    hull = cv.contourArea(cv.convexHull(c))
    sol = cv.contourArea(c) / hull if hull > 0 else 0.0
    # reward solidity; mild reward for a fuller (less tiny) droplet
    return sol * (0.6 + 0.4 * min(area, 0.15) / 0.15)


def segment(image, return_method=False):
    h, w = image.shape[:2]
    bm, bl_b = basic_seg(image); bm = clean(bm, bl_b); qb = quality(bm, (h, w))
    sm, bl_s = sam_seg(image);   sm = clean(sm, bl_s); qs = quality(sm, (h, w))

    if qb < 0 and qs < 0:
        out, bl, method = np.zeros((h, w), np.uint8), bl_s, "none"
    elif qs >= qb:
        out, bl, method = sm, bl_s, "sam"
    else:
        out, bl, method = bm, bl_b, "basic"
    return (out, bl, method) if return_method else (out, bl)
