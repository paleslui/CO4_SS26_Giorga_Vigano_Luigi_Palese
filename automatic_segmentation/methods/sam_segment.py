"""
sam_segment_v2.py - SAM v1 + two targeted fixes.

v1 had 34/39 "ok" but ~10 visual failures in two clear patterns:
  (a) Whole-frame catches (h2o-vetro 0.55, oct-plexig 0.59, ...) when SAM picked
      a too-big mask -> fix: cap selector area at 0.30.
  (b) Plate-edge catches (h2o-fog-5, h2o-print-*, oct-fog-*) when the seed point
      from basic_segment landed on the plate edge -> fix: pick the seed as the
      column with the most non-sky deviation just above the baseline.
Everything else (multimask, the 3-point-prompt structure) is identical to v1.
"""
import os, cv2 as cv, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor
from basic_segment import detect_baseline, per_column_deviation

_CKPT = os.path.expanduser("~/Library/Caches/sam_models/sam_vit_b_01ec64.pth")
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_PREDICTOR = None

def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        sam = sam_model_registry["vit_b"](checkpoint=_CKPT)
        sam.to(device=_DEVICE)
        _PREDICTOR = SamPredictor(sam)
    return _PREDICTOR

def _peak_seed(gray, baseline):
    """Seed point at the column with the most non-sky deviation in the band
    just above the baseline.  Avoids landing on a plate edge."""
    dev = per_column_deviation(gray, baseline)
    top = max(0, baseline - 300)
    col_sum = dev[top:baseline, :].astype(np.int64).sum(axis=0)
    cx = int(np.argmax(col_sum))
    return cx, max(0, baseline - 30)

def segment(image):
    predictor = _get_predictor()
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    gray = cv.medianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 5)
    baseline = detect_baseline(gray)

    sx, sy = _peak_seed(gray, baseline)
    points = np.array([[sx, sy], [10, 10], [w - 10, 10], [w // 2, 10]], dtype=np.float32)
    labels = np.array([1, 0, 0, 0])
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)

    best_i, best_score = -1, -1.0
    for i, m in enumerate(masks):
        m = m.astype(bool); af = m.mean()
        if af > 0.30 or af < 0.001:                                   # tighter cap
            continue
        ys, _ = np.where(m)
        if ys.size == 0: continue
        if ys.min() >= baseline: continue
        if (baseline - ys.max()) > 15 and not m[max(0, baseline - 15):baseline + 1, :].any():
            continue
        s = float(scores[i]) * (1.0 - max(0.0, af - 0.18))
        if s > best_score: best_score, best_i = s, i

    if best_i < 0:
        return np.zeros((h, w), np.uint8), baseline
    mask = masks[best_i].astype(np.uint8) * 255
    mask[baseline:, :] = 0
    return mask, baseline
