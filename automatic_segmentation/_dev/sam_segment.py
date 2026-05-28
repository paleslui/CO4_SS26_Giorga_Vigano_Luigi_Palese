"""
sam_segment.py - Segment Anything (Meta) used here as a generic boundary finder.

SAM is a foundation segmentation model: given an image and a point/box prompt,
it produces a mask for whatever object the prompt sits on, with no per-class
training. The lecture (Block 6-04) explicitly endorses this approach for
generalising beyond what classical CV can handle.

Our prompt strategy:
  1. Detect the baseline with the same simple Sobel-y peak as basic_segment.
  2. Run basic_segment to get a rough droplet centroid (it usually finds
     *some* structure near the droplet even when its outline is wrong).
  3. Use that centroid as a positive point prompt for SAM, and add a few
     negative (background) point prompts at the top corners.
  4. If basic_segment found nothing, use a default positive point just
     above the centre of the baseline.
"""
import os, cv2 as cv, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor
from basic_segment import segment as basic_segment, detect_baseline

_CKPT = os.path.expanduser("~/Library/Caches/sam_models/sam_vit_b_01ec64.pth")
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_SAM = None
_PREDICTOR = None


def _get_predictor():
    global _SAM, _PREDICTOR
    if _PREDICTOR is None:
        _SAM = sam_model_registry["vit_b"](checkpoint=_CKPT)
        _SAM.to(device=_DEVICE)
        _PREDICTOR = SamPredictor(_SAM)
    return _PREDICTOR


def _droplet_seed(image, baseline):
    initial, _ = basic_segment(image)
    if initial.sum() > 0:
        ys, xs = np.where(initial > 0)
        return int(np.median(xs)), int(np.median(ys))
    # default: horizontal centre, just above the baseline
    h, w = image.shape[:2]
    return w // 2, max(0, baseline - 30)


def segment(image):
    predictor = _get_predictor()
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv.medianBlur(gray, 5)
    baseline = detect_baseline(gray)

    sx, sy = _droplet_seed(image, baseline)
    points = np.array([[sx, sy], [10, 10], [w - 10, 10], [w // 2, 10]], dtype=np.float32)
    labels = np.array([1, 0, 0, 0])  # 1 = foreground, 0 = background

    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)

    # pick the mask with the best score among those that (a) touch the baseline,
    # (b) sit mostly above it, (c) are not the whole frame
    best_i, best_score = -1, -1.0
    for i, m in enumerate(masks):
        m = m.astype(bool)
        if m.mean() > 0.6 or m.mean() < 0.001:
            continue
        ys, xs = np.where(m)
        if ys.size == 0: continue
        if ys.min() >= baseline:           # entirely below baseline -> rubbish
            continue
        # must reach within ~10 px of the baseline
        if (baseline - ys.max()) > 15 and not m[max(0, baseline - 15):baseline + 1, :].any():
            continue
        s = scores[i] * (1.0 - max(0.0, m.mean() - 0.25))   # gentle penalty for huge masks
        if s > best_score:
            best_score, best_i = s, i

    if best_i < 0:
        return np.zeros((h, w), np.uint8), baseline
    mask = masks[best_i].astype(np.uint8) * 255
    mask[baseline:, :] = 0
    return mask, baseline
