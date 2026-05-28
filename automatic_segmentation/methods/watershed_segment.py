"""
watershed_segment.py - Watershed segmentation as a refinement step.

The lecture (Block 6-04) describes watershed as related to region growing: starting
from markers, the algorithm floods a topographic surface (the image gradient) and
the boundaries are drawn where the floods meet.  We use that to *refine* the result
from basic_segment.py:

  1. Get an initial droplet estimate from basic_segment (per-column sky subtraction).
  2. Erode that estimate strongly -> SURE foreground marker (definitely inside drop).
  3. Dilate it and take the complement (plus everything below the baseline)
     -> SURE background marker.
  4. Pixels that are neither foreground nor background remain "unknown".
  5. Apply watershed.  It re-assigns the unknown pixels to either foreground
     or background by following the image gradient, snapping to the actual
     droplet edge.

If basic_segment finds nothing, we return empty (watershed cannot conjure a
droplet out of nowhere).
"""
import cv2 as cv
import numpy as np
from basic_segment import segment as basic_segment


def segment(image):
    initial_mask, baseline = basic_segment(image)
    if initial_mask.sum() == 0:
        return initial_mask, baseline

    k = np.ones((11, 11), np.uint8)
    sure_fg = cv.erode(initial_mask, k, iterations=2)
    if sure_fg.sum() == 0:
        # initial mask was very thin; shrink the kernel and try once more
        sure_fg = cv.erode(initial_mask, np.ones((5, 5), np.uint8), iterations=1)
        if sure_fg.sum() == 0:
            return initial_mask, baseline

    sure_bg = cv.dilate(initial_mask, k, iterations=3)
    sure_bg = (sure_bg == 0).astype(np.uint8) * 255
    sure_bg[baseline:, :] = 255                                  # below the plate is background too

    markers = np.zeros(image.shape[:2], dtype=np.int32)
    markers[sure_bg > 0] = 1                                      # background label
    markers[sure_fg > 0] = 2                                      # droplet label
                                                                  # everything else stays 0 -> "unknown"

    img3 = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(img3, markers)

    mask = np.where(markers == 2, 255, 0).astype(np.uint8)
    mask[baseline:, :] = 0
    return mask, baseline
