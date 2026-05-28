import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r'''# Automatic Segmentation of Sessile Droplets — Classic CV

**Course:** MSLS / CO4 — Imaging for the Life Sciences (ZHAW)  
**Part:** Automatic segmentation + evaluation (Luigi)

We segment back-lit side-view droplet images using only classic computer-vision
operations (thresholding, edges, morphology) — no machine learning. The method
follows the four-step recipe suggested by the instructor:

1. **Background removal** — region-grow the background inward from the top corners.
2. **Support plate** — everything below the detected baseline is the plate.
3. **Mask logic** — the droplet is the region *enclosed* by background and plate.
4. **Refinement** — close gaps, fill holes (removes specular glints), keep the
   blob sitting on the baseline, cut flat at the baseline.

The images are first cleaned by the **preprocessing pipeline** (`preprocessing.py`),
whose background-subtraction step flattens the strong, uneven back-lighting so the
same segmentation logic works across very different substrates.''')

code(r'''import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import preprocessing as pp          # preprocessing pipeline (notebook "Preprocessing" stage)

%matplotlib inline
plt.rcParams["figure.dpi"] = 110
DATA = os.path.join("..", "fwdfoto")''')

md(r'''## 1. Preprocessing (recap)

Each raw photo is converted to a single channel (the most informative one),
the timestamp banner is cropped, the uneven illumination is divided out
(background subtraction), local contrast is enhanced (CLAHE) and the image is
denoised while preserving edges (bilateral filter). The same step also estimates
the **baseline** (the top edge of the support plate) and flags "blank" images
where the liquid spread completely flat (no droplet).''')

code(r'''sample = os.path.join(DATA, "h2o-metall.jpg")
pre = pp.preprocess_image(sample)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(cv.cvtColor(cv.imread(sample), cv.COLOR_BGR2RGB)); ax[0].set_title("original"); ax[0].axis("off")
ax[1].imshow(pre["preprocessed"], cmap="gray", vmin=0, vmax=255); ax[1].set_title("preprocessed"); ax[1].axis("off")
ov = cv.cvtColor(pre["preprocessed"], cv.COLOR_GRAY2RGB)
cv.line(ov, (0, pre["substrate_y"]), (ov.shape[1], pre["substrate_y"]), (255, 0, 0), 2)
ax[2].imshow(ov); ax[2].set_title(f"detected baseline (y={pre['substrate_y']})"); ax[2].axis("off")
plt.tight_layout(); plt.show()''')

md(r'''## 2. The segmentation function

`segment_core` implements steps 1–4 on the cleaned image. A pixel blocks the
background flood if it is **either** clearly darker than the background **or** an
edge (the droplet rim). If the flood is somehow blocked and almost the whole frame
is kept, a guardrail falls back to a simple "threshold the dark rim/body and fill"
strategy.''')

code(r'''def _fill_holes(m):
    h, w = m.shape
    ff = m.copy()
    cv.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    out = m.copy(); out[ff == 0] = 1
    return out

def _largest_on_baseline(m, yb, band=14):
    # keep the largest connected blob that actually sits on the baseline
    n, lab, st, _ = cv.connectedComponentsWithStats(m.astype(np.uint8), 8)
    y0 = max(0, yb - band); best, ba = 0, 0
    for i in range(1, n):
        if (lab[y0:yb + 1, :] == i).any() and st[i, cv.CC_STAT_AREA] > ba:
            best, ba = i, st[i, cv.CC_STAT_AREA]
    return (lab == best).astype(np.uint8) if best else np.zeros_like(m)

def _refine(cand, yb):
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cand = _fill_holes(cand)
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return _largest_on_baseline(cand, yb)

def segment_core(bg_sub, yb, is_blank=False, tol=30):
    # bg_sub: background-subtracted gray (uniform bright background)
    # yb: baseline row;  returns a 0/1 mask in the cropped coordinate system
    h, w = bg_sub.shape
    if is_blank:
        return np.zeros((h, w), np.uint8)
    above = np.zeros((h, w), np.uint8); above[:yb, :] = 1
    bg_level = float(np.median(bg_sub[:max(5, yb // 6), :]))

    # barriers for the flood: darker-than-background OR a clean edge
    med = float(np.median(bg_sub))
    edges = cv.Canny(bg_sub, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), 1)
    bright = bg_sub > (bg_level - tol)
    free = ((bright) & (edges == 0)).astype(np.uint8)
    free[yb:, :] = 0

    # step 1: region-grow the background from the top corners/edges
    ff = np.zeros((h + 2, w + 2), np.uint8); ff[yb + 1:, :] = 1
    flags = 4 | cv.FLOODFILL_MASK_ONLY | (255 << 8)
    fl = free.copy()
    for sx, sy in [(2, 2), (w // 2, 2), (w - 3, 2), (2, max(2, yb // 4)), (w - 3, max(2, yb // 4))]:
        if fl[sy, sx] == 1:
            cv.floodFill(fl, ff, (sx, sy), 2, 0, 0, flags)
    bg = ff[1:h + 1, 1:w + 1] == 255

    cand = ((above == 1) & (~bg)).astype(np.uint8)   # steps 2 + 3
    mask = _refine(cand, yb)                          # step 4

    # guardrail: flood blocked -> fall back to threshold-and-fill of the rim/body
    if mask.sum() / max(1, above.sum()) > 0.45:
        dark = ((bg_sub < bg_level - tol) & (above == 1)).astype(np.uint8)
        dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        dark = _fill_holes(dark)
        mask = _refine(dark, yb)
    return mask

def segment(image):
    # full entry point on an ORIGINAL BGR image -> full-size 0/255 mask
    crop = pp.crop_timestamp(image)
    gray = pp.extract_best_channel(crop)
    bg_sub = pp.subtract_background(gray)
    yb = pp.detect_substrate_line(gray)
    prep = pp.denoise(pp.apply_clahe(bg_sub))
    is_blank = pp.detect_blank_image(prep, yb)
    m = segment_core(bg_sub, yb, is_blank)            # cropped, 0/1
    full = np.zeros(image.shape[:2], np.uint8)
    full[:m.shape[0], :m.shape[1]] = (m * 255).astype(np.uint8)
    return full''')

md(r'''## 3. Results on a representative sample

Red = detected droplet outline, green = detected baseline. The set spans easy
cases (clear rims) and deliberately hard ones (faint flat domes, over-exposed
fabric) so we can see where the method holds up and where it struggles.''')

code(r'''names = ["h2o-teflon", "h2o-vetro", "h2o-metall", "h2o-print-5",
         "h2o-plexig", "h2o-rain-100", "h2o-fuoc", "h2o-tessuto"]

fig, axes = plt.subplots(4, 2, figsize=(13, 16))
for ax, nm in zip(axes.ravel(), names):
    img = cv.imread(os.path.join(DATA, nm + ".jpg"))
    mask = segment(img)
    yb = pp.detect_substrate_line(pp.extract_best_channel(pp.crop_timestamp(img)))
    ov = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov, cnts, -1, (255, 0, 0), 3)
    cv.line(ov, (0, yb), (ov.shape[1], yb), (0, 200, 0), 2)
    frac = (mask > 0).mean()
    ax.imshow(ov); ax.set_title(f"{nm}   area={frac:.3f}"); ax.axis("off")
plt.tight_layout(); plt.show()''')

md(r'''## 4. Evaluation metric (Dice)

To score the automatic masks against the manual ground-truth masks we use the
**Dice similarity coefficient**:

$$\mathrm{Dice}(A, B) = \frac{2\,|A \cap B|}{|A| + |B|}$$

It ranges from 0 (no overlap) to 1 (perfect overlap). We also report IoU
(Jaccard). The ground-truth masks are produced in the manual-segmentation step;
once they are available as clean binary masks we compute the mean and standard
deviation of the Dice score across the dataset here.''')

code(r'''def evaluate(mask1, mask2):
    # mask1, mask2: any arrays; treated as binary (>0). Returns Dice and IoU.
    a = mask1 > 0
    b = mask2 > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    denom = a.sum() + b.sum()
    dice = 1.0 if denom == 0 else 2.0 * inter / denom
    iou = 1.0 if union == 0 else inter / union
    return {"dice": float(dice), "iou": float(iou)}

# sanity check: a mask compared with itself scores 1.0
_m = segment(cv.imread(os.path.join(DATA, "h2o-metall.jpg")))
print("self-overlap (should be 1.0):", evaluate(_m, _m))
print("disjoint (should be ~0):", evaluate(_m, np.zeros_like(_m)))''')

md(r'''## 5. Status & next steps

- The region-growing + enclosure logic works well on droplets with a clear rim
  (teflon, print, metall, fuoc, rain).
- Faint flat domes (plexig) and over-exposed fabric (tessuto) are the hard cases:
  the rim is barely darker than the background, so the enclosure leaks or the image
  is flagged blank. These are expected to score lowest.
- **Pending:** real binary ground-truth masks from the manual step, to compute the
  Dice mean/std across the dataset and tune the few hard cases.''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
with open("v2_segmentation.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote v2_segmentation.ipynb with", len(cells), "cells")
