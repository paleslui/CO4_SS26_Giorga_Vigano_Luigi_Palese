"""
Build the method-comparison artefacts from the saved masks in results/masks/.

Outputs:
    results/overlays/comparison.jpg   side-by-side [original | basic | watershed | sam | automask]
                                      for a representative set of images
    results/compare_stats.json        per-method: ok-rate and mean solidity over all 39 images

Run AFTER results/masks/<method>/ are populated (see run_method.py).
"""
import os, glob, json
import numpy as np, cv2 as cv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "fwdfoto")
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
METHODS = ["basic", "watershed", "sam", "automask"]

def solidity(m):
    c, _ = cv.findContours((m > 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c = max(c, key=cv.contourArea); h = cv.contourArea(cv.convexHull(c))
    return cv.contourArea(c) / h if h > 0 else 0.0

def load_mask(method, nm):
    p = os.path.join(RESULTS, "masks", method, nm + ".png")
    return cv.imread(p, cv.IMREAD_GRAYSCALE) if os.path.exists(p) else None

names = sorted(os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(SRC, "*.jpg")) if os.path.basename(f) != "test.jpg")

# --- per-method stats over the whole dataset ---
stats = {}
for me in METHODS:
    oks, sols = 0, []
    for nm in names:
        m = load_mask(me, nm)
        if m is not None and m.sum() > 0:
            oks += 1; sols.append(solidity(m))
    stats[me] = {"n": len(names), "ok": oks, "ok_rate": round(oks / len(names), 3),
                 "mean_solidity_of_ok": round(float(np.mean(sols)) if sols else 0.0, 3)}
json.dump(stats, open(os.path.join(RESULTS, "compare_stats.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))

# --- representative comparison grid ---
rep = ["h2o-teflon", "i2-teflon", "h2o-metall", "i2-rain-100", "h2o-fuoc", "oct-lotus",
       "h2o-vetro", "i2-vetro", "oct-fog-100", "i2-tessuto", "h2o-fog-100", "oct-plexig"]
TW = 230
def cell(img, mask, title):
    ov = img.copy()
    if mask is not None and mask.sum() > 0:
        c, _ = cv.findContours((mask > 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(ov, c, -1, (0, 0, 255), 4)
    s = TW / ov.shape[1]; ov = cv.resize(ov, (TW, int(ov.shape[0] * s)))
    lab = np.zeros((20, TW, 3), np.uint8)
    cv.putText(lab, title, (3, 14), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    return np.vstack([lab, ov])

rows = []
for nm in rep:
    img = cv.imread(os.path.join(SRC, nm + ".jpg"))
    if img is None: continue
    cells = [cell(img, None, nm + " (orig)")]
    for me in METHODS:
        cells.append(cell(img, load_mask(me, nm), me))
    h = max(c.shape[0] for c in cells)
    cells = [cv.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)) for c in cells]
    rows.append(np.hstack(cells))
w = max(r.shape[1] for r in rows)
rows = [cv.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
cv.imwrite(os.path.join(RESULTS, "overlays", "comparison.jpg"), np.vstack(rows), [cv.IMWRITE_JPEG_QUALITY, 80])
print("saved results/overlays/comparison.jpg and results/compare_stats.json")
