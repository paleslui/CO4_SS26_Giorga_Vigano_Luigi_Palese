"""Run all three methods on all 39 images, save masks per method, build a
3-column comparison montage (basic | watershed | SAM) and print summary stats."""
import os, glob, time, json, cv2 as cv, numpy as np
from collections import Counter
from basic_segment import segment as basic_seg
from watershed_segment import segment as ws_seg
from sam_segment import segment as sam_seg, _get_predictor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")
OUT = os.path.dirname(os.path.abspath(__file__))
MASK_DIRS = {m: os.path.join(OUT, "masks_" + m) for m in ("basic", "watershed", "sam")}
for d in MASK_DIRS.values(): os.makedirs(d, exist_ok=True)

def solidity(m):
    c, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c = max(c, key=cv.contourArea); a = cv.contourArea(c); hull = cv.contourArea(cv.convexHull(c))
    return a/hull if hull > 0 else 0.0

print("loading SAM..."); t0 = time.time(); _get_predictor(); print(f"  loaded in {time.time()-t0:.1f}s")

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

stats = {m: [] for m in MASK_DIRS}
rows = []
for path in files:
    nm = os.path.basename(path).replace(".jpg", "")
    img = cv.imread(path)
    methods = {"basic": basic_seg(img), "watershed": ws_seg(img), "sam": sam_seg(img)}
    cells = []
    baseline_for_label = None
    for mname, (mask, baseline) in methods.items():
        cv.imwrite(os.path.join(MASK_DIRS[mname], nm + ".png"), mask)
        if baseline_for_label is None: baseline_for_label = baseline
        v = "ok" if mask.sum() > 0 else "empty"
        stats[mname].append((nm, (mask > 0).mean(), solidity(mask), v))
        ov = img.copy()
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(ov, cnts, -1, (0, 0, 255), 3)
        cv.line(ov, (0, baseline), (ov.shape[1], baseline), (0, 255, 0), 2)
        TW = 380; s = TW/ov.shape[1]; tile = cv.resize(ov, (TW, int(ov.shape[0]*s)))
        lab = np.zeros((22, TW, 3), np.uint8)
        cv.putText(lab, f"{mname}", (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cells.append(np.vstack([lab, tile]))
    H = max(c.shape[0] for c in cells)
    cells = [cv.copyMakeBorder(c, 0, H-c.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)) for c in cells]
    row_img = np.hstack(cells)
    # row label on left
    lab = np.zeros((H, 130, 3), np.uint8)
    cv.putText(lab, nm, (5, H//2), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
    rows.append(np.hstack([lab, row_img]))

H = max(r.shape[0] for r in rows)
W = max(r.shape[1] for r in rows)
rows = [cv.copyMakeBorder(r, 0, H-r.shape[0], 0, W-r.shape[1], cv.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
mont = np.vstack(rows)
out_path = os.path.join(OUT, "compare_all_overlays.jpg")
cv.imwrite(out_path, mont, [cv.IMWRITE_JPEG_QUALITY, 70])
print(f"saved {out_path} ({mont.shape[1]}x{mont.shape[0]})")

# summary
print(f"\n{'method':12s} {'n_ok':>5s} {'n_empty':>7s} {'mean_solid':>10s} {'mean_area':>10s}")
summary = {}
for mname in MASK_DIRS:
    sr = stats[mname]
    n_ok = sum(1 for r in sr if r[3] == "ok")
    n_em = sum(1 for r in sr if r[3] == "empty")
    sol = np.mean([r[2] for r in sr if r[3] == "ok"]) if n_ok else 0.0
    ar = np.mean([r[1] for r in sr if r[3] == "ok"]) if n_ok else 0.0
    summary[mname] = {"n_ok": n_ok, "n_empty": n_em, "mean_solidity": float(sol), "mean_area": float(ar)}
    print(f"{mname:12s} {n_ok:>5d} {n_em:>7d} {sol:>10.3f} {ar:>10.3f}")

# per-image dump for the notebook
with open(os.path.join(OUT, "compare_all_stats.json"), "w") as f:
    json.dump({"summary": summary, "per_image": {m: [list(r) for r in stats[m]] for m in stats}}, f, indent=2)
print("saved compare_all_stats.json")
