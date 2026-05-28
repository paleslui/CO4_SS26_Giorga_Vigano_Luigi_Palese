import os, glob, cv2 as cv, numpy as np
from collections import Counter
from watershed_segment import segment as ws_segment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")

def solidity(m):
    c, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c = max(c, key=cv.contourArea); a = cv.contourArea(c); hull = cv.contourArea(cv.convexHull(c))
    return a/hull if hull > 0 else 0.0

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

results, tiles = [], []
for path in files:
    nm = os.path.basename(path).replace(".jpg", "")
    img = cv.imread(path)
    mask, baseline = ws_segment(img)
    area = (mask > 0).mean(); sol = solidity(mask)
    v = "ok" if mask.sum() > 0 else "empty"
    results.append((nm, baseline, area, sol, v))

    ov = img.copy()
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov, cnts, -1, (0, 0, 255), 3)
    cv.line(ov, (0, baseline), (ov.shape[1], baseline), (0, 255, 0), 2)
    TW = 260; s = TW/ov.shape[1]; tile = cv.resize(ov, (TW, int(ov.shape[0]*s)))
    lab = np.zeros((22, TW, 3), np.uint8)
    color = (0, 255, 0) if v == "ok" else (0, 80, 255)
    cv.putText(lab, f"{nm}|{v}", (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    tiles.append(np.vstack([lab, tile]))

print(f"\n{'image':22s} {'verdict':8s} {'area':>6s} {'solid':>5s}")
for r in results: print(f"{r[0]:22s} {r[4]:8s} {r[2]:6.3f} {r[3]:5.2f}")
vc = Counter(r[4] for r in results)
print(f"\ntotal: {len(results)}  | counts: {dict(vc)}")
for sub in ("h2o", "i2", "oct"):
    sr = [r for r in results if r[0].startswith(sub)]
    print(f"  {sub}: n={len(sr)} {dict(Counter(r[4] for r in sr))}")

COLS = 5
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, H-t.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
while len(tiles) % COLS != 0: tiles.append(np.full_like(tiles[0], 30))
mont = np.vstack([np.hstack(tiles[i:i+COLS]) for i in range(0, len(tiles), COLS)])
cv.imwrite(os.path.join(os.path.dirname(__file__), "watershed_all_overlays.jpg"), mont, [cv.IMWRITE_JPEG_QUALITY, 72])
print(f"saved watershed_all_overlays.jpg ({mont.shape[1]}x{mont.shape[0]})")
