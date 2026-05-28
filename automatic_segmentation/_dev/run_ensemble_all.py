import os, glob, cv2 as cv, numpy as np
from collections import Counter
from ensemble_segment import segment as ens_segment, quality
from sam_segment_v2 import _get_predictor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")
print("loading SAM..."); _get_predictor(); print("ok")

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

results, tiles = [], []
for path in files:
    nm = os.path.basename(path).replace(".jpg", "")
    img = cv.imread(path)
    mask, baseline, method = ens_segment(img, return_method=True)
    area = (mask > 0).mean()
    v = method if mask.sum() > 0 else "empty"
    results.append((nm, method, area, v))

    ov = img.copy()
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov, cnts, -1, (0, 0, 255), 3)
    cv.line(ov, (0, baseline), (ov.shape[1], baseline), (0, 255, 0), 2)
    TW = 260; s = TW/ov.shape[1]; tile = cv.resize(ov, (TW, int(ov.shape[0]*s)))
    lab = np.zeros((22, TW, 3), np.uint8)
    col = {"sam": (0,255,0), "basic": (255,200,0), "none": (0,80,255)}[v if v in ("sam","basic","none") else "none"]
    cv.putText(lab, f"{nm}|{v}", (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    tiles.append(np.vstack([lab, tile]))
    print(f"  {nm:22s} -> {v:6s} area={area:.3f}")

print(f"\nmethod chosen: {dict(Counter(r[1] for r in results))}")
print(f"non-empty: {sum(1 for r in results if r[3] != 'none')}/{len(results)}")

COLS = 5
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, H-t.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30,30,30)) for t in tiles]
while len(tiles) % COLS != 0: tiles.append(np.full_like(tiles[0], 30))
mont = np.vstack([np.hstack(tiles[i:i+COLS]) for i in range(0, len(tiles), COLS)])
cv.imwrite(os.path.join(os.path.dirname(__file__), "ensemble_all_overlays.jpg"), mont, [cv.IMWRITE_JPEG_QUALITY, 72])
print(f"saved ensemble_all_overlays.jpg ({mont.shape[1]}x{mont.shape[0]})")
