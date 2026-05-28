import os, glob, time, cv2 as cv, numpy as np
from collections import Counter
from automask_segment import segment, get_generator, rough_baseline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks_auto")
os.makedirs(OUT, exist_ok=True)

print("loading SAM (CPU)...", flush=True)
gen = get_generator()
print("ok", flush=True)

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

results, tiles = [], []
t_start = time.time()
for k, path in enumerate(files):
    nm = os.path.basename(path).replace(".jpg", "")
    img = cv.imread(path)
    t0 = time.time()
    mask = segment(img, generator=gen)
    dt = time.time() - t0
    cv.imwrite(os.path.join(OUT, nm + "_auto.png"), mask)  # save full-size 0/255 mask
    area = (mask > 0).mean()
    v = "ok" if mask.sum() > 0 else "empty"
    results.append((nm, area, v))
    base = rough_baseline(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

    ov = img.copy()
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov, cnts, -1, (0, 0, 255), 3)
    TW = 260; s = TW / ov.shape[1]; tile = cv.resize(ov, (TW, int(ov.shape[0] * s)))
    lab = np.zeros((22, TW, 3), np.uint8)
    color = (0, 255, 0) if v == "ok" else (0, 80, 255)
    cv.putText(lab, f"{nm}|{v}", (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    tiles.append(np.vstack([lab, tile]))
    print(f"[{k+1:2d}/39] {nm:22s} {v:6s} area={area:.3f} ({dt:.1f}s)", flush=True)

print(f"\nTOTAL TIME {time.time()-t_start:.0f}s", flush=True)
vc = Counter(r[2] for r in results)
print(f"counts: {dict(vc)}", flush=True)
for sub in ("h2o", "i2", "oct"):
    sr = [r for r in results if r[0].startswith(sub)]
    print(f"  {sub}: n={len(sr)} {dict(Counter(r[2] for r in sr))}", flush=True)

COLS = 5
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
while len(tiles) % COLS != 0:
    tiles.append(np.full_like(tiles[0], 30))
mont = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
cv.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "automask_all_overlays.jpg"), mont, [cv.IMWRITE_JPEG_QUALITY, 72])
print("saved automask_all_overlays.jpg and masks_auto/*.png", flush=True)
