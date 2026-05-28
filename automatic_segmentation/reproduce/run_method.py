"""
Run one segmentation method over the whole dataset and save the masks + an overlay grid.

Usage (from the repo, with the project venv active):
    python automatic_segmentation/reproduce/run_method.py --method automask
    python automatic_segmentation/reproduce/run_method.py --method basic

Methods: basic | watershed | sam | automask   (automask = best, but CPU-only & slow)
Outputs:
    results/masks/<method>/<image>.png      one binary 0/255 mask per image
    results/overlays/<method>_all.jpg       a labelled grid of all overlays
"""
import sys, os, glob, time, argparse, importlib
import cv2 as cv, numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "methods"))          # make methods importable
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "fwdfoto")
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["basic", "watershed", "sam", "automask"])
args = ap.parse_args()

seg = importlib.import_module(f"{args.method}_segment").segment
maskdir = os.path.join(RESULTS, "masks", args.method)
os.makedirs(maskdir, exist_ok=True)
os.makedirs(os.path.join(RESULTS, "overlays"), exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

tiles, verdicts = [], []
t0 = time.time()
for k, path in enumerate(files):
    nm = os.path.basename(path).replace(".jpg", "")
    img = cv.imread(path)
    out = seg(img)
    mask = out[0] if isinstance(out, tuple) else out             # some methods also return baseline
    cv.imwrite(os.path.join(maskdir, nm + ".png"), mask)
    v = "ok" if mask.sum() > 0 else "empty"
    verdicts.append(v)
    ov = img.copy()
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov, cnts, -1, (0, 0, 255), 3)
    TW = 260; s = TW / ov.shape[1]; tile = cv.resize(ov, (TW, int(ov.shape[0] * s)))
    lab = np.zeros((22, TW, 3), np.uint8)
    cv.putText(lab, f"{nm}|{v}", (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42,
               (0, 255, 0) if v == "ok" else (0, 80, 255), 1)
    tiles.append(np.vstack([lab, tile]))
    print(f"[{k+1:2d}/{len(files)}] {nm:22s} {v}", flush=True)

COLS = 5
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
while len(tiles) % COLS: tiles.append(np.full_like(tiles[0], 30))
mont = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
cv.imwrite(os.path.join(RESULTS, "overlays", f"{args.method}_all.jpg"), mont, [cv.IMWRITE_JPEG_QUALITY, 72])
print(f"\n{args.method}: {dict(Counter(verdicts))}  in {time.time()-t0:.0f}s")
print(f"masks -> results/masks/{args.method}/   grid -> results/overlays/{args.method}_all.jpg")
