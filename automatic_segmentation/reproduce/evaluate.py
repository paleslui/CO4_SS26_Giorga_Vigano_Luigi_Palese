"""
Evaluation metrics for segmentation (Dice & IoU), per the lecture (Block 6-05).

    Dice(A,B) = 2|A n B| / (|A| + |B|)
    IoU(A,B)  =  |A n B| / |A u B|         (always <= Dice)

dice()/iou() take any two arrays and treat them as binary (>0).

As a script, it scores one method's masks against a folder of manual ground-truth
masks (matched by file name) and prints a per-image table plus the mean +- std.

    python automatic_segmentation/reproduce/evaluate.py \
        --method automask \
        --manual "/path/to/binary_manual_masks"

NOTE: the manual masks must be true binary masks (droplet = white, background =
black). Grayscale photo-like exports will NOT work (see README, "Manual masks").
"""
import os, sys, glob, argparse
import numpy as np, cv2 as cv

def dice(a, b):
    a = a > 0; b = b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)

def iou(a, b):
    a = a > 0; b = b > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)

def _load_binary(path):
    m = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8) * 255

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="automask")
    ap.add_argument("--manual", required=True, help="folder of BINARY manual masks")
    args = ap.parse_args()

    auto_dir = os.path.join(RESULTS, "masks", args.method)
    rows = []
    for ap_ in sorted(glob.glob(os.path.join(auto_dir, "*.png"))):
        nm = os.path.splitext(os.path.basename(ap_))[0]
        # try a few common manual-mask name patterns
        cand = [os.path.join(args.manual, nm + s) for s in
                (".png", "_mask.png", ".tif", "_mask.tif")]
        mp = next((c for c in cand if os.path.exists(c)), None)
        if mp is None:
            continue
        a = _load_binary(ap_); b = _load_binary(mp)
        if a.shape != b.shape:
            b = cv.resize(b, (a.shape[1], a.shape[0]), interpolation=cv.INTER_NEAREST)
        rows.append((nm, dice(a, b), iou(a, b)))

    if not rows:
        print("No matching manual masks found. Are they binary and named to match?")
        sys.exit(0)
    print(f"{'image':22s} {'Dice':>6s} {'IoU':>6s}")
    for nm, d, j in rows:
        print(f"{nm:22s} {d:6.3f} {j:6.3f}")
    ds = np.array([r[1] for r in rows]); js = np.array([r[2] for r in rows])
    print(f"\nN={len(rows)}  Dice {ds.mean():.3f}+-{ds.std():.3f}   IoU {js.mean():.3f}+-{js.std():.3f}")
