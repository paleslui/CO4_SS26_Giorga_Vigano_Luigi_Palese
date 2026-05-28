import os, cv2 as cv, numpy as np
import preprocessing as pp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")

# ---------- helpers ----------
def fill_holes(mask):
    h, w = mask.shape
    ff = mask.copy()
    cv.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 1)
    out = mask.copy(); out[ff == 0] = 1
    return out

def largest_on_baseline(mask, y_base, band=12):
    n, lab, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    y0 = max(0, y_base - band)
    best, best_area = 0, 0
    for i in range(1, n):
        if (lab[y0:y_base+1, :] == i).any() and stats[i, cv.CC_STAT_AREA] > best_area:
            best, best_area = i, stats[i, cv.CC_STAT_AREA]
    return (lab == best).astype(np.uint8) if best else np.zeros_like(mask)

# ---------- STEP 1: background removal by region growing ----------
def grow_background(prep, y_base, tol=20):
    h, w = prep.shape
    ffmask = np.zeros((h+2, w+2), np.uint8)
    ffmask[y_base+1:, :] = 1                      # block the plate region
    flags = 4 | cv.FLOODFILL_MASK_ONLY | (255 << 8)
    seeds = [(2,2),(w//2,2),(w-3,2),(2,max(2,y_base//4)),(w-3,max(2,y_base//4))]
    for sx, sy in seeds:
        if ffmask[sy+1, sx+1] == 0:
            cv.floodFill(prep, ffmask, (sx, sy), 0, tol, tol, flags)
    return (ffmask[1:h+1, 1:w+1] == 255)

# ---------- full segmentation (teacher's 4 steps) ----------
def segment_core(prep, y_base, is_blank):
    h, w = prep.shape
    if is_blank:
        return np.zeros((h, w), np.uint8), "blank"
    background = grow_background(prep, y_base, tol=20)        # step 1
    above = np.zeros((h, w), np.uint8); above[:y_base, :] = 1 # step 2 (plate=below)
    cand = ((above == 1) & (~background)).astype(np.uint8)    # step 3 (enclosed)
    # step 4 refine
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((9,9), np.uint8))
    cand = fill_holes(cand)
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = largest_on_baseline(cand, y_base)
    note = "ok"
    # guardrail: implausible area -> flag (fallback later)
    frac = mask.sum() / max(1, above.sum())
    if frac > 0.55 or frac < 0.0005:
        note = f"implausible frac={frac:.3f}"
    return (mask*255).astype(np.uint8), note

def overlay(bgr, mask, y_base, title):
    out = bgr.copy()
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(out, cnts, -1, (0,0,255), 3)
    cv.line(out, (0,y_base), (out.shape[1],y_base), (0,255,0), 2)
    return out

names = ["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5",
         "h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
tiles=[]
for nm in names:
    path = os.path.join(SRC, nm+".jpg")
    pre = pp.preprocess_image(path)
    prep, y_base, blank = pre["preprocessed"], pre["substrate_y"], pre["is_blank"]
    mask, note = segment_core(prep, y_base, blank)
    bgr_crop = pp.crop_timestamp(cv.imread(path))
    ov = overlay(bgr_crop, mask, y_base, nm)
    frac = mask.sum()/255/mask.size
    print(f"{nm:16s} y_base={y_base:4d} blank={blank} area_frac={frac:.4f} {note}")
    s = 360/ov.shape[1]; ov = cv.resize(ov, (360, int(ov.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1)
    cv.putText(ov, nm,(3,14), cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(ov)

rows=[np.hstack(tiles[i:i+2]) for i in range(0,8,2)]
mont=np.vstack(rows)
cv.imwrite(os.path.join(os.path.dirname(__file__),"v2_overlays.jpg"), mont,[cv.IMWRITE_JPEG_QUALITY,72])
print("saved v2_overlays.jpg")
