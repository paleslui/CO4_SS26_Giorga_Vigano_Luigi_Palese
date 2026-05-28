import os, cv2 as cv, numpy as np, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from basic_segment import detect_baseline

REPO = os.path.dirname(os.path.abspath(__file__)).replace("/automatic_segmentation","")
SRC = os.path.join(REPO, "fwdfoto")
CKPT = os.path.expanduser("~/Library/Caches/sam_models/sam_vit_b_01ec64.pth")
DEV = "cpu"

sam = sam_model_registry["vit_b"](checkpoint=CKPT); sam.to(DEV)
gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86,
                                stability_score_thresh=0.90, min_mask_region_area=500)

# the cases that "should be easy" but failed
names = ["h2o-metall","i2-rain-100","h20-lotus","h2o-rain-100","i2-rain-5","h2o-rain-5"]

def pick_droplet(masks, baseline, h, w):
    """Choose the SAM proposal that looks like a droplet sitting on the baseline."""
    cands = []
    for d in masks:
        m = d["segmentation"]
        area = m.sum() / (h*w)
        if area < 0.003 or area > 0.25:           # not tiny, not the whole scene
            continue
        ys, xs = np.where(m)
        top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        if m[:5, :].any():                          # touches image top -> it's background
            continue
        if (right-left) > 0.8*w:                    # spans full width -> background/plate band
            continue
        if abs(bot - baseline) > 40:                # must sit on the baseline
            continue
        if top >= baseline:                         # entirely below baseline -> plate
            continue
        # droplet score: prefer compact + sitting right on the line
        contour,_ = cv.findContours(m.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        c = max(contour, key=cv.contourArea)
        hull = cv.contourArea(cv.convexHull(c)); sol = cv.contourArea(c)/hull if hull>0 else 0
        cands.append((sol, area, m))
    if not cands: return None
    # prefer the most solid; tie-break larger
    cands.sort(key=lambda t:(round(t[0],2), t[1]), reverse=True)
    return cands[0][2]

tiles=[]
for nm in names:
    img = cv.imread(os.path.join(SRC, nm+".jpg"))
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h,w = img.shape[:2]
    gray = cv.medianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY),5)
    baseline = detect_baseline(gray)
    masks = gen.generate(rgb)
    pick = pick_droplet(masks, baseline, h, w)
    ov = img.copy()
    cv.line(ov,(0,baseline),(w,baseline),(0,255,0),2)
    if pick is not None:
        cnts,_ = cv.findContours(pick.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(ov, cnts, -1, (0,0,255), 3)
        v = f"ok n_masks={len(masks)}"
    else:
        v = f"MISS n_masks={len(masks)}"
    print(f"{nm:16s} baseline={baseline} -> {v}")
    TW=300; s=TW/w; tile=cv.resize(ov,(TW,int(h*s)))
    lab=np.zeros((22,TW,3),np.uint8); cv.putText(lab,nm,(3,15),cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(np.vstack([lab,tile]))

H=max(t.shape[0] for t in tiles)
tiles=[cv.copyMakeBorder(t,0,H-t.shape[0],0,0,cv.BORDER_CONSTANT,value=(30,30,30)) for t in tiles]
mont=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,6,3)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"automask_test.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,80])
print("saved automask_test.jpg")
