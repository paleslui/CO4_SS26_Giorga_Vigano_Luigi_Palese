import os, glob, cv2 as cv, numpy as np
from collections import Counter
import preprocessing as pp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")

def fill_holes(m):
    h,w=m.shape; ff=m.copy()
    cv.floodFill(ff,np.zeros((h+2,w+2),np.uint8),(0,0),1)
    o=m.copy(); o[ff==0]=1; return o
def largest_on_bottom(m, band=14):
    n,lab,st,_=cv.connectedComponentsWithStats(m.astype(np.uint8),8)
    h=m.shape[0]; best,ba=0,0
    for i in range(1,n):
        if (lab[h-band:,:]==i).any() and st[i,cv.CC_STAT_AREA]>ba: best,ba=i,st[i,cv.CC_STAT_AREA]
    return (lab==best).astype(np.uint8) if best else np.zeros_like(m)
def solidity(m):
    c,_=cv.findContours(m.astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c=max(c,key=cv.contourArea); a=cv.contourArea(c); hull=cv.contourArea(cv.convexHull(c))
    return a/hull if hull>0 else 0.0

def _seg_one(img):
    _, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    m = (th>0).astype(np.uint8)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, np.ones((9,9), np.uint8))
    m = fill_holes(m)
    m = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    return largest_on_bottom(m)
def _score(m):
    a = m.sum()/max(1,m.size)
    if a < 0.002 or a > 0.40: return -1.0
    return solidity(m) * 10 + min(a, 0.2)
def segment_roi(roi, is_blank=False):
    if is_blank: return np.zeros_like(roi)
    a = _seg_one(roi); b = _seg_one(255 - roi)
    sa, sb = _score(a), _score(b)
    if sa < 0 and sb < 0: return np.zeros_like(roi)
    return a if sa >= sb else b

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
files = [f for f in files if os.path.basename(f) != "test.jpg"]

results = []; tiles = []
for path in files:
    nm = os.path.basename(path).replace(".jpg","")
    try:
        pre = pp.preprocess_image(path)
        roi, yb, blank = pre['roi'], pre['substrate_y'], pre['is_blank']
        mask = segment_roi(roi, blank)
    except Exception as e:
        results.append((nm, 0, False, 0, 0, f"ERR:{type(e).__name__}"))
        continue
    if blank: v = "blank"
    elif mask.sum() == 0: v = "rejected"
    else: v = "ok"
    a = mask.sum()/max(1,mask.size); s = solidity(mask)
    results.append((nm, yb, blank, a, s, v))
    # overlay tile
    bgr_crop = pp.crop_timestamp(cv.imread(path))
    bgr_roi = bgr_crop[:yb, :].copy()
    cnts,_ = cv.findContours((mask*255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(bgr_roi, cnts, -1, (0,0,255), 2)
    cv.line(bgr_roi,(0,bgr_roi.shape[0]-2),(bgr_roi.shape[1],bgr_roi.shape[0]-2),(0,255,0),2)
    TW=260; s_=TW/bgr_roi.shape[1]; tile=cv.resize(bgr_roi,(TW,int(bgr_roi.shape[0]*s_)))
    lab=np.zeros((22,TW,3),np.uint8)
    color=(0,255,0) if v=="ok" else ((255,180,0) if v=="blank" else (0,80,255))
    cv.putText(lab,f"{nm}|{v}",(3,15),cv.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    tiles.append(np.vstack([lab,tile]))

# console summary
print(f"\n{'image':22s} {'verdict':8s} {'area':>6s} {'solid':>5s}")
for r in results: print(f"{r[0]:22s} {r[5]:8s} {r[3]:6.3f} {r[4]:5.2f}")
vc = Counter(r[5] for r in results)
print(f"\ntotal: {len(results)}  | counts: {dict(vc)}")
# per-substance breakdown
for sub in ("h2o","i2","oct"):
    sub_res = [r for r in results if r[0].startswith(sub)]
    sub_vc = Counter(r[5] for r in sub_res)
    print(f"  {sub}: n={len(sub_res)} {dict(sub_vc)}")

# montage 5 cols
COLS=5
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t,0,H-t.shape[0],0,0,cv.BORDER_CONSTANT,value=(30,30,30)) for t in tiles]
while len(tiles)%COLS!=0: tiles.append(np.full_like(tiles[0],30))
mont = np.vstack([np.hstack(tiles[i:i+COLS]) for i in range(0,len(tiles),COLS)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"all_overlays.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,72])
print(f"saved all_overlays.jpg ({mont.shape[1]}x{mont.shape[0]})")
