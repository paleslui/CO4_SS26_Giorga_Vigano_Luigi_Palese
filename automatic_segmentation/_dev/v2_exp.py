import os, cv2 as cv, numpy as np
import preprocessing as pp
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")

def fill_holes(mask):
    h,w=mask.shape; ff=mask.copy()
    cv.floodFill(ff, np.zeros((h+2,w+2),np.uint8),(0,0),1)
    out=mask.copy(); out[ff==0]=1; return out

def largest_on_baseline(mask,yb,band=12):
    n,lab,st,_=cv.connectedComponentsWithStats(mask.astype(np.uint8),8)
    y0=max(0,yb-band); best,ba=0,0
    for i in range(1,n):
        if (lab[y0:yb+1,:]==i).any() and st[i,cv.CC_STAT_AREA]>ba: best,ba=i,st[i,cv.CC_STAT_AREA]
    return (lab==best).astype(np.uint8) if best else np.zeros_like(mask)

def grow_bg_fixed(prep,yb,tol):
    h,w=prep.shape
    ff=np.zeros((h+2,w+2),np.uint8); ff[yb+1:,:]=1
    flags=4|cv.FLOODFILL_MASK_ONLY|cv.FLOODFILL_FIXED_RANGE|(255<<8)
    for sx,sy in [(2,2),(w//2,2),(w-3,2),(2,max(2,yb//4)),(w-3,max(2,yb//4))]:
        if ff[sy+1,sx+1]==0: cv.floodFill(prep,ff,(sx,sy),0,tol,tol,flags)
    return ff[1:h+1,1:w+1]==255

def seg(bgsub,yb,tol):
    h,w=bgsub.shape
    bg=grow_bg_fixed(bgsub,yb,tol)
    above=np.zeros((h,w),np.uint8); above[:yb,:]=1
    cand=((above==1)&(~bg)).astype(np.uint8)
    cand=cv.morphologyEx(cand,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
    cand=fill_holes(cand)
    cand=cv.morphologyEx(cand,cv.MORPH_OPEN,np.ones((5,5),np.uint8))
    m=largest_on_baseline(cand,yb)
    return m.sum()/max(1,above.sum())

names=["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5","h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
print(f"{'image':16s} {'bg_lvl':>6s} " + " ".join(f"tol{t:>3d}" for t in [25,35,45,60]))
for nm in names:
    img=cv.imread(os.path.join(SRC,nm+".jpg"))
    gray=pp.extract_best_channel(pp.crop_timestamp(img))
    bgsub=pp.subtract_background(gray)
    yb=pp.detect_substrate_line(gray)
    bglvl=int(np.median(bgsub[:30,:]))
    fr=[seg(bgsub.copy(),yb,t) for t in [25,35,45,60]]
    print(f"{nm:16s} {bglvl:6d} " + " ".join(f"{f:6.3f}" for f in fr))
