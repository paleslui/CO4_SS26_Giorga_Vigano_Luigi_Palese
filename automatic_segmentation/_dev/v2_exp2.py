import os, cv2 as cv, numpy as np
import preprocessing as pp
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fwdfoto")

def fill_holes(m):
    h,w=m.shape; ff=m.copy(); cv.floodFill(ff,np.zeros((h+2,w+2),np.uint8),(0,0),1)
    o=m.copy(); o[ff==0]=1; return o
def largest_on_baseline(m,yb,band=14):
    n,lab,st,_=cv.connectedComponentsWithStats(m.astype(np.uint8),8)
    y0=max(0,yb-band); best,ba=0,0
    for i in range(1,n):
        if (lab[y0:yb+1,:]==i).any() and st[i,cv.CC_STAT_AREA]>ba: best,ba=i,st[i,cv.CC_STAT_AREA]
    return (lab==best).astype(np.uint8) if best else np.zeros_like(m)

def segment(prep, yb):
    h,w=prep.shape
    med=float(np.median(prep))
    edges=cv.Canny(prep,int(max(0,0.66*med)),int(min(255,1.33*med)))
    edges=cv.dilate(edges,np.ones((3,3),np.uint8),1)
    barrier=edges.copy(); barrier[max(0,yb-2):yb+1,:]=255      # close bottom at baseline
    free=(barrier==0).astype(np.uint8); free[yb:,:]=0
    ff=np.zeros((h+2,w+2),np.uint8); ff[yb+1:,:]=1
    flags=4|cv.FLOODFILL_MASK_ONLY|(255<<8)
    fl=free.copy()
    for sx,sy in [(2,2),(w//2,2),(w-3,2),(2,max(2,yb//4)),(w-3,max(2,yb//4))]:
        if fl[sy,sx]==1 and ff[sy+1,sx+1]==0: cv.floodFill(fl,ff,(sx,sy),2,0,0,flags)
    bg=ff[1:h+1,1:w+1]==255
    above=np.zeros((h,w),np.uint8); above[:yb,:]=1
    cand=((above==1)&(~bg)).astype(np.uint8)
    cand=cv.morphologyEx(cand,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
    cand=fill_holes(cand)
    cand=cv.morphologyEx(cand,cv.MORPH_OPEN,np.ones((7,7),np.uint8))
    return largest_on_baseline(cand,yb)*255

names=["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5","h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
tiles=[]
for nm in names:
    path=os.path.join(SRC,nm+".jpg")
    pre=pp.preprocess_image(path)
    prep,yb=pre["preprocessed"],pre["substrate_y"]
    mask=segment(prep,yb)
    bgr=pp.crop_timestamp(cv.imread(path))
    ov=bgr.copy()
    cnts,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov,cnts,-1,(0,0,255),3); cv.line(ov,(0,yb),(ov.shape[1],yb),(0,255,0),2)
    fr=mask.sum()/255/mask.size
    print(f"{nm:16s} yb={yb:4d} area_frac={fr:.4f}")
    s=360/ov.shape[1]; ov=cv.resize(ov,(360,int(ov.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1); cv.putText(ov,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(ov)
mont=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"v2_overlays.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,72])
print("saved")
