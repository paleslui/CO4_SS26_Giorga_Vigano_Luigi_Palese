import os, cv2 as cv, numpy as np
import preprocessing as pp
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC=os.path.join(REPO,"fwdfoto")

def baseline_topmost(gray):
    h,w=gray.shape
    e=np.abs(cv.Sobel(gray,cv.CV_32F,0,1,ksize=5)).sum(1)
    e=np.convolve(e,np.ones(15)/15,mode="same")
    lo,hi=h//3,h-3; seg=e[lo:hi]
    strong=np.where(seg>0.45*seg.max())[0]
    return lo+int(strong[0]) if len(strong) else lo+int(np.argmax(seg))

def fill_holes(m):
    h,w=m.shape; ff=m.copy(); cv.floodFill(ff,np.zeros((h+2,w+2),np.uint8),(0,0),1)
    o=m.copy(); o[ff==0]=1; return o
def largest_on_baseline(m,yb,band=16):
    n,lab,st,_=cv.connectedComponentsWithStats(m.astype(np.uint8),8)
    y0=max(0,yb-band); best,ba=0,0
    for i in range(1,n):
        if (lab[y0:yb+1,:]==i).any() and st[i,cv.CC_STAT_AREA]>ba: best,ba=i,st[i,cv.CC_STAT_AREA]
    return (lab==best).astype(np.uint8) if best else np.zeros_like(m)

def segment_v3(bg_sub, yb, is_blank):
    h,w=bg_sub.shape
    if is_blank: return np.zeros((h,w),np.uint8)
    above=np.zeros((h,w),np.uint8); above[:yb,:]=1
    bg_level=float(np.median(bg_sub[:max(5,yb//6),:]))
    # rim = gradient ridges (sensitive); bg_sub bg is uniform so few false edges
    gx=cv.Sobel(bg_sub,cv.CV_32F,1,0,3); gy=cv.Sobel(bg_sub,cv.CV_32F,0,1,3)
    mag=cv.normalize(np.hypot(gx,gy),None,0,255,cv.NORM_MINMAX)
    thr=max(8.0, mag.mean()+0.8*mag.std())
    rim=(mag>thr).astype(np.uint8)
    dark=(bg_sub < bg_level-25).astype(np.uint8)        # dark-body cue (for opaque drops)
    seed=((rim|dark)&above).astype(np.uint8)
    seed[max(0,yb-2):yb+1,:]=1                            # close the bottom at baseline
    closed=cv.morphologyEx(seed,cv.MORPH_CLOSE,cv.getStructuringElement(cv.MORPH_ELLIPSE,(25,25)))
    filled=fill_holes(closed)
    filled=cv.morphologyEx(filled,cv.MORPH_OPEN,np.ones((9,9),np.uint8))
    filled[yb:,:]=0
    return largest_on_baseline(filled,yb)

names=["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5","h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
tiles=[]
for nm in names:
    path=os.path.join(SRC,nm+".jpg"); img=cv.imread(path)
    gray=pp.extract_best_channel(pp.crop_timestamp(img)); bgsub=pp.subtract_background(gray)
    yb=baseline_topmost(gray)
    blank=pp.detect_blank_image(pp.denoise(pp.apply_clahe(bgsub)),yb)
    m=segment_v3(bgsub,yb,blank)
    bgr=pp.crop_timestamp(img); ov=bgr.copy()
    cnts,_=cv.findContours((m*255).astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov,cnts,-1,(0,0,255),3); cv.line(ov,(0,yb),(ov.shape[1],yb),(0,255,0),2)
    print(f"{nm:16s} yb={yb:4d} blank={blank} area_frac={m.sum()/m.size:.4f}")
    s=360/ov.shape[1]; ov=cv.resize(ov,(360,int(ov.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1); cv.putText(ov,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(ov)
mont=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"v3_overlays.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,72]); print("saved")
