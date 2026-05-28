import os, cv2 as cv, numpy as np, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
REPO = os.path.dirname(os.path.abspath(__file__)).replace("/automatic_segmentation","")
SRC = os.path.join(REPO, "fwdfoto")
CKPT = os.path.expanduser("~/Library/Caches/sam_models/sam_vit_b_01ec64.pth")

sam = sam_model_registry["vit_b"](checkpoint=CKPT); sam.to("cpu")
gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86,
                                stability_score_thresh=0.90, min_mask_region_area=400)

def rough_baseline(gray):  # yellow = topmost sustained horizontal edge
    h,w=gray.shape; g=cv.medianBlur(gray,5)
    e=np.abs(cv.Sobel(g,cv.CV_32F,0,1,3)).mean(1)
    e=cv.GaussianBlur(e.reshape(-1,1),(1,11),0).ravel()
    lo,hi=int(0.45*h),int(0.98*h); seg=e[lo:hi]; thr=0.30*seg.max()
    run=np.convolve((seg>thr).astype(int),np.ones(8,int),'valid')
    return lo+int(np.argmax(run>=5))

def pick(masks, h, w, base):
    best=None; bestscore=-1
    for d in masks:
        m=d["segmentation"]; area=m.sum()/(h*w)
        if area<0.003 or area>0.25: continue
        ys,xs=np.where(m); top,bot,l,r=ys.min(),ys.max(),xs.min(),xs.max()
        if m[:int(0.05*h),:].any(): continue          # touches top -> background
        if (r-l)>0.75*w: continue                       # full width -> plate/bg band
        cnt,_=cv.findContours(m.astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        c=max(cnt,key=cv.contourArea); hull=cv.contourArea(cv.convexHull(c))
        sol=cv.contourArea(c)/hull if hull>0 else 0
        if sol<0.5: continue                            # not blob-like
        base_bonus=1.0-min(1.0,abs(bot-base)/300.0)     # soft: bottom near baseline
        score=sol*0.5 + min(area,0.10)/0.10*0.3 + base_bonus*0.2
        if score>bestscore: bestscore,best=score,m
    return best

names=["h2o-metall","i2-rain-100","h2o-rain-100","i2-rain-5","h2o-rain-5","h20-lotus",
       "h2o-teflon","oct-fuoc","i2-vetro","h2o-fuoc","i2-teflon","oct-lotus"]
tiles=[]
for nm in names:
    img=cv.imread(os.path.join(SRC,nm+".jpg")); h,w=img.shape[:2]
    rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    g=cv.cvtColor(img,cv.COLOR_BGR2GRAY); base=rough_baseline(g)
    masks=gen.generate(rgb); m=pick(masks,h,w,base)
    ov=img.copy()
    if m is not None:
        cnts,_=cv.findContours(m.astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(ov,cnts,-1,(0,0,255),3); v=f"ok({len(masks)})"
    else: v=f"MISS({len(masks)})"
    print(f"{nm:14s} base={base:4d} -> {v}")
    TW=300;s=TW/w;tile=cv.resize(ov,(TW,int(h*s)))
    lab=np.zeros((20,TW,3),np.uint8);cv.putText(lab,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
    tiles.append(np.vstack([lab,tile]))
H=max(t.shape[0] for t in tiles)
tiles=[cv.copyMakeBorder(t,0,H-t.shape[0],0,0,cv.BORDER_CONSTANT,value=(30,30,30)) for t in tiles]
mont=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,12,4)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"automask2_test.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,80])
print("saved automask2_test.jpg")
