import os, cv2 as cv, numpy as np
import preprocessing as pp
from dropletseg_v2 import segment_core
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC=os.path.join(REPO,"fwdfoto")
names=["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5","h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
tiles=[]
for nm in names:
    path=os.path.join(SRC,nm+".jpg"); img=cv.imread(path)
    gray=pp.extract_best_channel(pp.crop_timestamp(img)); bgsub=pp.subtract_background(gray)
    pre=pp.preprocess_image(path); prep,yb,blank=pre["preprocessed"],pre["substrate_y"],pre["is_blank"]
    m=segment_core(bgsub,prep,yb,blank)
    bgr=pp.crop_timestamp(img); ov=bgr.copy()
    cnts,_=cv.findContours((m*255).astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(ov,cnts,-1,(0,0,255),3); cv.line(ov,(0,yb),(ov.shape[1],yb),(0,255,0),2)
    print(f"{nm:16s} yb={yb:4d} blank={blank} area_frac={m.sum()/m.size:.4f}")
    s=360/ov.shape[1]; ov=cv.resize(ov,(360,int(ov.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1); cv.putText(ov,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(ov)
mont=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"v2_overlays.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,72]); print("saved")
