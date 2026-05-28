import os, cv2 as cv, numpy as np
import preprocessing as pp
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC=os.path.join(REPO,"fwdfoto")

def fill_holes(m):
    h,w=m.shape; ff=m.copy()
    cv.floodFill(ff,np.zeros((h+2,w+2),np.uint8),(0,0),1)
    o=m.copy(); o[ff==0]=1; return o

def largest_on_bottom(m, band=12):
    n,lab,st,_=cv.connectedComponentsWithStats(m.astype(np.uint8),8)
    h=m.shape[0]; best,ba=0,0
    for i in range(1,n):
        if (lab[h-band:,:]==i).any() and st[i,cv.CC_STAT_AREA]>ba:
            best,ba=i,st[i,cv.CC_STAT_AREA]
    return (lab==best).astype(np.uint8) if best else np.zeros_like(m)

def segment_roi(roi, is_blank=False):
    """Segment a preprocessed ROI: dark droplet on light background, plate removed."""
    h,w=roi.shape
    if is_blank: return np.zeros((h,w),np.uint8)
    # Otsu threshold then invert -> droplet (dark) = 1
    _, th = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    th = (th>0).astype(np.uint8)
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((9,9), np.uint8))
    th = fill_holes(th)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    return largest_on_bottom(th)

def solidity(m):
    c,_=cv.findContours(m.astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c=max(c,key=cv.contourArea); a=cv.contourArea(c); hull=cv.contourArea(cv.convexHull(c))
    return a/hull if hull>0 else 0.0

names=["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5","h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]
tiles=[]
print(f"{'image':16s} {'yb':>4s} {'roi_h':>5s} blank {'area':>6s} {'solid':>5s}")
for nm in names:
    path=os.path.join(SRC,nm+".jpg")
    pre = pp.preprocess_image(path)
    roi, yb, blank = pre['roi'], pre['substrate_y'], pre['is_blank']
    mask = segment_roi(roi, blank)
    fr = mask.sum()/max(1,mask.size); sol = solidity(mask)
    print(f"{nm:16s} {yb:4d} {roi.shape[0]:5d} {str(blank)[0]}    {fr:6.4f} {sol:5.2f}")
    # build overlay on the matching cropped BGR (timestamp removed, then crop above substrate)
    bgr = cv.imread(path)
    bgr_crop = pp.crop_timestamp(bgr)
    bgr_roi = bgr_crop[:yb, :].copy()
    cnts,_ = cv.findContours((mask*255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(bgr_roi, cnts, -1, (0,0,255), 3)
    cv.line(bgr_roi,(0,bgr_roi.shape[0]-2),(bgr_roi.shape[1],bgr_roi.shape[0]-2),(0,255,0),3)
    s = 360/bgr_roi.shape[1]; ov = cv.resize(bgr_roi, (360, int(bgr_roi.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1)
    cv.putText(ov, nm, (3,14), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    tiles.append(ov)

# pad tiles to same height
H = max(t.shape[0] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, H-t.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30,30,30)) for t in tiles]
mont = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"v5_overlays.jpg"), mont, [cv.IMWRITE_JPEG_QUALITY, 75])
print("saved v5_overlays.jpg")
