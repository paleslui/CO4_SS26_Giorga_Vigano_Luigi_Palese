import os, cv2 as cv, numpy as np
REPO = os.path.dirname(os.path.abspath(__file__)).replace("/automatic_segmentation","")
SRC = os.path.join(REPO, "fwdfoto")

def cur(gray):
    h,w=gray.shape
    e=np.abs(cv.Sobel(gray,cv.CV_32F,0,1,3)).mean(1)
    e=cv.GaussianBlur(e.reshape(-1,1),(1,11),0).ravel()
    lo,hi=int(0.40*h),int(0.97*h)
    return lo+int(np.argmax(e[lo:hi]))

def topmost_edge(gray):
    h,w=gray.shape
    g=cv.medianBlur(gray,5)
    e=np.abs(cv.Sobel(g,cv.CV_32F,0,1,3)).mean(1)
    e=cv.GaussianBlur(e.reshape(-1,1),(1,11),0).ravel()
    lo,hi=int(0.45*h),int(0.98*h)
    seg=e[lo:hi]; thr=0.30*seg.max()
    above=(seg>thr).astype(int)
    run=np.convolve(above,np.ones(8,int),'valid')
    idx=int(np.argmax(run>=5))
    return lo+idx

def median_transition(gray):
    h,w=gray.shape
    sky=np.median(gray[:int(0.10*h),:],axis=0).astype(int)
    dev=np.abs(gray.astype(int)-sky[None,:])
    start=int(0.30*h); thr=22
    trans=np.full(w,h)
    sub=(dev[start:,:]>thr).astype(int)
    K=15
    for x in range(w):
        run=np.convolve(sub[:,x],np.ones(K,int),'valid')
        j=int(np.argmax(run>=12))
        if run[j]>=12: trans[x]=start+j
    return int(np.median(trans))

names=["h2o-metall","i2-rain-100","h2o-rain-100","i2-rain-5","h2o-rain-5","h20-lotus",
       "h2o-teflon","oct-fuoc","i2-vetro"]
tiles=[]
for nm in names:
    img=cv.imread(os.path.join(SRC,nm+".jpg")); g=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    yc, yt, ym = cur(g), topmost_edge(g), median_transition(g)
    ov=img.copy(); w=ov.shape[1]
    cv.line(ov,(0,yc),(w,yc),(0,0,255),2)
    cv.line(ov,(0,yt),(w,yt),(0,255,255),2)
    cv.line(ov,(0,ym),(w,ym),(255,0,0),2)
    print(f"{nm:14s} cur(R)={yc:4d}  topmost(Y)={yt:4d}  median(B)={ym:4d}")
    TW=300; s=TW/w; tile=cv.resize(ov,(TW,int(ov.shape[0]*s)))
    lab=np.zeros((20,TW,3),np.uint8); cv.putText(lab,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
    tiles.append(np.vstack([lab,tile]))
H=max(t.shape[0] for t in tiles)
tiles=[cv.copyMakeBorder(t,0,H-t.shape[0],0,0,cv.BORDER_CONSTANT,value=(30,30,30)) for t in tiles]
mont=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)])
cv.imwrite(os.path.join(os.path.dirname(__file__),"baseline_test.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,80])
print("\nRED=current  YELLOW=topmost-edge  BLUE=median-transition")
