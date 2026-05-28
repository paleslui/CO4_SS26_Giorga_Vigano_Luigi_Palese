import os, glob, cv2 as cv, numpy as np, base64, importlib
import dropletseg; importlib.reload(dropletseg)
from dropletseg import segment

D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")
names = ["h2o-teflon","h2o-vetro","h2o-metall","h2o-print-5",
         "h2o-plexig","h2o-rain-100","h2o-fuoc","h2o-tessuto"]

def overlay(img, mask, y_base):
    out = img.copy()
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(out,cnts,-1,(0,0,255),3)            # red contour
    cv.line(out,(0,y_base),(out.shape[1],y_base),(0,255,0),2)  # green baseline
    area = int((mask>0).sum())
    cv.putText(out,f"area={area}",(10,40),cv.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
    return out

tiles=[]
for nm in names:
    img = cv.imread(os.path.join(D,nm+".jpg"))
    mask, dbg = segment(img, return_debug=True)
    ov = overlay(img, mask, dbg["y_base"])
    s = 360/ov.shape[1]; ov=cv.resize(ov,(360,int(ov.shape[0]*s)))
    cv.rectangle(ov,(0,0),(359,18),(0,0,0),-1)
    cv.putText(ov,nm,(3,14),cv.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    tiles.append(ov)

rows=[np.hstack(tiles[i:i+2]) for i in range(0,8,2)]
mont=np.vstack(rows)
cv.imwrite(os.path.join(os.path.dirname(__file__),"v1_overlays.jpg"),mont,[cv.IMWRITE_JPEG_QUALITY,72])
ok,buf=cv.imencode(".jpg",mont,[cv.IMWRITE_JPEG_QUALITY,42])
open(os.path.join(os.path.dirname(__file__),"v1.b64"),"w").write(base64.b64encode(buf.tobytes()).decode())
print("montage bytes", len(buf), "-> saved v1_overlays.jpg and v1.b64")
