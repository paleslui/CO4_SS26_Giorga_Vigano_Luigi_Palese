import os, cv2, numpy as np, base64
D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")
picks = ["h2o-teflon.jpg","h2o-metall.jpg","h2o-vetro.jpg","i2-fuoc.jpg"]
tiles=[]
for name in picks:
    im=cv2.imread(os.path.join(D,name))
    s=300/im.shape[1]; im=cv2.resize(im,(300,int(im.shape[0]*s)))
    cv2.rectangle(im,(0,0),(299,16),(0,0,0),-1)
    cv2.putText(im,name.replace('.jpg',''),(3,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
    tiles.append(im)
mont=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
ok,buf=cv2.imencode(".jpg",mont,[cv2.IMWRITE_JPEG_QUALITY,55])
open(os.path.join(os.path.dirname(__file__),"m4.b64"),"w").write(base64.b64encode(buf.tobytes()).decode())
print("bytes",len(buf),"wrote m4.b64")
