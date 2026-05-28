import os, cv2, numpy as np, base64
D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")
picks = ["h2o-teflon.jpg","h2o-metall.jpg","h2o-vetro.jpg",
         "oct-lotus.jpg","i2-plexig.jpg","oct-print-5.jpg"]
tiles=[]
for name in picks:
    im=cv2.imread(os.path.join(D,name))
    s=320/im.shape[1]; im=cv2.resize(im,(320,int(im.shape[0]*s)))
    cv2.rectangle(im,(0,0),(319,18),(0,0,0),-1)
    cv2.putText(im,name.replace('.jpg',''),(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    tiles.append(im)
row1=np.hstack(tiles[:3]); row2=np.hstack(tiles[3:]); mont=np.vstack([row1,row2])
ok,buf=cv2.imencode(".jpg",mont,[cv2.IMWRITE_JPEG_QUALITY,62])
b=buf.tobytes()
print("BYTES",len(b))
print("B64START")
print(base64.b64encode(b).decode())
print("B64END")
