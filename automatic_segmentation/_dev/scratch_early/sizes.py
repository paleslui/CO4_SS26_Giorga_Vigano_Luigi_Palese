import os, glob, cv2, base64

D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")
OUT = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(D, "*.jpg")))

def small(path, width, q):
    im = cv2.imread(path)
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0]*s)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

# representative spread: hydrophobic + hydrophilic + each substance + tricky
picks = ["h2o-metall.jpg","h2o-teflon.jpg","h2o-vetro.jpg","oct-lotus.jpg",
         "i2-plexig.jpg","oct-print-5.jpg"]
for name in picks:
    b = small(os.path.join(D,name), 600, 72)
    print(name, "jpeg_bytes", len(b), "b64_bytes", len(base64.b64encode(b)))

# also a full montage small
mb = open(os.path.join(OUT,"montage.jpg"),"rb").read()
print("montage.jpg bytes", len(mb), "b64", len(base64.b64encode(mb)))
print("DONE")
