import os, glob, cv2, numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")
OUT = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(D, "*.jpg")))
print("n images:", len(files))

rows = []
for f in files:
    im = cv2.imread(f)
    h, w = im.shape[:2]
    rows.append((os.path.basename(f), w, h, os.path.getsize(f) // 1024))
for name, w, h, kb in rows:
    print(f"{name:20s} {w}x{h}  {kb}KB")

THUMB_W = 240
cols = 7
imgs = []
for f in files:
    im = cv2.imread(f)
    s = THUMB_W / im.shape[1]
    im = cv2.resize(im, (THUMB_W, int(im.shape[0] * s)))
    cv2.putText(im, os.path.basename(f).replace(".jpg", ""), (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    imgs.append(im)

th = max(i.shape[0] for i in imgs)
imgs = [cv2.copyMakeBorder(i, 0, th - i.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for i in imgs]
rows_img = []
for r in range(0, len(imgs), cols):
    chunk = imgs[r:r + cols]
    while len(chunk) < cols:
        chunk.append(np.full_like(imgs[0], 30))
    rows_img.append(np.hstack(chunk))
montage = np.vstack(rows_img)
cv2.imwrite(os.path.join(OUT, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("montage:", montage.shape)
print("DONE")
