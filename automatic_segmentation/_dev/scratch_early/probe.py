import os, glob, cv2, numpy as np
D = os.path.join(os.path.dirname(__file__), "..", "fwdfoto")

def probe(name):
    im = cv2.imread(os.path.join(D, name))           # BGR
    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # background = top strip
    top = im[0:120, :, :].reshape(-1, 3).mean(0)     # BGR mean
    # "golden plate" candidate: yellowish hue, decent saturation, bright
    gold = ((H > 12) & (H < 38) & (S > 60) & (V > 90)).astype(np.uint8)
    gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cols_gold = gold.sum(0)
    rows_gold = gold.sum(1)
    plate_top = int(np.argmax(rows_gold > 0.3 * w)) if (rows_gold > 0.3 * w).any() else -1
    gold_frac = gold.mean()
    # brightness stats
    return dict(name=name, bg_BGR=[int(x) for x in top],
                gold_frac=round(float(gold_frac), 3),
                plate_top_row=plate_top,
                Vmean=int(V.mean()), Smean=int(S.mean()))

picks = ["h2o-teflon.jpg","h2o-metall.jpg","h2o-vetro.jpg","oct-lotus.jpg",
         "i2-plexig.jpg","oct-print-5.jpg","i2-fuoc.jpg","h2o-tessuto.jpg"]
for p in picks:
    print(probe(p))
