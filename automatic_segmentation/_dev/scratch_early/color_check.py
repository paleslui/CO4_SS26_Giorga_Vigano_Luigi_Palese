import os, cv2 as cv, numpy as np

ROOT = "/Users/Luigi/Library/CloudStorage/OneDrive-Persönlich/ZHAW/Master/2. Semester/CO4/CO4_SS26_Giorga_Vigano_Luigi_Palese"
MASKS = os.path.join(ROOT, "FOTO CO4 MASKS MANUAL")

for fn in sorted(os.listdir(MASKS)):
    if fn.startswith("."): continue
    m = cv.imread(os.path.join(MASKS, fn), cv.IMREAD_UNCHANGED)
    info = f"{fn:24s} shape={str(m.shape):18s}"
    if m.ndim == 3 and m.shape[2] >= 3:
        b, g, r = m[...,0].astype(int), m[...,1].astype(int), m[...,2].astype(int)
        # colored = channels differ a lot
        spread = np.abs(r-g) + np.abs(g-b) + np.abs(r-b)
        colored = spread > 60
        info += f" colored_px_frac={colored.mean():.4f}"
        if colored.any():
            ys, xs = np.where(colored)
            H, W = colored.shape
            info += f" bbox=({xs.min()/W:.2f},{ys.min()/H:.2f},{xs.max()/W:.2f},{ys.max()/H:.2f})"
            # dominant hue of colored pixels
            hsv = cv.cvtColor(m[...,:3], cv.COLOR_BGR2HSV)
            info += f" medHue={int(np.median(hsv[...,0][colored]))}"
        if m.shape[2] == 4:
            a = m[...,3]
            info += f" alpha[min={a.min()},max={a.max()},frac<255={(a<255).mean():.3f}]"
    print(info)
