import os, glob, cv2 as cv, numpy as np

ROOT = "/Users/Luigi/Library/CloudStorage/OneDrive-Persönlich/ZHAW/Master/2. Semester/CO4/CO4_SS26_Giorga_Vigano_Luigi_Palese"
MASKS = os.path.join(ROOT, "FOTO CO4 MASKS MANUAL")
SRC = os.path.join(ROOT, "fwdfoto")

def norm(fn):
    b = fn.lower().replace("_masks", "").replace("_mask", "")
    b = os.path.splitext(b)[0]
    b = b.replace("h20", "h2o")
    return b

for fn in sorted(os.listdir(MASKS)):
    if fn.startswith("."): continue
    p = os.path.join(MASKS, fn)
    m = cv.imread(p, cv.IMREAD_UNCHANGED)
    base = norm(fn)
    src = os.path.join(SRC, base + ".jpg")
    vals, counts = np.unique(m, return_counts=True)
    if m.ndim == 3:
        uniq = "3ch shape=%s" % (m.shape,)
    else:
        # show up to a few unique values
        uniq = list(zip(vals.tolist(), counts.tolist()))[:6]
    print(f"{fn:28s} -> {base+'.jpg':18s} exists={os.path.exists(src)} "
          f"dtype={m.dtype} shape={m.shape}")
    print(f"     uniq/frac: {uniq}")
