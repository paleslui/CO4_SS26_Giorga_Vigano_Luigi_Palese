import os, cv2 as cv, numpy as np

ROOT = "/Users/Luigi/Library/CloudStorage/OneDrive-Persönlich/ZHAW/Master/2. Semester/CO4/CO4_SS26_Giorga_Vigano_Luigi_Palese"
MASKS = os.path.join(ROOT, "FOTO CO4 MASKS MANUAL")
SRC = os.path.join(ROOT, "fwdfoto")

def norm(fn):
    b = os.path.splitext(fn.lower().replace("_masks","").replace("_mask",""))[0]
    return b  # keep h20 as-is; source uses h20-lotus

def bbox_frac(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0: return None
    H, W = binary.shape
    return (round(xs.min()/W,2), round(ys.min()/H,2), round(xs.max()/W,2),
            round(ys.max()/H,2), round(binary.mean(),3))

for fn in sorted(os.listdir(MASKS)):
    if fn.startswith("."): continue
    m = cv.imread(os.path.join(MASKS, fn), cv.IMREAD_UNCHANGED)
    if m.ndim == 3: m = m[...,0]
    base = norm(fn)
    src = os.path.join(SRC, base + ".jpg")
    line = f"{fn:24s}"
    if os.path.exists(src):
        g = cv.cvtColor(cv.imread(src), cv.COLOR_BGR2GRAY)
        diff = float(np.abs(g.astype(int) - m.astype(int)).mean())
        corr = float(np.corrcoef(g.ravel(), m.ravel())[0,1])
        line += f" vs-src: meanabsdiff={diff:6.1f} corr={corr:+.3f}"
    # where is the bright region (candidate droplet)?
    line += f"  bright(>200)bbox={bbox_frac(m>200)}"
    print(line)
