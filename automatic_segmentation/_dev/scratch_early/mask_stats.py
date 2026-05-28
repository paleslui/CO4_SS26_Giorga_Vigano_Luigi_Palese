import os, cv2 as cv, numpy as np

ROOT = "/Users/Luigi/Library/CloudStorage/OneDrive-Persönlich/ZHAW/Master/2. Semester/CO4/CO4_SS26_Giorga_Vigano_Luigi_Palese"
MASKS = os.path.join(ROOT, "FOTO CO4 MASKS MANUAL")
SRC = os.path.join(ROOT, "fwdfoto")

print("=== fwdfoto files containing 'lotus' ===")
for f in sorted(os.listdir(SRC)):
    if "lotus" in f.lower(): print("  ", f)

print("\n=== mask stats ===")
for fn in sorted(os.listdir(MASKS)):
    if fn.startswith("."): continue
    m = cv.imread(os.path.join(MASKS, fn), cv.IMREAD_UNCHANGED)
    if m.ndim == 3: m = m[..., 0]
    tot = m.size
    f0 = (m == 0).mean()
    fmax = (m == m.max()).mean()
    fhi = (m > 127).mean()
    print(f"{fn:26s} min={m.min():3d} max={m.max():3d} mean={m.mean():6.1f} "
          f"nuniq={len(np.unique(m)):3d} | frac==0:{f0:.3f} frac==max:{fmax:.3f} frac>127:{fhi:.3f}")
