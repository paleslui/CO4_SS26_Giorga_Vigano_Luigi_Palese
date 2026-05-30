"""
Compute droplet contact angles from binary segmentation masks.

Method: fit a circle to the UPPER part of the droplet contour using GEOMETRIC
least squares (Gauss-Newton on Euclidean residuals, initialised from Kasa
algebraic fit). The contact angle is then read off the circle:
        cos(theta) = (cy - baseline) / R    -- "through the liquid"
The baseline is the bottom row of the mask. We discard the bottom 6 pixels of
the contour to avoid the rasterised flat edge dragging the fit.

Runs over the 10 manual + 10 automask masks and compares against the lab
measurements in `angolo di contatto.xlsx`.

    python automatic_segmentation/reproduce/compute_contact_angles.py
"""
import os, json, cv2, numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.abspath(os.path.join(HERE, "..", ".."))
MAN    = os.path.join(REPO, "FOTO CO4 MASKS MANUAL", "manual_binary")
AUTO   = os.path.join(REPO, "automatic_segmentation", "results", "masks", "automask")
OUTJSON = os.path.join(REPO, "automatic_segmentation", "results", "contact_angles.json")

# real lab values from `angolo di contatto.xlsx` (Theta_mean per substrate x liquid)
REAL = {
    "h20-lotus":    146.7, "i2-lotus":     120.6,
    "h2o-metall":    79.6, "i2-metall":     51.3,
    "h2o-rain-100":  89.8, "i2-teflon":     74.5,
    "oct-teflon":    37.5, "oct-fog-100":   14.7,
    "oct-print-5":   48.4, "oct-rain-5":    29.1,
}

def kasa(xs, ys):
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]
    b = xs**2 + ys**2
    sol,*_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    return float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2))

def geom_fit(xs, ys, cx, cy, R, n_iter=50, tol=1e-6):
    """Gauss-Newton geometric circle fit (minimises sum of (Euclidean dist - R)^2)."""
    for _ in range(n_iter):
        dx, dy = xs - cx, ys - cy
        d  = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-10)
        J  = np.column_stack([-dx/d, -dy/d, -np.ones_like(d)])
        delta,*_ = np.linalg.lstsq(J, -(d - R), rcond=None)
        cx += delta[0]; cy += delta[1]; R += delta[2]
        if np.linalg.norm(delta) < tol: break
    return cx, cy, R

def contact_angle(mask, bottom_skip=6):
    if mask is None or mask.sum() == 0:
        return None
    ys_full = np.where(mask > 0)[0]
    base    = int(ys_full.max())
    cnts,_  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    upper   = contour[contour[:, 1] < base - bottom_skip]
    if len(upper) < 20: return None
    cx0, cy0, R0 = kasa(upper[:,0], upper[:,1])
    cx,  cy,  R  = geom_fit(upper[:,0], upper[:,1], cx0, cy0, R0)
    dy_bc = base - cy
    if R <= 0 or abs(dy_bc) > R: return None
    theta = float(np.degrees(np.arccos(np.clip(-dy_bc/R, -1, 1))))
    return dict(theta=theta, cx=cx, cy=cy, R=R, baseline=base)

# --- compute on all 10 ---
rows = []
for nm in sorted(REAL):
    m_man  = cv2.imread(os.path.join(MAN,  nm + ".png"), cv2.IMREAD_GRAYSCALE)
    m_auto = cv2.imread(os.path.join(AUTO, nm + ".png"), cv2.IMREAD_GRAYSCALE)
    r_man  = contact_angle(m_man);  theta_man  = r_man ["theta"] if r_man  else None
    r_auto = contact_angle(m_auto); theta_auto = r_auto["theta"] if r_auto else None
    rows.append((nm, REAL[nm], theta_man, theta_auto))

print(f"{'image':16s} {'REAL':>7s} {'manual':>8s} {'Δman':>6s}  {'auto':>7s} {'Δauto':>6s}")
print("-"*60)
dm, da = [], []
for nm, R, m, a in rows:
    em = (m - R) if m is not None else None
    ea = (a - R) if a is not None else None
    if em is not None: dm.append(em)
    if ea is not None: da.append(ea)
    fmt = lambda v, w: (f"{v:{w}.1f}" if v is not None else "  n/a".rjust(w))
    fmt_d = lambda v: (f"{v:+6.1f}" if v is not None else "  n/a")
    print(f"{nm:16s} {R:7.1f} {fmt(m,8)} {fmt_d(em)}  {fmt(a,7)} {fmt_d(ea)}")
dm, da = np.array(dm), np.array(da)
print("-"*60)
print(f"manual  vs real  (N={len(dm)}):  mean {dm.mean():+.1f}, std {dm.std():.1f}, MAE {np.abs(dm).mean():.1f} deg")
print(f"automask vs real (N={len(da)}):  mean {da.mean():+.1f}, std {da.std():.1f}, MAE {np.abs(da).mean():.1f} deg")

out = [{"image": nm, "real_deg": R,
        "manual_deg":  None if m is None else round(m, 2),
        "automask_deg":None if a is None else round(a, 2),
        "err_manual":  None if m is None else round(m - R, 2),
        "err_automask":None if a is None else round(a - R, 2)} for nm, R, m, a in rows]
json.dump(out, open(OUTJSON, "w"), indent=2)
print(f"\nsaved {OUTJSON}")
