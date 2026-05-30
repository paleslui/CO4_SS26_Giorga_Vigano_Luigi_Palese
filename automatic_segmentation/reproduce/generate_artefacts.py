"""
Generate all visualization and intermediate artefacts for the CO4 droplet
segmentation project (preprocessed images, per-step pipeline panels,
per-method overlays, contact-angle 3-panel figures).

Idempotent: skips any output file that already exists with size > 1 KB.
Uses cached segmentation masks under results/masks/{method}/; never re-runs SAM.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "automatic_segmentation"))

from methods.preprocessing import (
    crop_timestamp,
    extract_best_channel,
    detect_substrate_line,
    subtract_background,
    apply_clahe,
    denoise,
    preprocess_image,
)
from reproduce.compute_contact_angles import contact_angle


FWD_DIR     = REPO / "fwdfoto"
MANUAL_DIR  = REPO / "FOTO CO4 MASKS MANUAL" / "manual_binary"
MASKS_DIR   = REPO / "automatic_segmentation" / "results" / "masks"
PREP_DIR    = REPO / "data" / "preprocessed"
STEPS_DIR   = PREP_DIR / "pipeline_steps"
OVERLAYS    = REPO / "automatic_segmentation" / "results" / "overlays"
CA_DIR      = OVERLAYS / "contact_angle"

METHODS = ["basic", "watershed", "sam", "automask"]


def exists_nonempty(p: Path, min_bytes: int = 1024) -> bool:
    return p.exists() and p.stat().st_size > min_bytes


def list_source_images():
    """All 39 source JPGs (excluding test.jpg), sorted by basename."""
    files = sorted(FWD_DIR.glob("*.jpg"))
    return [f for f in files if f.stem != "test"]


# --------------------------------------------------------------------------
# Task 1 — Preprocessed images
# --------------------------------------------------------------------------

def task_preprocessed():
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] Preprocessed images  →  {PREP_DIR}")
    print("-" * 70)
    for src in list_source_images():
        out = PREP_DIR / (src.stem + ".png")
        if exists_nonempty(out):
            print(f"  {src.stem:<22} skip (cached)")
            continue
        result = preprocess_image(str(src))
        cv2.imwrite(str(out), result["preprocessed"])
        tag = "BLANK" if result["is_blank"] else f"substrate_y={result['substrate_y']}"
        print(f"  {src.stem:<22} {tag}")


# --------------------------------------------------------------------------
# Task 2 — Per-step preprocessing visualisations
# --------------------------------------------------------------------------

PANEL_W = 400  # target panel width in pixels

def _panel(gray: np.ndarray, title: str, sub_y: int, scale: float) -> np.ndarray:
    """One panel: scaled BGR image + red dashed substrate line + title bar."""
    h, w = gray.shape
    new_w = PANEL_W
    new_h = int(round(h * (new_w / w)))
    img = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # red dashed substrate line
    sy = int(round(sub_y * (new_h / h)))
    dash, gap = 12, 8
    x = 0
    while x < new_w:
        x2 = min(x + dash, new_w)
        cv2.line(bgr, (x, sy), (x2, sy), (0, 0, 255), 1, cv2.LINE_AA)
        x += dash + gap

    # title strip on top
    strip_h = 32
    strip = np.zeros((strip_h, new_w, 3), dtype=np.uint8)
    cv2.putText(strip, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, bgr])


def task_pipeline_steps():
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] Pipeline-step panels  →  {STEPS_DIR}")
    print("-" * 70)
    for stem in ["h2o-rain-100", "oct-fog-5", "i2-metall"]:
        out = STEPS_DIR / f"{stem}_steps.jpg"
        if exists_nonempty(out):
            print(f"  {stem:<22} skip (cached)")
            continue
        src = FWD_DIR / f"{stem}.jpg"
        img = cv2.imread(str(src))
        cropped = crop_timestamp(img)
        gray = extract_best_channel(cropped)
        sub_y = detect_substrate_line(gray)
        bg_sub = subtract_background(gray)
        clahe = apply_clahe(bg_sub)
        final = denoise(clahe)

        steps = [
            (gray,   "(a) Best channel (grayscale)"),
            (bg_sub, "(b) Background subtraction"),
            (clahe,  "(c) CLAHE"),
            (final,  "(d) Bilateral filter"),
        ]
        panels = [_panel(g, t, sub_y, 1.0) for g, t in steps]
        canvas = np.hstack(panels)
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {stem:<22} written ({canvas.shape[1]}x{canvas.shape[0]})")


# --------------------------------------------------------------------------
# Task 3 — Per-image overlays for 4 methods
# --------------------------------------------------------------------------

def task_overlays():
    print(f"\n[3/4] Per-image overlays  →  {OVERLAYS}/<method>/")
    print("-" * 70)
    for method in METHODS:
        out_dir = OVERLAYS / method
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = MASKS_DIR / method
        written = skipped = empty = 0
        for src in list_source_images():
            out = out_dir / f"{src.stem}.jpg"
            if exists_nonempty(out):
                skipped += 1
                continue
            photo = cv2.imread(str(src))
            mask_path = mask_dir / f"{src.stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

            if mask is not None and mask.sum() > 0:
                _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(photo, cnts, -1, (0, 0, 255), 4)
            else:
                empty += 1
                cv2.putText(photo, f"{method}: empty", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imwrite(str(out), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
            written += 1
        print(f"  {method:<10} written={written:<3} skipped={skipped:<3} empty_masks={empty}")


# --------------------------------------------------------------------------
# Task 4 — Contact-angle 3-panel visualisations
# --------------------------------------------------------------------------

CA_PANEL_W = 520
LABEL_H    = 40

def _label_strip(text: str, w: int) -> np.ndarray:
    strip = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    return strip


def _scale_to_panel(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = CA_PANEL_W
    new_h = int(round(h * (new_w / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _panel_1_segmentation(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = photo.copy()
    if mask is not None and mask.sum() > 0:
        _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (0, 0, 255), 4)
    return _scale_to_panel(img)


def _panel_2_contact_angle(photo: np.ndarray, mask: np.ndarray) -> tuple:
    """Return (panel_image, theta) where theta may be None."""
    img = photo.copy()
    theta = None
    if mask is not None and mask.sum() > 0:
        _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # droplet contour in orange
        cv2.drawContours(img, cnts, -1, (0, 165, 255), 3)

        res = contact_angle(binm)
        if res is not None:
            theta = res["theta"]
            cx, cy, R, base = int(round(res["cx"])), int(round(res["cy"])), int(round(res["R"])), int(res["baseline"])
            h, w = img.shape[:2]
            # fitted circle (blue)
            cv2.circle(img, (cx, cy), R, (255, 0, 0), 3, cv2.LINE_AA)
            # baseline (green)
            cv2.line(img, (0, base), (w - 1, base), (0, 255, 0), 3, cv2.LINE_AA)
            # contact points: intersection of circle with baseline
            dy = base - cy
            if abs(dy) <= R:
                dx = int(round((R * R - dy * dy) ** 0.5))
                cv2.circle(img, (cx - dx, base), 14, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx + dx, base), 14, (0, 0, 255), -1, cv2.LINE_AA)
            # top-left "theta = X deg" label in green on black box
            txt = f"theta = {theta:.1f} deg"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 16, 10 + th + 16), (0, 0, 0), -1)
            cv2.putText(img, txt, (18, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return _scale_to_panel(img), theta


def _panel_3_mask(photo_shape: tuple, mask: np.ndarray) -> np.ndarray:
    h, w = photo_shape[:2]
    if mask is None:
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return _scale_to_panel(bgr)


def task_contact_angle():
    CA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] Contact-angle 3-panel figures  →  {CA_DIR}")
    print("-" * 70)
    manual_files = sorted(MANUAL_DIR.glob("*.png"))
    for mf in manual_files:
        stem = mf.stem
        out = CA_DIR / f"{stem}_combined.jpg"
        if exists_nonempty(out):
            print(f"  {stem:<22} skip (cached)")
            continue

        photo = cv2.imread(str(FWD_DIR / f"{stem}.jpg"))
        automask = cv2.imread(str(MASKS_DIR / "automask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        manual = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)

        p1 = _panel_1_segmentation(photo, automask)
        p2, theta = _panel_2_contact_angle(photo, automask)
        p3 = _panel_3_mask(photo.shape, manual)

        # Force all panels to identical height (defensive)
        h = min(p1.shape[0], p2.shape[0], p3.shape[0])
        p1, p2, p3 = p1[:h], p2[:h], p3[:h]

        theta_txt = f"{theta:.0f}" if theta is not None else "n/a"
        labels = [
            _label_strip("1. Segmentation (automask)", p1.shape[1]),
            _label_strip(f"2. Contact angle (circle fit) -- theta={theta_txt} deg", p2.shape[1]),
            _label_strip("3. Binary mask (ground truth)", p3.shape[1]),
        ]
        col1 = np.vstack([labels[0], p1])
        col2 = np.vstack([labels[1], p2])
        col3 = np.vstack([labels[2], p3])
        canvas = np.hstack([col1, col2, col3])
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {stem:<22} theta={theta_txt}  ({canvas.shape[1]}x{canvas.shape[0]})")


# --------------------------------------------------------------------------
# Acceptance check
# --------------------------------------------------------------------------

def acceptance():
    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)
    stems = [s.stem for s in list_source_images()]
    manual_stems = sorted(p.stem for p in MANUAL_DIR.glob("*.png"))

    rows = [
        ("data/preprocessed/",                                      39,  PREP_DIR,    "*.png", stems),
        ("data/preprocessed/pipeline_steps/",                       3,   STEPS_DIR,   "*.jpg", ["h2o-rain-100_steps", "oct-fog-5_steps", "i2-metall_steps"]),
        ("automatic_segmentation/results/overlays/basic/",          39,  OVERLAYS / "basic",     "*.jpg", stems),
        ("automatic_segmentation/results/overlays/watershed/",      39,  OVERLAYS / "watershed", "*.jpg", stems),
        ("automatic_segmentation/results/overlays/sam/",            39,  OVERLAYS / "sam",       "*.jpg", stems),
        ("automatic_segmentation/results/overlays/automask/",       39,  OVERLAYS / "automask",  "*.jpg", stems),
        ("automatic_segmentation/results/overlays/contact_angle/",  10,  CA_DIR,      "*.jpg", [s + "_combined" for s in manual_stems]),
    ]

    header = f"| {'Directory':<55} | {'Expected':>8} | {'Found':>5} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    missing_report = []
    for label, expected, dirpath, glob_pat, expected_stems in rows:
        found_files = sorted(dirpath.glob(glob_pat)) if dirpath.exists() else []
        found_stems = {f.stem for f in found_files}
        n = len(found_files)
        print(f"| {label:<55} | {expected:>8} | {n:>5} |")
        if n != expected:
            missing = sorted(set(expected_stems) - found_stems)
            if missing:
                missing_report.append((label, missing))

    if missing_report:
        print("\nMISSING FILES:")
        for label, missing in missing_report:
            print(f"  {label}")
            for m in missing:
                print(f"    - {m}")
    else:
        print("\nAll expected files present.")


def main():
    task_preprocessed()
    task_pipeline_steps()
    task_overlays()
    task_contact_angle()
    acceptance()


if __name__ == "__main__":
    main()
