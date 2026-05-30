"""Replace cells §2 (Dataset) and §4 (Manual segmentation) with richer
content using Giorgia's notes. Idempotent: re-running overwrites cleanly."""
import sys

with open('build_notebook.py') as f:
    src = f.read()


# --- new §2 content ---
new_sec2 = '''cells.append(md(
"---",
"",
"## 2. Dataset",
"",
"**Origin and licence.** The images were acquired during a nanotechnology *Travaux Pratiques* (TP) at ZHAW Wädenswil [5]. The data are owned by us and are reused here with permission of all collaborators of the original work.",
"",
"**Acquisition.** Side-view (sessile-drop profile) macro photographs, taken with a digital camera mounted on an optical microscope.",
"",
"**Composition.** 40 JPEG images (1600 × 1200 px): one calibration blank (`test.jpg`, excluded from analysis) plus **3 liquids × 13 substrates = 39 measurement images**.",
"",
"**Liquids (3)**, chosen to span a wide range of surface tensions:",
"",
"| Liquid | Code | γ_lg [mN/m] | Polarity |",
"|---|---|---:|---|",
"| Water (H₂O)                | `h2o` | 72.8 | polar, high  |",
"| Diiodomethane (CH₂I₂)      | `i2`  | 50.8 | apolar, intermediate |",
"| Octanol (C₈H₁₇OH)          | `oct` | 21.6 | apolar, low  |",
"",
"Using three liquids with known surface tensions enables decomposition of the substrate's surface energy into dispersive (γᴰ) and polar (γᴾ) components via the Owens–Wendt method, and computation of the critical surface tension via the Zisman method [5]. A single liquid cannot resolve these components.",
"",
"**Substrates (13)**, categorised by water wettability:",
"",
"| Category | Substrates |",
"|---|---|",
"| Hydrophilic (θ_H₂O < 90°)  | bare glass · glass + antifog 5 mm/min · glass + antifog 100 mm/min · steel + anti-fingerprint 5 mm/min · bare steel · plexiglas |",
"| Hydrophobic (θ_H₂O > 90°)  | Lotus leaf · hydrophobic sol-gel coated fabric · glass + soot · glass + anti-rain 5 mm/min · glass + anti-rain 100 mm/min · steel + anti-fingerprint 100 mm/min · Teflon |",
"",
"(The `-5` / `-100` suffixes in image filenames refer to the dip-coating extraction speed in mm/min.)",
"",
"**Structure of interest.** The sessile droplet itself — specifically the **liquid–air interface** visible above the substrate surface. The droplet profile (its outer contour) encodes the contact angle θ, the fundamental physical quantity of the experiment. According to Young's equation, θ at the triple line (solid–liquid–gas) reflects the equilibrium between the three interfacial tensions γ_sg, γ_sl, γ_lg.",
"",
"**Special cases in the dataset:**",
"",
"- `i2-fuoc` — diiodomethane on soot exhibits complete wetting (θ ≈ 0°): no droplet is visible.",
"- `oct-metall` — contains two droplets in the same frame (our automatic pipeline currently picks one of them; per-droplet cropping is mentioned as further work in §9).",
"- Four octanol images on smooth or porous surfaces (`oct-plexig`, `oct-vetro`, `oct-tessuto`, `oct-teflon`) show severe spreading where the droplet becomes an almost invisible thin film.",
"",
"**Motivation for automatic segmentation.** In the original lab experiment [5], contact angles were measured manually in ImageJ by visually tracing tangent lines at the triple point. This introduces **inter-operator variability**: the perceived edge position depends on subjective judgement, producing slightly different measurements between operators. Automatic segmentation provides a reproducible, objective delineation of the liquid–air interface, removing human bias and enabling consistent contact-angle computation across the entire dataset. The reference contact-angle values used as ground truth in §7.2 come from this original lab work [5].",
"))
'''


# --- new §4 content ---
new_sec4 = '''cells.append(md(
"---",
"",
"## 4. Manual segmentation",
"",
"**Tool.** Fiji (ImageJ).",
"",
"**Structure segmented.** The sessile droplet — specifically the liquid–air interface above the substrate surface.",
"",
"**Annotation procedure.** The **Polygon Selection** tool was used to trace each droplet contour point by point. The curved upper profile was approximated with multiple anchor points around the dome, while the base was defined as a straight horizontal line at the substrate surface. Each selection was saved via the **ROI Manager** and exported as a PNG (foreground = 255, background = 0). Anti-aliasing at the polygon edges produced non-zero grey values in the background of the raw exports, so a threshold of 1 was applied programmatically to enforce strictly binary masks (all values forced to {0, 255}). The 10 final binary masks live in `FOTO CO4 MASKS MANUAL/manual_binary/`.",
"",
"**Time.** Several hours of focused work across the 10 droplets — most of the time was spent tracing the curved upper profile and revising anchor-point placement.",
"",
"**Difficulties encountered:**",
"",
"- **Drops with very low contact angle** (e.g. `oct-fog`) have an almost invisible edge, making precise manual delineation uncertain.",
"- **Drop transparency / reflectivity** — the boundary is not a sharp intensity step but a gradual transition, especially for water and octanol on smooth substrates.",
"- **Textured substrates** (`tessuto` = fabric, `fuoc` = soot deposit) have a surface texture that merges visually with the droplet base, blurring the contact line.",
"",
"These same factors limit *any* segmentation method on this dataset — manual or automatic — and they re-appear in the per-image evaluation in §7.",
"))
'''


def replace_md_cell(src, marker, new_content):
    """Replace the cells.append(md(...)) block whose body contains `marker`."""
    pos = src.find(marker)
    if pos == -1:
        raise ValueError(f"marker not found: {marker[:80]}")
    block_start = src.rfind("cells.append(md(", 0, pos)
    if block_start == -1:
        raise ValueError("no cells.append(md( opener before marker")
    block_end = src.find("\n))\n", pos)
    if block_end == -1:
        raise ValueError("no closing )) after marker")
    block_end += len("\n))\n")
    return src[:block_start] + new_content + src[block_end:]


src = replace_md_cell(src, '"## 2. Dataset",',            new_sec2)
print("OK: §2 Dataset cell replaced.")
src = replace_md_cell(src, '"## 4. Manual segmentation",', new_sec4)
print("OK: §4 Manual segmentation cell replaced.")

with open('build_notebook.py', 'w') as f:
    f.write(src)
print("\nbuild_notebook.py patched.")
