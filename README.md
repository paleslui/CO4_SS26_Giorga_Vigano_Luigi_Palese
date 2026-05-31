# Automatic Segmentation of Sessile Droplets — CO4 / MSLS (ZHAW, FS26)

**Authors:** Luigi Palese (automatic segmentation + evaluation), Giorgia Viganò (preprocessing + manual segmentation)
**Course:** Imaging for the Life Sciences (CO4), Master of Science in Life Sciences

This repository segments back-lit, side-view photographs of **sessile liquid droplets**
(water, iodine solution, octane) resting on ~13 different surfaces, so that the droplet
outline can later be used to measure **contact angles**. It documents four segmentation
methods of increasing capability, an honest comparison between them, and an evaluation
framework (Dice / IoU) against manual ground-truth masks.

> **If you are reading this to reproduce the work:** start at [Setup](#3-setup) then
> [Reproduce the results](#4-reproduce-the-results). If you only want to understand *what
> we did and why*, read [The methods](#6-the-four-methods) and [What we tried and why
> it failed](#7-what-we-tried-the-honest-history).

---

## 1. The problem

A droplet sitting on a surface is photographed from the side against a bright back-light.
The angle the droplet edge makes with the surface (the **contact angle**) tells you how
"wettable" the surface is. To measure it automatically you first need a clean **mask** of
the droplet (droplet = white, everything else = black).

This is harder than it sounds, for three reasons the data makes obvious:

1. **The droplets are usually transparent.** A water droplet is a clear lens: its
   *interior* is as bright as the background behind it. Only a thin **rim** (where the
   curved surface bends the light away) is darker. So "the droplet is the dark blob" —
   the assumption most simple methods make — is simply false here.
2. **Specular reflections.** The shiny gold/metal support plate and the droplet surface
   throw bright highlights that look like edges.
3. **Huge variability.** 3 liquids × ~13 surfaces. The same algorithm must cope with a
   near-spherical droplet on Teflon, a flat film on glass, and a barely-visible smear on
   fabric, with no per-image tuning.

The course instructor flagged this directly: *"depending on the droplet shape and the
specular reflections, it may be quite challenging,"* and suggested a four-step approach —
(1) background removal via region growing, (2) segment the support plate, (3) the droplet
is whatever is enclosed by background and plate, (4) refine. Our classical methods follow
exactly that recipe.

---

## 2. The dataset

* **39 images** in `fwdfoto/`, each 1600×1200 JPEG (plus a `test.jpg` we ignore).
* Naming: `<liquid>-<surface>[-<volume>].jpg`, e.g. `h2o-teflon.jpg`, `oct-rain-100.jpg`.
  * liquids: `h2o` (water), `i2` (iodine solution), `oct` (octane)
  * surfaces: lotus, fog, fuoc, metall, plexig, print, rain, teflon, tessuto, vetro, …
* A few images contain **no discernible droplet** (the liquid spread into an invisible
  film). For those, the correct output is an **empty mask**, not a guess.

Manual ground-truth masks (Giorgia's part) live in `automatic_segmentation/results/masks/manual/`. **See the
important note in [§9](#9-manual-masks--ground-truth) — the current exports are not yet in
a usable binary form.**

---

## 3. Setup

Requires Python 3.11+ (we used 3.14) and ~1.5 GB free disk (mostly PyTorch + the SAM model).

```bash
# 1. create and activate a virtual environment
python -m venv venv-co4
source venv-co4/bin/activate        # Windows: venv-co4\Scripts\activate

# 2. install dependencies
pip install -r automatic_segmentation/requirements.txt

# 3. download the SAM model checkpoint (~375 MB) used by Methods 3 & 4
mkdir -p ~/Library/Caches/sam_models
curl -L -o ~/Library/Caches/sam_models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The checkpoint path is set once at the top of `automatic_segmentation/methods/automask_segment.py`
and `sam_segment.py` (the `_CKPT` variable). Change it there if you store the weights
elsewhere. The weights are **not** committed to git (they are far over GitHub's 100 MB
limit — see `.gitignore`).

---

## 4. Reproduce the results

All commands are run from the repository root with the venv active.

```bash
# Run any single method over all 39 images.
# Saves one mask per image to results/masks/<method>/ and a grid to results/overlays/.
python automatic_segmentation/reproduce/run_method.py --method basic
python automatic_segmentation/reproduce/run_method.py --method watershed
python automatic_segmentation/reproduce/run_method.py --method sam
python automatic_segmentation/reproduce/run_method.py --method automask   # best; CPU-only, ~13 min

# Build the side-by-side comparison grid + per-method stats from the saved masks.
python automatic_segmentation/reproduce/run_compare.py

# Score a method against binary manual masks (once they exist — see §9).
python automatic_segmentation/reproduce/evaluate.py --method automask \
    --manual "FOTO CO4 MASKS MANUAL"
```

**Speed note:** Methods 1–2 are instant. Method 3 (SAM, point-prompted) runs on Apple-MPS
in ~1.3 s/image. Method 4 (SAM automatic mask generation) only runs on **CPU** here — the
automatic generator uses float64, which Apple-MPS does not support — so it takes
~20 s/image (~13 min for the full set). That is acceptable for a one-off batch; the masks
are saved so nothing needs re-running.

---

## 5. Repository structure

```
.
├── README.md                     ← this file
├── fwdfoto/                      ← 39 source images (input)
├── automatic_segmentation/results/masks/manual/        ← Giorgia's manual masks (see §9)
├── automatic_segmentation/
│   ├── requirements.txt
│   ├── methods/                  ← the segmentation library
│   │   ├── preprocessing.py      ← Giorgia's preprocessing pipeline
│   │   ├── basic_segment.py      ← Method 1: classical (per-column sky subtraction)
│   │   ├── watershed_segment.py  ← Method 2: classical refinement (watershed)
│   │   ├── sam_segment.py        ← Method 3: SAM with an automatic point prompt
│   │   └── automask_segment.py   ← Method 4: SAM automatic masks + shape selection ★ BEST
│   ├── reproduce/                ← scripts to regenerate everything
│   │   ├── run_method.py
│   │   ├── run_compare.py
│   │   └── evaluate.py
│   ├── results/
│   │   ├── masks/<method>/       ← 39 output masks per method (.png, 0/255)
│   │   ├── overlays/             ← per-method grids + comparison.jpg
│   │   └── compare_stats.json    ← ok-rate & mean solidity per method
│   └── _dev/                     ← full experimental history (v1–v7, tests) — archived, not needed
├── Rapport GV EM - Consegnato.pdf   ← previous (manual) contact-angle project, for reference
└── angolo di contatto.xlsx          ← previous contact-angle measurements, for reference
```

---

## 6. The four methods

All methods take a BGR image and return a full-size `uint8` mask (0 = background,
255 = droplet). The progression goes from "pure classical CV" to "foundation model."

### Method 1 — `basic_segment.py` (classical)

Pure OpenCV, follows the instructor's four-step recipe. Key idea: **per-column background
subtraction.** For each column of the image, the very top is clean back-light ("sky"); we
subtract that column's own sky value from every pixel below it. A droplet then shows up as
a *deviation* from the sky **whether it is darker or brighter** — which is the trick that
handles transparent droplets. Steps:

1. Grayscale + median blur.
2. Detect the **baseline** (top edge of the plate) as the strongest long horizontal
   gradient in the lower image.
3. Per-column sky subtraction → a "deviation" map.
4. Otsu-threshold the deviation, restrict to above the baseline, close gaps, fill the
   enclosed interior.
5. Keep the largest blob touching the baseline; reject blobs too flat to be a droplet.

**Verdict:** works on clear-rimmed droplets; fails where the droplet is flat and looks
like the plate, and is highly sensitive to the baseline being correct.

### Method 2 — `watershed_segment.py` (classical refinement)

Takes Method 1's mask, erodes it to a "sure droplet" seed and dilates it to a "sure
background" seed, then runs **watershed** to snap the boundary to the real image edge.
**Verdict:** sharper outlines (mean solidity 0.81 → 0.83) but it is a *refiner, not a
re-locator* — if Method 1 picked the wrong region, watershed just sharpens the wrong
region.

### Method 3 — `sam_segment.py` (SAM, prompted)

Uses Meta's **Segment Anything Model**. We give SAM one positive point prompt placed at
"the column with the most non-sky deviation just above the baseline," plus negative
prompts in the background corners, and pick the best returned mask. **Verdict:** much
cleaner masks (solidity 0.88) and it cracks cases classical CV never could (e.g. the
bright Teflon dome), but the single point prompt is brittle: on a transparent droplet the
bright *plate edge* deviates more than the clear droplet, so the point sometimes lands on
the plate.

### Method 4 — `automask_segment.py` (SAM automatic + shape selection) ★ recommended

The breakthrough. Instead of guessing one prompt point, we let SAM's **automatic mask
generator** propose *every* object in the image with no prompt, then **select the droplet
by shape**:

* reject masks that touch the top edge → that's the background,
* reject masks wider than 75 % of the frame → that's the plate/background band,
* reject masks that aren't compact (solidity > 0.5) → not a droplet,
* keep a plausible size (0.3 %–25 % of the frame),
* use a rough baseline only as a soft tie-breaker.

**Verdict:** the cleanest masks by a wide margin (solidity 0.96), and — crucially — it no
longer depends on a precise baseline, which was the single biggest source of failure (see
§7). This is the recommended method.

---

## 7. What we tried — the honest history

This section exists so the work can be understood, not just rerun. The short version:
**the bottleneck was never the segmentation model — it was a brittle hand-rolled baseline
detector feeding every method bad constraints.**

* **First classical attempts** assumed "droplet = darker than background." This works only
  for the rare opaque droplet (e.g. iodine on soot) and fails on every transparent one.
* **Adding the friend's preprocessing** (background subtraction + polarity normalisation +
  cropping below the plate) helped a lot and produced the per-column-deviation idea, but
  the polarity heuristic occasionally voted the wrong way, and flat droplets still got
  confused with the plate's specular line.
* **An "ensemble" router** (run several methods, keep the best by a quality score) was
  built — and then abandoned. The quality score was a pile of hand-picked thresholds that
  overfit to the failures we had already seen; it is *not* a principled metric, and a
  compact wrong blob could out-score a slightly ragged correct one. Lesson: selecting the
  least-bad of several bad outputs does not fix bad outputs.
* **The real diagnosis:** clearly-visible droplets (h2o-metall, i2-rain-100, …) were being
  *missed entirely*. We traced it to the baseline detector landing ~250 px too low (on the
  plate's bottom rim instead of the contact line), which made every method discard the
  correct droplet for "not sitting on the baseline." Three different baseline detectors all
  topped out around ~55 % correct — gradient-based baseline detection is fundamentally
  unreliable on shiny plates.
* **The fix was to stop depending on the baseline.** Method 4 selects the droplet by shape
  among SAM's proposals and uses the baseline only as a soft hint. That single change fixed
  the bulk of the dataset at once.

---

## 8. Results

Over all 39 images (from `results/compare_stats.json`):

| Method     | segmented (ok) | mean solidity of masks | notes |
|------------|:--------------:|:----------------------:|-------|
| basic      | 35 / 39        | 0.81 | many "ok" masks are actually wrong (a line on the plate) |
| watershed  | 35 / 39        | 0.83 | same locations as basic, sharper edges |
| sam        | 34 / 39        | 0.88 | cleaner; brittle point prompt |
| **automask** | **35 / 39**  | **0.96** | cleanest by far; baseline-independent |

"ok" only means a non-empty mask was produced; **solidity** (mask area ÷ convex-hull area)
is the better quality proxy because it penalises ragged/plate-smeared masks. The steady
climb in solidity is the real story. Method 4's 4 "empties" (h2o-fog-100, h2o-fog-5,
oct-plexig, oct-vetro) are images with no clearly discernible droplet — empty is the
honest answer there. See `results/overlays/comparison.jpg` for the visual side-by-side.

---

## 9. Manual masks / ground truth

`evaluate.py` computes **Dice** and **IoU** of an automatic method against manual masks:

```
Dice(A,B) = 2|A ∩ B| / (|A| + |B|)        IoU(A,B) = |A ∩ B| / |A ∪ B|   (IoU ≤ Dice)
```

**Important:** the current files in `automatic_segmentation/results/masks/manual/` are **not usable** as ground
truth — they were exported as grayscale/photo-like images (continuous values 0–255), not
binary masks. A real mask must be **pure black/white** (droplet = 255, background = 0). To
re-export correctly in Fiji/ImageJ: trace the droplet → `Edit ▸ Selection ▸ Create Mask` →
`File ▸ Save As ▸ PNG`, same base name + `_mask`, no crop/resize so it stays 1600×1200.
Trace flat along the contact line at the bottom (to match where the automatic methods cut).
Once those exist, `evaluate.py` produces the Dice/IoU table automatically.

---

## 10. Limitations & honest notes

* **Classical methods (1–2) do not generalise** across surface types — exactly the
  limitation the lecture (Block 6-04) lists for thresholding/region-growing.
* **Method 4 needs CPU** here (~20 s/image) because Apple-MPS lacks float64 for SAM's
  automatic generator. On a CUDA GPU it would be far faster.
* **A few images genuinely have no droplet** to find; all methods (correctly) return empty.
* **The baseline detector is the weak classical component** and is kept only for Methods
  1–3 and as a soft hint in Method 4.

---

## 11. Key takeaway

The most transferable lesson from this project: *before blaming the model, check what
you're feeding it.* Days were spent tuning thresholds and quality scores, when the actual
fault was an upstream baseline detector silently throwing away correct results. Removing
that dependency — by letting a foundation model propose objects and selecting by shape —
fixed most of the dataset in a single change.
