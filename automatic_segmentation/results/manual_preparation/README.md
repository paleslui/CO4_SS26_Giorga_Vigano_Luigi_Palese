# manual_preparation/

Behind-the-scenes intermediate files from preparing the manual ground-truth masks.
These are **not part of the final deliverable**, but kept for transparency about
how the binary masks in `../masks/manual/` were obtained.

## Workflow

1. Polygon contours were traced in Fiji (ImageJ) using the **Polygon Selection** tool
   on each of the 10 chosen droplet images. Each selection was saved as a PNG (or, in
   one case, a TIFF) with the droplet filled white on a black background.

2. The exported `*_mask.png` / `*_mask.tif` files in this folder are those raw exports.
   They contain anti-aliasing artefacts (non-zero grey values at the polygon edges),
   so they are not directly usable as binary masks.

3. A small Python cleanup pipeline thresholded each raw export at value 1, kept the
   largest connected blob, filled internal holes, and (for three images where the
   contour included part of the substrate) cut at a hand-set baseline. The clean
   binary outputs live in `../masks/manual/` and are what `evaluate.py` and the
   notebook actually use.

## Contents

- `*_mask.png`, `h2o-metall_mask.tif`: raw Fiji exports (one per ground-truth image)
- `RAW_*.png`: smaller raw exports kept for two specific cases
- `BASELINE_PICK_*.jpg`: diagnostic figures showing the baseline cut chosen by
   the cleanup pipeline on the three masks that needed cutting
- `CUT_CHECK_*.jpg`: before/after visualisations for the same three cases
- `FINAL_binary_masks.jpg`, `FINAL_color_overlay.jpg`: contact-sheet visualisations
   of the final 10 binary masks alongside the originals
- `FOR_GIORGIA_mask_status.jpg`: status sheet used during the prep process
- `manual_binary_check.jpg`: post-cleanup overlay check on all 10 images
