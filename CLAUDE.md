# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Insect CT reorientation pipeline: takes raw 3-D micro-CT volumes (`.tif` stacks) of insects, detects the head position using DINOv2 patch-token similarity, and rotates the volume so the head points in a canonical direction. Species codes map to folders under `data/original_photos/` (e.g. `AC`, `BC`, `BF`, …).

## Running the pipeline

### Full pipeline (all species, overwrite existing results)

```bash
python run_pipeline.py
```

### Individual steps

Each step is a standalone CLI script. Run from the repo root:

```bash
# 1. Segment volumes and render 6 canonical views per individual
python segment_original_photos.py --input-root data/original_photos --output-root data/new_photos/segmented --overwrite

# 2. Render annotated head-marker views for the reference set
python visualize_annotated_heads.py

# 3. Find top-3 likely head patches per view using DINOv2 cosine similarity
python top3_head_patches.py --animal AC   # or omit --animal for all

# 4. Fuse head position across views via camera-ray triangulation
python fuse_head_position.py

# 5. Rotate volumes so head points up (+Y)
python rotate_head_up.py --overwrite

# 5b. (Optional) Head-up AND roll about Y so the bilateral symmetry plane is canonical.
#     Add --composite to also save a finished 6-view QA image per volume.
python rotate_head_up_symmetry.py --overwrite --composite

# 6. Render 6-view composites of rotated volumes
python segment_sixview_composite.py --input-root data/finished_photos/rotated --output-root data/finished_photos/composite --overwrite

# Regenerate annotations JSON (only needed if annotation dict changes)
python annotations/save_annotations.py
```

Limit scope with `--animal AC` or `--max-files 3`.

## Key dependencies

`numpy`, `scipy`, `scikit-image` (`skimage`), `tifffile`, `vedo`, `PIL` (Pillow), `torch`, `transformers` (HuggingFace DINOv2), `sklearn`.

No package manager config exists in the repo; install manually. `vedo` uses a VTK backend (`settings.default_backend = "vtk"`).

## Architecture

### Coordinate convention

Vedo loads volumes as `[x, y, z]` numpy arrays. All internal xyz coordinates follow this order. `volume_shape_xyz` in metadata JSONs refers to `[x, y, z]`.

### Segmentation (`segment_largest_component`)

Used in every pipeline script. Current implementation (multi-Otsu, class 2):
1. `threshold_multiotsu(volume, classes=3)` → top intensity class (`regions == 2`)
2. `binary_dilation(iterations=5)` + `binary_fill_holes` to close gaps before labelling
3. Label connected components, pick largest by **intensity-weighted** sum (`sum_labels`)

Returns a boolean mask. Scripts zero out the volume outside the mask before rendering.

### Head detection flow

```
segment_original_photos  →  6 PNG views + metadata.json (camera params)
        ↓
visualize_annotated_heads →  head_projection.json (ground-truth head patch coords per view)
        ↓
top3_head_patches          →  DINOv2 patch tokens; PCA-reduced prototypes; cosine similarity
                               ranks patches; emits top3_heads JSON per angle
        ↓
fuse_head_position         →  camera-ray triangulation → fused_head.json
                               (estimated_head_xyz_voxel, confidence, reprojection error)
        ↓
rotate_head_up             →  affine rotation aligning head→CoM vector to +Y axis
        ↓
rotate_head_up_symmetry    →  (optional) head-up + roll about Y so the bilateral
                               symmetry plane is canonical (see below)
```

### Symmetry-axis roll correction (`rotate_head_up_symmetry.py`)

`rotate_head_up` only fixes the head→tail direction (+Y); a roll degree of freedom about
Y remains. `rotate_head_up_symmetry.py` (a copy of `rotate_head_up.py`) additionally
detects the bilateral symmetry plane and rolls the volume so it becomes canonical.

- **Symmetry metric** (same as the `symmetry.ipynb` winner): segment → project the
  head-up volume along Y (head-tail axis) with **`sum`** → centre the cross-section →
  scan the mirror axis and pick the angle θ minimizing `1 − ncc(img, reflection)`
  (normalized cross-correlation). `sum` (not `max`) avoids spurious minima from sharp
  leg-tips; `ncc` is intensity-scale-invariant and gives the cleanest single minimum.
- **Must run on the segmented volume.** Raw rotated volumes are ~90% non-animal voxels
  (background/holder) which otherwise dominate and bias the axis.
- **Roll** `= 90 − θ` about Y; the head-up rotation and roll are composed into one affine
  (single interpolation on the raw data). Canonical frame: head-tail = Y, dorsoventral = Z,
  left-right = X.
- **Confidence** `= (median − min)/std` of the score curve. Below
  `--symmetry-conf-threshold` (default 2.0) the individual is flagged
  (`low_symmetry_confidence` in JSON + `[LOW-CONF]` in the log) — this catches curved or
  twisted specimens that have no clean planar symmetry; symmetry alone cannot fix those.
- Measurement runs on a downsampled copy (`--symmetry-downsample`, default 2) for speed;
  the output tif is full resolution. `--no-symmetry` reverts to plain head-up behaviour.
- `--composite` reuses `render_six_views`/`compose_panel` from `segment_sixview_composite`
  to save a finished 6-view QA image (title shows roll + confidence).
- `symmetry.ipynb` is the exploration notebook: 2D metric comparison (absdiff/iou/ncc ×
  max/sum), a more robust 3D slice-wise reflection alternative, and a confidence survey.

### Data layout

```
data/
  original_photos/<SPECIES>/<individual>.tif
  new_photos/
    segmented/<SPECIES>/<individual>/      ← 6 PNGs + metadata.json (camera params)
    head_visualizations/<SPECIES>/<individual>/  ← head_projection.json
    head_top3/<SPECIES>/<individual>/json/ ← *_top3_heads.json per angle
    head_fused/<SPECIES>/<individual>/json/fused_head.json
  finished_photos/
    rotated/<SPECIES>/tif/<individual>.tif
    composite/<SPECIES>/images/
    rotated_symmetry/<SPECIES>/         ← rotate_head_up_symmetry.py output
      tif/<individual>.tif              ← head-up + symmetry-roll volume
      json/<individual>_rotation.json   ← incl. symmetry_axis_deg, roll_angle_deg, symmetry_confidence
      composite/<individual>_sixview.png ← with --composite
annotations/
  annotations_output/image_annotations.json  ← ground-truth head voxels [x,y,z]
```

### Annotation format

`image_annotations.json`: `{ "SPECIES": { "filename.tif": [x, y, z], ... }, ... }`. Coordinates are voxel xyz. Regenerate with `python annotations/save_annotations.py` after editing the dicts in that file.
