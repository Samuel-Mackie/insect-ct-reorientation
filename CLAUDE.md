# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Insect micro-CT reorientation pipeline. Input is raw 3-D `.tif` volume stacks of insects
(one folder per species code under `data/original_photos/<SPECIES>/`). The pipeline segments
the specimen, finds the head via DINOv3 patch-token similarity, triangulates the 3-D head
position from multiple rendered views, and rotates each volume so the head points along +Y.

The **entire working pipeline now lives in `full_pipeline_dinov3.ipynb`** (single notebook,
top-to-bottom). The older one-script-per-step version (`run_pipeline.py`,
`segment_original_photos.py`, etc.) has been retired to `gamle_filer/` — do not edit those;
treat them as reference only.

## Running

There is no package manager config, test suite, or CLI. You run the notebook.

- Open `full_pipeline_dinov3.ipynb` and run all cells top-to-bottom. The orchestrator is the
  `run_all(original_path, info_path)` function near the "# Run all" heading; the cell after it
  sets `info_path = Path("data/test_dinov3_test_base_model")` and calls it.
- Headless re-run from the repo root:
  `jupyter nbconvert --to notebook --execute --inplace full_pipeline_dinov3.ipynb`
- Scope a run by editing the `animals` list (cell 2) or pointing `run_all` at a smaller
  `original_path`. Token extraction is cached (skips if `tokens.npy` exists), so re-runs are
  cheap; segmentation/rendering and rotation are **not** cached and re-overwrite outputs.

### DINOv3 weights are gated

`model_name` (cell 2) is `facebook/dinov3-vitb16-pretrain-lvd1689m` (the `-vits16` small
variant is commented out). These weights require an approved HuggingFace license. Before a
fresh run on a new machine: accept the license on the model page, then authenticate with
`huggingface-cli login` or `$env:HF_TOKEN = "hf_..."`. DINOv3 has register tokens between CLS
and patch tokens — patch extraction skips `1 + num_register_tokens` prefix tokens (read
dynamically from `model.config`).

## Key dependencies

`numpy`, `scipy` (`ndimage`), `scikit-image` (`threshold_otsu`), `tifffile`, `vedo` (VTK
backend, `settings.default_backend = "vtk"`), `Pillow`, `torch`, `transformers`
(`AutoImageProcessor`/`AutoModel`), `scikit-learn` (`cosine_similarity`), `matplotlib`.
Install manually; vedo renders offscreen via VTK.

## Architecture

### Coordinate convention (important)

vedo loads a volume as a numpy array in `[x, y, z]` order, and that order is used everywhere:
`volume_shape`, `center_of_mass_xyz`, the triangulated `head_xyz`, and the rotation are all in
the same voxel xyz frame. `ndimage.center_of_mass(mask)` returns indices in that same array
order, so COM and head live in one consistent space and the head→COM vector can be rotated to
+Y directly. Annotations in `image_annotations.json` are voxel `[x, y, z]`.

### Pipeline stages (functions, in notebook order)

```
process_volume            load .tif -> segment_largest_component -> COM ->
                          zero outside mask -> render_views (6 canonical views) ->
                          metadata.json (camera params per view + COM + shape)
get_head_information      project the annotated head xyz into each view with
                          project_point_to_pixel -> head_projection.json (patch row/col)
extract_patch_tokens      DINOv3 forward per rendered PNG -> tokens.npy (cached)
build_prototype_animal    gather the head-patch token from every annotated individual of a
                          species, average -> prototype_vector.npy   (needs ALL individuals)
save_top_k_patches        cosine(prototype, foreground patch tokens) -> top_k_patches.json
build_rays_from_individual  top-k patches + camera params -> 3-D rays (build_ray_from_pixel)
ransac_fuse               exhaustive-pair triangulation (triangulate_rays, lstsq) + refine
                          -> fused 3-D head point
rotation_matrix_from_vectors / rotate_volume
                          rotate head->COM vector onto +Y (Rodrigues) -> affine_transform
                          -> data/.../final/<SPECIES>/<ind>.tif, then re-segment for QA
```

`project_point_to_pixel` (forward) and `build_ray_from_pixel` (inverse) are an exact
projection/back-projection pair sharing `camera_basis`; keep them consistent if you touch one.

### Render geometry coupling

Rendering uses `render_size = (960, 960)` = `grid_w * patch_size` with `grid_w = grid_h = 60`,
`patch_size = 16`. The processor runs with `do_resize=False`, so the rendered image must stay
exactly `grid * patch_size` or the patch grid / token count will mismatch. Patch indexing is
`flat_idx = row * grid_w + col` throughout — `grid_w` is the row stride.

### Data layout (under `info_path`, e.g. `data/test_dinov3_test_base_model/`)

```
segmented/<SPECIES>/<ind>/  <ind>_<ANGLE>.png  (6 views) + metadata.json + head_projection.json
tokens/<SPECIES>/<ind>/<ind>_<ANGLE>/tokens.npy + top_k_patches.json
tokens/<SPECIES>/prototype_vector.npy
final/<SPECIES>/<ind>.tif            rotated (head-up) volume
final_segmented/<SPECIES>/<ind>/     QA re-render of the rotated volume
```

`VIEWS` defines the 6 canonical camera directions/up-vectors; `AXIS_TO_VECTOR` maps axis
labels to unit vectors (+Y is the head-up target).

## Supporting notebooks

- `HEAD_LOCALIZATION_MATH..ipynb` — derivation/checks for the projection and triangulation math.
- `symmetry.ipynb` — exploration of a bilateral-symmetry roll correction (not wired into the
  main pipeline here).
- `Annoteringer/save_annotations.py` — regenerates `Annoteringer/annotations_output/image_annotations.json`
  from in-file dicts; rerun after editing annotations.
