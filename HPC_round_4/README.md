# HPC round 4 — PCA-based reorientation (segment → PCA → Rodrigues)

A purely geometric baseline for reorienting the insect volumes, to compare against
the DINOv3 head-localization sweep (rounds 1–3). It **replaces DINO entirely**: no
model, no GPU, no token extraction, no triangulation.

## What it does (per individual)

1. **Segment** the specimen with the same segmentation the DINO baseline uses —
   `pipeline.segment_largest_component` (otsu threshold → dilation → hole-fill →
   intensity-weighted largest connected component).
2. **PCA** on the segmented voxel coordinates. The **largest principal component**
   (top eigenvector of the coordinate covariance) is the body's long axis.
3. **Rodrigues rotate** that principal axis onto the volume's largest axis, using
   the pipeline's existing `rotation_matrix_from_vectors` + `rotate_volume`,
   rotating about the segmented centroid.
4. **QA re-render** the rotated volume in the 6 canonical views and build a 6-panel
   composite — the same manual-inspection artifact as the DINO rounds.

## Axis alignment only — no head/tail direction

A PCA principal component is an undirected line; it does not know which end is the
head, and the evaluation individuals are not annotated. So this round **only aligns
the long axis** — it does **not** resolve head-vs-tail. **Expect ~50% of specimens
to come out head-down.** That is inherent to PCA, not a bug. In `summary.csv`,
`n_ok` counts specimens that *produced a rotated output*; whether the head ends up
**up** is counted **by hand** in `composites/`, exactly like the DINO rounds.

## Run

Locally (repo root, same venv as the other rounds):

```bash
python HPC_round_4/run_pca_test.py --animals GH \
    --original data/original_photos \
    --results  HPC_round_4/experiments
```

On the DTU HPC cluster (submit from the repo root `$PROJ`; logs and results both
land under the `HPC_round_4/` subfolder):

```bash
cd $PROJ
mkdir -p HPC_round_4/logs
bsub < HPC_round_4/run_pca_test_lsf.sh      # all 12 species, one job
# or a subset:
SPECIES_LIST="GH MA" bsub < HPC_round_4/run_pca_test_lsf.sh

tail -f HPC_round_4/logs/pca4_*.out         # live log once it is RUNning
```

PCA itself uses no GPU, but the submit script requests the **`gpuv100` queue + a
GPU** on purpose: it pins the job to the same node type as the DINO rounds so
Pass 1/3 (CPU-bound segment + software VTK render) are timed on the same hardware
and are directly comparable to the DINO `summary.csv` (an earlier run on the `hpc`
CPU queue showed a ~14% Pass 3 node confound). The GPU just sits idle. **`xvfb` is
still required** — the QA re-render uses VTK offscreen rendering, so everything runs
under `xvfb-run -a` (the submit script handles this).

## Output (per species, mirrors the DINO rounds)

```
experiments/<SPECIES>/
├── summary.csv                 # SAME columns as the DINO summary.csv (see below)
└── pca_seg/
    ├── config.json             # method + parameters
    ├── metrics.json            # per-individual status, n_voxels, elongation, sub-timings
    └── composites/
        └── <SPECIES>__<ind>.png   # 6-panel QA render of the rotated volume
```

### Timing — same `summary.csv` columns and 3-pass layout as `run_experiments.py`

The header is imported verbatim from `run_experiments.py`, so a PCA row stacks
straight onto the DINO rows (unused knob columns — `model` / `top_k` / `anno_mode`
— are blank; `views=6`, `grid=60`, `threshold=otsu` are the fixed PCA settings).
The three passes map onto the DINO passes:

| Column    | Pass | What it measures | DINO analogue |
|-----------|------|------------------|---------------|
| `pass1_s` | 1 (parallel) | load + **segment** | DINO Pass 1 (segment + render-6-views) — PCA needs no pre-render, so it is legitimately smaller |
| `pass2_s` | 2 (serial) | **PCA** localization: coordinate covariance + eigen-decompose + largest principal component | DINO Pass 2 (patch tokens + prototype + top-k) |
| `pass3_s` | 3 (parallel) | **Rodrigues rotate** + write `.tif` + QA re-render 6 views | DINO Pass 3 (triangulate + rotate + QA render) — no triangulation here |
| `load_s`  | — | `0` (no model to load) | DINO model load time |
| `total_s` | — | wall clock for the config | **headline comparison** — PCA should be far cheaper |

## Evaluation set

Identical individuals to the earlier rounds: an individual is evaluated iff it (a)
shares a name prefix with an annotated individual (e.g. `gras_1_*`) **and** (b) its
trailing index `_NNN` is in `--eval-range` (default 11–20). Annotations are used
**only** for this selection — PCA itself needs none. The annotated individuals
(`gras_1_000..004`) are not processed here (no prototype to build).

## Files

- `run_pca_test.py` — the harness (standalone; imports `pipeline.py` +
  `run_experiments.py` helpers, edits neither).
- `run_pca_test_lsf.sh` — LSF submit script (all 12 species, `xvfb-run`, CPU queue).
