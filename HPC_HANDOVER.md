# HPC Handover — Insect CT reorientation parameter sweep

This document describes how the parameter-sweep experiment is run on the DTU HPC
cluster (LSF / `bsub`), what was set up, the decisions made, and the gotchas we
hit. It is meant to let someone re-run, modify, or debug the sweep from scratch.

---

## 1. What the experiment does

`run_experiments.py` measures **timing** and produces **visual output for manual
correctness checking** across variations of the reorientation pipeline
(`pipeline.py`, the importable copy of `full_pipeline_dinov3.ipynb`). It runs the
same three passes as `run_parallel.py` and times each:

- **Pass 1** — segment + render the 6 (or 4) views (+ project the head annotation)
- **Pass 2** — DINOv3 patch tokens + prototype + top-k head patches
- **Pass 3** — triangulate the head, rotate the volume, QA re-render

### Knobs swept (one-factor-at-a-time around a baseline → 14 configs)

| Axis | Levels | Baseline |
|------|--------|----------|
| views | 6, 4 | 6 |
| model | small, base, large, huge (DINOv3 ViT-S/B/L/H+) | base |
| grid (resolution) | 30, 60, 90 (render = grid × 16 px) | 60 |
| segmentation threshold | otsu, 96, 97, 98 | otsu |
| top-k head patches | 3, 5, 8 | 5 |
| annotations (prototype source) | 1_image, 1_animal, 5_animals | 5_animals |

OFAT = baseline + one varied level at a time = **14 configs**. `--mode grid`
switches to the full Cartesian product (2592 configs) but is off by default.

### Two non-obvious design decisions

- **Thresholds 96/97/98 are PERCENTILES**, not absolute intensities: voxels below
  that percentile of the volume intensity count as background. `otsu` uses
  skimage's automatic threshold.
- **Correctness is checked by hand** from the composites — there is no automatic
  accuracy metric. Use `--verify-annotations` (below) to confirm the 3-D head
  annotation is correct before trusting a sweep.

---

## 2. Cluster layout (what lives where)

| Thing | Path |
|-------|------|
| Repo / working dir | `/zhome/1c/0/216847/Desktop/fagprojekt` (`$PROJ`) |
| Python venv | `$PROJ/venv` |
| HF model cache | `$PROJ/hf_cache` (weights in `$PROJ/hf_cache/hub/`) |
| Results output | `$PROJ/experiments/<SPECIES>` (one subfolder per species) |
| Job logs | `$PROJ/logs/experiments_<JOBID>.{out,err}` |
| **Insect data (do NOT copy)** | `/dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512` |

The BugNIST volumes already live on the shared `/dtu/...` filesystem and are
readable from the compute nodes, so `--original` points straight at them — no
data transfer needed. Only the **code** and the **DINOv3 weights** were
transferred from a laptop.

### Data naming (important)

Each individual is a single `.tif` directly under the species folder:
`bugnist_512/GH/gras_<group>_<NNN>.tif` (e.g. `gras_1_011.tif`, `gras_17_013.tif`).
There are ~713 GH volumes across many scan groups (`gras_1_*`, `gras_3_*`,
`gras_17_*`, ...).

---

## 3. One-time setup

### 3a. Connect
```bash
ssh sXXXXXX@login1.hpc.dtu.dk          # off-campus: connect DTU VPN (vpn.dtu.dk) first if it times out
```

### 3b. DINOv3 weights
The four DINOv3 models (`vits16`, `vitb16`, `vitl16`, `vith16plus`) are gated on
HuggingFace. They were downloaded with an `HF_TOKEN` and placed in
`$PROJ/hf_cache/hub/` as `models--facebook--dinov3-*`. Two ways to get them:

- **Download on the login node** (has internet; compute nodes do not):
  ```bash
  export HF_HOME=$PROJ/hf_cache HF_TOKEN=hf_...
  python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov3-vith16plus-pretrain-lvd1689m')"
  ```
  (repeat for vits16 / vitb16 / vitl16)
- **Or transfer a local cache** with `scp -r ...models--facebook--dinov3-*` into
  `$PROJ/hf_cache/hub/`.

The job sets `HF_HOME=$PROJ/hf_cache` and `HF_HUB_OFFLINE=1` so it reads the
cache without touching the network.

### 3c. Python environment
```bash
cd $PROJ
module load python3/3.11.9          # use `module avail python3` to find a real version
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-image scikit-learn tifffile pillow vedo transformers huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install filelock                 # transitive dep that pip missed once (see Gotchas)
pip check                            # report any other missing/broken deps and install them
```

---

## 4. Pre-flight checks (do these before a full submit)

Run on an **interactive GPU node** (`voltash`, or `a100sh`):
```bash
voltash
cd $PROJ && source venv/bin/activate
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1

# CUDA present?
python -c "import torch; print('cuda:', torch.cuda.is_available())"

# Rendering works? (needs xvfb; verify-annotations needs no DINO model)
which xvfb-run
xvfb-run -a python run_experiments.py --verify-annotations --animals GH \
    --original /dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512 \
    --results  $PWD/experiments
```
Inspect `experiments/verify_annotations/GH__*.png`: a **red circle should sit on
the head** in the views where the head is visible. That confirms (a) rendering
works and (b) the head annotation projects correctly. Then test one real config:
```bash
xvfb-run -a python run_experiments.py --only baseline --animals GH \
    --original /dtu/.../bugnist_512 --results $PWD/experiments
```
Look for `Pass 1/3: segment + render 15 volumes (10 eval + 5 annotated)` — the
**"10 eval + 5 annotated"** is the key sanity number (see Gotcha #1). Then `exit`.

---

## 5. Submit the full sweep

A **single job** runs all 12 species back-to-back (loop inside
`run_experiments_lsf.sh`), each species writing to `experiments/<SPECIES>/`.

```bash
cd $PROJ
mkdir -p logs                        # so LSF can write the output file
rm -rf experiments                   # optional: start clean

bsub < run_experiments_lsf.sh        # all 12 species, one job

bjobs                                # PEND -> RUN
bjobs -l <JOBID>                     # pending reason / queue position
tail -f logs/experiments_<JOBID>.out # once RUN; or `bpeek <JOBID>` while running
```

`run_experiments_lsf.sh` requests queue `gpuv100`, 1 GPU exclusive, 8 cores,
4 GB/core (= 32 GB, per-core on DTU LSF), **24 h walltime** (12 species on one
GPU is ~20 h). It `cd`s to `$PROJ`, activates the venv, sets the offline HF
cache, and loops `python run_experiments.py` over each species wrapped in
`xvfb-run -a`.

Results are flushed after every config, so if the job is killed or hits the
walltime the finished species survive on disk. Rerun only the remaining ones:
```bash
SPECIES_LIST="ML PP SL WO" bsub < run_experiments_lsf.sh
```
To go faster (or to parallelise), raise `#BSUB -n`, or submit one job per
species with `SPECIES_LIST=<one code>`.

---

## 6. Output structure

```
experiments/
└── GH/                             # one subfolder per species
    ├── summary.csv                 # one row per config: timings + n_ok + status
    ├── baseline/
    │   ├── config.json             # exact parameters of this config
    │   ├── metrics.json            # per-individual status / n_rays / n_inliers
    │   └── composites/
    │       ├── GH__gras_1_011.png  # 6-panel image of the rotated volume (one individual)
    │       ├── GH__gras_1_012.png
    │       └── ...                 # 10 individuals (gras_1_011..020)
    ├── views=4/composites/ ...
    ├── model=large/composites/ ...
    └── ...                         # one folder per config (14 total)
```

≈ 14 configs × 10 individuals = **140 composite images per species**. Each composite always
shows the 6 canonical views (even for the 4-view config) so configs are
visually comparable. **Correctness counting is manual**: open each config's
`composites/`, count how many heads point the right way, compare across configs,
and weigh against the timings in `summary.csv`.

The heavy intermediates (`work/`: segmentations, tokens, rotated `.tif`s) are
deleted per config to save disk. Add `--keep-work` to keep them.

Retrieve to a laptop:
```powershell
scp -r sXXXXXX@transfer.gbar.dtu.dk:/zhome/1c/0/216847/Desktop/fagprojekt/experiments .
```

---

## 7. Gotchas (things that bit us)

1. **Individual selection must match the annotation prefix.** Files are
   `gras_<group>_<NNN>` across many groups. Selecting by the trailing `_NNN`
   alone matched `_011.._020` in *every* group → hundreds of volumes. The fix:
   an individual is in the eval set only if it (a) shares a prefix with an
   annotated individual (e.g. `gras_1_*`) **and** (b) its `_NNN` is in
   `--eval-range` (default 11–20). So eval = `gras_1_011..020`, prototype =
   annotated `gras_1_000..004`. Sanity check is the "10 eval + 5 annotated" log
   line.

2. **Rendering needs `xvfb-run`.** Compute nodes have no `DISPLAY`; vedo/VTK
   offscreen rendering prints `bad X server connection. DISPLAY=` and produces
   blank images without it. The job and all interactive renders must be wrapped
   in `xvfb-run -a`.

3. **`No module named 'filelock'`** — a transitive dependency of
   `transformers`/`huggingface_hub` that pip did not pull in once. Fix:
   `pip install filelock`, then `pip check` for anything else.

4. **HF offline mode is strict.** With `HF_HUB_OFFLINE=1`, a model missing from
   the cache makes that config fail (logged as `FAILED` in `summary.csv`; the
   run continues). Confirm all four `models--facebook--dinov3-*` dirs are present
   before submitting.

5. **Home (`/zhome`) has a disk quota.** Weights (~5 GB) + outputs fit, but if
   you hit "disk quota exceeded", move `hf_cache` and `experiments` to
   `/work3/$USER/` and update `HF_HOME` and `--results` in the script.

6. **GPU queue pending is normal.** `STAT PEND` with reason
   "ngpus_physical not satisfied" just means it is waiting for a free GPU on
   `gpuv100`. If it waits for hours, try another queue you have access to
   (`gpua100`, `gpua10`) by editing `#BSUB -q`.

7. **Multiprocessing uses `spawn`** (forced in the script), because Pass 3's
   process pool is created after Pass 2 initialises CUDA in the main process, and
   forking after a live CUDA context is fragile on Linux.

---

## 8. Useful variations

```bash
# fewer/more species (must have annotations in Annoteringer/.../image_annotations.json)
--animals GH MA ML

# different evaluation individuals
--eval-range 11 30

# rerun a single config (id = "axis=level" or "baseline")
--only baseline model=large grid=90

# keep raw intermediates for inspection
--keep-work
```

Per-pass progress is printed live (`Pass 1/3: ...`, `Pass 1 done in NNs`), so a
silent terminal during a long pass is expected — check that
`experiments/<config>/work/` is growing if unsure whether it is stuck.

---

## 9. Current status (as of handover)

- Job `28638051` submitted to `gpuv100`, **PENDING** for a free GPU (normal).
- The full 14-config sweep was validated end-to-end **locally** (all `status=ok`,
  including grid 30/90, all four model sizes, percentile thresholds, the
  annotation modes, and `views=4`).
- Open question for the first cluster run: whether `xvfb`/Mesa software rendering
  is fast enough on the 512³ BugNIST volumes. Check `Pass 1`/`Pass 3` timings in
  the first configs; if too slow, raise `#BSUB -n` (workers) and/or `#BSUB -W`
  (walltime), or investigate GPU-accelerated offscreen rendering (EGL/OSMesa).
```
