# HPC round 5 — pipeline timing, QA re-render removed

Re-measures the per-pass timings of the **actual pipeline** (both the DINO sweep and
PCA) with the final **QA re-render removed**. The QA re-render (a 6-view re-render of
the already-rotated volume) is only a manual-correctness aid — not part of producing
the reoriented volume — but it dominated Pass 3 in the earlier rounds. Dropping it
gives a clean measurement of the real reorientation cost. Correctness was already
established in rounds 1–4, so **no composites are produced here** — only timings.

## What is / isn't removed

| Method | Pass 1 | Pass 2 | Pass 3 | QA re-render |
|--------|--------|--------|--------|:------------:|
| **DINO** | segment + **render 6 views** *(required — tokens are extracted from these)* | DINO tokens + prototype + top-k | triangulate + rotate + write | **removed** |
| **PCA** | segment | PCA principal axis | rotate + write | **removed** |

⚠️ **DINO's Pass 1 render stays** — it is intrinsic to the method (tokens come from
those images), so the DINO sweep is *not* dramatically faster; only the optional
Pass 3 QA render is gone (~120 s/config saved). **PCA does no rendering at all** now,
so it finishes in seconds.

## Run

Local (repo root, same venv):

```bash
python HPC_round_5/run_timing_noqa.py --animals GH \
    --original data/original_photos --results HPC_round_5/experiments
```

DTU HPC (submit from repo root `$PROJ`; logs + results under `HPC_round_5/`):

```bash
cd $PROJ
mkdir -p HPC_round_5/logs
bsub < HPC_round_5/run_timing_noqa_lsf.sh          # all 12 species, DINO + PCA
tail -f HPC_round_5/logs/timing5_*.out
```

Subsets (env vars honored by the submit script):

```bash
SPECIES_LIST="GH MA" bsub < HPC_round_5/run_timing_noqa_lsf.sh   # fewer species
METHODS="pca"        bsub < HPC_round_5/run_timing_noqa_lsf.sh   # PCA only (fast)
```

Script flags: `--methods dino pca` (default both) · `--only baseline grid=90`
(subset DINO configs) · `--mode ofat|grid` · `--eval-range LO HI` · `--keep-work`.

## Setup / requirements

- **`gpuv100` queue + a GPU** — required for DINO's Pass 2 (patch tokens); also pins
  the job to the same node type as the earlier rounds so the timings are comparable.
  PCA uses no GPU.
- **`xvfb`** — still required: DINO's Pass 1 render needs a virtual X display. The
  submit script wraps everything in `xvfb-run -a`.
- **HF weights cache** (`$PROJ/hf_cache`) — the DINO models, same as the earlier
  rounds. Not needed if you run `METHODS="pca"`.

## Output

Same `summary.csv` columns as `run_experiments.py` (imported verbatim), so DINO rows
and the PCA row stack together and line up with the earlier rounds:

```
experiments/<SPECIES>/
├── summary.csv                 # one row per DINO config + one pca_seg row
├── baseline/config.json + metrics.json
├── grid=90/ ...
└── pca_seg/config.json + metrics.json
```

Note `pass3_s` here is **rotate + write only** (no render). For DINO, `pass1_s`
still includes the required 6-view render.

## Files

- `run_timing_noqa.py` — the harness (imports `pipeline.py` + `run_experiments.py`,
  edits neither; re-implements only the QA-free Pass 3 and per-config driver).
- `run_timing_noqa_lsf.sh` — LSF submit script (gpuv100 + GPU + `xvfb-run`).
