#!/bin/bash
# DTU HPC (LSF / bsub) submit script for HPC round 5: pipeline TIMING ONLY, with the
# QA re-render removed. Runs BOTH the DINO sweep and PCA (via
# HPC_round_5/run_timing_noqa.py) for every species in SPECIES_LIST, writing to
# <RESULTS>/<SPECIES>/. Submit from the repo root:
#
#     cd $PROJ && mkdir -p HPC_round_5/logs
#     bsub < HPC_round_5/run_timing_noqa_lsf.sh
#
# Same node/queue setup as the DINO rounds (gpuv100 + a GPU): the GPU is REQUIRED for
# DINO's Pass 2 (patch tokens), and running on the same node type keeps the timings
# comparable to the earlier rounds. xvfb IS still needed -- DINO's Pass 1 renders the
# 6 views the tokens are extracted from (that render is part of the method and is
# kept; only the optional Pass 3 QA re-render is removed). PCA uses neither GPU nor
# rendering, but shares the job so its Pass 3 is timed on the same node.
#
# Env-var overrides (all optional):
#   SPECIES_LIST  species to run            (default: all 12)
#   RESULTS_NAME  output dir under $PROJ    (default: HPC_round_5/experiments)
#   METHODS       "dino pca" | "dino" | "pca"  (default: dino pca)

#BSUB -J insect_timing5
#BSUB -q gpuv100             # SAME queue/node type as the DINO rounds (comparable timings)
#BSUB -gpu "num=1:mode=exclusive_process"   # REQUIRED for DINO Pass 2 tokens
#BSUB -n 8                    # CPU cores -> ProcessPool workers (passes 1 & 3)
#BSUB -R "rusage[mem=4GB]"    # PER CORE on DTU LSF -> 8 x 4 = 32 GB total
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00               # walltime hh:mm (Pass 1 render still dominates the DINO sweep)
# Logs + results both live under the HPC_round_5/ subfolder. HPC_round_5/logs/ must
# exist at submit time (mkdir -p HPC_round_5/logs before bsub).
#BSUB -o HPC_round_5/logs/timing5_%J.out
#BSUB -e HPC_round_5/logs/timing5_%J.err

set -euo pipefail
PROJ=/zhome/1c/0/216847/Desktop/fagprojekt   # repo location on the cluster
cd "$PROJ"
mkdir -p HPC_round_5/logs

# --- environment ----------------------------------------------------------
source "$PROJ/venv/bin/activate"
export HF_HOME="$PROJ/hf_cache"     # DINO weights cache (needed for Pass 2)
export HF_HUB_OFFLINE=1

# --- run ------------------------------------------------------------------
DATA=/dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512
SPECIES_LIST="${SPECIES_LIST:-AC BC BF BL BP CF GH MA ML PP SL WO}"
RESULTS_NAME="${RESULTS_NAME:-HPC_round_5/experiments}"
METHODS="${METHODS:-dino pca}"

for sp in $SPECIES_LIST; do
    echo "===================== $sp ====================="
    # '|| ...' so one bad species does not abort the whole job (set -e).
    xvfb-run -a python HPC_round_5/run_timing_noqa.py \
        --original "$DATA" \
        --results  "$PROJ/$RESULTS_NAME/$sp" \
        --animals  "$sp" \
        --methods  $METHODS \
        --eval-range 11 20 \
        || echo "WARNING: species $sp exited non-zero, continuing with the next one"
done
echo "All species done."
