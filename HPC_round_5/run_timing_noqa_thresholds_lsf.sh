#!/bin/bash
# DTU HPC (LSF / bsub) submit script for HPC round 5: render-free TIMING of the two
# extra percentile thresholds 99 and 99.5 only. Submit from the repo root:
#
#     cd $PROJ && mkdir -p HPC_round_5/logs
#     bsub < HPC_round_5/run_timing_noqa_thresholds_lsf.sh
#
# Runs threshold=99 and threshold=99.5 (DINO only) via
# HPC_round_5/run_timing_noqa_thresholds.py, appending the rows to the round-5
# summary.csv per species (same format, QA re-render removed). Same node/queue setup
# as the rest of round 5: gpuv100 + a GPU (required for DINO Pass 2 tokens) and
# xvfb-run (DINO's Pass 1 render still runs). Only 2 configs/species, so it is quick.
#
# Env-var overrides (all optional):
#   SPECIES_LIST  species to run            (default: all 12)
#   RESULTS_NAME  output dir under $PROJ    (default: HPC_round_5/experiments)

#BSUB -J insect_thr5
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00                 # 2 configs/species -> quick
#BSUB -o HPC_round_5/logs/thr5_%J.out
#BSUB -e HPC_round_5/logs/thr5_%J.err

set -euo pipefail
PROJ=/zhome/1c/0/216847/Desktop/fagprojekt
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

for sp in $SPECIES_LIST; do
    echo "===================== $sp ====================="
    # '|| ...' so one bad species does not abort the whole job (set -e).
    xvfb-run -a python HPC_round_5/run_timing_noqa_thresholds.py \
        --original "$DATA" \
        --results  "$PROJ/$RESULTS_NAME/$sp" \
        --animals  "$sp" \
        --eval-range 11 20 \
        || echo "WARNING: species $sp exited non-zero, continuing with the next one"
done
echo "All species done."
