#!/bin/bash
# DTU HPC (LSF / bsub) submit script for HPC round 3: the segmentation-threshold
# test at percentiles 99 and 99.5 only. Submit from the repo root:
#
#     bsub < HPC_round_3/run_threshold_test_lsf.sh
#
# This runs three configs per species -- baseline (otsu, a within-job timing
# anchor), threshold=99 and threshold=99.5 -- for every species in SPECIES_LIST,
# reusing the unchanged run_experiments.py harness via
# HPC_round_3/run_threshold_test.py. Far cheaper than the full sweep (3 configs
# per species instead of 14), so the walltime is short.
#
# Env-var overrides (all optional):
#   SPECIES_LIST  species to run            (default: all 12)
#   RESULTS_NAME  output dir under $PROJ    (default: HPC_round_3/experiments)

#BSUB -J insect_thr3
#BSUB -q gpuv100              # GPU queue (use your available GPU queue)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8                    # CPU cores -> ProcessPool workers (passes 1 & 3)
#BSUB -R "rusage[mem=4GB]"    # PER CORE on DTU LSF -> 8 x 4 = 32 GB total
#BSUB -R "span[hosts=1]"
#BSUB -W 8:00                 # walltime hh:mm (3 configs/species is quick)
#BSUB -o logs/threshold3_%J.out
#BSUB -e logs/threshold3_%J.err

set -euo pipefail
PROJ=/zhome/1c/0/216847/Desktop/fagprojekt   # repo location on the cluster
cd "$PROJ"
mkdir -p logs

# --- environment ----------------------------------------------------------
source "$PROJ/venv/bin/activate"
export HF_HOME="$PROJ/hf_cache"
export HF_HUB_OFFLINE=1

# --- run ------------------------------------------------------------------
# The BugNIST volumes already live on the shared project filesystem; point
# --original straight at them (no copy needed). xvfb-run gives the offscreen VTK
# renderer a virtual X display. Each species writes to <RESULTS_NAME>/<SPECIES>/.
DATA=/dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512
SPECIES_LIST="${SPECIES_LIST:-AC BC BF BL BP CF GH MA ML PP SL WO}"
RESULTS_NAME="${RESULTS_NAME:-HPC_round_3/experiments}"   # output dir under $PROJ

for sp in $SPECIES_LIST; do
    echo "===================== $sp ====================="
    # '|| ...' so one bad species does not abort the whole job (set -e).
    xvfb-run -a python HPC_round_3/run_threshold_test.py \
        --original "$DATA" \
        --results  "$PROJ/$RESULTS_NAME/$sp" \
        --animals  "$sp" \
        --eval-range 11 20 \
        || echo "WARNING: species $sp exited non-zero, continuing with the next one"
done
echo "All species done."
