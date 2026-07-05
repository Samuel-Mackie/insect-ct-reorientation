#!/bin/bash
# DTU HPC (LSF / bsub) submit script for HPC round 4: the PCA-based reorientation
# test (segment -> PCA principal axis -> Rodrigues rotate). Submit from the repo
# root:
#
#     bsub < HPC_round_4/run_pca_test_lsf.sh
#
# This runs the single `pca_seg` config for every species in SPECIES_LIST via
# HPC_round_4/run_pca_test.py, writing to <RESULTS>/<SPECIES>/. It is much cheaper
# than the DINO sweep: no GPU, no model load, no token extraction, no triangulation
# -- just segment + PCA + rotate, then the 6-view QA re-render.
#
# NO GPU is needed (PCA runs on the CPU). xvfb IS still needed: the QA re-render
# uses vedo/VTK offscreen rendering, which needs a virtual X display on the compute
# node -- hence the `xvfb-run -a` wrapper, same as the earlier rounds.
#
# Env-var overrides (all optional):
#   SPECIES_LIST  species to run            (default: all 12)
#   RESULTS_NAME  output dir under $PROJ    (default: HPC_round_4/experiments)

#BSUB -J insect_pca4
#BSUB -q hpc                  # CPU queue (no GPU needed). If `hpc` is unavailable
                              # to you, use a GPU queue that works, e.g. gpuv100 --
                              # the GPU will simply sit idle.
#BSUB -n 8                    # CPU cores -> ProcessPool workers (both passes)
#BSUB -R "rusage[mem=4GB]"    # PER CORE on DTU LSF -> 8 x 4 = 32 GB total
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00                 # walltime hh:mm (1 cheap config/species -> finishes early)
# Everything for this round lives under the HPC_round_4/ subfolder (logs + results).
# These -o/-e paths are relative to the directory you SUBMIT from (= $PROJ, the repo
# root), so HPC_round_4/logs/ must already exist at submit time -- create it with
#     mkdir -p HPC_round_4/logs
# before `bsub < HPC_round_4/run_pca_test_lsf.sh`.
#BSUB -o HPC_round_4/logs/pca4_%J.out
#BSUB -e HPC_round_4/logs/pca4_%J.err

set -euo pipefail
PROJ=/zhome/1c/0/216847/Desktop/fagprojekt   # repo location on the cluster
cd "$PROJ"
mkdir -p HPC_round_4/logs

# --- environment ----------------------------------------------------------
source "$PROJ/venv/bin/activate"
# HF cache/offline are irrelevant here (no model is loaded) but harmless to set.
export HF_HOME="$PROJ/hf_cache"
export HF_HUB_OFFLINE=1

# --- run ------------------------------------------------------------------
# The BugNIST volumes already live on the shared project filesystem; point
# --original straight at them (no copy needed). xvfb-run gives the offscreen VTK
# renderer a virtual X display. Each species writes to <RESULTS_NAME>/<SPECIES>/.
DATA=/dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512
SPECIES_LIST="${SPECIES_LIST:-AC BC BF BL BP CF GH MA ML PP SL WO}"
RESULTS_NAME="${RESULTS_NAME:-HPC_round_4/experiments}"   # output dir under $PROJ

for sp in $SPECIES_LIST; do
    echo "===================== $sp ====================="
    # '|| ...' so one bad species does not abort the whole job (set -e).
    xvfb-run -a python HPC_round_4/run_pca_test.py \
        --original "$DATA" \
        --results  "$PROJ/$RESULTS_NAME/$sp" \
        --animals  "$sp" \
        --eval-range 11 20 \
        || echo "WARNING: species $sp exited non-zero, continuing with the next one"
done
echo "All species done."
