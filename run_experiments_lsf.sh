#!/bin/bash
# DTU HPC (LSF / bsub) submit script: runs the experiment sweep for ALL species
# in a single job, one species after another. Submit from the repo root:
#
#     bsub < run_experiments_lsf.sh
#
# All 12 species sequentially is long (~20 h on one GPU), so the walltime is 24 h.
# Results are written per species and flushed after every config, so if the job
# is killed or times out the finished species survive -- rerun only the rest with
#     SPECIES_LIST="ML PP SL WO" bsub < run_experiments_lsf.sh
# (To parallelise instead, submit one job per species; see the loop at the bottom
#  of this comment.)
#
# Three env-var overrides (all optional):
#   SPECIES_LIST  species to run            (default: all 12)
#   RESULTS_NAME  output dir under $PROJ    (default: experiments) -> $PROJ/<name>/<SPECIES>
#   ONLY          run only one config id    (e.g. ONLY=views=4 -> just the 4-view config)
# Example -- only the 4-view config for every species, into a fresh dir:
#     RESULTS_NAME=experiments_2 ONLY=views=4 bsub < run_experiments_lsf.sh

#BSUB -J insect_all
#BSUB -q gpuv100              # GPU queue (use your available GPU queue)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8                    # CPU cores -> ProcessPool workers (passes 1 & 3)
#BSUB -R "rusage[mem=4GB]"    # PER CORE on DTU LSF -> 8 x 4 = 32 GB total
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00                # walltime hh:mm (must be <= gpuv100 hard RUNLIMIT; check: bqueues -l gpuv100)
#BSUB -o logs/experiments_%J.out
#BSUB -e logs/experiments_%J.err

set -euo pipefail
PROJ=/zhome/1c/0/216847/Desktop/fagprojekt   # repo location on the cluster
cd "$PROJ"
mkdir -p logs

# --- environment ----------------------------------------------------------
# module load cuda/12.4         # adjust to a module that exists (module avail cuda)
source "$PROJ/venv/bin/activate"
export HF_HOME="$PROJ/hf_cache"
export HF_HUB_OFFLINE=1

# --- run ------------------------------------------------------------------
# The BugNIST volumes already live on the shared project filesystem; point
# --original straight at them (no copy needed). xvfb-run gives the offscreen VTK
# renderer a virtual X display (compute nodes have no DISPLAY; without it the
# renders come out blank). Each species writes to experiments/<SPECIES>/.
DATA=/dtu/3d-imaging-center/projects/2022_QIM_55_BugNIST/analysis/BugNIST3D/bugnist_512
SPECIES_LIST="${SPECIES_LIST:-AC BC BF BL BP CF GH MA ML PP SL WO}"
RESULTS_NAME="${RESULTS_NAME:-experiments}"   # output dir under $PROJ
ONLY="${ONLY:-}"                              # empty -> all OFAT configs; else one id

for sp in $SPECIES_LIST; do
    echo "===================== $sp ====================="
    only_arg=""
    [ -n "$ONLY" ] && only_arg="--only $ONLY"
    # '|| ...' so one bad species does not abort the whole job (set -e).
    xvfb-run -a python run_experiments.py \
        --original "$DATA" \
        --results  "$PROJ/$RESULTS_NAME/$sp" \
        --animals  "$sp" \
        --eval-range 11 20 \
        --mode ofat \
        $only_arg \
        || echo "WARNING: species $sp exited non-zero, continuing with the next one"
done
echo "All species done."
