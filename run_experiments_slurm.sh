#!/bin/bash
# DTU Sophia (SLURM) submit script for the experiment sweep.
# Submit from the repo root:   sbatch run_experiments_slurm.sh

#SBATCH --job-name=insect_experiments
#SBATCH --partition=gpu          # GPU partition on your allocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8        # -> ProcessPool workers (passes 1 & 3)
#SBATCH --mem=16G                # total for the node (SLURM --mem is not per-core)
#SBATCH --time=48:00:00          # finishes earlier; margin for huge + software rendering
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err

set -euo pipefail
mkdir -p logs

# module load python/3.11 cuda/12.x   # adjust to the cluster's module names
# source ~/envs/fagprojekt/bin/activate

# Pre-download the gated DINOv3 weights ON THE LOGIN NODE first (see
# run_experiments_lsf.sh for the huggingface-cli commands). The job then reads
# them from the cache offline:
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

python run_experiments.py \
    --original data/original_photos \
    --results  data/experiments \
    --animals GH \
    --eval-range 11 20 \
    --mode ofat
