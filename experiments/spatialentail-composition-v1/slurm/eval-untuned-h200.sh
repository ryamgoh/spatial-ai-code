#!/usr/bin/env bash
#SBATCH --job-name=composition-untuned
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-composition-v1/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-composition-v1/logs/%x-%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh
bash experiments/spatialentail-composition-v1/slurm/eval-common.bash untuned
