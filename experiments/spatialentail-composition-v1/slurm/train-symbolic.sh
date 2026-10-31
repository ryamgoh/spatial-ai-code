#!/usr/bin/env bash
#SBATCH --job-name=spatialentail-symbolic
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/spatialentail-composition-v1/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-composition-v1/logs/%x-%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh
bash experiments/spatialentail-composition-v1/slurm/train-common.bash symbolic
