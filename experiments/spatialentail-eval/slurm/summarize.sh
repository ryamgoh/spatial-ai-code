#!/bin/bash
#SBATCH --job-name=spatialentail-eval-summary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/spatialentail-eval/prepared-final/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-eval/prepared-final/logs/%x-%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"
bash experiments/spatialentail-eval/jobs/summarize.sh
