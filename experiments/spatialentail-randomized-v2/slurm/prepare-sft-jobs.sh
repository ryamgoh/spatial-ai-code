#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-jobs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

candidate=${TRAINING_CANDIDATE:-experiments/spatialentail-randomized-v2/prepared-training-candidate}
output=${SFT_JOBS_OUT:-experiments/spatialentail-randomized-v2/prepared-sft}

uv run python -m spatial.v2.study.randomized_jobs \
  "$candidate" \
  "$candidate/token-audit.json" \
  "$output" \
  --repo-root "$SLURM_SUBMIT_DIR"
