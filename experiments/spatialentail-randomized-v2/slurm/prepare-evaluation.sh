#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-eval-prep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

benchmark=${BENCHMARK_JSONL:-experiments/spatialentail-randomized-v2/prepared-benchmark-candidate/source-natural.jsonl}
output=${EVAL_OUT:-experiments/spatialentail-randomized-v2/prepared-eval}

uv run python -m spatial.v2.study.randomized_evaluation \
  "$benchmark" \
  "$output" \
  --repo-root "$SLURM_SUBMIT_DIR"
