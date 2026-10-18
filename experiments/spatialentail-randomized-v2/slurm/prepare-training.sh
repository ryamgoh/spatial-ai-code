#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

benchmark_root=${BENCHMARK_ROOT:-experiments/spatialentail-randomized-v2/prepared-benchmark-candidate}
training_out=${TRAINING_OUT:-experiments/spatialentail-randomized-v2/prepared-training-candidate}

[[ -f "$benchmark_root/source-natural.jsonl" ]] || {
  echo "Missing benchmark source: $benchmark_root/source-natural.jsonl" >&2
  exit 1
}

echo "Preparing randomized SpatialEntail v2 training candidate"
echo "benchmark=$benchmark_root/source-natural.jsonl"
echo "output=$training_out"
echo "workers=${SLURM_CPUS_PER_TASK:-1}"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.randomized_v2 \
  --training-out "$training_out" \
  --benchmark-jsonl "$benchmark_root/source-natural.jsonl" \
  --workers "${SLURM_CPUS_PER_TASK:-1}"

[[ -f "$training_out/manifest.json" ]] || {
  echo "Training preparation did not produce a manifest" >&2
  exit 1
}

echo "completed=$(date --iso-8601=seconds)"
