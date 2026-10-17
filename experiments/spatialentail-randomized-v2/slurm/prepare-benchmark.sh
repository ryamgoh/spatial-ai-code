#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-benchmark
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

calibration_root=${CALIBRATION_ROOT:-experiments/spatialentail-randomized-v2/prepared-calibration}
benchmark_out=${BENCHMARK_OUT:-experiments/spatialentail-randomized-v2/prepared-benchmark-candidate}

[[ -f "$calibration_root/manifest.json" ]] || {
  echo "Missing completed calibration manifest: $calibration_root/manifest.json" >&2
  exit 1
}

echo "Preparing randomized SpatialEntail v2 benchmark candidate"
echo "calibration=$calibration_root"
echo "output=$benchmark_out"
echo "workers=${SLURM_CPUS_PER_TASK:-1}"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.randomized_v2 \
  --benchmark-out "$benchmark_out" \
  --calibration-manifest "$calibration_root/manifest.json" \
  --workers "${SLURM_CPUS_PER_TASK:-1}"

echo "completed=$(date --iso-8601=seconds)"
