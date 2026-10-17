#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

diagnostic_root=${DIAGNOSTIC_ROOT:-experiments/spatialentail-randomized-v2/prepared-diagnostics}
mkdir -p "$diagnostic_root"

echo "Preparing randomized SpatialEntail v2 diagnostics"
echo "output=$diagnostic_root"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.randomized_v2 \
  --special-out "$diagnostic_root/special-responses" \
  --consistency-out "$diagnostic_root/consistency"

echo "completed=$(date --iso-8601=seconds)"
