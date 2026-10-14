#!/bin/bash
#SBATCH --job-name=spatialentail-allpossible-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-allpossible-pilot/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-allpossible-pilot/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

pilot_out=${PILOT_OUT:-experiments/spatialentail-allpossible-pilot/prepared}
samples_per_cell=${SAMPLES_PER_CELL:-25}

echo "Preparing SpatialEntail-ALL-POSSIBLE pilot"
echo "output=$pilot_out"
echo "samples_per_cell=$samples_per_cell"
echo "allocated_cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.possibility_pilot \
  spatial/v2/all-possible-cells.yaml \
  --out "$pilot_out" \
  --samples-per-cell "$samples_per_cell" \
  --workers "${SLURM_CPUS_PER_TASK:-1}"

[[ -f "$pilot_out/jobs.json" ]] || {
  echo "Pilot preparation did not produce $pilot_out/jobs.json" >&2
  exit 1
}

echo "completed=$(date --iso-8601=seconds)"
echo "Prepared pilot is ready at $pilot_out"
