#!/bin/bash
#SBATCH --job-name=spatialentail-prepare-spatial-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"
uv run python -m spatial.v2.study.spatial_specialist_evaluation \
  experiments/spatialentail-randomized-v2/prepared-general-eval \
  experiments/spatialentail-randomized-v2/prepared-spatial-eval
