#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

calibration_out=${CALIBRATION_OUT:-experiments/spatialentail-randomized-v2/prepared-calibration}
study_manifest=${STUDY_MANIFEST:-experiments/spatialentail-randomized-v2/study-manifest.json}
train_max_l=${TRAIN_MAX_L:-6}
train_max_x=${TRAIN_MAX_X:-6}
eval_max_l=${EVAL_MAX_L:-8}
eval_max_x=${EVAL_MAX_X:-8}

echo "Preparing randomized SpatialEntail v2 calibration"
echo "output=$calibration_out"
echo "workers=${SLURM_CPUS_PER_TASK:-1}"
echo "train_max_l=$train_max_l"
echo "train_max_x=$train_max_x"
echo "eval_max_l=$eval_max_l"
echo "eval_max_x=$eval_max_x"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.randomized_v2 \
  --manifest-out "$study_manifest" \
  --calibration-out "$calibration_out" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \
  --train-max-l "$train_max_l" \
  --train-max-x "$train_max_x" \
  --eval-max-l "$eval_max_l" \
  --eval-max-x "$eval_max_x"

[[ -f "$calibration_out/manifest.json" ]] || {
  echo "Calibration did not produce a manifest" >&2
  exit 1
}

echo "completed=$(date --iso-8601=seconds)"
