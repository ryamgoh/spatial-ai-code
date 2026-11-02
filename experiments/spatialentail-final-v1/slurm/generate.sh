#!/usr/bin/env bash
#SBATCH --job-name=spatialentail-final-generate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu

set -euo pipefail
: "${FINAL_RUN_DIR:?Use slurm/submit-generation.sh}"
: "${FINAL_EXPERIMENT_DIR:?Use slurm/submit-generation.sh}"
: "${FINAL_GENERATION_MODE:?Use slurm/submit-generation.sh}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
ROOT=$(dirname "$(dirname "$FINAL_EXPERIMENT_DIR")")
source "$ROOT/slurm/lib/pin-srun-cpus.sh"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORKERS=${FINAL_GENERATION_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$WORKERS" -le "${SLURM_CPUS_PER_TASK:-1}" ]] || {
    echo "Workers must fit SLURM_CPUS_PER_TASK." >&2; exit 2;
}
echo "Generation only: mode=$FINAL_GENERATION_MODE workers=$WORKERS output=$FINAL_RUN_DIR"
srun uv --native-tls run --python 3.12 --no-project \
    --script "$FINAL_EXPERIMENT_DIR/generate.py" build \
    --run-dir "$FINAL_RUN_DIR" --mode "$FINAL_GENERATION_MODE" --workers "$WORKERS"
srun uv --native-tls run --python 3.12 --no-project \
    --script "$FINAL_EXPERIMENT_DIR/generate.py" verify --run-dir "$FINAL_RUN_DIR"
echo "Candidate generation finished. No training job was submitted."
