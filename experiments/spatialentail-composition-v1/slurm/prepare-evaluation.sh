#!/usr/bin/env bash
#SBATCH --job-name=composition-eval-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-composition-v1/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-composition-v1/logs/%x-%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec srun uv --native-tls run --python 3.12 --no-project \
    --with transformers==5.16.1 --with tokenizers==0.23.1 \
    --with z3-solver==4.16.0.0 --with scipy==1.18.1 --with pyyaml \
    python experiments/spatialentail-composition-v1/prepare_evaluation.py
