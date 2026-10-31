#!/usr/bin/env bash
# Generate questions, audit evidence, and prepare both training arms.
#SBATCH --job-name=spatialentail-prepare
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
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${PREPARED:?Use slurm/submit.sh to choose a fresh output}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# No pre-generated data or model weights are required.
srun uv --native-tls run --python 3.12 --no-project \
    --with 'transformers==5.16.1' --with 'tokenizers==0.23.1' \
    --with 'jinja2==3.1.6' --with 'pyyaml==6.0.3' \
    --with 'z3-solver==4.16.0.0' --with 'scipy==1.18.1' \
    python experiments/spatialentail-composition-v1/generate_and_prepare.py \
    --prepared "$PREPARED" --workers 4
srun uv --native-tls run --python 3.12 --no-project --with 'pyyaml==6.0.3' \
    python -m spatial.v2.entailment_thinking_training verify --output "$PREPARED"

# Install the shared Linux training environment once, before both GPU jobs.
srun uv --native-tls sync --frozen
printf 'Prepared both training arms: %s\n' "$PREPARED"
