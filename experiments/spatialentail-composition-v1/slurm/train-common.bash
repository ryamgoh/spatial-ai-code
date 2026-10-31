#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Must run inside a Slurm allocation}"
: "${PREPARED:?Use slurm/submit.sh}"
case "${1:-}" in
    natural|symbolic) style=$1 ;;
    *) echo "Expected natural or symbolic" >&2; exit 2 ;;
esac
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [[ -e "$PREPARED/models/$style-seed-42" ]]; then
    echo "Refusing to overwrite an existing training run: $style" >&2
    exit 2
fi
srun uv run --frozen --no-sync python -m spatial.v2.entailment_thinking_training \
    verify --output "$PREPARED"
echo "Training $style on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
srun nvidia-smi
exec srun uv run --frozen --no-sync python finetune/finetune.py \
    "$PREPARED/configs/$style.yaml"
