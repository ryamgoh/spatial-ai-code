#!/bin/bash
# Evaluate the latest retained 8K checkpoint against the exported best adapter.
# No training. Requires checkpoint-* directories to still exist on the cluster.
#SBATCH --job-name=spatial13-8k-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=experiments/13-iterative-hardening/logs/%x-%j.out
#SBATCH --error=experiments/13-iterative-hardening/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/13-iterative-hardening
MODEL=$EXP/models/qwen3.5-4b-v13-sft-8000

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

latest_checkpoint=$(
  find "$MODEL" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null \
    | sed 's#^.*/checkpoint-##' \
    | sort -n \
    | tail -1
)
if [[ -z "$latest_checkpoint" ]]; then
  echo "No retained checkpoint-* directory under $MODEL."
  echo "The final-checkpoint audit cannot be recovered from the exported adapter alone."
  exit 2
fi
CHECKPOINT=$MODEL/checkpoint-$latest_checkpoint
if ! has_adapter "$CHECKPOINT"; then
  echo "Latest checkpoint is not a complete LoRA adapter: $CHECKPOINT"
  exit 1
fi
echo "Auditing final retained checkpoint: $CHECKPOINT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/generated"
nvidia-smi -L || true

uv run --no-project python "$EXP/scripts/make_checkpoint_eval_config.py" \
  --source "$EXP/eval-sft-4b-8000-v13.yaml" \
  --output "$EXP/generated/eval-sft-4b-8000-final-v13.yaml" \
  --lora-path "../$CHECKPOINT" || exit 1
uv run --no-project python "$EXP/scripts/make_checkpoint_eval_config.py" \
  --source "$EXP/eval-breakpoint-sft-8000.yaml" \
  --output "$EXP/generated/eval-breakpoint-sft-8000-final.yaml" \
  --lora-path "../$CHECKPOINT" || exit 1

cd eval
uv sync || exit 1
for row in \
  "sft-8000-final-v13-stage1 eval-sft-4b-8000-final-v13.yaml" \
  "breakpoint-sft-8k-final-stage1 eval-breakpoint-sft-8000-final.yaml"; do
  set -- $row
  tag=$1 config=$2 out=../$EXP/results/$1
  if [[ "${FORCE_EVAL:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip $tag (already complete) ==="
    continue
  fi
  uv run python eval_new.py \
    --config ../$EXP/generated/"$config" \
    --stages 1 --output-dir "$out" || exit 1
done
cd "$SLURM_SUBMIT_DIR"

uv run --no-project python "$EXP/scripts/summarize_sft_8000.py" || true
uv run --no-project python "$EXP/scripts/summarize_checkpoint_audit.py" || exit 1
echo "Audit: $EXP/results/SFT-8000-CHECKPOINT-AUDIT.md"
