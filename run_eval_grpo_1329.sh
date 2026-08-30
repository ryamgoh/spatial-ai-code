#!/bin/bash
# Separate from GRPO train. 1329 cleaned SpatialMap two-pass eval.
# Default: merged SFT + GRPO LoRA from the H100-96 2000-prompt run.
# Override adapter:
#   sbatch --export=ALL,ADAPTER=experiments/05-grpo/models/deepseek-r1-qwen3-8b-grpo-vllm run_eval_grpo_1329.sh
#SBATCH --job-name=spatialeval-grpo-1329
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MERGED=experiments/05-grpo/models/deepseek-r1-qwen3-8b-merged
ADAPTER="${ADAPTER:-experiments/05-grpo/models/deepseek-r1-qwen3-8b-grpo-h100}"

if [[ ! -f "$MERGED/config.json" ]]; then
  echo "Merged SFT weights missing at $MERGED"
  exit 1
fi

if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  latest=$(ls -d "$ADAPTER"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [[ -n "${latest}" && -f "$latest/adapter_config.json" ]]; then
    echo "Final adapter missing; using $latest"
    ADAPTER="$latest"
  else
    echo "GRPO adapter missing at $ADAPTER — wait for train to finish"
    exit 1
  fi
fi

echo "eval merged=$MERGED"
echo "eval lora=$ADAPTER"
echo "eval task=spatial_eval_gen_cleaned_1329"

cd eval
srun uv sync

CFG=../experiments/05-grpo/eval-grpo-1329.yaml
RUN_CFG="$CFG"
# Yaml lora_path is relative to eval/. Rewrite if we resolved a checkpoint.
if [[ "$ADAPTER" != "experiments/05-grpo/models/deepseek-r1-qwen3-8b-grpo-h100" ]]; then
  RUN_CFG=$(mktemp ../experiments/05-grpo/eval-grpo-1329.XXXX.yaml)
  sed "s|lora_path:.*|lora_path: ../${ADAPTER#./}|" "$CFG" > "$RUN_CFG"
  trap 'rm -f "$RUN_CFG"' EXIT
fi

srun --cpu-bind=cores uv run python eval_new.py --config "$RUN_CFG"
