#!/bin/bash
# Exp 13 bridge: evaluate untuned/1.5K/6K on the original V12 2K test using
# the one-pass protocol. No training. H200 lives on gpu; gpu max is 03:00:00.
#
#   sbatch experiments/13-iterative-hardening/slurm/eval-v12-stage1-bridge-h200.sh
#   ONLY=v12-4b-1.5k sbatch experiments/13-iterative-hardening/slurm/eval-v12-stage1-bridge-h200.sh
#SBATCH --job-name=spatial13-v12bridge
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
RESULTS=$EXP/results
TEST=data/spatial_sft_v12_2000_test.jsonl
CELLS=(
  "baseline      eval-v12test-baseline-stage1.yaml -"
  "v12-4b-1.5k  eval-v12test-v12-4b-1500-stage1.yaml experiments/12-v6-mix-reweight/models/qwen3.5-4b-sft-v12-1500"
  "v12-4b-6k    eval-v12test-v12-4b-6000-stage1.yaml experiments/12-v6-mix-reweight/models/qwen3.5-4b-sft-v12-6000"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  [[ -z "${ONLY:-}" ]] && return 0
  local IFS=, candidate
  for candidate in $ONLY; do
    [[ "$candidate" == "$tag" ]] && return 0
  done
  return 1
}

mkdir -p "$EXP/logs" "$RESULTS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
nvidia-smi -L || true

cd eval
srun uv sync || exit 1
cd "$SLURM_SUBMIT_DIR"

if [[ ! -s "$TEST" ]]; then
  echo "=== Rebuild the frozen Exp 12 split ==="
  POOL=data/spatial_sft_v12_21000_pool.jsonl
  if [[ ! -s "$POOL" ]]; then
    cd finetune
    srun --cpu-bind=cores uv run --no-project --with typer python \
      ../spatial/generate_all_v6.py \
      --out ../data/spatial_sft_v12_21000_pool.jsonl \
      --test-split 0 \
      --seed 52 \
      --system-prompt-file ../experiments/12-v6-mix-reweight/system_prompt.txt \
      --shuffle-special \
      --shuffle-none-phrase \
      --num-type0-1-answer 1167 \
      --num-type0-2-answer 1167 \
      --num-type0-undetermined 1166 \
      --num-type0-cycle 1166 \
      --num-type0-incomplete-pair 1166 \
      --num-type0-none 1167 \
      --num-type1-1-answer 1401 \
      --num-type1-2-answer 1401 \
      --num-type1-3-answer 1401 \
      --num-type1-4-answer 1399 \
      --num-type1-none 1399 \
      --num-type2 3500 \
      --num-type2-none 3500 || exit 1
    cd "$SLURM_SUBMIT_DIR"
    if [[ -s data/spatial_sft_v12_21000_pool_train.jsonl ]]; then
      mv data/spatial_sft_v12_21000_pool_train.jsonl "$POOL"
    fi
  fi
  uv run --no-project python experiments/12-v6-mix-reweight/scripts/make_v12_data.py || exit 1
fi
if [[ ! -s "$TEST" ]]; then
  echo "Missing $TEST after rebuild"
  exit 1
fi

selected=0
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 config=$2 adapter=$3
  want_tag "$tag" || continue
  selected=1
  if [[ "$adapter" != "-" ]] && ! has_adapter "$adapter"; then
    echo "Missing frozen adapter for $tag at $adapter"
    exit 1
  fi
  out=$RESULTS/v12test-$tag-stage1
  if [[ "${FORCE:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip $tag (already complete) ==="
    continue
  fi
  mkdir -p "$out"
  cd eval
  srun --cpu-bind=cores uv run python eval_new.py \
    --config "../$EXP/$config" \
    --stages 1 \
    --output-dir "../$out" || exit 1
  cd "$SLURM_SUBMIT_DIR"
done

if [[ "$selected" -eq 0 ]]; then
  echo "No cells selected (ONLY=${ONLY-})"
  exit 1
fi
uv run --no-project python "$EXP/scripts/summarize_v12_bridge.py" || exit 1
echo "Summary: $RESULTS/V12-STAGE1-BRIDGE.md"
