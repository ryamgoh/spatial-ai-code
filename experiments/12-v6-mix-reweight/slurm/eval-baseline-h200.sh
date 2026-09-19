#!/bin/bash
# Exp 12 — BASELINE evals: untuned Qwen3.5-4B (no LoRA), no training.
# Prompt family in experiments/prompts/ (round 5 = Exp 12 letter-rules prompt;
# nonshot_5.txt is canonical, its inline copies checked by
# experiments/prompts/check_prompt_sync.py).
#   baseline           — zero-shot, letter-rules prompt nonshot_5 (same as SFT
#                        cells; only the adapter differs, isolates SFT gain).
#   baseline-oneshot   — nonshot_5 + 1 demo (dir-2, from 1k val).
#   baseline-threeshot — nonshot_5 + 3 demos (dir-2, dir-cycle, count-omit).
# All: matched 2k holdout + SpatialMap v6, two-stage protocol.
#
#   sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh
#   ONLY=baseline,baseline-oneshot sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh
#   FORCE=1 sbatch experiments/12-v6-mix-reweight/slurm/eval-baseline-h200.sh
#SBATCH --job-name=spatial12-baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=experiments/12-v6-mix-reweight/logs/%x-%j.out
#SBATCH --error=experiments/12-v6-mix-reweight/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/12-v6-mix-reweight
EVAL_OUT=$SLURM_SUBMIT_DIR/$EXP/results
PROMPT=$SLURM_SUBMIT_DIR/$EXP/system_prompt.txt
POOL=$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool.jsonl
SYNTH_TEST=$SLURM_SUBMIT_DIR/data/spatial_sft_v12_2000_test.jsonl
VAL=$SLURM_SUBMIT_DIR/data/spatial_sft_v12_1000_val.jsonl
SMAP=$SLURM_SUBMIT_DIR/data/spatialeval_v6_corr.jsonl

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi
echo "FORCE=${FORCE-0}"
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

if [[ ! -s "$SMAP" ]]; then
  echo "=== Build SpatialMap v6 corr ==="
  cd finetune
  uv run --no-project --with typer python \
    ../experiments/11-v6-synthetic/scripts/make_spatialmap_v6.py || exit 1
  cd "$SLURM_SUBMIT_DIR"
fi

if [[ ! -s "$POOL" ]]; then
  echo "=== Generate 21000 v12 traces (seed 52, hierarchical uniform) ==="
  cd finetune
  mkdir -p "$SLURM_SUBMIT_DIR/data"
  exec 9>"$SLURM_SUBMIT_DIR/data/.spatial_sft_v12.lock"
  flock 9
  uv run --no-project --with typer python ../spatial/generate_all_v6.py \
    --out ../data/spatial_sft_v12_21000_pool.jsonl \
    --test-split 0 \
    --seed 52 \
    --system-prompt-file "../$PROMPT" \
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
    --num-type2-none 3500 || {
    echo "v12 pool generation FAILED — aborting."
    exit 1
  }
  flock -u 9
  cd "$SLURM_SUBMIT_DIR"
  if [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool_train.jsonl" ]]; then
    mv "$SLURM_SUBMIT_DIR/data/spatial_sft_v12_21000_pool_train.jsonl" "$POOL"
  fi
fi
if [[ ! -s "$POOL" ]]; then
  echo "Missing $POOL after generation; aborting"
  exit 1
fi

if [[ ! -s "$SYNTH_TEST" ]]; then
  echo "=== Slice 2k test / 1k val / train from pool ==="
  uv run --no-project python "$EXP/scripts/make_v12_data.py" || exit 1
fi

if [[ ! -s "$SMAP" ]]; then
  echo "Missing $SMAP"
  exit 1
fi
if [[ ! -s "$SYNTH_TEST" ]]; then
  echo "Missing $SYNTH_TEST (Exp 12 matched 2k holdout)"
  exit 1
fi

# tag  eval_yaml
CELLS=(
  "baseline           eval-baseline-4b.yaml"
  "baseline-oneshot   eval-baseline-4b-oneshot.yaml"
  "baseline-threeshot eval-baseline-4b-threeshot.yaml"
)

want_tag() {
  local tag=$1
  if [[ -n "${ONLY:-}" ]]; then
    local IFS=, t
    for t in $ONLY; do
      [[ "$t" == "$tag" ]] && return 0
    done
    return 1
  fi
  return 0
}

cd eval
srun uv sync
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 ev=$2
  want_tag "$tag" || continue
  out=$EVAL_OUT/$tag
  if [[ "${FORCE:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip eval $tag (already have results.json) ==="
    continue
  fi
  echo "=== eval $tag (untuned 4B, $ev) ==="
  mkdir -p "$out"
  srun --cpu-bind=cores uv run python eval_new.py \
    --config "../$EXP/$ev" \
    --output-dir "$out" || {
    echo "Exp 12 baseline eval $tag FAILED"
    exit 1
  }
  echo "  eval $tag OK -> $out/"
done
cd "$SLURM_SUBMIT_DIR"

uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py || true
echo "    uv run --no-project python experiments/12-v6-mix-reweight/scripts/summarize.py"
