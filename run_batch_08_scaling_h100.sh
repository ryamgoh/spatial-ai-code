#!/bin/bash
# Exp 8 — Dual Scaling Laws.
# Instruct Qwen3.5 only; Single-answer traces; eval TQA-Corr-Single.
#
#   0. generate_all.py  20k single-answer  -> data/spatial_sft_single_scale_20000_train.jsonl
#      make_scale_data.py nested 5k/1.5k   -> data/spatial_sft_single_scale_{5000,1500}_train.jsonl
#   1. axolotl train    4B×{1.5k,5k,20k} + {2B,0.8B}×20k
#   2. eval_new.py      all five on TQA-Corr-Single
#   3. summarize.py
#
# Two-GPU split (preferred):
#   sbatch run_batch_08_scaling_h100_96.sh   # 4B × {1.5k, 5k, 20k}
#   sbatch run_batch_08_scaling_h100_47.sh   # 0.8B-20k + 2B-20k
# Sequential fallback (this file): all five cells on one h100-47.
# Overrides: SKIP_TRAIN=1 SKIP_EVAL=1 SKIP_2_1=1 SKIP_2_2=1 ONLY=4b-5k,2b-20k
#SBATCH --job-name=spatial8-scale
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/08-dual-scaling
POOL=data/spatial_sft_single_scale_20000_train.jsonl
A_POOL=$SLURM_SUBMIT_DIR/$POOL
EVAL_OUT=$EXP/results/scaling
A_EVAL_OUT=$SLURM_SUBMIT_DIR/$EVAL_OUT
mkdir -p "$A_EVAL_OUT"

# tag  train_yaml  adapter_rel  eval_yaml  n_train
CELLS=(
  "4b-1.5k  train-sft-4b-1500.yaml   models/qwen3.5-4b-sft-1500    eval-sft-4b-1500.yaml   1500"
  "4b-5k    train-sft-4b-5000.yaml   models/qwen3.5-4b-sft-5000    eval-sft-4b-5000.yaml   5000"
  "4b-20k   train-sft-4b-20000.yaml  models/qwen3.5-4b-sft-20000   eval-sft-4b-20000.yaml  20000"
  "0.8b-20k train-sft-0.8b-20000.yaml models/qwen3.5-0.8b-sft-20000 eval-sft-0.8b-20000.yaml 20000"
  "2b-20k   train-sft-2b-20000.yaml  models/qwen3.5-2b-sft-20000   eval-sft-2b-20000.yaml  20000"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  if [[ -n "${ONLY:-}" ]]; then
    local IFS=,
    local t
    for t in $ONLY; do
      [[ "$t" == "$tag" ]] && return 0
    done
    return 1
  fi
  case "$tag" in
    4b-1.5k|4b-5k)
      [[ "${SKIP_2_1:-0}" == "1" ]] && return 1
      ;;
    0.8b-20k|2b-20k)
      [[ "${SKIP_2_2:-0}" == "1" ]] && return 1
      ;;
  esac
  return 0
}

selected=()
for row in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  tag=$1 cfg=$2 adapter=$3 ev=$4 n=$5
  if want_tag "$tag"; then
    selected+=("$tag $cfg $adapter $ev $n")
  fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No cells selected (ONLY=${ONLY-} SKIP_2_1=${SKIP_2_1-} SKIP_2_2=${SKIP_2_2-})"
  exit 1
fi
echo "Selected cells:"
printf '  %s\n' "${selected[@]}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  export PYTHONPATH="$SLURM_SUBMIT_DIR/finetune${PYTHONPATH:+:$PYTHONPATH}"
  srun uv sync

  # flock: 96 and 47 jobs may both hit this. Winner generates; loser waits
  # and reuses. Nested 1.5k/5k slices are deterministic from the 20k pool.
  mkdir -p "$SLURM_SUBMIT_DIR/data"
  exec 9>"$SLURM_SUBMIT_DIR/data/.spatial_sft_single_scale.lock"
  flock 9
  if [[ -s "$A_POOL" ]]; then
    echo "=== [0] 20k pool already at $POOL ==="
  else
    echo "=== [0] Generate 20000 single-answer traces (seed 42) ==="
    srun uv run python generate_all.py \
      --out ../data/spatial_sft_single_scale_20000.jsonl \
      --test-split 0 \
      --seed 42 \
      --num-type0-1-answer 6397 \
      --num-type0-2-answer 0 \
      --num-type0-4-answer 0 \
      --num-type1-1-answer 5819 \
      --num-type1-2-answer 0 \
      --num-type2 7784 || {
      echo "SFT data generation FAILED — aborting."
      exit 1
    }
  fi
  if [[ ! -s "$A_POOL" ]]; then
    echo "Missing $A_POOL after generate; aborting"
    exit 1
  fi

  echo "=== [0] Nested 1.5k / 5k slices ==="
  srun uv run python ../experiments/08-dual-scaling/scripts/make_scale_data.py || {
    echo "make_scale_data.py FAILED — aborting."
    exit 1
  }
  flock -u 9

  train_one() {
    local tag=$1 cfg=$2 dest=$3
    if has_adapter "$dest"; then
      echo "=== skip SFT $tag (adapter complete) ==="
      return 0
    fi
    echo "=== SFT QLoRA $tag ==="
    srun --cpu-bind=cores uv run axolotl train "$cfg" --launcher python || {
      echo "SFT $tag FAILED — aborting."
      exit 1
    }
    if ! has_adapter "$dest"; then
      echo "SFT $tag did not produce a complete adapter at $dest; aborting"
      exit 1
    fi
  }

  for row in "${selected[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    tag=$1 cfg=$2 adapter=$3
    train_one "$tag" "../experiments/08-dual-scaling/$cfg" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  srun uv sync

  run_eval() {
    local tag=$1 cfg=$2 adapter=$3
    if [[ -n "$adapter" ]] && ! has_adapter "$adapter"; then
      echo "WARN: no adapter at $adapter — skip eval $tag"
      return 0
    fi
    if [[ -f "$A_EVAL_OUT/$tag/results.json" ]]; then
      echo "=== skip eval $tag (already have results.json) ==="
      return 0
    fi
    echo "=== eval $tag ($cfg) ==="
    mkdir -p "$A_EVAL_OUT/$tag"
    local status=0
    srun --cpu-bind=cores uv run python eval_new.py \
      --config "$cfg" \
      --output-dir "$A_EVAL_OUT/$tag" || status=$?
    if [[ $status -ne 0 ]]; then
      echo "  eval $tag FAILED (status $status)"
      return "$status"
    fi
    echo "  eval $tag OK -> $A_EVAL_OUT/$tag/"
  }

  for row in "${selected[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    tag=$1 cfg=$2 adapter=$3 ev=$4
    run_eval "$tag" "../experiments/08-dual-scaling/$ev" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
fi

echo "=== Exp 8 cells done. After both GPU jobs finish:"
echo "    cd eval && uv run --no-project python ../experiments/08-dual-scaling/scripts/summarize.py"
