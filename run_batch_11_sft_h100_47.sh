#!/bin/bash
# Exp 11 — v6 synthetic SFT body (1 GPU, --launcher python).
# Sourced by the three h100-47 wrappers. Do not sbatch this file.
#
#   0. generate_all_v6 20k train + 4k test, nested 1.5k/6k, SpatialMap v6
#   1. axolotl train
#   2. eval_new.py on synthetic 20% + SpatialMap v6
#
# Overrides: SKIP_TRAIN=1 SKIP_EVAL=1 ONLY=2b-1.5k,2b-6k
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# #SBATCH --cpus-per-task and SLURM_CPUS_PER_TASK must never differ (Slurm 23+).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm_pin_srun_cpus.sh"

EXP=experiments/11-v6-synthetic
POOL=data/spatial_sft_v6_scale_20000_train.jsonl
TEST=data/spatial_sft_v6_scale_test.jsonl
SMAP=data/spatialeval_v6_corr.jsonl
A_POOL=$SLURM_SUBMIT_DIR/$POOL
A_TEST=$SLURM_SUBMIT_DIR/$TEST
A_SMAP=$SLURM_SUBMIT_DIR/$SMAP
EVAL_OUT=$EXP/results
A_EVAL_OUT=$SLURM_SUBMIT_DIR/$EVAL_OUT
mkdir -p "$A_EVAL_OUT"

# tag  train_yaml  adapter_rel  eval_yaml  n_train
CELLS=(
  "2b-1.5k  train-sft-2b-1500.yaml   models/qwen3.5-2b-sft-v6-1500    eval-sft-2b-1500.yaml   1500"
  "2b-6k    train-sft-2b-6000.yaml   models/qwen3.5-2b-sft-v6-6000    eval-sft-2b-6000.yaml   6000"
  "2b-20k   train-sft-2b-20000.yaml  models/qwen3.5-2b-sft-v6-20000   eval-sft-2b-20000.yaml  20000"
  "4b-1.5k  train-sft-4b-1500.yaml   models/qwen3.5-4b-sft-v6-1500    eval-sft-4b-1500.yaml   1500"
  "4b-6k    train-sft-4b-6000.yaml   models/qwen3.5-4b-sft-v6-6000    eval-sft-4b-6000.yaml   6000"
  "4b-20k   train-sft-4b-20000.yaml  models/qwen3.5-4b-sft-v6-20000   eval-sft-4b-20000.yaml  20000"
)

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

want_tag() {
  local tag=$1
  if [[ -z "${ONLY:-}" ]]; then
    return 0
  fi
  local IFS=,
  local t
  for t in $ONLY; do
    [[ "$t" == "$tag" ]] && return 0
  done
  return 1
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
  echo "No cells selected (ONLY=${ONLY-})"
  exit 1
fi
echo "Selected cells (H100-47:1, 1-GPU python, cpus-per-task pinned):"
printf '  %s\n' "${selected[@]}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0"
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  export PYTHONPATH="$SLURM_SUBMIT_DIR/finetune${PYTHONPATH:+:$PYTHONPATH}"
  uv sync || exit 1

  mkdir -p "$SLURM_SUBMIT_DIR/data"
  exec 9>"$SLURM_SUBMIT_DIR/data/.spatial_sft_v6_scale.lock"
  flock 9

  if [[ -s "$A_SMAP" ]]; then
    echo "=== SpatialMap v6 already at $SMAP ==="
  else
    echo "=== Build SpatialMap v6 corr ==="
    uv run --no-project --with typer python \
      ../experiments/11-v6-synthetic/scripts/make_spatialmap_v6.py || exit 1
  fi

  if [[ -s "$A_POOL" ]]; then
    echo "=== [0] 20k v6 train pool already at $POOL ==="
  else
    echo "=== [0] Generate 20000 v6 traces (seed 42, no test split) ==="
    uv run --no-project --with typer python generate_all_v6.py \
      --out ../data/spatial_sft_v6_scale_20000.jsonl \
      --test-split 0 \
      --seed 42 \
      --shuffle-special \
      --shuffle-none-phrase \
      --num-type0-1-answer 2000 \
      --num-type0-2-answer 1667 \
      --num-type0-undetermined 1000 \
      --num-type0-cycle 667 \
      --num-type0-incomplete-pair 667 \
      --num-type0-none 666 \
      --num-type1-1-answer 2000 \
      --num-type1-2-answer 1667 \
      --num-type1-3-answer 1000 \
      --num-type1-4-answer 667 \
      --num-type1-none 1333 \
      --num-type2 5333 \
      --num-type2-none 1333 || {
      echo "SFT data generation FAILED — aborting."
      exit 1
    }
    if [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v6_scale_20000_train.jsonl" ]]; then
      : # generate_all_v6 with test-split 0 still writes _train suffix
    elif [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v6_scale_20000.jsonl" ]]; then
      mv "$SLURM_SUBMIT_DIR/data/spatial_sft_v6_scale_20000.jsonl" "$A_POOL"
    fi
  fi
  if [[ ! -s "$A_POOL" ]]; then
    echo "Missing $A_POOL after generate; aborting"
    exit 1
  fi

  if [[ -s "$A_TEST" ]]; then
    echo "=== [0] v6 test split already at $TEST ==="
  else
    echo "=== [0] Generate 4000 v6 test traces (seed 43) ==="
    uv run --no-project --with typer python generate_all_v6.py \
      --out ../data/spatial_sft_v6_scale_test.jsonl \
      --test-split 0 \
      --seed 43 \
      --shuffle-special \
      --shuffle-none-phrase \
      --num-type0-1-answer 400 \
      --num-type0-2-answer 333 \
      --num-type0-undetermined 200 \
      --num-type0-cycle 133 \
      --num-type0-incomplete-pair 133 \
      --num-type0-none 134 \
      --num-type1-1-answer 400 \
      --num-type1-2-answer 333 \
      --num-type1-3-answer 200 \
      --num-type1-4-answer 133 \
      --num-type1-none 267 \
      --num-type2 1067 \
      --num-type2-none 267 || {
      echo "test split generation FAILED — aborting."
      exit 1
    }
    if [[ -s "$SLURM_SUBMIT_DIR/data/spatial_sft_v6_scale_test_train.jsonl" ]]; then
      mv "$SLURM_SUBMIT_DIR/data/spatial_sft_v6_scale_test_train.jsonl" "$A_TEST"
    fi
  fi

  echo "=== [0] Nested 1.5k / 6k slices ==="
  uv run --no-project python ../experiments/11-v6-synthetic/scripts/make_v6_scale_data.py || {
    echo "make_v6_scale_data.py FAILED — aborting."
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
    uv run axolotl train "$cfg" --launcher python || {
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
    train_one "$tag" "../experiments/11-v6-synthetic/$cfg" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  if [[ ! -s "$A_TEST" ]]; then
    echo "Missing $A_TEST — cannot eval synth"
    exit 1
  fi
  if [[ ! -s "$A_SMAP" ]]; then
    echo "Missing $A_SMAP — cannot eval SpatialMap v6"
    exit 1
  fi
  cd eval
  uv sync || exit 1

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
    uv run python eval_new.py \
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
    run_eval "$tag" "../experiments/11-v6-synthetic/$ev" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
  uv run --no-project python experiments/11-v6-synthetic/scripts/summarize.py || true
fi

echo "=== Exp 11 cells done. After all three 47 jobs finish:"
echo "    cd eval && uv run --no-project python ../experiments/11-v6-synthetic/scripts/summarize.py"
