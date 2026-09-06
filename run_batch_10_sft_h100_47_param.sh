#!/bin/bash
# Exp 10 — 0.8B / 2B Full-mix body (1 GPU, --launcher python).
# Sourced by the 0.8B/2B 47 wrappers. Not DDP. Do not sbatch this file.
#
# Overrides: SKIP_TRAIN=1 SKIP_EVAL=1 ONLY=0.8b-1.5k,0.8b-5k
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/10-option-e-full
POOL=data/spatial_sft_full_scale_20000_train.jsonl
FULL=data/spatialeval_corr_full.jsonl
A_POOL=$SLURM_SUBMIT_DIR/$POOL
A_FULL=$SLURM_SUBMIT_DIR/$FULL
EVAL_OUT=$EXP/results/full
A_EVAL_OUT=$SLURM_SUBMIT_DIR/$EVAL_OUT
mkdir -p "$A_EVAL_OUT"

CELLS=(
  "0.8b-1.5k  train-sft-0.8b-1500.yaml   models/qwen3.5-0.8b-sft-full1500   eval-sft-0.8b-1500.yaml   1500"
  "0.8b-5k    train-sft-0.8b-5000.yaml   models/qwen3.5-0.8b-sft-full5000   eval-sft-0.8b-5000.yaml   5000"
  "0.8b-20k   train-sft-0.8b-20000.yaml  models/qwen3.5-0.8b-sft-full20000  eval-sft-0.8b-20000.yaml  20000"
  "2b-1.5k    train-sft-2b-1500.yaml     models/qwen3.5-2b-sft-full1500     eval-sft-2b-1500.yaml     1500"
  "2b-5k      train-sft-2b-5000.yaml     models/qwen3.5-2b-sft-full5000     eval-sft-2b-5000.yaml     5000"
  "2b-20k     train-sft-2b-20000.yaml    models/qwen3.5-2b-sft-full20000    eval-sft-2b-20000.yaml    20000"
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
echo "Selected cells (H100-47:1, 1-GPU python):"
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
  srun uv sync

  mkdir -p "$SLURM_SUBMIT_DIR/data"
  exec 9>"$SLURM_SUBMIT_DIR/data/.spatial_sft_full_scale.lock"
  flock 9

  if [[ -s "$A_FULL" ]]; then
    echo "=== Full jsonl already at $FULL ==="
  else
    echo "=== Build SpatialMap-TQA-Corr-Full ==="
    srun uv run python ../experiments/10-option-e-full/scripts/make_corr_full.py || exit 1
  fi

  if [[ -s "$A_POOL" ]]; then
    echo "=== [0] 20k Full-mix pool already at $POOL ==="
  else
    echo "=== [0] Generate 20000 Full-mix traces (seed 42) ==="
    srun uv run python generate_all.py \
      --out ../data/spatial_sft_full_scale_20000.jsonl \
      --test-split 0 \
      --seed 42 \
      --include-option-e \
      --num-type0-1-answer 4428 \
      --num-type0-2-answer 1240 \
      --num-type0-4-answer 0 \
      --num-type1-1-answer 4028 \
      --num-type1-2-answer 0 \
      --num-type1-4-answer 0 \
      --num-type2 5384 \
      --num-none-dir 1000 \
      --num-none-which 2640 \
      --num-none-count 1280 || {
      echo "SFT data generation FAILED — aborting."
      exit 1
    }
  fi
  if [[ ! -s "$A_POOL" ]]; then
    echo "Missing $A_POOL after generate; aborting"
    exit 1
  fi

  echo "=== [0] Nested 1.5k / 5k slices ==="
  srun uv run python ../experiments/10-option-e-full/scripts/make_full_scale_data.py || {
    echo "make_full_scale_data.py FAILED — aborting."
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
    train_one "$tag" "../experiments/10-option-e-full/$cfg" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  if [[ ! -s "$A_FULL" ]]; then
    echo "Missing $A_FULL — cannot eval"
    exit 1
  fi
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
    run_eval "$tag" "../experiments/10-option-e-full/$ev" "$SLURM_SUBMIT_DIR/$EXP/$adapter"
  done
fi

echo "=== Exp 10 param cells done."
echo "    cd eval && uv run --no-project python ../experiments/10-option-e-full/scripts/summarize.py"
