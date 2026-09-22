#!/bin/bash
# V14 GRPO feasibility: GPU0 policy training, GPU1 vLLM rollout server.
# Calibration runs first and aborts before training if rewards are degenerate.
#SBATCH --job-name=spatial14-grpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:2
#SBATCH --output=experiments/14-v13-grpo-feasibility/logs/%x-%j.out
#SBATCH --error=experiments/14-v13-grpo-feasibility/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/14-v13-grpo-feasibility
SFT_ADAPTER=experiments/13-iterative-hardening/models/qwen3.5-4b-v13-sft-8000-delta
MERGED=$EXP/models/qwen3.5-4b-delta-merged
GRPO_ADAPTER=$EXP/models/qwen3.5-4b-grpo-probe
TRAIN=data/spatial_v14_grpo_pool_train.jsonl
HOLDOUT=data/spatial_v14_grpo_pool_eval.jsonl
MANIFEST=data/spatial_v14_grpo_pool_manifest.json
CALIBRATION=$EXP/results/CALIBRATION.json
CFG=../$EXP/train-grpo-4b-probe.yaml
TRAIN_DEV=0
VLLM_DEV=1

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

mkdir -p "$EXP/logs" "$EXP/results" "$EXP/models" data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
fi

cd finetune
srun uv sync || exit 1
cd "$SLURM_SUBMIT_DIR"
if [[ $(CUDA_VISIBLE_DEVICES=0,1 uv run python -c 'import torch; print(torch.cuda.device_count())') -lt 2 ]]; then
  echo "Need two distinct H200 devices"
  exit 1
fi
if ! has_adapter "$SFT_ADAPTER"; then
  echo "Missing delta-state SFT adapter: $SFT_ADAPTER"
  exit 1
fi

if [[ ! -s "$TRAIN" || ! -s "$HOLDOUT" || ! -s "$MANIFEST" ]]; then
  uv run --no-project python "$EXP/scripts/make_rl_pool.py" || exit 1
fi
uv run --no-project python "$EXP/scripts/make_rl_pool.py" --validate-only || exit 1

if [[ ! -f "$MERGED/config.json" ]]; then
  echo "=== merge delta-state SFT QLoRA into BF16 policy base ==="
  cd finetune
  CUDA_VISIBLE_DEVICES="" uv run python merge_sft.py \
    --base Qwen/Qwen3.5-4B \
    --adapter ../"$SFT_ADAPTER" \
    --out ../"$MERGED" || exit 1
  cd "$SLURM_SUBMIT_DIR"
fi

if [[ "${SKIP_CALIBRATION:-0}" != "1" ]]; then
  echo "=== calibrate four-rollout reward variance on delta SFT ==="
  set +e
  CUDA_VISIBLE_DEVICES=$VLLM_DEV uv run python \
    "$EXP/scripts/calibrate_rollouts.py" \
    --data "$TRAIN" --model "$MERGED" --out "$CALIBRATION"
  calibration_status=$?
  set -e
  if [[ $calibration_status -eq 3 ]]; then
    echo "Calibration gate failed; inspect $CALIBRATION. GRPO was not started."
    exit 3
  elif [[ $calibration_status -ne 0 ]]; then
    exit $calibration_status
  fi
elif [[ ! -f "$CALIBRATION" ]]; then
  echo "SKIP_CALIBRATION=1 requires an existing $CALIBRATION"
  exit 1
fi

if [[ "${CALIBRATE_ONLY:-0}" == "1" ]]; then
  echo "Calibration-only run complete: $CALIBRATION"
  exit 0
fi

cd finetune
CUDA_VISIBLE_DEVICES=$VLLM_DEV \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run axolotl vllm-serve "$CFG" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 90); do
  if curl -sfL http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
    || curl -sfL http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ok=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server exited during startup"
    exit 1
  fi
  sleep 10
done
[[ $ok -eq 1 ]] || { echo "vLLM server did not become ready"; exit 1; }

resume=()
if compgen -G "../$GRPO_ADAPTER/checkpoint-*" >/dev/null; then
  resume=(--resume)
fi
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
CUDA_VISIBLE_DEVICES=$TRAIN_DEV uv run python finetune.py "$CFG" "${resume[@]}" || exit 1
kill "$VLLM_PID" 2>/dev/null || true
trap - EXIT
cd "$SLURM_SUBMIT_DIR"
if ! has_adapter "$GRPO_ADAPTER"; then
  echo "GRPO did not produce a complete adapter at $GRPO_ADAPTER"
  exit 1
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  uv sync || exit 1
  for row in \
    "sft-holdout-stage1 eval-sft-holdout.yaml" \
    "grpo-holdout-stage1 eval-grpo-holdout.yaml"; do
    set -- $row
    tag=$1 config=$2 out=../$EXP/results/$1
    [[ -f "$out/results.json" && "${FORCE_EVAL:-0}" != "1" ]] && continue
    CUDA_VISIBLE_DEVICES=$TRAIN_DEV uv run python eval_new.py \
      --config ../$EXP/"$config" --stages 1 --output-dir "$out" || exit 1
  done
  if [[ "${RUN_RETENTION:-0}" == "1" ]]; then
    for row in \
      "grpo-v13-stage1 eval-grpo-v13.yaml" \
      "grpo-breakpoint-stage1 eval-grpo-breakpoint.yaml"; do
      set -- $row
      tag=$1 config=$2 out=../$EXP/results/$1
      [[ -f "$out/results.json" && "${FORCE_EVAL:-0}" != "1" ]] && continue
      CUDA_VISIBLE_DEVICES=$TRAIN_DEV uv run python eval_new.py \
        --config ../$EXP/"$config" --stages 1 --output-dir "$out" || exit 1
    done
  fi
  cd "$SLURM_SUBMIT_DIR"
fi

uv run --no-project python "$EXP/scripts/summarize.py" || true
echo "V14 summary: $EXP/results/SUMMARY.md"
