#!/bin/bash
# Exp 4b — same 4a Qwen3.5-4B smoke, Axolotl CLI instead of finetune.py.
# 2x H100-47 (MIG 3g.47gb).
#
#   1. axolotl train          SFT QLoRA     -> models/qwen3.5-4b-sft
#   2. axolotl merge-lora     bf16 merge    -> models/qwen3.5-4b-sft/merged
#   3. generate_grpo.py       ~300 prompts  -> ../spatial_grpo_data_smoke4b.jsonl
#   4. axolotl vllm-serve + axolotl train   -> models/qwen3.5-4b-grpo-smoke
#   5. eval_new.py base/sft/grpo            -> results/smoke16/{base,sft,grpo}/
#
# --launcher python: CLI default is accelerate, which would claim both MIG
# slices. SFT and GRPO train are one GPU each (GRPO: GPU0 serve, GPU1 train).
#
# Submit:  sbatch run_batch_4b_cli_h100.sh
#SBATCH --job-name=spatial4b-cli
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100-47:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/04b-cli
SFT_ADAPTER=$EXP/models/qwen3.5-4b-sft
MERGED=$SFT_ADAPTER/merged
GRPO_ADAPTER=$EXP/models/qwen3.5-4b-grpo-smoke
GRPO_DATA=spatial_grpo_data_smoke4b.jsonl
SMOKE_OUT=$EXP/results/smoke16
A_SFT_ADAPTER=$SLURM_SUBMIT_DIR/$SFT_ADAPTER
A_MERGED=$SLURM_SUBMIT_DIR/$MERGED
A_GRPO_ADAPTER=$SLURM_SUBMIT_DIR/$GRPO_ADAPTER
A_GRPO_DATA=$SLURM_SUBMIT_DIR/$GRPO_DATA
A_SMOKE_OUT=$SLURM_SUBMIT_DIR/$SMOKE_OUT

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

n_mig=$(nvidia-smi -L | grep -c 'MIG-' || true)
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${n_mig}" -lt 2 && "${n_gpu}" -lt 2 ]]; then
  echo "Need 2 GPUs, saw MIG=${n_mig} GPU=${n_gpu}"
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0,1 (vLLM cannot parse UUIDs)"
  export CUDA_VISIBLE_DEVICES=0,1
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  cd finetune
  srun uv sync --extra vllm

  CFG_SFT=../experiments/04b-cli/train-sft-4b.yaml
  CFG_GRPO=../experiments/04b-cli/train-grpo-4b-smoke.yaml

  if has_adapter "$A_SFT_ADAPTER"; then
    echo "=== [1/5] SFT adapter already complete, skipping ==="
  else
    echo "=== [1/5] SFT QLoRA on Qwen3.5-4B (axolotl train) ==="
    CUDA_VISIBLE_DEVICES=0 srun --cpu-bind=cores \
      uv run axolotl train "$CFG_SFT" --launcher python || {
      echo "SFT training FAILED — aborting."
      exit 1
    }
  fi
  if ! has_adapter "$A_SFT_ADAPTER"; then
    echo "SFT did not produce a complete adapter at $A_SFT_ADAPTER; aborting"
    exit 1
  fi

  if [[ -f "$A_MERGED/config.json" ]]; then
    echo "=== [2/5] Merged SFT weights already present, skipping ==="
  else
    if [[ -d "$A_MERGED" ]]; then
      echo "Removing partial merge output at $A_MERGED"
      find "$A_MERGED" -depth -delete 2>/dev/null || true
    fi
    echo "=== [2/5] Merge SFT LoRA (axolotl merge-lora -> $MERGED) ==="
    CUDA_VISIBLE_DEVICES="" srun uv run axolotl merge-lora "$CFG_SFT" || {
      echo "Merge FAILED — aborting. Delete $A_MERGED before re-running."
      exit 1
    }
  fi

  echo "=== [3/5] Generate GRPO prompt data (~300 rows, seeded) ==="
  if [[ -s "$A_GRPO_DATA" ]]; then
    echo "GRPO data already at $A_GRPO_DATA; skipping generate"
  else
    srun uv run python generate_grpo.py --n 300 --seed 42 --out "../$GRPO_DATA"
  fi
  srun uv run python generate_grpo.py --annotate --out "../$GRPO_DATA"

  echo "=== [4/5] GRPO 20 steps (GPU0 vllm-serve, GPU1 axolotl train) ==="
  CUDA_VISIBLE_DEVICES=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    uv run axolotl vllm-serve "$CFG_GRPO" &
  VLLM_PID=$!
  trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

  ok=0
  for _ in $(seq 1 90); do
    if curl -sfL "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1 \
      || curl -sfL "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      ok=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vllm-serve exited during startup — check the log, aborting"
      exit 1
    fi
    sleep 10
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "vLLM server did not become ready"
    exit 1
  fi

  RESUME=()
  if compgen -G "$A_GRPO_ADAPTER/checkpoint-*" > /dev/null; then
    latest=$(ls -d "$A_GRPO_ADAPTER"/checkpoint-* | sort -t- -k2 -n | tail -1)
    echo "Resuming GRPO from $latest"
    RESUME=(--resume-from-checkpoint "$latest")
  fi

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
  CUDA_VISIBLE_DEVICES=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    uv run axolotl train "$CFG_GRPO" --launcher python "${RESUME[@]}" || {
    echo "GRPO training FAILED — aborting."
    exit 1
  }
  kill "$VLLM_PID" 2>/dev/null || true
  cd ..
else
  echo "SKIP_TRAIN=1 — using existing adapters/models"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  cd eval
  srun uv sync

  grpo_lora=$A_GRPO_ADAPTER
  if ! has_adapter "$grpo_lora"; then
    latest=$(ls -d "$grpo_lora"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [[ -n "${latest:-}" ]] && has_adapter "$latest"; then
      echo "Final GRPO adapter missing; using $latest"
      grpo_lora=$latest
    else
      echo "WARN: no complete GRPO adapter at $grpo_lora — grpo eval skipped"
      grpo_lora=""
    fi
  fi

  run_eval() {
    local tag=$1 cfg=$2 lora=$3
    local cfg_to_use=$cfg
    if [[ -n "$lora" && "$lora" != "$A_GRPO_ADAPTER" ]]; then
      cfg_to_use=$(mktemp "../$EXP/eval-grpo-4b.XXXX.yaml")
      sed "s|lora_path:.*|lora_path: ../${lora#$SLURM_SUBMIT_DIR/}|" "$cfg" > "$cfg_to_use"
      trap "rm -f '$cfg_to_use'" EXIT
    fi
    echo "=== [5/5] Eval $tag ($cfg_to_use) ==="
    local status=0
    CUDA_VISIBLE_DEVICES=0 srun --cpu-bind=cores uv run python eval_new.py --config "$cfg_to_use" || status=$?
    local latest
    latest=$(ls -dt "$SLURM_SUBMIT_DIR/$EXP"/results/2*/ 2>/dev/null | head -1 || true)
    mkdir -p "$A_SMOKE_OUT/$tag"
    if [[ -n "${latest:-}" && -f "${latest}results.json" ]]; then
      mv "${latest}"* "$A_SMOKE_OUT/$tag/"
      rmdir "${latest}" 2>/dev/null || true
      echo "  -> $A_SMOKE_OUT/$tag/"
    fi
    [[ $status -ne 0 ]] && echo "  eval $tag FAILED (status $status)" || echo "  eval $tag OK"
  }

  run_eval base  ../experiments/04b-cli/eval-base-4b.yaml  ""
  run_eval sft   ../experiments/04b-cli/eval-sft-4b.yaml   ""
  if [[ -n "${grpo_lora:-}" ]]; then
    run_eval grpo  ../experiments/04b-cli/eval-grpo-4b.yaml "$grpo_lora"
  fi

  srun uv run --no-project python ../experiments/04b-cli/scripts/summarize.py \
    || echo "summarize failed — check $SMOKE_OUT manually"
fi

echo "=== Exp 4b CLI pipeline done. Summary: $A_SMOKE_OUT/SUMMARY.md ==="
