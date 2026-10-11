#!/bin/bash

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

load_wandb_credentials() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return
  fi
  local config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
  local credentials="${WANDB_ENV_FILE:-$config_home/spatialentail/wandb.env}"
  if [[ -r "$credentials" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$credentials"
    set +a
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Missing WANDB_API_KEY; create $credentials with mode 600" >&2
    return 1
  fi
}

normalize_single_cuda_device() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
    export CUDA_VISIBLE_DEVICES=0
  fi
}

sync_uv_project() {
  local project=$1
  exec 9>"$project/.uv-sync.lock"
  flock 9
  (
    cd "$project"
    uv sync
  )
  flock -u 9
}
