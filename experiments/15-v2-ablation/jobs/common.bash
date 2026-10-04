#!/bin/bash

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
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
