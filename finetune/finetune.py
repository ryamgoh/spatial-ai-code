"""
Finetune a model using Axolotl API.

Usage:
    cd finetune && uv run python finetune.py <config_name> [--resume]

Example:
    cd finetune && uv run python finetune.py qwen3-7b-lora
    cd finetune && uv run python finetune.py qwen3-7b-lora --resume
"""

import os

# Axolotl 0.13.2 ships without telemetry/whitelist.yaml. If tracking is left
# at the default (on), import crashes on that missing file after a 10s sleep.
os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
os.environ.setdefault("AXOLOTL_NO_TELEMETRY", "1")

import argparse
import signal
import sys
from pathlib import Path

import yaml
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.train import setup_signal_handler, train
from axolotl.utils.dict import DictDefault
from axolotl.utils import set_pytorch_cuda_alloc_conf


def _patch_grpo_vllm_max_model_length() -> None:
    """Axolotl 0.13.2 never forwards vllm.max_model_len to TRL colocate.

    TRL then infers the native R1/Qwen context (131072) and vLLM demands ~18GiB
    of KV cache. Pass yaml `vllm.max_model_len` through as vllm_max_model_length.
    """
    original = GRPOStrategy.set_training_args_kwargs

    @classmethod
    def patched(cls, cfg):
        kwargs = original.__func__(cls, cfg)
        max_len = None
        if cfg.vllm is not None:
            max_len = getattr(cfg.vllm, "max_model_len", None)
        if max_len:
            kwargs["vllm_max_model_length"] = int(max_len)
        return kwargs

    GRPOStrategy.set_training_args_kwargs = patched


def _patch_vllm_bnb_weight_reload() -> None:
    """vLLM 0.11.0 bitsandbytes reload_weights asserts on existing bnb_quant_state.

    TRL colocate QLoRA loads the engine once, then collective_rpc("reload_weights")
    on the first generate. Allow attribute overwrite so LoRA sync can proceed.
    """
    try:
        import vllm.model_executor.model_loader.bitsandbytes_loader as bnb_loader
    except ImportError:
        return

    def set_weight_attrs(weight, attrs):
        for key, val in attrs.items():
            setattr(weight, key, val)

    bnb_loader.set_weight_attrs = set_weight_attrs


CONFIGS_DIR = Path(__file__).parent / "config"


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model using Axolotl")
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the config file (without .yaml extension)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint"
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> DictDefault:
    """Load a YAML config file into a DictDefault for Axolotl."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DictDefault(config_dict)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint in the output directory."""
    if not output_dir.exists():
        return None

    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return checkpoints[-1]


def main():
    args = parse_args()

    config_path = CONFIGS_DIR / f"{args.config_name}.yaml"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        print(f"Available configs in {CONFIGS_DIR}:")
        for f in CONFIGS_DIR.glob("*.yaml"):
            print(f"  - {f.stem}")
        sys.exit(1)

    print(f"Loading config: {config_path}")

    config = load_yaml_config(config_path)
    cfg = load_cfg(config)
    if cfg.rl and cfg.trl and cfg.trl.use_vllm:
        _patch_grpo_vllm_max_model_length()

    use_vllm = bool(cfg.trl and cfg.trl.use_vllm)
    # vLLM colocate sleep/memory pool cannot use expandable_segments.
    if not (cfg.rl and use_vllm):
        set_pytorch_cuda_alloc_conf()

    if args.resume:
        checkpoint = find_latest_checkpoint(Path(cfg.output_dir))
        if checkpoint:
            cfg.resume_from_checkpoint = str(checkpoint)
            print(f"Resuming from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from lora_model_dir / base")

    print("Loading datasets...")
    if cfg.rl:
        dataset_meta = load_preference_datasets(cfg=cfg)
        print(f"RL dataset path ({cfg.rl}): preference/prompt loader")
    else:
        dataset_meta = load_datasets(cfg=cfg)

    print(f"Starting training: {args.config_name}")
    print(f"Output directory: {cfg.output_dir}")
    if cfg.resume_from_checkpoint:
        print(f"Resuming from: {cfg.resume_from_checkpoint}")
    print("-" * 50)

    model, tokenizer, trainer = train(
        cfg=cfg,
        dataset_meta=dataset_meta,
    )

    print("-" * 50)
    print(f"Training complete! Model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
