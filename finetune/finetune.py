"""
Finetune a model using Axolotl API.

Usage:
    cd finetune && uv run python finetune.py <config_name> [--resume]

Example:
    cd finetune && uv run python finetune.py qwen3-7b-lora
    cd finetune && uv run python finetune.py qwen3-7b-lora --resume
"""

import argparse
import signal
import sys
from pathlib import Path

import yaml
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import setup_signal_handler, train
from axolotl.utils.dict import DictDefault
from axolotl.utils import set_pytorch_cuda_alloc_conf

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "models"


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

    set_pytorch_cuda_alloc_conf()

    print("Loading datasets...")
    dataset_meta = load_datasets(cfg=cfg)

    resume_from_checkpoint = None
    if args.resume:
        output_dir = Path(cfg.output_dir)
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            resume_from_checkpoint = str(checkpoint)
            print(f"Resuming from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting fresh training")

    print(f"Starting training: {args.config_name}")
    print(f"Output directory: {cfg.output_dir}")
    if resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")
    print("-" * 50)

    model, tokenizer, trainer = train(
        cfg=cfg,
        dataset_meta=dataset_meta,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    print("-" * 50)
    print(f"Training complete! Model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
