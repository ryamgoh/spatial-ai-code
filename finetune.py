"""
Finetune a model using Axolotl API.
Usage:
    uv run finetune <config_name>
Example:
    uv run finetune qwen3-7b-lora
This will load configs/models/qwen3-7b-lora.yaml and run Axolotl training.
"""

import sys
from pathlib import Path
import yaml
from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.utils import set_pytorch_cuda_alloc_conf
from axolotl.common.datasets import load_datasets
from axolotl.train import train

CONFIGS_DIR = Path("configs/models")


def load_yaml_config(config_path: Path) -> DictDefault:
    """Load a YAML config file into a DictDefault for Axolotl."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DictDefault(config_dict)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run finetune <config_name>")
        print("Example: uv run finetune qwen3-7b-lora")
        sys.exit(1)

    config_name = sys.argv[1]
    config_path = CONFIGS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        print(f"Available configs in {CONFIGS_DIR}:")
        for f in CONFIGS_DIR.glob("*.yaml"):
            print(f"  - {f.stem}")
        sys.exit(1)

    print(f"Loading config: {config_path}")

    # Load YAML config into DictDefault
    config = load_yaml_config(config_path)

    # Validate the configuration
    cfg = load_cfg(config)

    # Set PyTorch CUDA allocation config
    set_pytorch_cuda_alloc_conf()

    # Load, parse and tokenize the datasets
    print("Loading datasets...")
    dataset_meta = load_datasets(cfg=cfg)

    print(f"Starting training: {config_name}")
    print(f"Output directory: {cfg.output_dir}")
    print("-" * 50)

    # Run training
    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

    print("-" * 50)
    print(f"Training complete! Model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
