"""
Finetune a model using Axolotl API.

Usage:
    cd finetune && uv run python finetune.py <config> [--resume]

Config may be a path to a yaml or a bare name searched recursively under
experiments/.

Example:
    cd finetune && uv run python finetune.py ../experiments/03-sft-vs-baseline/train-sft-8b.yaml
    cd finetune && uv run python finetune.py train-grpo-8b-vllm-h100 --resume
"""

import os

# Axolotl 0.13.2 ships without telemetry/whitelist.yaml. If tracking is left
# at the default (on), import crashes on that missing file after a 10s sleep.
os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
os.environ.setdefault("AXOLOTL_NO_TELEMETRY", "1")


def _patch_torch_single_gpu_index() -> None:
    """CUDA_VISIBLE_DEVICES=N (N!=0) remaps to one GPU, but Torch dynamo
    still indexes properties[N]. That IndexError is raised from has_triton
    during `import transformers` / torchao, before training starts.
    """
    import torch
    from torch._dynamo import device_interface
    from torch.utils import _triton

    orig_has_triton = _triton.has_triton

    def has_triton():
        try:
            return orig_has_triton()
        except IndexError:
            if not torch.cuda.is_available():
                return False
            return torch.cuda.get_device_capability(0)[0] >= 7

    _triton.has_triton = has_triton

    worker = device_interface.CudaInterface.Worker
    orig_props = worker.get_device_properties

    def get_device_properties(device=None):
        try:
            return orig_props(device)
        except IndexError:
            props = device_interface.caching_worker_device_properties.get("cuda") or []
            if not props:
                return torch.cuda.get_device_properties(0)
            return props[0]

    worker.get_device_properties = get_device_properties


_patch_torch_single_gpu_index()

import sys
from pathlib import Path

import typer
import yaml
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.train import train
from axolotl.utils.dict import DictDefault

try:
    from axolotl.utils import set_pytorch_cuda_alloc_conf
except ImportError:

    def set_pytorch_cuda_alloc_conf():
        return


def _ensure_fsdp_flag(sync_fn):
    def sync_weights(self, *args, **kwargs):
        if not hasattr(self, "is_fsdp_enabled"):
            self.is_fsdp_enabled = False
        return sync_fn(self, *args, **kwargs)

    return sync_weights


def _patch_vllm_generation_fsdp_flag() -> None:
    """Axolotl 0.18 wrap reads Trainer.is_fsdp_enabled on VLLMGeneration.

    Trainer init re-applies that wrap after any earlier patch, so we hook the
    factory and the current method.
    """
    try:
        import axolotl.monkeypatch.trainer.trl_vllm as trl_vllm
        from trl.generation.vllm_generation import VLLMGeneration
    except ImportError:
        return

    orig_make = trl_vllm._make_batched_sync_weights

    def make_batched_sync_weights(original_sync_weights):
        return _ensure_fsdp_flag(orig_make(original_sync_weights))

    trl_vllm._make_batched_sync_weights = make_batched_sync_weights
    VLLMGeneration.sync_weights = _ensure_fsdp_flag(VLLMGeneration.sync_weights)


EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


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


def resolve_config(arg: str) -> Path:
    """Resolve a config arg to an existing yaml.

    Accepts a path (with or without .yaml) or a bare config name, which is
    searched recursively under the top-level experiments/ directory.
    """
    p = Path(arg)
    if p.suffix == ".yaml" or p.suffix == ".yml":
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {p}")
    if "/" in arg or "\\" in arg:
        # Path-like without extension: try as-is, then with .yaml.
        for cand in (p, p.with_suffix(".yaml")):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Config not found: {p} or {p.with_suffix('.yaml')}")

    candidates = sorted(EXPERIMENTS_DIR.rglob(f"{arg}.yaml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        print(f"Error: Config not found: {arg!r}")
        print(f"Available configs under {EXPERIMENTS_DIR}:")
        for f in sorted(EXPERIMENTS_DIR.rglob("*.yaml")):
            print(f"  - {f.relative_to(EXPERIMENTS_DIR.parent)}")
        sys.exit(1)
    print(f"Error: {arg!r} is ambiguous; use a path:")
    for f in candidates:
        print(f"  - {f}")
    sys.exit(1)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config_name: str = typer.Argument(
        help=(
            "Path to a config yaml (with or without .yaml) or a bare config "
            f"name searched recursively under {EXPERIMENTS_DIR}"
        ),
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from the latest checkpoint"),
) -> None:
    """Finetune a model using Axolotl."""
    config_path = resolve_config(config_name)

    print(f"Loading config: {config_path}")

    config = load_yaml_config(config_path)
    cfg = load_cfg(config)
    if cfg.rl and cfg.trl and cfg.trl.use_vllm:
        _patch_vllm_generation_fsdp_flag()

    use_vllm = bool(cfg.trl and cfg.trl.use_vllm)
    # vLLM colocate sleep/memory pool cannot use expandable_segments.
    if not (cfg.rl and use_vllm):
        set_pytorch_cuda_alloc_conf()

    if resume:
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

    print(f"Starting training: {config_name}")
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
    app()
