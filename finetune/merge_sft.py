"""Merge SFT QLoRA into a new bf16 directory. Never writes into the adapter dir.

Usage:
    cd finetune && uv run python merge_sft.py
    cd finetune && uv run python merge_sft.py \\
        --adapter ./outputs/deepseek-r1-qwen3-8b \\
        --out ./outputs/deepseek-r1-qwen3-8b-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SFT LoRA into a new folder")
    p.add_argument("--base", default=BASE)
    p.add_argument("--adapter", default="./outputs/deepseek-r1-qwen3-8b")
    p.add_argument("--out", default="./outputs/deepseek-r1-qwen3-8b-merged")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = Path(args.adapter).resolve()
    out = Path(args.out).resolve()
    if adapter == out or out.is_relative_to(adapter):
        raise SystemExit(
            f"Refusing to write merge into the adapter tree:\n  adapter={adapter}\n  out={out}"
        )
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter}")
    if (out / "config.json").exists():
        print(f"already merged: {out}")
        return

    print(f"base     {args.base}")
    print(f"adapter  {adapter}  (read-only)")
    print(f"out      {out}")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"wrote merge to {out}")
    print(f"SFT LoRA still at {adapter}")


if __name__ == "__main__":
    main()
