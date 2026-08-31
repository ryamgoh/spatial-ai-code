"""Merge SFT QLoRA into a new bf16 directory. Never writes into the adapter dir.

--base is required: there is no default model, and merging a LoRA into the
wrong base silently produces a broken model.

Usage:
    cd finetune && uv run python merge_sft.py \\
        --base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \\
        --adapter ../experiments/03-sft-vs-baseline/models/deepseek-r1-qwen3-8b \\
        --out ../experiments/05-grpo/models/deepseek-r1-qwen3-8b-merged
"""

from __future__ import annotations

from pathlib import Path

import typer
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    base: str = typer.Option(..., "--base", help="HF base model id to merge the LoRA into (required)"),
    adapter: str = typer.Option(..., "--adapter", help="Path to the SFT LoRA adapter dir to merge (required)"),
    out: str = typer.Option(..., "--out", help="Where to write the merged bf16 model (required)"),
) -> None:
    """Merge SFT LoRA into a new folder."""
    adapter = Path(adapter).resolve()
    out = Path(out).resolve()
    if adapter == out or out.is_relative_to(adapter):
        raise SystemExit(
            f"Refusing to write merge into the adapter tree:\n  adapter={adapter}\n  out={out}"
        )
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter}")
    if (out / "config.json").exists():
        print(f"already merged: {out}")
        return

    print(f"base     {base}")
    print(f"adapter  {adapter}  (read-only)")
    print(f"out      {out}")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
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
    app()
