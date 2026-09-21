"""Copy an eval YAML while replacing its LoRA path without YAML parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def render(source: Path, output: Path, lora_path: str) -> None:
    text = source.read_text()
    rendered, count = re.subn(
        r"(?m)^(\s*lora_path:)\s*.*$",
        rf"\1 {lora_path}",
        text,
    )
    if count != 1:
        raise ValueError(f"expected one lora_path in {source}, found {count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-path", required=True)
    args = parser.parse_args()
    render(args.source, args.output, args.lora_path)
    print(f"wrote {args.output} for {args.lora_path}")


if __name__ == "__main__":
    main()
