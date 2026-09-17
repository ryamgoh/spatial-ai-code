#!/usr/bin/env python3
"""Copy an eval yaml and point model_args.lora_path at a checkpoint."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: rewrite_eval_lora.py in.yaml out.yaml /abs/lora")
    src, dst, lora = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    text = src.read_text(encoding="utf-8")
    new, n = re.subn(
        r"^(\s*lora_path:\s*).+$",
        rf"\1{lora}",
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"expected 1 lora_path line, got {n} in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new, encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
