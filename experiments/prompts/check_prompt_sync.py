#!/usr/bin/env python3
"""Check that inline system prompts in eval yamls match the canonical files.

Canonical prompt family lives in experiments/prompts/ (round-indexed; a new
iteration gets a new index, files are never edited in place). Eval yamls and
training stamps carry inline copies of those prompts, so this guards against
drift between the canonical file and its inline copies.

    uv run --no-project python experiments/prompts/check_prompt_sync.py

Exit 1 on any mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "experiments" / "prompts"
EXP12 = ROOT / "experiments" / "12-v6-mix-reweight"

# (inline source, yaml key)  ->  canonical prompt file
CHECKS: list[tuple[Path, str, Path]] = [
    (EXP12 / "system_prompt.txt", "file", PROMPTS / "nonshot_5.txt"),
    (EXP12 / "eval-sft-4b-1500.yaml", "yaml", PROMPTS / "nonshot_5.txt"),
    (EXP12 / "eval-sft-4b-6000.yaml", "yaml", PROMPTS / "nonshot_5.txt"),
    (EXP12 / "eval-sft-4b-18000.yaml", "yaml", PROMPTS / "nonshot_5.txt"),
    (EXP12 / "eval-baseline-4b.yaml", "yaml", PROMPTS / "nonshot_5.txt"),
    (EXP12 / "eval-baseline-4b-oneshot.yaml", "yaml", PROMPTS / "oneshot_5.txt"),
    (EXP12 / "eval-baseline-4b-threeshot.yaml", "yaml", PROMPTS / "threeshot_5.txt"),
]


def inline_of(path: Path, kind: str) -> str:
    if kind == "file":
        return path.read_text(encoding="utf-8").strip()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(cfg.get("system_instruction") or "").strip()


def main() -> int:
    bad = 0
    for src, kind, canon in CHECKS:
        if not src.is_file():
            print(f"SKIP (missing): {src.relative_to(ROOT)}")
            continue
        a = inline_of(src, kind).rstrip("\n")
        b = canon.read_text(encoding="utf-8").strip().rstrip("\n")
        status = "OK" if a == b else "MISMATCH"
        if a != b:
            bad += 1
        print(f"{status}: {src.relative_to(ROOT)}  ==  {canon.name}")
    if bad:
        print(f"check_prompt_sync: {bad} mismatch(es)")
        return 1
    print("check_prompt_sync: all in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
