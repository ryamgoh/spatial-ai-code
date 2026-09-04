#!/usr/bin/env python3
"""Slice nested 1.5k / 5k single-answer SFT sets from the 20k pool.

1.5k ⊂ 5k ⊂ 20k, stratified to the TQA-Corr-Single type mix
(count 404 / dir 332 / which 302 of 1,038). The 20k file is produced by
generate_all.py (seed 42, those three 1-answer counts, --test-split 0).

Usage (from repo root or finetune/):
    uv run python experiments/08-dual-scaling/scripts/make_scale_data.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
POOL = DATA / "spatial_sft_single_scale_20000_train.jsonl"
SEED = 42

# n -> (dir, which, count). 20k is the generate_all.py request;
# 5k / 1.5k are prefixes of those buckets.
MIX = {
    20000: {"dir": 6397, "which": 5819, "count": 7784},
    5000: {"dir": 1599, "which": 1455, "count": 1946},
    1500: {"dir": 480, "which": 436, "count": 584},
}


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def user_text(row: dict) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def out_path(n: int) -> Path:
    return DATA / f"spatial_sft_single_scale_{n}_train.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not POOL.is_file() or POOL.stat().st_size == 0:
        raise SystemExit(f"missing 20k pool at {POOL} — generate it first")

    buckets: dict[str, list[dict]] = defaultdict(list)
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            buckets[qtype(user_text(row))].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    need = MIX[20000]
    missing = [t for t, n in need.items() if counts.get(t, 0) < n]
    if missing:
        raise SystemExit(
            f"{POOL} type counts {counts} cannot cover 20k mix {need}; "
            f"short: {missing}"
        )
    extra = {t: n for t, n in counts.items() if t not in need or t == "?"}
    if extra.get("?"):
        print(f"WARN: {extra['?']} untyped rows ignored", file=sys.stderr)

    # Nested prefixes: 1.5k = first MIX[1500] of each 20k bucket, etc.
    nested: dict[int, list[dict]] = {}
    for n, mix in MIX.items():
        if n == 20000:
            continue
        chosen: list[dict] = []
        for t, k in mix.items():
            chosen.extend(buckets[t][:k])
        rng = random.Random(SEED)
        rng.shuffle(chosen)
        nested[n] = chosen
        dest = out_path(n)
        write_jsonl(dest, chosen)
        got = defaultdict(int)
        for row in chosen:
            got[qtype(user_text(row))] += 1
        print(f"wrote {dest} n={len(chosen)} mix={dict(got)}")

    n20 = sum(need[t] for t in need)
    print(f"pool {POOL} n={sum(counts.values())} (requested {n20}) mix={counts}")
    # Nested check: every 1.5k row is in 5k; every 5k row is in the 20k prefixes.
    s15 = {json.dumps(r, sort_keys=True) for r in nested[1500]}
    s5 = {json.dumps(r, sort_keys=True) for r in nested[5000]}
    prefixes = []
    for t, k in MIX[5000].items():
        prefixes.extend(buckets[t][:k])
    s5_src = {json.dumps(r, sort_keys=True) for r in prefixes}
    if not s15 <= s5:
        raise SystemExit("1.5k is not a subset of 5k")
    if not s5 <= s5_src:
        raise SystemExit("5k is not a prefix-subset of the 20k buckets")
    print("nested OK: 1500 ⊂ 5000 ⊂ 20000")


if __name__ == "__main__":
    main()
