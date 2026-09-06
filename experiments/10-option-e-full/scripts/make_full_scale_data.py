#!/usr/bin/env python3
"""Slice nested 1.5k / 5k Full-mix SFT sets from the 20k pool.

1.5k ⊂ 5k ⊂ 20k, stratified to SpatialMap-TQA-Corr-Full
(dir-1 332 / dir-2 93 / dir-E 75 / which-1 302 / which-E 198 /
count-1 404 / count-E 96 of 1,500). The 20k file is produced by
generate_all.py (seed 42, --include-option-e, those seven counts,
--test-split 0).

Usage (from repo root or finetune/):
    uv run python experiments/10-option-e-full/scripts/make_full_scale_data.py
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
POOL = DATA / "spatial_sft_full_scale_20000_train.jsonl"
SEED = 42
ANSWER_RE = re.compile(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", re.I)

# n -> kind counts. 20k is the generate_all.py request;
# 5k / 1.5k are prefixes of those buckets (10/3 and 1× the Full census).
MIX = {
    20000: {
        "dir-1": 4428,
        "dir-2": 1240,
        "dir-E": 1000,
        "which-1": 4028,
        "which-E": 2640,
        "count-1": 5384,
        "count-E": 1280,
    },
    5000: {
        "dir-1": 1107,
        "dir-2": 310,
        "dir-E": 250,
        "which-1": 1007,
        "which-E": 660,
        "count-1": 1346,
        "count-E": 320,
    },
    1500: {
        "dir-1": 332,
        "dir-2": 93,
        "dir-E": 75,
        "which-1": 302,
        "which-E": 198,
        "count-1": 404,
        "count-E": 96,
    },
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


def assistant_text(row: dict) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def kind(row: dict) -> str:
    t = qtype(user_text(row))
    asst = assistant_text(row)
    m = list(ANSWER_RE.finditer(asst))
    gold = m[-1].group(1) if m else ""
    letters = [p for p in re.split(r"[,;| ]+", gold.upper()) if p in "ABCDE"]
    if letters == ["E"]:
        return f"{t}-E"
    n = len([x for x in letters if x in "ABCD"])
    return f"{t}-{n}"


def out_path(n: int) -> Path:
    return DATA / f"spatial_sft_full_scale_{n}_train.jsonl"


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
            buckets[kind(row)].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    need = MIX[20000]
    missing = [t for t, n in need.items() if counts.get(t, 0) < n]
    if missing:
        raise SystemExit(
            f"{POOL} kind counts {counts} cannot cover 20k mix {need}; "
            f"short: {missing}"
        )

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
        got: dict[str, int] = defaultdict(int)
        for row in chosen:
            got[kind(row)] += 1
        print(f"wrote {dest} n={len(chosen)} mix={dict(sorted(got.items()))}")

    print(f"pool {POOL} n={sum(counts.values())} mix={dict(sorted(counts.items()))}")
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
