#!/usr/bin/env python3
"""Exp 12 slicer: 21k equal pool → 2k test + 1k val + 1.5k ⊂ 6k ⊂ 18k train.

Peel matched test, then matched val, then nested train from the leftover 18k.
Each file is concatenated by quota then shuffled (no round-robin). Axolotl NLL
val is 200 shuffled rows of the 1k. Checkpoint pick uses the full 1k val,
never the 2k test.

    uv run --no-project python experiments/12-v6-mix-reweight/scripts/make_v12_data.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinds import HARD, kind, user_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
EXP = Path(__file__).resolve().parents[1]
PROMPT = EXP / "system_prompt.txt"
POOL = DATA / "spatial_sft_v12_21000_pool.jsonl"
TEST_PATH = DATA / "spatial_sft_v12_2000_test.jsonl"
VAL_PATH = DATA / "spatial_sft_v12_1000_val.jsonl"
NLL_PATH = DATA / "spatial_sft_v12_val_nll.jsonl"
SEED = 52

# 2k TEST: ~1/3 types, equal subtypes (leftovers on which-1/2/3).
HOLD = {
    "dir-1": 111,
    "dir-2": 111,
    "dir-undetermined": 111,
    "dir-cycle": 111,
    "dir-incomplete": 111,
    "dir-omit": 111,
    "which-1": 134,
    "which-2": 134,
    "which-3": 134,
    "which-4": 133,
    "which-0": 133,
    "count-1": 333,
    "count-omit": 333,
}

# 1k VAL (overfit / ckpt pick). Same two-level idea; leftover on dir-1/2/omit and count.
VAL = {
    "dir-1": 56,
    "dir-2": 56,
    "dir-undetermined": 55,
    "dir-cycle": 55,
    "dir-incomplete": 55,
    "dir-omit": 56,
    "which-1": 67,
    "which-2": 67,
    "which-3": 67,
    "which-4": 66,
    "which-0": 66,
    "count-1": 167,
    "count-omit": 167,
}

# Hierarchical uniform: 1/3 dir, 1/3 which, 1/3 count; equal subtypes
# inside each type. Leftover 1–2 dir rows at 1.5k/6k go to dir-1 and dir-2
# so 1.5k ⊂ 6k ⊂ 18k still holds by bucket.
MIX = {
    18000: {
        "dir-1": 1000,
        "dir-2": 1000,
        "dir-undetermined": 1000,
        "dir-cycle": 1000,
        "dir-incomplete": 1000,
        "dir-omit": 1000,
        "which-1": 1200,
        "which-2": 1200,
        "which-3": 1200,
        "which-4": 1200,
        "which-0": 1200,
        "count-1": 3000,
        "count-omit": 3000,
    },
    6000: {
        "dir-1": 334,
        "dir-2": 334,
        "dir-undetermined": 333,
        "dir-cycle": 333,
        "dir-incomplete": 333,
        "dir-omit": 333,
        "which-1": 400,
        "which-2": 400,
        "which-3": 400,
        "which-4": 400,
        "which-0": 400,
        "count-1": 1000,
        "count-omit": 1000,
    },
    1500: {
        "dir-1": 84,
        "dir-2": 84,
        "dir-undetermined": 83,
        "dir-cycle": 83,
        "dir-incomplete": 83,
        "dir-omit": 83,
        "which-1": 100,
        "which-2": 100,
        "which-3": 100,
        "which-4": 100,
        "which-0": 100,
        "count-1": 250,
        "count-omit": 250,
    },
}


def _dump(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def stamp_prompt(row: dict, prompt: str) -> dict:
    messages = []
    stamped = False
    for msg in row.get("messages") or []:
        if msg.get("role") == "system":
            messages.append({**msg, "content": prompt})
            stamped = True
        else:
            messages.append(msg)
    if not stamped:
        messages = [{"role": "system", "content": prompt}, *messages]
    out = dict(row)
    out["messages"] = messages
    return out


def take_shuffle(buckets: dict[str, list], mix: dict[str, int], rng: random.Random) -> list:
    out: list = []
    for tag, need in mix.items():
        have = list(buckets.get(tag, []))
        if len(have) < need:
            raise SystemExit(f"{tag}: need {need} have {len(have)}")
        out.extend(have[:need])
    rng.shuffle(out)
    return out


def main() -> None:
    prompt = PROMPT.read_text(encoding="utf-8").strip()
    if POOL.is_file():
        rows = [
            stamp_prompt(json.loads(line), prompt)
            for line in POOL.read_text().splitlines()
            if line.strip()
        ]
        buckets: dict[str, list] = defaultdict(list)
        for row in rows:
            buckets[kind(row)].append(row)
        print("pool kinds:", {k: len(v) for k, v in sorted(buckets.items())})

        rng = random.Random(SEED)
        for tag, items in buckets.items():
            rng.shuffle(items)

        test_b, rest = _peel(buckets, HOLD)
        val_b, train_b = _peel(rest, VAL)
        for tag, need in MIX[18000].items():
            if len(train_b.get(tag, [])) < need:
                raise SystemExit(
                    f"{tag}: train need {need} have {len(train_b.get(tag, []))}"
                )
        test_rows = take_shuffle(test_b, HOLD, random.Random(SEED + 2000))
        val_rows = take_shuffle(val_b, VAL, random.Random(SEED + 1000))
        _dump(TEST_PATH, test_rows)
        _dump(VAL_PATH, val_rows)
        nll = list(val_rows)
        random.Random(SEED + 200).shuffle(nll)
        _dump(NLL_PATH, nll[:200])
        for n in (1500, 6000, 18000):
            chosen = take_shuffle(train_b, MIX[n], random.Random(SEED + n))
            _dump(DATA / f"spatial_sft_v12_{n}_train.jsonl", chosen)
    else:
        print(f"WARN: missing {POOL} — skip test/val/train slices")

    if not VAL_PATH.is_file():
        print(f"WARN: missing {VAL_PATH} — skip hard slice")
        return
    hard = []
    for line in VAL_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if kind(row) in HARD:
            hard.append(row)
    rng2 = random.Random(SEED)
    rng2.shuffle(hard)
    _dump(DATA / "spatial_sft_v12_hard_eval.jsonl", hard)
    print("hard-from-val kinds:", {k: sum(1 for r in hard if kind(r) == k) for k in HARD})
    print("overlap train18k vs test2k:", _overlap_files(
        DATA / "spatial_sft_v12_18000_train.jsonl", TEST_PATH
    ))
    print("overlap train18k vs val1k:", _overlap_files(
        DATA / "spatial_sft_v12_18000_train.jsonl", VAL_PATH
    ))
    print("overlap val1k vs test2k:", _overlap_files(VAL_PATH, TEST_PATH))


def _peel(buckets: dict[str, list], spec: dict[str, int]) -> tuple[dict[str, list], dict[str, list]]:
    taken: dict[str, list] = {}
    rest: dict[str, list] = {}
    for tag, n in spec.items():
        have = buckets.get(tag, [])
        if len(have) < n:
            raise SystemExit(f"{tag}: need {n} have {len(have)}")
        taken[tag] = have[:n]
        rest[tag] = have[n:]
    for tag, have in buckets.items():
        if tag not in rest:
            rest[tag] = have
    return taken, rest


def _users(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        user_text(json.loads(l))
        for l in path.read_text().splitlines()
        if l.strip()
    }


def _overlap_files(a: Path, b: Path) -> int:
    return len(_users(a) & _users(b))


if __name__ == "__main__":
    main()
