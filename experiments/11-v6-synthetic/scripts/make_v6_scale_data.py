#!/usr/bin/env python3
"""Slice nested 1.5k ⊂ 6k ⊂ 20k from the v6 20k train pool.

Usage (repo root or finetune/):
    uv run python experiments/11-v6-synthetic/scripts/make_v6_scale_data.py
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
POOL = DATA / "spatial_sft_v6_scale_20000_train.jsonl"
SEED = 42
ANSWER_RE = re.compile(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", re.I)

MIX = {
    20000: {
        "dir-1": 2000,
        "dir-2": 1667,
        "dir-undetermined": 1000,
        "dir-cycle": 667,
        "dir-incomplete": 667,
        "dir-omit": 666,
        "which-1": 2000,
        "which-2": 1667,
        "which-3": 1000,
        "which-4": 667,
        "which-0": 1333,
        "count-1": 5333,
        "count-omit": 1333,
    },
    6000: {
        "dir-1": 600,
        "dir-2": 500,
        "dir-undetermined": 300,
        "dir-cycle": 200,
        "dir-incomplete": 200,
        "dir-omit": 200,
        "which-1": 600,
        "which-2": 500,
        "which-3": 300,
        "which-4": 200,
        "which-0": 400,
        "count-1": 1600,
        "count-omit": 400,
    },
    1500: {
        "dir-1": 150,
        "dir-2": 125,
        "dir-undetermined": 75,
        "dir-cycle": 50,
        "dir-incomplete": 50,
        "dir-omit": 50,
        "which-1": 150,
        "which-2": 125,
        "which-3": 75,
        "which-4": 50,
        "which-0": 100,
        "count-1": 400,
        "count-omit": 100,
    },
}


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


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def gold_letters(asst: str) -> list[str]:
    m = list(ANSWER_RE.finditer(asst))
    if not m:
        return []
    return [p.strip() for p in m[-1].group(1).split(",") if p.strip()]


def kind(row: dict) -> str:
    user = user_text(row)
    asst = assistant_text(row)
    t = qtype(user)
    n = len(gold_letters(asst))
    if t == "dir":
        if "contradiction" in asst:
            return "dir-cycle"
        if "only one of the two remaining compounds is listed" in asst:
            return "dir-incomplete"
        if "the proven direction is not listed" in asst or "No listed option is correct" in user:
            if n == 1 and "Cannot be determined" not in (asst[asst.rfind("Answer:"):] if "Answer:" in asst else ""):
                pass
        gold = gold_letters(asst)
        asst_tail = asst[asst.find("### Final Deduction"):] if "### Final Deduction" in asst else asst
        inns = re.findall(r"- ([A-E])\.\s*(.+?) — .+\. (In|Out)\.", asst_tail)
        in_vals = [v.strip() for let, v, mark in inns if mark == "In"]
        if any("Cannot be determined" in v for v in in_vals):
            if "contradiction" in asst_tail:
                return "dir-cycle"
            if "only one of the two remaining" in asst_tail:
                return "dir-incomplete"
            return "dir-undetermined"
        if any(
            "not listed" in (asst_tail.lower()) and mark == "In"
            for *_, mark in inns
        ) or any(
            "none of" in v.lower() or "not among" in v.lower() or "no listed option" in v.lower()
            for v in in_vals
        ):
            return "dir-omit"
        if n == 1:
            return "dir-1"
        if n == 2:
            return "dir-2"
        return "dir-?"
    if t == "which":
        inns = re.findall(
            r"- ([A-E])\.\s*(.+?) — .+\. (In|Out)\.",
            asst[asst.find("### Final Deduction"):] if "### Final Deduction" in asst else asst,
        )
        in_vals = [v.strip() for _, v, mark in inns if mark == "In"]
        if any(
            "none of" in v.lower() or "not among" in v.lower() or "no listed" in v.lower()
            for v in in_vals
        ):
            return "which-0"
        k = sum(
            1
            for v in in_vals
            if "none of" not in v.lower()
            and "not among" not in v.lower()
            and "cannot be determined" not in v.lower()
            and "no listed" not in v.lower()
        )
        if k in (1, 2, 3, 4):
            return f"which-{k}"
        return "which-?"
    if t == "count":
        inns = re.findall(
            r"- ([A-E])\.\s*(.+?) — .+\. (In|Out)\.",
            asst[asst.find("### Final Deduction"):] if "### Final Deduction" in asst else asst,
        )
        in_vals = [v.strip() for _, v, mark in inns if mark == "In"]
        if any(
            "none of" in v.lower() or "not among" in v.lower() or "no listed" in v.lower()
            for v in in_vals
        ):
            return "count-omit"
        return "count-1"
    return "?"


def main() -> None:
    if not POOL.is_file():
        raise SystemExit(f"missing {POOL}")
    rows = [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]
    buckets: dict[str, list] = defaultdict(list)
    for row in rows:
        buckets[kind(row)].append(row)
    print("pool kinds:", {k: len(v) for k, v in sorted(buckets.items())})
    rng = random.Random(SEED)
    for k, v in buckets.items():
        rng.shuffle(v)

    for n, mix in MIX.items():
        out = DATA / f"spatial_sft_v6_scale_{n}_train.jsonl"
        chosen: list = []
        for k, need in mix.items():
            have = buckets.get(k, [])
            if len(have) < need:
                raise SystemExit(f"{k}: need {need} have {len(have)}")
            chosen.extend(have[:need])
        rng.shuffle(chosen)
        with out.open("w", encoding="utf-8") as f:
            for row in chosen:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {out} n={len(chosen)}")
        if n == 20000 and len(chosen) != len(rows):
            print(f"note: sliced {len(chosen)} of pool {len(rows)}")


if __name__ == "__main__":
    main()
