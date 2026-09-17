#!/usr/bin/env python3
"""Pick the checkpoint on the 1k VAL (not the 2k test).

Reads results/<tag>/ckpts/<name>/responses_spatial_eval_v12_val.jsonl
and kinds from data/spatial_sft_v12_1000_val.jsonl. Writes
results/<tag>/CKPT_PICK.md.

    uv run --no-project python experiments/12-v6-mix-reweight/scripts/score_checkpoints.py 4b-1.5k
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinds import HARD, kind, user_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
VAL_JSONL = REPO / "data" / "spatial_sft_v12_1000_val.jsonl"


def letters(raw: object) -> list[str]:
    s = str(raw or "").strip().upper()
    if not s:
        return []
    return [p for p in re.split(r"[,;| ]+", s) if p and p in "ABCDE"]


def pred_of(sample: dict) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        inner = fr[0]
        if isinstance(inner, list):
            return str(inner[0] if inner else "")
        return str(inner or "")
    return ""


def gold_of(sample: dict) -> str:
    doc = sample.get("doc") or {}
    return str(doc.get("oracle_option") or "")


def index_kinds() -> dict[str, str]:
    out = {}
    if not VAL_JSONL.is_file():
        return out
    for line in VAL_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[user_text(row)] = kind(row)
    return out


def score(path: Path, index: dict[str, str]) -> dict[str, tuple[int, int]]:
    tallies: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        text = str((sample.get("doc") or {}).get("text") or "")
        tag = index.get(text, "?")
        g = set(letters(gold_of(sample)))
        p = set(letters(pred_of(sample)))
        tallies[tag][1] += 1
        if g and p == g:
            tallies[tag][0] += 1
    return {k: (v[0], v[1]) for k, v in tallies.items()}


def overall(buckets: dict[str, tuple[int, int]]) -> float | None:
    ok = sum(v[0] for v in buckets.values())
    n = sum(v[1] for v in buckets.values())
    if n <= 0:
        return None
    return round(100.0 * ok / n, 1)


def macro(buckets: dict[str, tuple[int, int]]) -> float | None:
    vals = []
    for tag in HARD:
        ok, n = buckets.get(tag, (0, 0))
        if n:
            vals.append(ok / n)
    if not vals:
        return None
    return round(100.0 * (sum(vals) / len(vals)), 1)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "4b-1.5k"
    ckpt_root = ROOT / "results" / tag / "ckpts"
    index = index_kinds()
    rows = []
    if not ckpt_root.is_dir():
        raise SystemExit(f"missing {ckpt_root}")
    for d in sorted(ckpt_root.iterdir()):
        resp = d / "responses_spatial_eval_v12_val.jsonl"
        if not resp.is_file():
            continue
        buckets = score(resp, index)
        rows.append((d.name, overall(buckets), macro(buckets), buckets))
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0), -(r[2] or 0)))
    lines = [
        f"# Exp 12 checkpoint pick ({tag})",
        "",
        "On the **1k VAL** (not the 2k test). Pick = best overall strict; hard macro is the Type-0 holes.",
        "",
        "| ckpt | val overall | hard macro | dir-2 | incomplete | cycle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, ov, m, buckets in rows:
        def cell(t: str) -> str:
            ok, n = buckets.get(t, (0, 0))
            return f"{round(100.0 * ok / n, 1)}% ({ok}/{n})" if n else "—"

        lines.append(
            f"| `{name}` | {ov if ov is not None else '—'}% | "
            f"{m if m is not None else '—'}% | "
            f"{cell('dir-2')} | {cell('dir-incomplete')} | {cell('dir-cycle')} |"
        )
    if rows and rows[0][1] is not None:
        lines += ["", f"**pick:** `{rows[0][0]}` (val overall {rows[0][1]}%)", ""]
    out = ROOT / "results" / tag / "CKPT_PICK.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
