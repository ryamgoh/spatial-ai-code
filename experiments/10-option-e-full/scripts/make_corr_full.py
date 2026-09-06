#!/usr/bin/env python3
"""Build SpatialMap-TQA-Corr-Full from spatialeval_cleaned.jsonl.

Appends ``E. None of these is proven`` to every item.
- 171 empty oracle_option → gold E
- which-4 (198): type-1 fallback, 0 of A–D proven → gold E
- remaining nonempty A–D golds keep their letters; E is a distractor

Writes data/spatialeval_corr_full.jsonl (1500 rows).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "spatialeval_cleaned.jsonl"
DST = ROOT / "data" / "spatialeval_corr_full.jsonl"
NONE_LABEL = "None of these is proven"
E_LINE = f"E. {NONE_LABEL}"
E_GOLD = "E"
E_LINE_RE = re.compile(
    r"\bE\.\s*(None of these is proven|None of the Options)", re.I
)


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def attach_e(text: str) -> str:
    if E_LINE_RE.search(text):
        return E_LINE_RE.sub(E_LINE, text, count=1)
    t = text.rstrip()
    if t.endswith("."):
        t = t[:-1]
    return t + "\n" + E_LINE + "."


def set_gold_e(row: dict, note: str) -> None:
    row["oracle_option"] = E_GOLD
    row["oracle_answer"] = NONE_LABEL
    row["oracle_full_answer"] = E_LINE
    row["full_note"] = note


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    counts: Counter[str] = Counter()
    n = 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            gold = str(row.get("oracle_option") or "").strip()
            row["text"] = attach_e(text)
            t = qtype(text)
            if not gold:
                set_gold_e(row, "empty-oracle → E")
                counts[f"{t}-E"] += 1
            else:
                letters = [p for p in re.split(r"[,;| ]+", gold.upper()) if p in "ABCD"]
                if t == "which" and len(letters) == 4:
                    set_gold_e(row, "which-4 fallback (0 proven) → E")
                    counts["which-E"] += 1
                else:
                    row["oracle_option"] = ",".join(letters) if letters else gold
                    counts[f"{t}-{len(letters)}"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {DST} n={n}")
    print("mix:", dict(sorted(counts.items())))
    assert n == 1500, n
    assert counts["dir-E"] == 75
    assert counts["count-E"] == 96
    assert counts["which-E"] == 198
    assert counts["dir-E"] + counts["count-E"] + counts["which-E"] == 369


if __name__ == "__main__":
    main()
