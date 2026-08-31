"""Summarize Exp 4a smoke evals into one comparison table.

Reads experiments/04a-smoke/results/smoke16/<tag>/results.json for each
config tag (base / sft / grpo) and writes a SUMMARY.md next to them, plus
prints it to stdout. Run from anywhere:

    uv run --no-project python experiments/04a-smoke/scripts/summarize.py

A <tag> folder is any subdir of smoke16/ that contains a results.json.
The smoke16/ folders are created by the batch launcher (one per config),
so the table always reflects exactly the configs that ran.
"""
import json
from pathlib import Path

SMOKE = Path(__file__).resolve().parents[1] / "results" / "smoke16"
TASK = "spatial_eval_4a_smoke16"


def acc(results: dict, metric: str):
    res = results.get("results", {})
    # Match the task name (suffix) since some runners prefix it.
    for name, row in res.items():
        if name == TASK or name.endswith(TASK):
            val = row.get(f"{metric},extract-answer")
            if val is None:
                val = row.get(metric)
            if val is None:
                return None
            return round(float(val) * 100, 1)
    return None


def main() -> None:
    if not SMOKE.is_dir():
        raise SystemExit(f"no results yet at {SMOKE}")
    rows = []
    for d in sorted(SMOKE.iterdir()):
        rj = d / "results.json"
        if not (d.is_dir() and rj.exists()):
            continue
        data = json.loads(rj.read_text())
        rows.append((d.name, acc(data, "strict_acc"), acc(data, "loose_acc")))

    lines = [
        "# Exp 4a — Qwen3.5-4B smoke (16-row SpatialMap-TQA-CORR)",
        "",
        "| config | strict acc | loose acc |",
        "|---|---|---|",
    ]
    for tag, strict, loose in rows:
        s = f"{strict}%" if strict is not None else "—"
        l = f"{loose}%" if loose is not None else "—"
        lines.append(f"| {tag} | {s} | {l} |")

    # base -> sft / grpo deltas
    base = next((s for t, s, _ in rows if t == "base"), None)
    if base is not None:
        for tag, strict, _ in rows:
            if strict is not None and tag != "base":
                lines.append("")
                lines.append(f"{tag} vs base (strict): {strict - base:+.1f} pp")

    out = "\n".join(lines) + "\n"
    (SMOKE / "SUMMARY.md").write_text(out)
    print(out)
    print(f"wrote {SMOKE / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
