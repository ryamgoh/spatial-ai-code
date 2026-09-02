"""Summarize Exp 4b CLI-pipeline evals into one comparison table.

Reads experiments/04b-cli/results/smoke16/<tag>/results.json for each
config tag (base / sft / grpo) and writes SUMMARY.md next to them.
"""
import json
from pathlib import Path

SMOKE = Path(__file__).resolve().parents[1] / "results" / "smoke16"
TASK = "spatial_eval_4a_smoke16"


def acc(results: dict, metric: str):
    res = results.get("results", {})
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
        "# Exp 4b — Qwen3.5-4B CLI pipeline (16-row SpatialMap-TQA-CORR)",
        "",
        "| config | strict acc | loose acc |",
        "|---|---|---|",
    ]
    for tag, strict, loose in rows:
        s = f"{strict}%" if strict is not None else "—"
        l = f"{loose}%" if loose is not None else "—"
        lines.append(f"| {tag} | {s} | {l} |")

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
