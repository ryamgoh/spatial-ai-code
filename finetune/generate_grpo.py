"""Generate prompt-only GRPO data from the SFT synthetic generator.

Writes conversational `prompt` + `oracle_option` JSONL. No gold traces.

Usage:
    cd finetune && uv run python generate_grpo.py
    cd finetune && uv run python generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import typer

from generate_all import generate_sample

ANSWER_RE = re.compile(r"Answer:\s*([A-D](?:\s*,\s*[A-D])*)", re.IGNORECASE)

# Match cleaned SpatialMap mix (~32% direction, ~38% which-object, ~30% count)
# rather than the SFT 40/40/20 uncertainty-heavy mix.
DEFAULT_PLAN = [
    # (question_type, target_num_answers, count, label)
    (0, 1, 1050, "Type0-1ans"),
    (0, 2, 280, "Type0-2ans"),
    (1, 1, 1100, "Type1-1ans"),
    (1, 2, 360, "Type1-2ans"),
    (2, None, 1210, "Type2-count"),
]


def oracle_from_assistant(content: str) -> str | None:
    matches = ANSWER_RE.findall(content)
    if not matches:
        return None
    letters = [tok.strip().upper() for tok in matches[-1].split(",")]
    letters = [tok for tok in letters if tok in {"A", "B", "C", "D"}]
    if not letters:
        return None
    return ",".join(letters)


# SFT gold always ends with: </think>\nAnswer: A   (or "A, B")
ANSWER_LINE = (
    "\n\nAfter your reasoning, close with </think> and a final line "
    "exactly like the examples: `Answer: A` or `Answer: A, C`."
)


def attach_answer_line(prompt: list[dict]) -> list[dict]:
    out = []
    for msg in prompt:
        msg = dict(msg)
        if msg.get("role") == "user" and ANSWER_LINE not in (msg.get("content") or ""):
            msg["content"] = (msg.get("content") or "") + ANSWER_LINE
        out.append(msg)
    return out


def to_grpo_row(sample: dict) -> dict | None:
    messages = sample.get("messages") or []
    prompt = [m for m in messages if m.get("role") in {"system", "user"}]
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if not prompt or assistant is None:
        return None
    oracle = oracle_from_assistant(assistant.get("content") or "")
    if not oracle:
        return None
    return {"prompt": attach_answer_line(prompt), "oracle_option": oracle}


def generate_rows(plan: list[tuple], seed: int) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []
    for q_type, tgt_ans, target_count, label in plan:
        if target_count <= 0:
            continue
        generated = 0
        attempts = 0
        while generated < target_count:
            n_ent = random.randint(5, 10)
            n_sent = random.randint(n_ent, n_ent + 5)
            sample = generate_sample(
                num_entities=n_ent,
                num_sentences=n_sent,
                target_num_answers=tgt_ans,
                question_type=q_type,
            )
            attempts += 1
            if not sample:
                continue
            row = to_grpo_row(sample)
            if not row:
                continue
            rows.append(row)
            generated += 1
            if generated % 100 == 0:
                print(
                    f"[{label}] {generated}/{target_count} "
                    f"(attempts {attempts})",
                    flush=True,
                )
        print(f"done {label}: {generated} in {attempts} attempts", flush=True)
    random.shuffle(rows)
    return rows


def annotate_jsonl(path: Path) -> int:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prompt"] = attach_answer_line(row["prompt"])
            rows.append(row)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def scaled_plan(n: int) -> list[tuple]:
    total = sum(item[2] for item in DEFAULT_PLAN)
    plan = []
    allocated = 0
    for i, (q_type, tgt_ans, count, label) in enumerate(DEFAULT_PLAN):
        if i == len(DEFAULT_PLAN) - 1:
            scaled = n - allocated
        else:
            scaled = int(round(count * n / total))
            allocated += scaled
        plan.append((q_type, tgt_ans, max(0, scaled), label))
    return plan


app = typer.Typer(add_completion=False)


@app.command()
def main(
    out: str = typer.Option(
        "../spatial_grpo_data.jsonl", "--out", help="Output JSONL path (relative to cwd)",
    ),
    seed: int = typer.Option(42, "--seed"),
    n: int = typer.Option(4000, "--n", help="If set, scale DEFAULT_PLAN to this many rows (default 4000)"),
    annotate: bool = typer.Option(
        False, "--annotate",
        help="Patch an existing JSONL in place with the Answer-line instruction",
    ),
) -> None:
    """Generate GRPO prompt-only JSONL."""
    out_path = Path(out)
    if annotate:
        n_rows = annotate_jsonl(out_path)
        print(f"annotated {n_rows} rows in {out_path.resolve()}")
        return
    plan = scaled_plan(n)
    print("plan:", [(p[3], p[2]) for p in plan], file=sys.stderr)
    rows = generate_rows(plan, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    app()
