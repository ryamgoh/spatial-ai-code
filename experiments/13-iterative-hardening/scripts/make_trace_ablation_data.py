"""Derive matched full-state and delta-state datasets from frozen V13.1 8K."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "spatial"))

from spatial_generation_v13 import (
    TraceFormat,
    render_trace,
)
from spatial_solver_v13 import SpatialSolverV13

VERSION = "v13.1-trace-ablation-v1"
EXPECTED_TRAIN = 8000
EXPECTED_VAL = 129


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_text(row: dict) -> str:
    return next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def assistant_text(row: dict) -> str:
    return next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
    )


def paths(output: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        output.with_name(output.stem + "_full_train.jsonl"),
        output.with_name(output.stem + "_full_val.jsonl"),
        output.with_name(output.stem + "_delta_train.jsonl"),
        output.with_name(output.stem + "_delta_val.jsonl"),
        output.with_name(output.stem + "_manifest.json"),
    )


def convert(rows: list[dict], trace_format: TraceFormat) -> list[dict]:
    solver = SpatialSolverV13()
    converted = []
    for row in rows:
        result = json.loads(json.dumps(row))
        solved = solver.solve_and_analyze(user_text(row))
        if solved.grade.raw != str(row.get("oracle_option") or ""):
            raise ValueError("frozen 8K row no longer agrees with V13 solver")
        for message in result.get("messages") or []:
            if message.get("role") == "assistant":
                message["content"] = render_trace(solved, trace_format)
        result["trace_format"] = trace_format.value
        result["trace_ablation_version"] = VERSION
        converted.append(result)
    return converted


def validate(
    full_train_path: Path,
    full_val_path: Path,
    delta_train_path: Path,
    delta_val_path: Path,
    manifest_path: Path,
    source_train: Path,
    source_val: Path,
) -> None:
    full_train, full_val = load_jsonl(full_train_path), load_jsonl(full_val_path)
    delta_train, delta_val = load_jsonl(delta_train_path), load_jsonl(delta_val_path)
    source_train_rows, source_val_rows = load_jsonl(source_train), load_jsonl(source_val)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("trace ablation manifest version mismatch")
    if len(full_train) != EXPECTED_TRAIN or len(delta_train) != EXPECTED_TRAIN:
        raise ValueError("trace ablation training size mismatch")
    if len(full_val) != EXPECTED_VAL or len(delta_val) != EXPECTED_VAL:
        raise ValueError("trace ablation validation size mismatch")
    if manifest.get("source_train_sha256") != sha256(source_train):
        raise ValueError("trace ablation source train hash mismatch")
    if manifest.get("source_val_sha256") != sha256(source_val):
        raise ValueError("trace ablation source validation hash mismatch")

    for source, full, delta in zip(
        source_train_rows + source_val_rows,
        full_train + full_val,
        delta_train + delta_val,
        strict=True,
    ):
        if user_text(source) != user_text(full) or user_text(source) != user_text(delta):
            raise ValueError("trace ablation prompt/order mismatch")
        for key in ("oracle_option", "semantic_subtype", "difficulty", "generation_cell"):
            if source.get(key) != full.get(key) or source.get(key) != delta.get(key):
                raise ValueError(f"trace ablation metadata mismatch: {key}")
        if full.get("trace_format") != TraceFormat.FULL_STATE.value:
            raise ValueError("full-state row label mismatch")
        if delta.get("trace_format") != TraceFormat.DELTA_STATE.value:
            raise ValueError("delta-state row label mismatch")
        if assistant_text(full).split("Answer:")[-1] != assistant_text(delta).split("Answer:")[-1]:
            raise ValueError("trace ablation answer mismatch")
        if "**Final X-State**:" not in assistant_text(delta):
            raise ValueError("delta-state row lacks final state")
        if "**X-State**:" in assistant_text(delta):
            raise ValueError("delta-state row contains repeated full-state marker")
        if assistant_text(source) != assistant_text(full):
            raise ValueError("full-state control differs from frozen source trace")


def generate(output: Path, source_train: Path, source_val: Path) -> tuple[Path, ...]:
    output_paths = paths(output)
    full_train, full_val, delta_train, delta_val, manifest_path = output_paths
    source_train_rows, source_val_rows = load_jsonl(source_train), load_jsonl(source_val)
    write_jsonl(full_train, convert(source_train_rows, TraceFormat.FULL_STATE))
    write_jsonl(full_val, convert(source_val_rows, TraceFormat.FULL_STATE))
    write_jsonl(delta_train, convert(source_train_rows, TraceFormat.DELTA_STATE))
    write_jsonl(delta_val, convert(source_val_rows, TraceFormat.DELTA_STATE))

    full_train_rows, delta_train_rows = load_jsonl(full_train), load_jsonl(delta_train)
    full_chars = sum(len(assistant_text(row)) for row in full_train_rows)
    delta_chars = sum(len(assistant_text(row)) for row in delta_train_rows)
    manifest = {
        "version": VERSION,
        "train_rows_per_arm": len(full_train_rows),
        "validation_rows_per_arm": len(load_jsonl(full_val)),
        "source_train_sha256": sha256(source_train),
        "source_val_sha256": sha256(source_val),
        "full_assistant_chars": full_chars,
        "delta_assistant_chars": delta_chars,
        "delta_to_full_char_ratio": delta_chars / full_chars,
        "cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in full_train_rows).items())
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(*output_paths, source_train, source_val)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "data" / "spatial_v13_trace_ablation.jsonl",
    )
    parser.add_argument(
        "--source-train", type=Path,
        default=REPO / "data" / "spatial_v13_sft_8000_train.jsonl",
    )
    parser.add_argument(
        "--source-val", type=Path,
        default=REPO / "data" / "spatial_v13_sft_8000_val.jsonl",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    missing = [str(path) for path in (args.source_train, args.source_val) if not path.is_file()]
    if missing:
        raise SystemExit("missing source data: " + ", ".join(missing))
    output_paths = paths(args.out)
    if args.validate_only:
        validate(*output_paths, args.source_train, args.source_val)
        print("valid trace ablation: " + ", ".join(map(str, output_paths)))
        return
    output_paths = generate(args.out, args.source_train, args.source_val)
    print("wrote " + ", ".join(map(str, output_paths)))


if __name__ == "__main__":
    main()
