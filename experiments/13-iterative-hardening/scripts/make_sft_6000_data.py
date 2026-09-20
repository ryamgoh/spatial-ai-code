"""Generate the frozen nested native V13 6K train and validation set.

The exact V13 1.5K training file is retained as the prefix of the 6K file.
Each generation cell then receives three times its original row count, making
the final cell distribution exactly four times the 1.5K distribution.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFT15_PATH = HERE / "make_sft_1500_data.py"
SPEC = importlib.util.spec_from_file_location("v13_sft_6000_helpers", SFT15_PATH)
assert SPEC and SPEC.loader
SFT15 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SFT15)

V1 = SFT15.V1
REPO = SFT15.REPO
SEED = 13600
VERSION = "v13-sft-6000-nested-v1"
EXPECTED_TRAIN = 6000
EXPECTED_ADDITIONS = 4500
EXPECTED_VAL = SFT15.EXPECTED_VAL
EXPECTED_CELLS = SFT15.EXPECTED_CELLS


def paths(output: Path) -> tuple[Path, Path, Path]:
    return V1.probe_paths(output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(V1.user_text(row)) for row in rows}


def additional_cells() -> list:
    """Return the 4.5K extension: three additional quotas per 1.5K quota."""
    cells = [
        dataclasses.replace(cell, count=cell.count * 3) for cell in SFT15.build_cells()
    ]
    assert len(cells) == EXPECTED_CELLS
    assert sum(cell.count for cell in cells) == EXPECTED_ADDITIONS
    return cells


def expected_train_cells() -> Counter:
    return Counter({cell.name: cell.count * 4 for cell in SFT15.build_cells()})


def validate(
    train_path: Path,
    val_path: Path,
    manifest_path: Path,
    base_train_path: Path,
    base_val_path: Path,
    base_manifest_path: Path,
    held_out: list[Path],
) -> None:
    SFT15.validate(base_train_path, base_val_path, base_manifest_path, held_out)
    train = V1.load_jsonl(train_path)
    val = V1.load_jsonl(val_path)
    base_train = V1.load_jsonl(base_train_path)
    base_val = V1.load_jsonl(base_val_path)
    external = [row for path in held_out for row in V1.load_jsonl(path)]
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("version") != VERSION or manifest.get("seed") != SEED:
        raise ValueError("6K manifest version or seed mismatch")
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise ValueError(f"unexpected sizes: train={len(train)} val={len(val)}")
    if len(base_train) != SFT15.EXPECTED_TRAIN or len(base_val) != EXPECTED_VAL:
        raise ValueError("unexpected frozen 1.5K input sizes")

    # Strong nesting contract: exact rows, exact order, and exact serialized
    # bytes from the frozen 1.5K file are retained at the start of 6K.
    if train[: len(base_train)] != base_train:
        raise ValueError("6K does not contain the exact 1.5K rows as its prefix")
    if val != base_val:
        raise ValueError("6K validation is not the exact frozen 1.5K validation")
    if not train_path.read_bytes().startswith(base_train_path.read_bytes()):
        raise ValueError("6K serialized prefix differs from frozen 1.5K bytes")
    if val_path.read_bytes() != base_val_path.read_bytes():
        raise ValueError("6K validation bytes differ from frozen 1.5K validation")
    if manifest.get("base_train_sha256") != sha256(base_train_path):
        raise ValueError("6K manifest has stale 1.5K train hash")
    if manifest.get("base_validation_sha256") != sha256(base_val_path):
        raise ValueError("6K manifest has stale 1.5K validation hash")
    if manifest.get("base_manifest_sha256") != sha256(base_manifest_path):
        raise ValueError("6K manifest has stale 1.5K manifest hash")

    train_prompts = _hashes(train, V1.fingerprint)
    val_prompts = _hashes(val, V1.fingerprint)
    external_prompts = _hashes(external, V1.fingerprint)
    train_worlds = _hashes(train, V1.world_fingerprint)
    val_worlds = _hashes(val, V1.world_fingerprint)
    external_worlds = _hashes(external, V1.world_fingerprint)
    if len(train_prompts) != EXPECTED_TRAIN or len(val_prompts) != EXPECTED_VAL:
        raise ValueError("duplicate prompt in 6K data")
    if len(train_worlds) != EXPECTED_TRAIN or len(val_worlds) != EXPECTED_VAL:
        raise ValueError("duplicate premise world in 6K data")
    if train_prompts & val_prompts or (train_prompts | val_prompts) & external_prompts:
        raise ValueError("6K prompt leakage")
    if train_worlds & val_worlds or (train_worlds | val_worlds) & external_worlds:
        raise ValueError("6K world leakage")
    if any(
        max(
            row["difficulty"].get("x_depth") or 0, row["difficulty"].get("y_depth") or 0
        )
        >= 5
        for row in train + val
    ):
        raise ValueError("depth 5 must remain held out")
    if Counter(row["generation_cell"] for row in train) != expected_train_cells():
        raise ValueError("6K train cell distribution mismatch")
    category_counts = Counter(
        "closed_loop_rows"
        if row["generation_cell"].endswith("-closed")
        else "open_chain_rows"
        if row["generation_cell"].endswith("-open-control")
        else "ordinary_consistent_rows"
        for row in train
    )
    expected_categories = Counter(
        {
            "ordinary_consistent_rows": 3760,
            "closed_loop_rows": 1120,
            "open_chain_rows": 1120,
        }
    )
    if category_counts != expected_categories:
        raise ValueError("6K category distribution mismatch")
    if any(manifest.get(key) != value for key, value in expected_categories.items()):
        raise ValueError("6K manifest category distribution mismatch")
    if Counter(row["generation_cell"] for row in val) != Counter(
        {cell.name: cell.count for cell in SFT15.build_cells(validation=True)}
    ):
        raise ValueError("6K validation cell distribution mismatch")


def generate(
    output: Path,
    base_train_path: Path,
    base_val_path: Path,
    base_manifest_path: Path,
    held_out: list[Path],
) -> tuple[Path, Path, Path]:
    SFT15.validate(base_train_path, base_val_path, base_manifest_path, held_out)
    train_path, val_path, manifest_path = paths(output)
    base_train = V1.load_jsonl(base_train_path)
    base_val = V1.load_jsonl(base_val_path)
    external = [row for path in held_out for row in V1.load_jsonl(path)]

    forbidden = external + base_train + base_val
    additions_path = output.with_name(output.stem + "_additions.tmp.jsonl")
    rejections = SFT15.generate_unique_partition(
        additional_cells(),
        additions_path,
        SEED,
        _hashes(forbidden, V1.fingerprint),
        _hashes(forbidden, V1.world_fingerprint),
    )
    additions = V1.load_jsonl(additions_path)
    if len(additions) != EXPECTED_ADDITIONS:
        raise ValueError(
            f"expected {EXPECTED_ADDITIONS} additions, found {len(additions)}"
        )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as target:
        base_bytes = base_train_path.read_bytes()
        target.write(base_bytes)
        if base_bytes and not base_bytes.endswith(b"\n"):
            target.write(b"\n")
        for row in additions:
            target.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    val_path.write_bytes(base_val_path.read_bytes())
    additions_path.unlink()

    train = V1.load_jsonl(train_path)
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "train_rows": len(train),
        "nested_base_rows": len(base_train),
        "new_rows": len(additions),
        "validation_rows": len(base_val),
        "ordinary_consistent_rows": 3760,
        "closed_loop_rows": 1120,
        "open_chain_rows": 1120,
        "generation_cells": EXPECTED_CELLS,
        "base_train_sha256": sha256(base_train_path),
        "base_validation_sha256": sha256(base_val_path),
        "base_manifest_sha256": sha256(base_manifest_path),
        "train_cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in train).items())
        ),
        "held_out_sources": [str(path) for path in held_out],
        "prompt_overlap": 0,
        "world_overlap": 0,
        "addition_generation_rejections": dict(sorted(rejections.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(
        train_path,
        val_path,
        manifest_path,
        base_train_path,
        base_val_path,
        base_manifest_path,
        held_out,
    )
    return train_path, val_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "spatial_v13_sft_6000.jsonl"
    )
    parser.add_argument(
        "--base-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_train.jsonl",
    )
    parser.add_argument(
        "--base-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_val.jsonl",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_1500_manifest.json",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl",
    )
    parser.add_argument(
        "--probe-v1-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_train.jsonl",
    )
    parser.add_argument(
        "--probe-v1-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_val.jsonl",
    )
    parser.add_argument(
        "--probe-v2-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_v2_train.jsonl",
    )
    parser.add_argument(
        "--probe-v2-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_probe_v2_val.jsonl",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    held_out = [
        args.diagnostic,
        args.probe_v1_train,
        args.probe_v1_val,
        args.probe_v2_train,
        args.probe_v2_val,
    ]
    required = [args.base_train, args.base_val, args.base_manifest, *held_out]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required input: " + ", ".join(missing))
    output_paths = paths(args.out)
    if args.validate_only:
        validate(
            *output_paths, args.base_train, args.base_val, args.base_manifest, held_out
        )
        print("valid nested 6K data: " + ", ".join(str(path) for path in output_paths))
        return
    output_paths = generate(
        args.out, args.base_train, args.base_val, args.base_manifest, held_out
    )
    print("wrote " + ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
