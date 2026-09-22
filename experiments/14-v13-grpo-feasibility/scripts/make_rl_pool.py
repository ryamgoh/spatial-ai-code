"""Generate a solver-verified hard V13 prompt pool for GRPO feasibility."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V13_SCRIPTS = REPO / "experiments" / "13-iterative-hardening" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BREAK = load_module(
    "v14_rl_breakpoint_helpers", V13_SCRIPTS / "make_breakpoint_data.py"
)
V1 = load_module("v14_rl_probe_helpers", V13_SCRIPTS / "make_probe_data.py")

SEED = 14001
VERSION = "v14-v13-grpo-feasibility-v1"
EXPECTED_TRAIN = 680
EXPECTED_EVAL = 300
ANSWER_LINE = (
    "\n\nUse compact delta-state reasoning. For each premise, track only the "
    "new X/Y extraction, the affected component, and whether a global conflict "
    "is present. Do not rewrite the complete map after every premise. After "
    "your reasoning, close with </think> and end with exactly one final line "
    "like `Answer: A`, `Answer: A, C`, or `Answer: E`."
)


def depth_cells(count: int, *, evaluation: bool) -> list:
    cells = []
    suffix = "eval" if evaluation else "train"
    symmetric_count = 8 if evaluation else 24
    unequal_count = 6 if evaluation else 16
    for depth in (10, 12, 14):
        for mode in (V1.RelationMode.CARDINAL, V1.RelationMode.MIXED):
            for policy in (
                V1.DistractorPolicy.DISCONNECTED,
                V1.DistractorPolicy.QUERY_BRANCH,
            ):
                cell = BREAK.depth_cell(
                    f"rl-{suffix}-d{depth}-{policy.value}",
                    mode,
                    depth,
                    depth,
                    policy,
                )
                cells.append(dataclasses.replace(cell, count=symmetric_count))
    for x_depth, y_depth in ((8, 12), (12, 8), (10, 14), (14, 10)):
        for mode in (V1.RelationMode.CARDINAL, V1.RelationMode.MIXED):
            cell = BREAK.depth_cell(
                f"rl-{suffix}-unequal",
                mode,
                x_depth,
                y_depth,
                V1.DistractorPolicy.QUERY_BRANCH,
            )
            cells.append(dataclasses.replace(cell, count=unequal_count))
    assert sum(cell.count for cell in cells) == count
    return cells


def loop_configurations() -> list[tuple]:
    values = []
    configs = BREAK.LOOP_CONFIGURATIONS
    for family_index, (family, subtype) in enumerate(BREAK.LOOP_FAMILIES):
        for length_index, length in enumerate((10, 12, 14)):
            for placement_index, placement in enumerate(V1.CyclePlacement):
                mode, axes = configs[
                    (family_index + length_index + placement_index) % len(configs)
                ]
                values.append((family, subtype, mode, axes, placement, length))
    assert len(values) == 18
    return values


def loop_cells(count: int, *, evaluation: bool) -> list:
    per_condition = 3 if evaluation else 6
    suffix = "eval" if evaluation else "train"
    cells = []
    for index, (family, subtype, mode, axes, placement, length) in enumerate(
        loop_configurations(), 1
    ):
        closed, opened = BREAK.loop_pair(family, subtype, mode, axes, placement, length)
        cells.extend(
            (
                dataclasses.replace(
                    closed,
                    name=f"rl-{suffix}-loop-{index}-{family}-l{length}-closed",
                    count=per_condition,
                ),
                dataclasses.replace(
                    opened,
                    name=f"rl-{suffix}-loop-{index}-{family}-l{length}-open-control",
                    count=per_condition,
                ),
            )
        )
    assert sum(cell.count for cell in cells) == count
    return cells


def semantic_cells(count: int, *, evaluation: bool) -> list:
    suffix = "eval" if evaluation else "train"
    per_cell = 6
    configurations = (
        (V1.SemanticSubtype.WHICH_1, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.WHICH_2, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_3, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_4, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.WHICH_0, V1.RelationMode.CARDINAL),
        (V1.SemanticSubtype.COUNT_1, V1.RelationMode.MIXED),
        (V1.SemanticSubtype.COUNT_OMIT, V1.RelationMode.MIXED),
    )
    cells = [
        V1.GenerationCell(
            name=f"rl-{suffix}-large-{index}-{subtype.value}",
            spec=V1.GenerationSpec(
                semantic_subtype=subtype,
                relation_mode=mode,
                num_entities=32,
                num_relations=44,
                max_attempts=3000,
                entity_pool=BREAK.CHALLENGE_ENTITY_NAMES,
            ),
            count=per_cell,
        )
        for index, (subtype, mode) in enumerate(configurations, 1)
    ]
    assert sum(cell.count for cell in cells) == count
    return cells


def build_cells(*, evaluation: bool) -> list:
    if evaluation:
        cells = (
            depth_cells(144, evaluation=True)
            + loop_cells(108, evaluation=True)
            + semantic_cells(48, evaluation=True)
        )
        assert sum(cell.count for cell in cells) == EXPECTED_EVAL
    else:
        cells = (
            depth_cells(416, evaluation=False)
            + loop_cells(216, evaluation=False)
            + semantic_cells(48, evaluation=False)
        )
        assert sum(cell.count for cell in cells) == EXPECTED_TRAIN
    assert len({cell.name for cell in cells}) == len(cells)
    return cells


def user_text(row: dict) -> str:
    return next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def world_fingerprint(text: str) -> str:
    return BREAK.world_fingerprint(text)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hashes(rows: list[dict], function) -> set[str]:
    return {function(user_text(row)) for row in rows}


def requested_scale(row: dict) -> int:
    cell = str(row.get("generation_cell") or "")
    loop = re.search(r"-l(\d+)-(?:closed|open-control)$", cell)
    if loop:
        return int(loop.group(1))
    depth = re.search(r"-x(\d+)-y(\d+)-", cell)
    if depth:
        return max(map(int, depth.groups()))
    return 0


def generate_partition(
    cells: list, seed: int, forbidden: list[dict]
) -> tuple[list[dict], Counter]:
    accepted_prompts = _hashes(forbidden, fingerprint)
    accepted_worlds = _hashes(forbidden, world_fingerprint)
    engine = V1.SpatialGenerator()
    seed_rng = random.Random(seed)
    rows = []
    rejections: Counter = Counter()
    for cell in cells:
        accepted = 0
        attempts = 0
        while accepted < cell.count:
            attempts += 1
            if attempts > cell.count * 3000:
                raise ValueError(
                    f"unable to fill {cell.name}: {accepted}/{cell.count}; "
                    f"rejections={dict(rejections)}"
                )
            example = engine.generate(
                cell.spec, random.Random(seed_rng.randrange(2**63))
            )
            row = example.to_row(generation_cell=cell.name)
            row["generator_version"] = VERSION
            prompt_hash = fingerprint(user_text(row))
            world_hash = world_fingerprint(user_text(row))
            if prompt_hash in accepted_prompts:
                rejections["prompt_collision"] += 1
                continue
            if world_hash in accepted_worlds:
                rejections["world_collision"] += 1
                continue
            rows.append(row)
            accepted_prompts.add(prompt_hash)
            accepted_worlds.add(world_hash)
            accepted += 1
    seed_rng.shuffle(rows)
    return rows, rejections


def to_rl_row(row: dict) -> dict:
    prompt = []
    for message in row["messages"]:
        if message.get("role") not in {"system", "user"}:
            continue
        item = dict(message)
        if item["role"] == "user":
            item["content"] = str(item.get("content") or "") + ANSWER_LINE
        prompt.append(item)
    return {
        "prompt": prompt,
        "oracle_option": row["oracle_option"],
        "generation_cell": row["generation_cell"],
        "difficulty": row["difficulty"],
    }


def paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(output.stem + "_train.jsonl"),
        output.with_name(output.stem + "_eval.jsonl"),
        output.with_name(output.stem + "_manifest.json"),
    )


def validate(
    train_path: Path, eval_path: Path, manifest_path: Path, held_out: list[Path]
) -> None:
    train, evaluation = load_jsonl(train_path), load_jsonl(eval_path)
    external = [row for path in held_out for row in load_jsonl(path)]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or manifest.get("seed") != SEED:
        raise ValueError("RL pool manifest version or seed mismatch")
    if len(train) != EXPECTED_TRAIN or len(evaluation) != EXPECTED_EVAL:
        raise ValueError("RL pool size mismatch")
    train_prompts = {
        fingerprint(user_text({"messages": row["prompt"]})) for row in train
    }
    eval_prompts = _hashes(evaluation, fingerprint)
    external_prompts = _hashes(external, fingerprint)
    train_worlds = {
        world_fingerprint(user_text({"messages": row["prompt"]})) for row in train
    }
    eval_worlds = _hashes(evaluation, world_fingerprint)
    external_worlds = _hashes(external, world_fingerprint)
    if len(train_prompts) != EXPECTED_TRAIN or len(train_worlds) != EXPECTED_TRAIN:
        raise ValueError("duplicate RL train prompt or world")
    if len(eval_prompts) != EXPECTED_EVAL or len(eval_worlds) != EXPECTED_EVAL:
        raise ValueError("duplicate RL eval prompt or world")
    if train_prompts & eval_prompts or train_worlds & eval_worlds:
        raise ValueError("RL train/eval leakage")
    if (train_prompts | eval_prompts) & external_prompts:
        raise ValueError("RL prompt leakage into frozen V13 data")
    if (train_worlds | eval_worlds) & external_worlds:
        raise ValueError("RL world leakage into frozen V13 data")
    if any(
        requested_scale(row) < 10
        and not str(row["generation_cell"]).startswith("rl-eval-large-")
        for row in evaluation
    ):
        raise ValueError("RL holdout contains an unexpectedly easy structural row")


def generate(output: Path, held_out: list[Path]) -> tuple[Path, Path, Path]:
    train_path, eval_path, manifest_path = paths(output)
    external = [row for path in held_out for row in load_jsonl(path)]
    train_rows, train_rejections = generate_partition(
        build_cells(evaluation=False), SEED, external
    )
    eval_rows, eval_rejections = generate_partition(
        build_cells(evaluation=True), SEED + 1, external + train_rows
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        "".join(
            json.dumps(to_rl_row(row), ensure_ascii=False) + "\n" for row in train_rows
        )
    )
    eval_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows)
    )
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in train_rows).items())
        ),
        "eval_cell_counts": dict(
            sorted(Counter(row["generation_cell"] for row in eval_rows).items())
        ),
        "held_out_sources": [str(path) for path in held_out],
        "train_generation_rejections": dict(sorted(train_rejections.items())),
        "eval_generation_rejections": dict(sorted(eval_rejections.items())),
        "prompt_overlap": 0,
        "world_overlap": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate(train_path, eval_path, manifest_path, held_out)
    return train_path, eval_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "spatial_v14_grpo_pool.jsonl"
    )
    parser.add_argument(
        "--v13-diagnostic",
        type=Path,
        default=REPO / "data" / "spatial_v13_diagnostic_test.jsonl",
    )
    parser.add_argument(
        "--v13-breakpoint",
        type=Path,
        default=REPO / "data" / "spatial_v13_breakpoint_test.jsonl",
    )
    parser.add_argument(
        "--v13-sft-train",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_8000_train.jsonl",
    )
    parser.add_argument(
        "--v13-sft-val",
        type=Path,
        default=REPO / "data" / "spatial_v13_sft_8000_val.jsonl",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    held_out = [
        args.v13_diagnostic,
        args.v13_breakpoint,
        args.v13_sft_train,
        args.v13_sft_val,
    ]
    missing = [str(path) for path in held_out if not path.is_file()]
    if missing:
        raise SystemExit("missing held-out source: " + ", ".join(missing))
    output_paths = paths(args.out)
    if args.validate_only:
        validate(*output_paths, held_out)
        print("valid V14 GRPO pool: " + ", ".join(map(str, output_paths)))
        return
    output_paths = generate(args.out, held_out)
    print("wrote " + ", ".join(map(str, output_paths)))


if __name__ == "__main__":
    main()
