"""Validate that a generated file is the current frozen V13 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_ROWS = 2256
EXPECTED_CELLS = 233
EXPECTED_SCHEMA = 4
EXPECTED_GENERATOR = "v13.6-orthogonal-taxonomy"
SEMANTIC_SUBTYPES = {
    "dir-1",
    "dir-2",
    "dir-undetermined",
    "dir-incomplete",
    "dir-omit",
    "which-1",
    "which-2",
    "which-3",
    "which-4",
    "which-0",
    "count-1",
    "count-omit",
}


def validate(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    cells = {str(row.get("generation_cell") or "") for row in rows}
    if len(cells) != EXPECTED_CELLS or "" in cells:
        raise ValueError(f"expected {EXPECTED_CELLS} named cells, found {len(cells)}")
    schemas = {row.get("difficulty_schema_version") for row in rows}
    if schemas != {EXPECTED_SCHEMA}:
        raise ValueError(f"expected schema {EXPECTED_SCHEMA}, found {schemas}")
    generators = {row.get("generator_version") for row in rows}
    if generators != {EXPECTED_GENERATOR}:
        raise ValueError(
            f"expected generator {EXPECTED_GENERATOR}, found {generators}"
        )
    for index, row in enumerate(rows, 1):
        subtype = row.get("semantic_subtype")
        difficulty = row.get("difficulty") or {}
        difficulty_subtype = difficulty.get("semantic_subtype")
        if not subtype or subtype != difficulty_subtype:
            raise ValueError(
                f"row {index} has inconsistent semantic subtype: "
                f"row={subtype!r}, difficulty={difficulty_subtype!r}"
            )
        if subtype not in SEMANTIC_SUBTYPES:
            raise ValueError(f"row {index} contains invalid semantic subtype {subtype!r}")
        if difficulty.get("question_family") not in {"direction", "which", "count"}:
            raise ValueError(f"row {index} has invalid question family")
        if difficulty.get("world_consistency") not in {"consistent", "inconsistent"}:
            raise ValueError(f"row {index} has invalid world consistency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        validate(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid V13 diagnostic: {exc}")
        raise SystemExit(1) from exc
    print(f"valid V13 diagnostic: {args.path}")


if __name__ == "__main__":
    main()
