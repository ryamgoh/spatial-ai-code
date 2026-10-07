"""Audit released SpatialMap-TQA rows and derive SpatialMap-TQA-Corr."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from spatial_grading_v2 import AnswerMode, encode_menu_answer, resolve_answer
from spatial_solver_v2 import (
    Direction,
    DirectionAnalysis,
    DirectionQuery,
    QueryAnalysis,
    SpatialProblem,
    SpatialSolverV2,
    WhichAnalysis,
    WhichQuery,
    conjunctive_atoms,
    direction_between,
)
from spatialeval_adapter_v2 import SpatialEvalAdapter, SpatialEvalAudit, audit

AUDIT_FILE = "spatialeval_audit.jsonl"
CORRECTED_FILE = "spatialmap_tqa_corr.jsonl"
SUMMARY_FILE = "spatialeval_audit_summary.json"
UNDETERMINED = "Cannot be determined"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class AuditArtifacts:
    audit: Path
    corrected: Path
    summary: Path


def _json_bytes(value: Any, *, lines: bool = False) -> bytes:
    if lines:
        return "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in value
        ).encode()
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("input dataset is empty")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not row_id for row_id in ids):
        raise ValueError("every input row must have an ID")
    if len(ids) != len(set(ids)):
        raise ValueError("input dataset contains duplicate IDs")
    return rows, content


def _render_value(value: Direction | str | int) -> str:
    return value.value if isinstance(value, Direction) else str(value)


def _possible_values(analysis: QueryAnalysis) -> tuple[Direction | str | int, ...]:
    if isinstance(analysis, DirectionAnalysis):
        return analysis.possible_directions
    if isinstance(analysis, WhichAnalysis):
        return analysis.possible_entities
    return analysis.possible_counts


def _entailed_values(analysis: QueryAnalysis) -> tuple[Direction | str | int, ...]:
    if isinstance(analysis, WhichAnalysis):
        return analysis.entailed_entities
    possible = _possible_values(analysis)
    return possible if len(possible) == 1 else ()


def _query_kind(problem: SpatialProblem) -> str:
    if isinstance(problem.query, DirectionQuery):
        return "direction"
    if isinstance(problem.query, WhichQuery):
        return "which"
    return "count"


def _coordinates(value: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    return {name: list(point) for name, point in sorted(value.items())}


def _validate_premises(
    problem: SpatialProblem,
    coordinates: dict[str, tuple[int, int]],
) -> None:
    atoms = conjunctive_atoms(problem.premise)
    if atoms is None:
        raise ValueError("SpatialEval audit requires positive conjunctive premises")
    for atom in atoms:
        actual = direction_between(
            coordinates[atom.subject],
            coordinates[atom.reference],
        )
        if actual not in atom.allowed:
            raise ValueError("solver witness violates a SpatialEval premise")


def _validate_witnesses(problem: SpatialProblem, analysis: QueryAnalysis) -> None:
    for answer, coordinates in analysis.witnesses.items():
        _validate_premises(problem, coordinates)
        query = problem.query
        if isinstance(query, DirectionQuery):
            actual: Direction | str | int = direction_between(
                coordinates[query.target], coordinates[query.reference]
            )
        elif isinstance(query, WhichQuery):
            actual = answer
            if direction_between(coordinates[answer], coordinates[query.reference]) not in query.directions:
                raise ValueError("Which witness does not realize its possible entity")
        else:
            actual = sum(
                direction_between(coordinates[candidate], coordinates[query.reference])
                in query.directions
                for candidate in query.candidates
            )
        if actual != answer:
            raise ValueError("solver witness does not realize its answer")


def _audit_record(row: dict[str, Any], result: SpatialEvalAudit) -> dict[str, Any]:
    analysis = result.analysis
    _validate_witnesses(result.case.problem, analysis)
    witnesses = {
        _render_value(answer): _coordinates(coordinates)
        for answer, coordinates in analysis.witnesses.items()
    }
    possible = [_render_value(value) for value in _possible_values(analysis)]
    entailed = [_render_value(value) for value in _entailed_values(analysis)]
    return {
        "id": result.case.row_id,
        "text": row["text"],
        "query_kind": _query_kind(result.case.problem),
        "original_oracle": {
            "answer": row.get("oracle_answer"),
            "option": row.get("oracle_option"),
            "full_answer": row.get("oracle_full_answer"),
        },
        "status": result.status,
        "consistent": analysis.consistent,
        "error": analysis.error,
        "possible_answers": possible,
        "entailed_answers": entailed,
        "possibility_count": len(possible),
        "objects": list(result.case.problem.objects),
        "premise_count": len(conjunctive_atoms(result.case.problem.premise) or ()),
        "options": result.case.options,
        "resolution": {
            "status": result.menu_answer.resolution.status.value,
            "values": [
                _render_value(value) for value in result.menu_answer.resolution.values
            ],
            "menu_status": result.menu_answer.status,
            "letters": sorted(result.menu_answer.letters),
            "error": result.menu_answer.error,
        },
        "solver_engine": analysis.engine,
        "witnesses": witnesses,
    }


def _append_undetermined_option(text: str, options: dict[str, str]) -> str:
    if "E" in options:
        raise ValueError("original SpatialEval row already contains option E")
    return text.rstrip() + f"\nE. {UNDETERMINED}."


def _corrected_row(
    row: dict[str, Any],
    result: SpatialEvalAudit,
    adapter: SpatialEvalAdapter,
) -> dict[str, Any]:
    if result.status not in {"exact-match", "underdetermined-oracle-possible"}:
        raise ValueError(
            f"cannot correct {result.case.row_id}: audit status is {result.status}"
        )
    corrected = dict(row)
    corrected["text"] = _append_undetermined_option(
        str(row["text"]), result.case.options
    )
    options = {**result.case.options, "E": UNDETERMINED}
    resolution = resolve_answer(result.analysis, AnswerMode.SINGLE)
    menu_answer = encode_menu_answer(resolution, options)
    if not menu_answer.is_resolved or len(menu_answer.letters) != 1:
        raise ValueError(
            f"cannot encode corrected answer for {result.case.row_id}: "
            f"{menu_answer.error or menu_answer.status}"
        )
    letter = next(iter(menu_answer.letters))
    corrected["oracle_option"] = letter
    corrected["oracle_answer"] = options[letter]
    corrected["oracle_full_answer"] = f"{letter}. {options[letter]}"
    validation_row = {**row, "text": corrected["text"]}
    parsed = adapter.parse(validation_row, AnswerMode.SINGLE)
    if parsed.problem != result.case.problem or parsed.options != options:
        raise ValueError(
            f"corrected prompt changed the problem for {result.case.row_id}"
        )
    reparsed_answer = encode_menu_answer(
        resolve_answer(result.analysis, AnswerMode.SINGLE),
        parsed.options,
    )
    if reparsed_answer.letters != menu_answer.letters:
        raise ValueError(
            f"corrected prompt changed the answer for {result.case.row_id}"
        )
    return corrected


def _artifact_paths(output_dir: Path) -> AuditArtifacts:
    return AuditArtifacts(
        output_dir / AUDIT_FILE,
        output_dir / CORRECTED_FILE,
        output_dir / SUMMARY_FILE,
    )


def _write_artifacts(
    artifacts: AuditArtifacts,
    contents: dict[Path, bytes],
) -> None:
    artifacts.audit.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=artifacts.audit.parent,
        prefix=".spatialeval-audit-",
    ) as temporary:
        temporary_dir = Path(temporary)
        staged = {}
        for destination, content in contents.items():
            path = temporary_dir / destination.name
            path.write_bytes(content)
            staged[destination] = path
        for destination, path in staged.items():
            os.replace(path, destination)


def audit_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "z3",
    replace: bool = False,
    expected_rows: int | None = None,
) -> AuditArtifacts:
    """Audit one released JSONL dataset and write deterministic artifacts."""
    input_path = Path(input_file)
    artifacts = _artifact_paths(Path(output_dir))
    existing = [path for path in vars(artifacts).values() if path.exists()]
    if existing and not replace:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"audit outputs already exist: {names}")

    rows, input_content = _read_rows(input_path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")

    adapter = SpatialEvalAdapter()
    solver = SpatialSolverV2(backend=backend)
    audit_rows = []
    corrected_rows = []
    status_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    status_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    possibility_counts: Counter[int] = Counter()

    for row in rows:
        case = adapter.parse(row, AnswerMode.SINGLE)
        result = audit(case, solver.analyze(case.problem))
        record = _audit_record(row, result)
        corrected = _corrected_row(row, result, adapter)
        audit_rows.append(record)
        corrected_rows.append(corrected)
        query_kind = record["query_kind"]
        status_counts[result.status] += 1
        query_counts[query_kind] += 1
        status_by_query[query_kind][result.status] += 1
        possibility_counts[record["possibility_count"]] += 1

    audit_content = _json_bytes(audit_rows, lines=True)
    corrected_content = _json_bytes(corrected_rows, lines=True)
    summary = {
        "schema": "spatialeval-v2-audit-1",
        "input": {
            "file": input_path.name,
            "rows": len(rows),
            "sha256": _sha256(input_content),
        },
        "solver": {"backend": backend, "answer_mode": AnswerMode.SINGLE.value},
        "counts": {
            "status": dict(sorted(status_counts.items())),
            "query_kind": dict(sorted(query_counts.items())),
            "status_by_query": {
                kind: dict(sorted(counts.items()))
                for kind, counts in sorted(status_by_query.items())
            },
            "possibility_count": {
                str(count): total for count, total in sorted(possibility_counts.items())
            },
        },
        "outputs": {
            "audit": {
                "file": artifacts.audit.name,
                "rows": len(audit_rows),
                "sha256": _sha256(audit_content),
            },
            "corrected": {
                "file": artifacts.corrected.name,
                "rows": len(corrected_rows),
                "sha256": _sha256(corrected_content),
            },
        },
    }
    summary_content = _json_bytes(summary)
    _write_artifacts(
        artifacts,
        {
            artifacts.audit: audit_content,
            artifacts.corrected: corrected_content,
            artifacts.summary: summary_content,
        },
    )
    return artifacts


@app.command()
def main(
    input_file: Path = typer.Option(  # noqa: B008
        Path("data/spatialeval_org.jsonl"),
        "--input",
        help="Released SpatialEval JSONL file.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("results/part1-spatialeval-audit"),
        "--output-dir",
        help="Directory for audit, corrected data, and summary artifacts.",
    ),
    backend: str = typer.Option("z3", help="Solver backend: z3 or reference."),
    replace: bool = typer.Option(False, help="Replace this command's outputs."),
    expected_rows: int = typer.Option(1500, min=1),
) -> None:
    try:
        artifacts = audit_dataset(
            input_file,
            output_dir,
            backend=backend,
            replace=replace,
            expected_rows=expected_rows,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"audit: {artifacts.audit}")
    typer.echo(f"corrected: {artifacts.corrected}")
    typer.echo(f"summary: {artifacts.summary}")


if __name__ == "__main__":
    app()
