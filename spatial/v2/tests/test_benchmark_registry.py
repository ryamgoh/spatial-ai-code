"""Public-contract tests for the SpatialEntail benchmark-cell registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial.v2.legacy.benchmark_registry import (
    CapabilityLens,
    CellReadiness,
    EvidenceObligation,
    ReasoningStructure,
    SemanticStatus,
    generate_registry_pilot,
    load_benchmark_registry,
    probe_registry,
)
from spatial.v2.legacy.check_registry import main
from spatial.v2.legacy.generation import QueryKind


REGISTRY_YAML = """
version: 1
cells:
  - id: DIR-TRC-UNI
    readiness: ready
    query_kind: direction
    capability_lenses: [RR, MTR]
    reasoning_structures: [TRC]
    semantic_status: UNI
    evidence_obligations: [positive-proof, consistency-witness, candidate-exclusion]
    generation:
      semantic_shape: unique
      boolean_shape: atomic
      num_entities: 3
      num_premises: 2
      depth: 2
      omit_direct_query_relation: true

  - id: CNT-CSR-JCR-UNI
    readiness: blocked
    query_kind: count
    capability_lenses: [LR, MTR]
    reasoning_structures: [CSR]
    semantic_status: UNI
    evidence_obligations: [possibility-witness, candidate-exclusion, joint-count]
    generation:
      semantic_shape: unique
      boolean_shape: case-split
      generation_mode: proof-template-first
      num_entities: 6
      num_premises: 7
    prerequisites:
      - Add a generator admission check for non-factorizable count dependencies.
"""


def _write_registry(tmp_path: Path, source: str = REGISTRY_YAML) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(source)
    return path


def test_registry_loads_typed_ready_and_blocked_cells(tmp_path: Path) -> None:
    registry = load_benchmark_registry(_write_registry(tmp_path))

    assert registry.version == 1
    assert [cell.id for cell in registry.cells] == [
        "DIR-TRC-UNI",
        "CNT-CSR-JCR-UNI",
    ]
    ready, blocked = registry.cells
    assert ready.readiness is CellReadiness.READY
    assert ready.query_kind is QueryKind.DIRECTION
    assert ready.capability_lenses == (
        CapabilityLens.RELATIONAL,
        CapabilityLens.MODEL_THEORETIC,
    )
    assert ready.reasoning_structures == (ReasoningStructure.TRANSITIVE,)
    assert ready.semantic_status is SemanticStatus.UNIQUE
    assert ready.evidence_obligations == (
        EvidenceObligation.POSITIVE_PROOF,
        EvidenceObligation.CONSISTENCY_WITNESS,
        EvidenceObligation.CANDIDATE_EXCLUSION,
    )
    assert blocked.readiness is CellReadiness.BLOCKED
    assert blocked.prerequisites


def test_registry_rejects_duplicate_cell_ids(tmp_path: Path) -> None:
    duplicate = REGISTRY_YAML + REGISTRY_YAML.split("cells:\n", 1)[1].split(
        "  - id: CNT", 1
    )[0]

    with pytest.raises(ValueError, match="duplicate benchmark cell IDs"):
        load_benchmark_registry(_write_registry(tmp_path, duplicate))


def test_registry_rejects_lowercase_cell_id(tmp_path: Path) -> None:
    source = REGISTRY_YAML.split("\n  - id: CNT", 1)[0].replace(
        "DIR-TRC-UNI", "dir-trc-uni"
    )

    with pytest.raises(ValueError, match="uppercase dash-separated code"):
        load_benchmark_registry(_write_registry(tmp_path, source))


def test_ready_cell_requires_a_valid_generation_policy(tmp_path: Path) -> None:
    source = REGISTRY_YAML.split("\n  - id: CNT", 1)[0].replace(
        "num_entities: 3\n      num_premises: 2",
        "num_entities: 2\n      num_premises: 1",
    )

    with pytest.raises(ValueError, match="ready generation policy is invalid"):
        load_benchmark_registry(_write_registry(tmp_path, source))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("semantic_status: UNI", "semantic_status: ALT", "requires semantic_shape"),
        ("reasoning_structures: [TRC]", "reasoning_structures: [MP]", "requires boolean_shape"),
        ("readiness: ready", "readiness: blocked", "blocked cell.*prerequisites"),
    ],
)
def test_registry_rejects_taxonomy_policy_disagreement(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    first_cell = REGISTRY_YAML.split("\n  - id: CNT", 1)[0]
    source = first_cell.replace(old, new)

    with pytest.raises(ValueError, match=message):
        load_benchmark_registry(_write_registry(tmp_path, source))


def test_probe_registry_generates_ready_cells_and_reports_blockers(
    tmp_path: Path,
) -> None:
    registry = load_benchmark_registry(_write_registry(tmp_path))

    report = probe_registry(registry, seed=1901, max_attempts=100)

    assert report.ready == ("DIR-TRC-UNI",)
    assert report.failed == ()
    assert report.blocked == {
        "CNT-CSR-JCR-UNI": (
            "Add a generator admission check for non-factorizable count dependencies.",
        )
    }


def test_canonical_registry_has_no_allocations_and_covers_public_dimensions() -> None:
    path = Path(__file__).parents[1] / "benchmark-cells.yaml"

    registry = load_benchmark_registry(path)

    assert {cell.query_kind for cell in registry.cells} == set(QueryKind)
    assert {
        lens for cell in registry.cells for lens in cell.capability_lenses
    } == set(CapabilityLens)
    assert {
        structure for cell in registry.cells for structure in cell.reasoning_structures
    } == set(ReasoningStructure)
    assert {cell.semantic_status for cell in registry.cells} == set(SemanticStatus)
    assert all("count" not in cell.generation for cell in registry.cells)
    assert registry.cells
    assert all(cell.readiness is CellReadiness.READY for cell in registry.cells)


def test_registry_cli_reports_static_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_registry(tmp_path)

    exit_code = main([str(path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "blocked": {
            "CNT-CSR-JCR-UNI": [
                "Add a generator admission check for non-factorizable count dependencies."
            ]
        },
        "failed": [],
        "ready": ["DIR-TRC-UNI"],
        "version": 1,
    }


def test_registry_cli_writes_a_pilot_receipt(tmp_path: Path) -> None:
    path = _write_registry(tmp_path)
    output = tmp_path / "pilot.jsonl"

    exit_code = main(
        [
            str(path),
            "--pilot-out",
            str(output),
            "--samples-per-ready-cell",
            "2",
            "--seed",
            "1902",
            "--max-attempts",
            "100",
        ]
    )

    assert exit_code == 0
    assert len(output.read_text().splitlines()) == 2
    assert (tmp_path / "pilot_manifest.json").exists()


def test_registry_pilot_persists_checked_rows_and_audit_manifest(
    tmp_path: Path,
) -> None:
    registry = load_benchmark_registry(_write_registry(tmp_path))

    rows_path, manifest_path = generate_registry_pilot(
        registry,
        tmp_path / "pilot.jsonl",
        samples_per_ready_cell=2,
        seed=1902,
        max_attempts=100,
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert len(rows) == 2
    assert {row["metadata"]["benchmark_cell"] for row in rows} == {
        "DIR-TRC-UNI"
    }
    assert all(row["metadata"]["trace_format"] == "natural" for row in rows)
    assert all(row["metadata"]["expected_process_valid"] is True for row in rows)
    assert all("audit" in row for row in rows)
    assert manifest["samples_per_ready_cell"] == 2
    assert manifest["requested_ready_samples"] == 2
    assert manifest["generated_ready_samples"] == 2
    assert manifest["cell_counts"] == {"DIR-TRC-UNI": 2}
    assert manifest["blocked"] == {
        "CNT-CSR-JCR-UNI": [
            "Add a generator admission check for non-factorizable count dependencies."
        ]
    }
    assert manifest["rows_sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        generate_registry_pilot(
            registry,
            tmp_path / "pilot.jsonl",
            samples_per_ready_cell=2,
            seed=1902,
            max_attempts=100,
        )


def test_all_possible_registry_cells_are_set_valued_alternatives() -> None:
    path = Path(__file__).parents[1] / "all-possible-cells.yaml"
    registry = load_benchmark_registry(path)

    assert len(registry.cells) == 34
    for cell in registry.cells:
        policy = cell.generation_policy()
        assert cell.semantic_status.value == "ALT"
        assert policy.answer_mode.value == "all-possible"
        assert policy.menu_coverage.value == "full"
        assert policy.possibility_skeleton is not None


def test_all_possible_hardening_registry_is_separate_and_composed() -> None:
    path = Path(__file__).parents[1] / "all-possible-hardening-cells.yaml"
    registry = load_benchmark_registry(path)

    assert len(registry.cells) == 14
    assert all("-COMP-" in cell.id or "-SHIFT-" in cell.id for cell in registry.cells)
    assert any(
        cell.generation.get("possibility_skeleton") == "case-chain"
        for cell in registry.cells
    )
    assert any(
        cell.generation.get("possibility_skeleton") == "case-count"
        for cell in registry.cells
    )
    for cell in registry.cells:
        policy = cell.generation_policy()
        assert policy.answer_mode.value == "all-possible"
        assert policy.ordinary_option_target == 5


def test_all_possible_count_hardening_registry_is_targeted() -> None:
    path = Path(__file__).parents[1] / "all-possible-count-hardening-cells.yaml"
    registry = load_benchmark_registry(path)

    assert len(registry.cells) == 2
    assert {cell.generation["possibility_gold_size"] for cell in registry.cells} == {
        3,
        4,
    }
    for cell in registry.cells:
        assert cell.query_kind.value == "count"
        assert cell.generation["possibility_skeleton"] == "factor-count"
        assert cell.generation_policy().ordinary_option_target == 5


def test_all_possible_final_registry_has_frozen_cell_matrix() -> None:
    path = Path(__file__).parents[1] / "all-possible-final-cells.yaml"
    registry = load_benchmark_registry(path)

    assert len(registry.cells) == 30
    assert sum("-COMP-" in cell.id for cell in registry.cells) == 9
    assert sum("-SHIFT-" in cell.id for cell in registry.cells) == 21
    assert {
        kind.value: sum(cell.query_kind is kind for cell in registry.cells)
        for kind in QueryKind
    } == {"direction": 8, "which": 11, "count": 11}
    assert "CNT-SHIFT-IFF-JCR-ALT-G3" in {
        cell.id for cell in registry.cells
    }
    assert "CNT-SHIFT-IFF-JCR-ALT-G4" in {
        cell.id for cell in registry.cells
    }
    for kind in QueryKind:
        depths = {
            cell.generation["possibility_depth"]
            for cell in registry.cells
            if cell.query_kind is kind
            and cell.generation.get("possibility_skeleton") == "depth"
        }
        assert depths == {5, 6, 7, 8, 9}
        gold_sizes = {
            cell.generation["possibility_gold_size"]
            for cell in registry.cells
            if cell.query_kind is kind
            and "possibility_gold_size" in cell.generation
        }
        assert gold_sizes == {2, 3, 4}
