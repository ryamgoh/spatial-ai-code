"""Typed benchmark-cell registry and bounded generation feasibility probes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from spatial.v2.generation import (
    BooleanShape,
    GenerationPolicy,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
)

EnumValue = TypeVar("EnumValue", bound=Enum)


class CapabilityLens(str, Enum):
    RELATIONAL = "RR"
    LOGICAL = "LR"
    MODEL_THEORETIC = "MTR"


class ReasoningStructure(str, Enum):
    ATOMIC = "ARC"
    TRANSITIVE = "TRC"
    AXIS_COMPOSITION = "AXC"
    MODUS_PONENS = "MP"
    EQUIVALENCE = "IFF"
    NEGATION = "NEG"
    DISJUNCTIVE_ELIMINATION = "DJE"
    CASE_SPLIT = "CSR"


class SemanticStatus(str, Enum):
    UNIQUE = "UNI"
    ALTERNATIVES = "ALT"
    MULTIPLE_CERTAIN = "MUL"
    NO_MATCH = "NOM"


class EvidenceObligation(str, Enum):
    POSITIVE_PROOF = "positive-proof"
    CONSISTENCY_WITNESS = "consistency-witness"
    POSSIBILITY_WITNESS = "possibility-witness"
    CANDIDATE_EXCLUSION = "candidate-exclusion"
    JOINT_COUNT = "joint-count"


class CellReadiness(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BenchmarkCell:
    id: str
    readiness: CellReadiness
    query_kind: QueryKind
    capability_lenses: tuple[CapabilityLens, ...]
    reasoning_structures: tuple[ReasoningStructure, ...]
    semantic_status: SemanticStatus
    evidence_obligations: tuple[EvidenceObligation, ...]
    generation: dict[str, Any]
    prerequisites: tuple[str, ...] = ()

    def generation_policy(self) -> GenerationPolicy:
        """Build the representative policy used by a bounded feasibility probe."""
        raw = dict(self.generation)
        depth = raw.pop("depth", None)
        if depth is not None:
            field = (
                "min_axis_depth"
                if self.query_kind is QueryKind.DIRECTION
                else "min_membership_depth"
            )
            maximum = (
                "max_axis_depth"
                if self.query_kind is QueryKind.DIRECTION
                else "max_membership_depth"
            )
            raw[field] = int(depth)
            raw[maximum] = int(depth)
        return GenerationPolicy(query_kind=self.query_kind, **raw)


@dataclass(frozen=True)
class BenchmarkRegistry:
    version: int
    cells: tuple[BenchmarkCell, ...]
    fingerprint: str


@dataclass(frozen=True)
class ProbeFailure:
    cell_id: str
    error: str


@dataclass(frozen=True)
class RegistryProbeReport:
    ready: tuple[str, ...]
    blocked: dict[str, tuple[str, ...]]
    failed: tuple[ProbeFailure, ...]


_ROOT_KEYS = {"version", "cells"}
_CELL_KEYS = {
    "id",
    "readiness",
    "query_kind",
    "capability_lenses",
    "reasoning_structures",
    "semantic_status",
    "evidence_obligations",
    "generation",
    "prerequisites",
}
_GENERATION_KEYS = {
    "composition",
    "semantic_shape",
    "boolean_shape",
    "generation_mode",
    "num_entities",
    "num_premises",
    "depth",
    "ambiguity_size",
    "omit_direct_query_relation",
    "require_independent_axes",
    "distractor_premises",
    "require_joint_count_dependency",
    "query_direction",
    "target_direction",
    "answer_mode",
    "possibility_skeleton",
    "possibility_depth",
    "possibility_gold_size",
    "possibility_candidate_count",
    "ordinary_option_target",
    "visible_gold_size",
    "menu_option_count",
}
_QUERY_CODE = {
    QueryKind.DIRECTION: "DIR",
    QueryKind.WHICH: "SEL",
    QueryKind.COUNT: "CNT",
}
_CELL_ID = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)+$")
_STATUS_SHAPE = {
    SemanticStatus.UNIQUE: SemanticShape.UNIQUE,
    SemanticStatus.ALTERNATIVES: SemanticShape.AMBIGUOUS,
    SemanticStatus.MULTIPLE_CERTAIN: SemanticShape.MULTIPLE_CERTAIN,
    SemanticStatus.NO_MATCH: SemanticShape.NO_MATCH,
}
_STRUCTURE_SHAPE = {
    ReasoningStructure.ATOMIC: BooleanShape.ATOMIC,
    ReasoningStructure.TRANSITIVE: BooleanShape.ATOMIC,
    ReasoningStructure.AXIS_COMPOSITION: BooleanShape.ATOMIC,
    ReasoningStructure.MODUS_PONENS: BooleanShape.MODUS_PONENS,
    ReasoningStructure.EQUIVALENCE: BooleanShape.IFF,
    ReasoningStructure.NEGATION: BooleanShape.DOUBLE_NEGATION,
    ReasoningStructure.DISJUNCTIVE_ELIMINATION: BooleanShape.DISJUNCTIVE_SYLLOGISM,
    ReasoningStructure.CASE_SPLIT: BooleanShape.CASE_SPLIT,
}
_RELATIONAL_STRUCTURES = {
    ReasoningStructure.ATOMIC,
    ReasoningStructure.TRANSITIVE,
    ReasoningStructure.AXIS_COMPOSITION,
}
_LOGICAL_STRUCTURES = set(ReasoningStructure) - _RELATIONAL_STRUCTURES


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return value


def _enum(value: Any, enum_type: type[EnumValue], field: str) -> EnumValue:
    normalized = str(value).strip().lower()
    for member in enum_type:
        if str(member.value).lower() == normalized:
            return member
    raise ValueError(f"unknown {field}: {value}")


def _unique_enums(
    value: Any, enum_type: type[EnumValue], field: str
) -> tuple[EnumValue, ...]:
    items = tuple(_enum(item, enum_type, field) for item in _sequence(value, field))
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate {field} values")
    return items


def _cell(raw: dict[str, Any]) -> BenchmarkCell:
    unknown = set(raw) - _CELL_KEYS
    if unknown:
        raise ValueError("unknown benchmark cell keys: " + ", ".join(sorted(unknown)))
    cell_id = str(raw.get("id", "")).strip()
    if _CELL_ID.fullmatch(cell_id) is None:
        raise ValueError("benchmark cell id must be an uppercase dash-separated code")
    query_kind = _enum(raw.get("query_kind"), QueryKind, "query kind")
    if not cell_id.startswith(_QUERY_CODE[query_kind] + "-"):
        raise ValueError(f"cell {cell_id}: id prefix does not match query kind")
    generation = _mapping(raw.get("generation"), f"cell {cell_id} generation")
    unknown_generation = set(generation) - _GENERATION_KEYS
    if unknown_generation:
        raise ValueError(
            f"cell {cell_id}: unknown generation keys: "
            + ", ".join(sorted(unknown_generation))
        )
    prerequisites_raw = raw.get("prerequisites", [])
    if not isinstance(prerequisites_raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in prerequisites_raw
    ):
        raise ValueError(f"cell {cell_id}: prerequisites must be a list of text")
    cell = BenchmarkCell(
        id=cell_id,
        readiness=_enum(raw.get("readiness"), CellReadiness, "cell readiness"),
        query_kind=query_kind,
        capability_lenses=_unique_enums(
            raw.get("capability_lenses"), CapabilityLens, "capability lens"
        ),
        reasoning_structures=_unique_enums(
            raw.get("reasoning_structures"),
            ReasoningStructure,
            "reasoning structure",
        )
        if generation.get("composition") is None
        else tuple(
            _enum(item, ReasoningStructure, "reasoning structure")
            for item in _sequence(
                raw.get("reasoning_structures"), "reasoning structures"
            )
        ),
        semantic_status=_enum(
            raw.get("semantic_status"), SemanticStatus, "semantic status"
        ),
        evidence_obligations=_unique_enums(
            raw.get("evidence_obligations"),
            EvidenceObligation,
            "evidence obligation",
        ),
        generation=dict(generation),
        prerequisites=tuple(item.strip() for item in prerequisites_raw),
    )
    _validate_cell(cell)
    if cell.readiness is CellReadiness.READY:
        try:
            cell.generation_policy()
        except ValueError as exc:
            raise ValueError(
                f"cell {cell.id}: ready generation policy is invalid: {exc}"
            ) from exc
    return cell


def _validate_cell(cell: BenchmarkCell) -> None:
    if cell.readiness is CellReadiness.READY and cell.prerequisites:
        raise ValueError(f"cell {cell.id}: ready cell cannot declare prerequisites")
    if cell.readiness is CellReadiness.BLOCKED and not cell.prerequisites:
        raise ValueError(f"cell {cell.id}: blocked cell requires prerequisites")
    if (
        cell.semantic_status is SemanticStatus.MULTIPLE_CERTAIN
        and cell.query_kind is not QueryKind.WHICH
    ):
        raise ValueError(f"cell {cell.id}: MUL applies only to Selection")
    if (
        EvidenceObligation.JOINT_COUNT in cell.evidence_obligations
        and cell.query_kind is not QueryKind.COUNT
    ):
        raise ValueError(f"cell {cell.id}: joint-count applies only to Count")

    semantic_shape = _enum(
        cell.generation.get("semantic_shape"), SemanticShape, "semantic_shape"
    )
    expected_shape = _STATUS_SHAPE[cell.semantic_status]
    if semantic_shape is not expected_shape:
        raise ValueError(
            f"cell {cell.id}: {cell.semantic_status.value} requires "
            f"semantic_shape {expected_shape.value}"
        )

    boolean_shape = _enum(
        cell.generation.get("boolean_shape", "atomic"),
        BooleanShape,
        "boolean_shape",
    )
    composition = cell.generation.get("composition")
    skeleton = cell.generation.get("possibility_skeleton")
    if skeleton is not None:
        from spatial.v2.possibility import POSSIBILITY_STRUCTURES

        if cell.semantic_status is not SemanticStatus.ALTERNATIVES:
            raise ValueError(f"cell {cell.id}: possibility skeletons require ALT")
        if set(cell.reasoning_structures) != set(POSSIBILITY_STRUCTURES[skeleton]):
            raise ValueError(
                f"cell {cell.id}: possibility skeleton disagrees with structures"
            )
    elif composition is not None:
        from spatial.v2.composition import COMPOSITIONS, composition_structures

        expected = (
            tuple(
                _enum(code, ReasoningStructure, "composition structure")
                for code in composition_structures(composition)
            )
            if composition in COMPOSITIONS
            else ()
        )
        if composition not in COMPOSITIONS or cell.reasoning_structures != expected:
            raise ValueError(
                f"cell {cell.id}: composition disagrees with ordered structures"
            )
        if boolean_shape is not BooleanShape.ATOMIC:
            raise ValueError(f"cell {cell.id}: composition requires atomic base shape")
    else:
        for structure in cell.reasoning_structures:
            expected_shape = _STRUCTURE_SHAPE[structure]
            if boolean_shape is not expected_shape:
                raise ValueError(
                    f"cell {cell.id}: {structure.value} requires boolean_shape {expected_shape.value}"
                )
        if len(cell.reasoning_structures) > 1 and cell.readiness is CellReadiness.READY:
            raise ValueError(
                f"cell {cell.id}: mixed structures need an explicit blocked prerequisite"
            )

    structures = set(cell.reasoning_structures)
    lenses = set(cell.capability_lenses)
    if structures & _RELATIONAL_STRUCTURES and CapabilityLens.RELATIONAL not in lenses:
        raise ValueError(f"cell {cell.id}: relational structure requires RR")
    if structures & _LOGICAL_STRUCTURES and CapabilityLens.LOGICAL not in lenses:
        raise ValueError(f"cell {cell.id}: logical structure requires LR")
    if (
        cell.semantic_status
        in {
            SemanticStatus.ALTERNATIVES,
            SemanticStatus.MULTIPLE_CERTAIN,
            SemanticStatus.NO_MATCH,
        }
        or EvidenceObligation.JOINT_COUNT in cell.evidence_obligations
    ) and CapabilityLens.MODEL_THEORETIC not in lenses:
        raise ValueError(f"cell {cell.id}: semantic obligation requires MTR")

    depth = cell.generation.get("depth")
    if skeleton is not None:
        depth = None
    if ReasoningStructure.ATOMIC in structures and depth not in {None, 1}:
        raise ValueError(f"cell {cell.id}: ARC requires depth 1 or omitted")
    if (
        ReasoningStructure.TRANSITIVE in structures
        and composition is None
        and skeleton is None
    ):
        if depth is None or int(depth) < 2:
            raise ValueError(f"cell {cell.id}: TRC requires depth at least 2")
        if not cell.generation.get("omit_direct_query_relation", False):
            raise ValueError(f"cell {cell.id}: TRC requires direct-relation omission")
    if ReasoningStructure.AXIS_COMPOSITION in structures:
        if depth is None or int(depth) < 2:
            raise ValueError(f"cell {cell.id}: AXC requires depth at least 2")
        if not cell.generation.get("require_independent_axes", False):
            raise ValueError(f"cell {cell.id}: AXC requires independent axes")

    if cell.semantic_status is SemanticStatus.ALTERNATIVES and (
        EvidenceObligation.POSSIBILITY_WITNESS not in cell.evidence_obligations
    ):
        raise ValueError(f"cell {cell.id}: ALT requires possibility-witness evidence")
    if cell.semantic_status is SemanticStatus.NO_MATCH and (
        EvidenceObligation.CANDIDATE_EXCLUSION not in cell.evidence_obligations
    ):
        raise ValueError(f"cell {cell.id}: NOM requires candidate-exclusion evidence")
    if (
        cell.query_kind is QueryKind.DIRECTION
        and cell.semantic_status is SemanticStatus.UNIQUE
        and (
            EvidenceObligation.POSITIVE_PROOF not in cell.evidence_obligations
            or EvidenceObligation.CONSISTENCY_WITNESS not in cell.evidence_obligations
        )
    ):
        raise ValueError(
            f"cell {cell.id}: unique Direction requires proof and consistency witness"
        )


def load_benchmark_registry(path: str | Path) -> BenchmarkRegistry:
    """Load and validate a versioned benchmark-cell registry without generating."""
    source_text = Path(path).read_text(encoding="utf-8")
    try:
        source = yaml.safe_load(source_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid registry YAML: {exc}") from exc
    root = _mapping(source, "benchmark registry")
    unknown = set(root) - _ROOT_KEYS
    if unknown:
        raise ValueError("unknown registry keys: " + ", ".join(sorted(unknown)))
    if root.get("version") != 1:
        raise ValueError("benchmark registry version must be 1")
    cells = tuple(
        _cell(_mapping(item, "benchmark cell"))
        for item in _sequence(root.get("cells"), "cells")
    )
    identifiers = [cell.id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate benchmark cell IDs")
    return BenchmarkRegistry(
        version=1,
        cells=cells,
        fingerprint=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
    )


def probe_registry(
    registry: BenchmarkRegistry,
    *,
    seed: int = 42,
    max_attempts: int = 100,
) -> RegistryProbeReport:
    """Generate one checked sample per ready cell and retain every failure."""
    generator = SpatialGeneratorV2(seed=seed)
    ready = []
    blocked = {}
    failed = []
    for cell in registry.cells:
        if cell.readiness is CellReadiness.BLOCKED:
            blocked[cell.id] = cell.prerequisites
            continue
        try:
            generator.generate(cell.generation_policy(), max_attempts=max_attempts)
        except (RuntimeError, ValueError) as exc:
            failed.append(ProbeFailure(cell.id, str(exc)))
        else:
            ready.append(cell.id)
    return RegistryProbeReport(tuple(ready), blocked, tuple(failed))


def generate_registry_pilot(
    registry: BenchmarkRegistry,
    output_file: str | Path,
    *,
    samples_per_ready_cell: int,
    seed: int = 42,
    max_attempts: int = 500,
) -> tuple[Path, Path]:
    """Persist checked Natural rows for every ready cell and an audit manifest."""
    if samples_per_ready_cell <= 0:
        raise ValueError("samples_per_ready_cell must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    rows_path = Path(output_file)
    stem = rows_path.with_suffix("") if rows_path.suffix else rows_path
    manifest_path = stem.with_name(stem.name + "_manifest").with_suffix(".json")
    existing = [path for path in (rows_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "pilot output already exists: " + ", ".join(str(path) for path in existing)
        )

    rows: list[dict[str, Any]] = []
    cell_counts: dict[str, int] = {}
    ready_cells = [
        cell for cell in registry.cells if cell.readiness is CellReadiness.READY
    ]
    for index, cell in enumerate(ready_cells):
        generator = SpatialGeneratorV2(seed + index)
        for _ in range(samples_per_ready_cell):
            try:
                sample = generator.generate(
                    cell.generation_policy(),
                    max_attempts=max_attempts,
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"cell {cell.id}: pilot generation failed before any output "
                    f"was written: {exc}"
                ) from exc
            row = sample.as_sft_row(include_audit=True)
            row["metadata"].update(
                {
                    "benchmark_cell": cell.id,
                    "capability_lenses": [
                        lens.value for lens in cell.capability_lenses
                    ],
                    "reasoning_structures": [
                        structure.value for structure in cell.reasoning_structures
                    ],
                    "semantic_status": cell.semantic_status.value,
                    "evidence_obligations": [
                        obligation.value for obligation in cell.evidence_obligations
                    ],
                    "registry_fingerprint": registry.fingerprint,
                }
            )
            rows.append(row)
        cell_counts[cell.id] = samples_per_ready_cell

    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    rows_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "spatialentail-registry-pilot-v1",
        "registry_version": registry.version,
        "registry_fingerprint": registry.fingerprint,
        "seed": seed,
        "samples_per_ready_cell": samples_per_ready_cell,
        "requested_ready_samples": len(ready_cells) * samples_per_ready_cell,
        "generated_ready_samples": len(rows),
        "cell_counts": cell_counts,
        "blocked": {
            cell.id: list(cell.prerequisites)
            for cell in registry.cells
            if cell.readiness is CellReadiness.BLOCKED
        },
        "rows_sha256": rows_hash,
    }
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(serialized, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path
