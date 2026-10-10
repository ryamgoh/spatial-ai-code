"""Command-line inspection and bounded probing for benchmark cells."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from spatial.v2.benchmark_registry import (
    CellReadiness,
    RegistryProbeReport,
    generate_registry_pilot,
    load_benchmark_registry,
    probe_registry,
)


def _payload(version: int, report: RegistryProbeReport) -> dict[str, object]:
    return {
        "version": version,
        "ready": list(report.ready),
        "blocked": {
            cell_id: list(reasons) for cell_id, reasons in report.blocked.items()
        },
        "failed": [
            {"cell_id": failure.cell_id, "error": failure.error}
            for failure in report.failed
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a SpatialEntail benchmark-cell registry."
    )
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("benchmark-cells.yaml"),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--probe",
        action="store_true",
        help="Generate one checked example for every ready cell.",
    )
    action.add_argument(
        "--pilot-out",
        type=Path,
        help="Write checked Natural pilot rows and their manifest.",
    )
    parser.add_argument("--samples-per-ready-cell", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args(argv)

    registry = load_benchmark_registry(args.registry)
    if args.pilot_out is not None:
        _, manifest_path = generate_registry_pilot(
            registry,
            args.pilot_out,
            samples_per_ready_cell=args.samples_per_ready_cell,
            seed=args.seed,
            max_attempts=args.max_attempts,
        )
        print(manifest_path.read_text(encoding="utf-8"), end="")
        return 0
    if args.probe:
        report = probe_registry(
            registry,
            seed=args.seed,
            max_attempts=args.max_attempts,
        )
    else:
        report = RegistryProbeReport(
            ready=tuple(
                cell.id
                for cell in registry.cells
                if cell.readiness is CellReadiness.READY
            ),
            blocked={
                cell.id: cell.prerequisites
                for cell in registry.cells
                if cell.readiness is CellReadiness.BLOCKED
            },
            failed=(),
        )
    print(json.dumps(_payload(registry.version, report), indent=2, sort_keys=True))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
