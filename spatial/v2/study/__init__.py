"""Release assembly for the bounded SpatialEntail learning study."""

from pathlib import Path

ROLE_FILENAMES = {
    "bench-comp": "bench-composition-matched.jsonl",
    "bench-composition": "bench-composition-heldout.jsonl",
}


def role_path(release: Path, role: str) -> Path:
    """Resolve a persisted role while preserving sealed internal role keys."""
    return release / ROLE_FILENAMES.get(role, f"{role}.jsonl")
