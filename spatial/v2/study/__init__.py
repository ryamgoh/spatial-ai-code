"""Release assembly for the bounded SpatialEntail learning study."""

import json
from pathlib import Path

ROLE_FILENAMES = {
    "bench-comp": "bench-composition.jsonl",
    "bench-composition": "bench-shift.jsonl",
    "bench-depth": "bench-shift.jsonl",
}
SHIFT_ROLES = ("bench-composition", "bench-depth")


def role_path(release: Path, role: str) -> Path:
    """Resolve a persisted role while preserving sealed internal role keys."""
    return release / ROLE_FILENAMES.get(role, f"{role}.jsonl")


def read_role_text(release: Path, role: str) -> str:
    """Read one sealed role, filtering the shared SHIFT artifact when needed."""
    text = role_path(release, role).read_text()
    if role not in SHIFT_ROLES:
        return text
    return "".join(
        line + "\n"
        for line in text.splitlines()
        if json.loads(line)["metadata"]["study_role"] == role
    )


def write_role_texts(output: Path, serialized: dict[str, str]) -> None:
    """Write internal roles into the public CORE/COMP/SHIFT artifact layout."""
    for role, text in serialized.items():
        if role not in SHIFT_ROLES:
            role_path(output, role).write_text(text)
    shift = "".join(serialized.get(role, "") for role in SHIFT_ROLES)
    if shift:
        role_path(output, SHIFT_ROLES[0]).write_text(shift)
