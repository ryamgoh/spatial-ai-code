"""Deterministic JSON-compatible serialization for typed spatial artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


def tagged_dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Serialize a dataclass and retain its top-level runtime type."""
    payload = _json_value(value)
    if not isinstance(payload, dict):
        raise TypeError("tagged serialization requires a dataclass value")
    return {"type": type(value).__name__, **payload}
