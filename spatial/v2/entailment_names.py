"""Story-consistent ordinary names over unchanged verified entailment scenes."""

import random
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256

from spatial.v2.generation import ENTITY_NAMES

VERSION = "ordinary-place-names-v1"
NAMES = (*ENTITY_NAMES, "Courthouse")


def story_name_map(objects, story_id):
    if len(objects) != len(set(objects)) or len(objects) > len(NAMES):
        raise ValueError("scene requires distinct names within the fixed vocabulary")
    pool = list(NAMES)
    seed = int.from_bytes(sha256(f"{VERSION}/{story_id}".encode()).digest(), "big")
    random.Random(seed).shuffle(pool)
    return dict(zip(sorted(objects), pool))


def rename_structure(value, names):
    """Rename exact entity values and keys, never substrings or record identities."""
    if isinstance(value, Enum):
        return value
    if isinstance(value, str):
        return names.get(value, value)
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: rename_structure(getattr(value, field.name), names)
                for field in fields(value)
            },
        )
    if isinstance(value, dict):
        return {
            rename_structure(key, names): rename_structure(item, names)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return type(value)(rename_structure(item, names) for item in value)
    return value
