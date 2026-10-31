"""Story-consistent ordinary names over unchanged verified entailment scenes."""

import random
import re
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256

from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_solver_reuse import QuestionSolver
from spatial.v2.entailment_contracts import Contract, render_prompt
from spatial.v2.entailment_dataset import audit_row
from spatial.v2.entailment_evidence import render_pair
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.generation import ENTITY_NAMES
from spatial.v2.serialization import tagged_dataclass_to_dict

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


def semantic_record(record):
    result = {
        k: v
        for k, v in record.items()
        if k not in ("tokens", "rows", "entity_name_map", "naming_policy")
    }
    result["rows"] = [
        {k: v for k, v in row.items() if k not in ("prompt", "references")}
        for row in record["rows"]
    ]
    return result


def rename_record(original, tokenizer):
    if original.get("prompt_version") is not None:
        raise ValueError(
            "scoped naming revision cannot revise all-scene records; "
            "name objects before canonical all-scene admission"
        )
    old_evidence = load_evidence(original)
    names = story_name_map(old_evidence.semantics.problem.objects, original["story_id"])
    inverse = {name: old for old, name in names.items()}
    evidence = rename_structure(old_evidence, names)
    if rename_structure(evidence, inverse) != old_evidence:
        raise ValueError("entity naming changed typed evidence beyond a bijection")
    result = rename_structure(original, names)
    result["problem"] = tagged_dataclass_to_dict(evidence.semantics.problem)
    result["tokens"] = []
    solver = QuestionSolver()
    for row in result["rows"]:
        menu, contract = tuple(row["menu"]), Contract(row["contract"])
        row["prompt"] = render_prompt(evidence.semantics.problem, menu, contract)
        row["references"] = render_pair(evidence, menu, contract, solver=solver)
        audit_row(row, evidence, solver=solver)
        for text in (row["prompt"], *row["references"].values()):
            if re.search(r"\bObject\d+\b", text):
                raise ValueError("technical object identifier leaked into visible text")
        for style in ("natural", "symbolic"):
            measurement = token_measurement(tokenizer, row, style)
            if not measurement["admitted"]:
                raise ValueError("ordinary names exceed the unchanged token budget")
            result["tokens"].append(measurement)
    if rename_structure(semantic_record(result), inverse) != semantic_record(original):
        raise ValueError(
            "naming changed labels, proof metadata, structure or allocation"
        )
    result["entity_name_map"] = names
    result["naming_policy"] = VERSION
    return result
