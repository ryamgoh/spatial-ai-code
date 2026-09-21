from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "spatial"))

from spatial_generation_v13 import (
    DepthRange,
    DistractorPolicy,
    DistractorSpec,
    GenerationSpec,
    RelationMode,
    SemanticSubtype,
    SpatialGenerator,
    StructuralConstraints,
)

MODULE = Path(__file__).with_name("make_trace_ablation_data.py")
SPEC = importlib.util.spec_from_file_location("make_trace_ablation", MODULE)
assert SPEC and SPEC.loader
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def make_source() -> dict:
    generator = SpatialGenerator()
    example = generator.generate(
        GenerationSpec(
            semantic_subtype=SemanticSubtype.DIR_1,
            relation_mode=RelationMode.MIXED,
            constraints=StructuralConstraints(
                require_independent_axes=True,
                x_depth=DepthRange.exact(4),
                y_depth=DepthRange.exact(4),
                distractors=DistractorSpec(
                    policy=DistractorPolicy.QUERY_BRANCH, count=3
                ),
            ),
            num_entities=11,
            num_relations=11,
        ),
        random.Random(13900),
    )
    return example.to_row(generation_cell="trace-test")


def test_trace_conversion_changes_only_assistant_representation() -> None:
    source = make_source()
    full = DATA.convert([source], DATA.TraceFormat.FULL_STATE)[0]
    delta = DATA.convert([source], DATA.TraceFormat.DELTA_STATE)[0]

    assert DATA.user_text(full) == DATA.user_text(delta) == DATA.user_text(source)
    assert full["oracle_option"] == delta["oracle_option"]
    assert full["difficulty"] == delta["difficulty"]
    assert DATA.assistant_text(full) == DATA.assistant_text(source)
    assert DATA.assistant_text(delta) != DATA.assistant_text(source)
    assert "**Final X-State**:" in DATA.assistant_text(delta)
    assert len(DATA.assistant_text(delta)) < len(DATA.assistant_text(full))


def test_trace_ablation_files_are_row_aligned(tmp_path: Path, monkeypatch) -> None:
    source = make_source()
    train = tmp_path / "source-train.jsonl"
    val = tmp_path / "source-val.jsonl"
    train.write_text(json.dumps(source) + "\n")
    val.write_text(json.dumps(source) + "\n")
    monkeypatch.setattr(DATA, "EXPECTED_TRAIN", 1)
    monkeypatch.setattr(DATA, "EXPECTED_VAL", 1)

    output_paths = DATA.generate(tmp_path / "trace.jsonl", train, val)
    manifest = json.loads(output_paths[-1].read_text())

    DATA.validate(*output_paths, train, val)
    assert manifest["delta_to_full_char_ratio"] < 1
