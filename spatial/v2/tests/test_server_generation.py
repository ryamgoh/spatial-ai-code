"""Exercise server generation with tracked inputs and no candidate payloads."""

import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_pilot import story_key
from spatial.v2.entailment_replacement_allocation import schedule
from spatial.v2.entailment_thinking_training import tokenize

EXP = Path(__file__).resolve().parents[3] / "experiments/spatialentail-composition-v1"


@pytest.fixture(scope="module")
def server():
    sys.path.insert(0, str(EXP))
    return importlib.import_module("generate_and_prepare")


def test_only_tracked_inputs_needed(server):
    policy, held_out = server.inputs()
    assert len(schedule(policy)) == 6000
    assert len(held_out["base_ids"]) == 4800
    assert server.digest(EXP / "generation-exclusions-v1.json") == server.EXCLUSION_SHA


def test_existing_output_refused(server, tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        server.generate(tmp_path, 1)


def test_generated_cases_reach_native_training_without_saved_payloads(server):
    from transformers import AutoTokenizer

    from spatial.v2.composition_training import messages

    old = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-4B",
        revision=server.tokenizer_loader.REVISION,
        local_files_only=True,
    )
    server.tokenizer_loader.TOKENIZER = old
    active = json.loads((EXP / "active-model.json").read_text())
    thinking = AutoTokenizer.from_pretrained(
        active["model_id"],
        revision=active["revision"],
        local_files_only=True,
    )
    policy, _ = server.inputs()
    chosen = {}
    for slot in schedule(policy):
        if slot.span in (1, 8) or slot.contingent:
            chosen.setdefault(
                (slot.kind, slot.category, slot.span, slot.contingent), slot
            )
    assert len(chosen) >= 20
    for slot in chosen.values():
        story, problem, anchor = server.build.construct(slot, 0)
        record = server.generate_one(
            {
                "slot": asdict(slot),
                "alternative": 0,
                "query_key": query_profile(problem)["key"],
                "scene_key": story_key(story.objects, story.premise),
                "anchor_key": anchor,
            }
        )
        assert record["independent_audit"] == "passed"
        row = record["rows"][0]
        for style in ("natural", "symbolic"):
            tokenized, _, _ = tokenize(
                thinking,
                [
                    *messages(row, style),
                    {"role": "assistant", "content": row["references"][style]},
                ],
                row["answer"],
                active,
            )
            assert len(tokenized["input_ids"]) <= 8192
