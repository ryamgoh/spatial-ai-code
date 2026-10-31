"""All named scene objects count; menus and supporting roles cannot restrict scope."""

from dataclasses import replace

import pytest

from spatial.v2.composition_training import messages
from spatial.v2.entailment_contracts import (
    VERSION as INTERFACE,
)
from spatial.v2.entailment_contracts import (
    Contract,
    analyze,
    encode,
)
from spatial.v2.entailment_contracts import (
    parse_prompt as parse_scoped,
)
from spatial.v2.entailment_contracts import (
    render_prompt as render_scoped,
)
from spatial.v2.entailment_evidence import (
    REFERENCE_VERSION,
    build_evidence,
    render_pair,
)
from spatial.v2.entailment_names import rename_structure
from spatial.v2.entailment_scene_protocol import (
    COMMON_INSTRUCTION,
    SCOPE,
    VERSION,
    all_scene_problem,
    display_references,
    parse_prompt,
    parse_row,
    render_prompt,
)
from spatial.v2.solver import (
    And,
    CountQuery,
    Direction,
    DirectionQuery,
    SpatialProblem,
    WhichQuery,
    direction_constraint,
)


def scoped_problem(kind="which", direction=Direction.NORTH):
    candidates = ("Library", "Museum", "Bank", "Cafe", "Hotel")
    query = (
        DirectionQuery("Library", "Park")
        if kind == "direction"
        else (WhichQuery if kind == "which" else CountQuery)(
            frozenset((direction,)), "Park", candidates
        )
    )
    return SpatialProblem(
        ("Park", "Clinic", *candidates),
        And(
            tuple(
                direction_constraint(a, b, d)
                for a, b, d in (
                    ("Clinic", "Park", Direction.NORTH),
                    ("Library", "Clinic", Direction.NORTH),
                    ("Museum", "Clinic", Direction.NORTH),
                    ("Bank", "Park", Direction.SOUTH),
                    ("Cafe", "Park", Direction.SOUTH),
                    ("Hotel", "Park", Direction.SOUTH),
                )
            )
        ),
        query,
    )


def menu_for(kind):
    return (
        ("North", "East", "South", "West")
        if kind == "direction"
        else (0, 1, 2, 3)
        if kind == "count"
        else ("Library", "Museum", "Bank", "Cafe")
    )


def test_supporting_object_is_an_answer_even_when_unlisted():
    old = scoped_problem()
    new = all_scene_problem(old)
    assert analyze(old).entailed == {"Library", "Museum"}
    semantics = analyze(new)
    assert semantics.entailed == {"Clinic", "Library", "Museum"}
    assert set(new.query.candidates) == set(new.objects) - {"Park"}
    assert tuple(encode(semantics, menu_for("which"), c).letters for c in Contract) == (
        "E",
        "AB",
        "F",
    )


def test_count_changes_when_supporting_object_enters_scope():
    old = scoped_problem("count")
    assert analyze(old).entailed == {2}
    assert analyze(all_scene_problem(old)).entailed == {3}


@pytest.mark.parametrize("kind", ("direction", "which", "count"))
@pytest.mark.parametrize("contract", tuple(Contract))
@pytest.mark.parametrize("direction", tuple(Direction))
def test_plain_prompt_roundtrip_for_every_query_contract_direction(
    kind, contract, direction
):
    problem = all_scene_problem(scoped_problem(kind, direction))
    prompt = render_prompt(problem, menu_for(kind), contract)
    assert prompt.splitlines()[1].count(" is to the ") == len(problem.premise.operands)
    assert prompt.splitlines()[2] == ""
    assert parse_prompt(prompt) == (problem, menu_for(kind), contract)
    for forbidden in ("Eligible:", "Objects:", "Query:", "Interface:", "Contract:"):
        assert forbidden not in prompt
    with pytest.raises(ValueError):
        parse_scoped(prompt)


def test_scoped_renderer_is_preserved_but_new_renderer_refuses_subset_scope():
    old = scoped_problem()
    prompt = render_scoped(old, menu_for("which"), Contract.LISTED)
    assert parse_scoped(prompt) == (old, menu_for("which"), Contract.LISTED)
    with pytest.raises(ValueError, match="all-scene scope"):
        render_prompt(old, menu_for("which"), Contract.LISTED)
    with pytest.raises(ValueError):
        parse_prompt(prompt)


def test_saved_line_layout_remains_readable():
    problem = all_scene_problem(scoped_problem())
    old = render_prompt(problem, menu_for("which"), Contract.LISTED, layout="lines")
    assert parse_prompt(old) == (problem, menu_for("which"), Contract.LISTED)
    paragraph = render_prompt(problem, menu_for("which"), Contract.LISTED)
    assert " ".join(old.splitlines()[1:7]) == paragraph.splitlines()[1]
    assert parse_prompt(paragraph)[0] == problem


def test_multiword_reference_and_native_external_names_need_no_visible_alias_legend():
    old = rename_structure(scoped_problem(), {"Park": "City Hall"})
    problem = all_scene_problem(old)
    assert (
        parse_prompt(render_prompt(problem, menu_for("which"), Contract.LISTED))[0]
        == problem
    )
    aliases = {name: f"Entity{i}" for i, name in enumerate(problem.objects)}
    aliased = all_scene_problem(rename_structure(problem, aliases))
    names = {alias: name for name, alias in aliases.items()}
    names[aliases["Clinic"]] = "Narwhal's Novelties"
    menu = tuple(aliases[name] for name in menu_for("which"))
    prompt = render_prompt(aliased, menu, Contract.LISTED, display_names=names)
    assert "Entity" not in prompt
    assert "Narwhal's Novelties" in prompt
    assert parse_prompt(prompt, display_names=names) == (aliased, menu, Contract.LISTED)
    evidence = build_evidence(aliased)
    pair = display_references(render_pair(evidence, menu, Contract.LISTED), names)
    assert all("Entity" not in text for text in pair.values())
    assert all(text.endswith("Answer: AB") for text in pair.values())


def test_common_rules_are_in_system_message_not_user_scene():
    problem = all_scene_problem(scoped_problem())
    row = {
        "interface": INTERFACE,
        "prompt_version": VERSION,
        "scope_policy": SCOPE,
        "prompt": render_prompt(problem, menu_for("which"), Contract.LISTED),
    }
    for style in ("natural", "symbolic"):
        source = messages(row, style)
        assert COMMON_INSTRUCTION in source[0]["content"]
        assert source[1]["content"] == row["prompt"]
        assert "fixed global x/y axes" not in source[1]["content"]


def test_undeclared_scope_extensions_and_tampered_questions_fail():
    old = scoped_problem()
    with pytest.raises(ValueError, match="every object"):
        all_scene_problem(replace(old, objects=(*old.objects, "Unmentioned")))
    prompt = render_prompt(all_scene_problem(old), menu_for("which"), Contract.LISTED)
    with pytest.raises(ValueError):
        parse_prompt(prompt.replace("are guaranteed", "could be"))
    with pytest.raises(ValueError):
        parse_prompt(prompt.replace("D. Cafe", "D. Unmentioned"))


def test_alias_substitution_does_not_confuse_entity_one_with_entity_ten():
    pair = {"natural": "Entity1 and Entity10. Answer: AB"}
    assert display_references(pair, {"Entity1": "Bank", "Entity10": "City Hall"}) == {
        "natural": "Bank and City Hall. Answer: AB"
    }


def test_plain_admission_saved_evidence_and_independent_audit(tokenizer):
    from types import SimpleNamespace

    from spatial.v2.entailment_dataset import build_attempt
    from spatial.v2.entailment_positions import balanced_metadata, render_record
    from spatial.v2.entailment_saved import load_evidence
    from spatial.v2.tests.test_entailment_artifact_audit import AUDIT

    problem = all_scene_problem(scoped_problem())
    story = SimpleNamespace(
        story_id="plain-scope-development",
        source_family="plain-scope-development",
        recipe="plain-scope-development",
        objects=problem.objects,
        candidates=problem.query.candidates,
        premise=problem.premise,
        distractors=(),
    )
    record = build_attempt(story, problem, None, tokenizer, prompt_version=VERSION)
    assert record["status"] == "accepted", record.get("reason")
    assert load_evidence(record).semantics.entailed == {"Clinic", "Library", "Museum"}
    assert AUDIT.audit_record(record) == (3, 3)
    revised = render_record(record, balanced_metadata(record), tokenizer)
    assert AUDIT.audit_record(revised) == (3, 3)
    for row in record["rows"]:
        assert parse_row(row)[0] == problem
        assert COMMON_INSTRUCTION in messages(row, "natural")[0]["content"]
        assert row["reference_version"] == REFERENCE_VERSION
        assert row["prompt"] == render_prompt(
            problem, tuple(row["menu"]), row["contract"]
        )
        assert "P1:" not in row["prompt"]
        assert row["references"]["symbolic"].startswith("Premises used")
        assert "P1: Clinic is to the North of Park." in row["references"]["symbolic"]
    record["scope_policy"] = "unknown"
    with pytest.raises(ValueError, match="views disagree"):
        load_evidence(record)
    record["scope_policy"] = SCOPE
    record["rows"][0]["answer"] = "A"
    with pytest.raises(ValueError, match="gold mismatch"):
        AUDIT.audit_record(record)


@pytest.fixture(scope="module")
def tokenizer():
    from spatial.v2.tests.test_entailment_dataset import tokenizer as fixture

    return fixture.__wrapped__()


@pytest.mark.parametrize("field", ("scope_policy", "prompt_version"))
def test_protocol_metadata_is_required_in_parse_and_messages(field):
    row = {
        "interface": INTERFACE,
        "prompt_version": VERSION,
        "scope_policy": SCOPE,
        "prompt": render_prompt(
            all_scene_problem(scoped_problem()), menu_for("which"), Contract.LISTED
        ),
    }
    row[field] = "unknown"
    with pytest.raises(ValueError):
        parse_row(row)
    with pytest.raises(ValueError):
        messages(row, "natural")


def test_old_irrelevant_context_is_not_silently_accepted_under_expanded_scope(
    tokenizer,
):
    from types import SimpleNamespace

    from spatial.v2.entailment_dataset import build_attempt

    old = scoped_problem()
    old = replace(
        old,
        objects=(*old.objects, "Church"),
        premise=And(
            (
                *old.premise.operands,
                direction_constraint("Church", "Park", Direction.NORTH),
            )
        ),
    )
    story = SimpleNamespace(
        story_id="scope-context-development",
        source_family="scope-context-development",
        recipe="scope-context-development",
        objects=old.objects,
        premise=old.premise,
        candidates=old.query.candidates,
        distractors=(len(old.premise.operands) - 1,),
    )
    assert build_attempt(story, old, None, tokenizer)["status"] == "accepted"
    new = build_attempt(
        story, all_scene_problem(old), None, tokenizer, prompt_version=VERSION
    )
    assert new["status"] == "unsupported"
    assert "distractor changes all-scene answers" in new["reason"]
