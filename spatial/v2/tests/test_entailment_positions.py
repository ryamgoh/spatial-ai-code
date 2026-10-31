"""Position views preserve base semantics and expose every required letter mask."""

import json

from spatial.v2.composition_production import summary_record
from spatial.v2.entailment_contracts import Contract, encode, render_prompt
from spatial.v2.entailment_evidence import build_evidence
from spatial.v2.entailment_positions import balanced_metadata
from spatial.v2.entailment_saved import load_evidence
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.tests.test_pivot_evidence import BY_ID


def fixture_record(name, menu):
    problem = BY_ID[name].problem()
    evidence = build_evidence(problem)
    rows = []
    for contract in Contract:
        decision = encode(evidence.semantics, menu, contract)
        entailed_letters = "".join(
            letter
            for letter, value in zip("ABCD", menu)
            if value in evidence.semantics.entailed
        )
        rows.append(
            {
                "interface": "spatialentail-entailment-v1",
                "id": f"{name}-{contract.value}",
                "base_id": name,
                "bundle_id": name,
                "query_kind": "which",
                "contract": contract.value,
                "menu": list(menu),
                "menu_role": "primary",
                "answer": decision.letters,
                "entailed_menu_letters": entailed_letters,
                "negative_menu_categories": {
                    letter: "impossible"
                    for letter in "ABCD"
                    if letter not in entailed_letters
                },
                "prompt": render_prompt(problem, menu, contract),
            }
        )
    return {
        "interface": "spatialentail-entailment-v1",
        "base_id": name,
        "family_id": "same-family",
        "query_kind": "which",
        "semantics": {"entailed": list(evidence.semantics.entailed)},
        "problem": tagged_dataclass_to_dict(problem),
        "evidence": tagged_dataclass_to_dict(evidence),
        "rows": rows,
    }


def test_two_entailed_case_covers_six_masks_without_new_bases():
    record = fixture_record("S3", ("U", "V", "W", "T"))
    result = balanced_metadata(summary_record(record))
    rows = [r for r in result["rows"] if r["contract"] == Contract.LISTED.value]
    assert {r["answer"] for r in rows} == {"AB", "AC", "AD", "BC", "BD", "CD"}
    assert {r["base_id"] for r in result["rows"]} == {"S3"}
    assert result["family_id"] == record["family_id"]
    assert record["rows"][0]["menu"] == ["U", "V", "W", "T"]


def test_saved_evidence_roundtrip_is_lossless():
    record = fixture_record("S3", ("U", "V", "W", "T"))
    saved = json.loads(json.dumps(record))
    restored = load_evidence(saved)
    assert restored.semantics.entailed == {"U", "V"}
    assert tagged_dataclass_to_dict(restored) == record["evidence"]


def test_singleton_has_no_unnecessary_position_expansion():
    record = fixture_record("S1", ("U", "V", "W", "T"))
    result = balanced_metadata(summary_record(record))
    assert len(result["rows"]) == 3
