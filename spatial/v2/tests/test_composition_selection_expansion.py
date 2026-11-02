import hashlib
import json
import random
from collections import Counter

import pytest

from spatial.v2.composition_balance import query_profile
from spatial.v2.composition_contracts import analyze
from spatial.v2.composition_depth import measure_depth
from spatial.v2.composition_selection_expansion import (
    PATTERNS,
    deduplicate,
    make_story,
    schedule,
)
from spatial.v2.serialization import tagged_dataclass_to_dict


def test_fixed_bounded_schedule():
    entries = schedule()
    assert entries == schedule()
    assert len(entries) == 60
    assert len({s.story_id for s, _ in entries}) == 60
    assert Counter((s.recipe, d) for s, d in entries) == {
        (p, d): 2 for p in PATTERNS for d in range(1, 11)
    }
    assert len({s.source_family for s, _ in entries}) == 6
    assert all(not s.distractors for s, _ in entries)


def test_historical_schedule_fingerprint_is_unchanged():
    data = [
        (s.story_id, s.source_family, d, tagged_dataclass_to_dict(s.problems()[1]))
        for s, d in schedule()
    ]
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == (
        "7568a21d0b5631d0aab639c932cfc0d2f2e2675dde9eab8b984a35829a8cdcf4"
    )


def test_bad_direction_rejected():
    with pytest.raises(ValueError, match="must be a Direction"):
        make_story("match-chain", 3, 0, random.Random(1), direction="Northeast")


@pytest.mark.parametrize("index", range(60))
def test_unique_match_and_verified_requested_depth(index):
    story, depth = schedule()[index]
    problem = story.problems()[1]
    semantics = analyze(problem)
    assert semantics.consistent
    assert semantics.possible == semantics.certain == {story.target}
    measured = measure_depth(problem)
    assert measured["depth"] == depth
    assert measured["dependence_check"] == "passed-not-a-minimality-proof"
    assert any(a["answer_changed"] for a in measured["ablations"])
    positive = [m for m in measured["memberships"] if m["qualifies"]]
    negative = [m for m in measured["memberships"] if not m["qualifies"]]
    if story.recipe == "match-chain":
        assert positive[0]["depth"] == depth
        assert all(m["depth"] == 1 for m in negative)
    else:
        assert positive[0]["depth"] == 1
        assert max(m["depth"] for m in negative) == depth
        if story.recipe == "shared-exclusion":
            assert all(m["depth"] == depth for m in negative)


def test_no_duplicate_signature_inflates_added_supply():
    old = [{"status": "accepted", "structure": {"key": "old"}}]
    additions = [
        {"status": "accepted", "base_id": id_, "structure": {"key": key}}
        for id_, key in (("a", "old"), ("b", "new"), ("c", "new"))
    ]
    kept, duplicates = deduplicate(old, additions)
    assert [r["base_id"] for r in kept] == ["b"]
    assert duplicates == ["a", "c"]


def test_deeper_patterns_change_query_structure():
    entries = [(s, d) for s, d in schedule() if d == 5]
    keys = {
        pattern: {
            query_profile(s.problems()[1])["core_topology"]
            for s, _ in entries
            if s.recipe == pattern
        }
        for pattern in PATTERNS
    }
    # Shared rival support is a branching graph, unlike the two single-chain forms.
    assert keys["shared-exclusion"].isdisjoint(keys["match-chain"])
    assert keys["shared-exclusion"].isdisjoint(keys["rival-chain"])
