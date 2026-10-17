import pytest

from spatial.v2.consistency import (
    check_consistency_certificate,
    generate_consistency_pair,
)
from spatial.v2.generation import ENTITY_NAMES


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("cycle_length", (2, 5, 8))
def test_consistency_pairs_are_clean_dirty_and_deletion_minimal(
    axis: str, cycle_length: int
) -> None:
    pair = generate_consistency_pair(
        names=ENTITY_NAMES,
        cycle_length=cycle_length,
        distractor_count=3,
        axis=axis,
        seed=cycle_length,
    )

    check_consistency_certificate(pair.certificate)
    clean, dirty = pair.rows()
    assert clean["metadata"]["label"] == "consistent"
    assert dirty["metadata"]["label"] == "inconsistent"
    assert clean["metadata"]["pair_id"] == dirty["metadata"]["pair_id"]
    assert clean["metadata"]["cycle_length"] == cycle_length
    assert clean["messages"][-1]["content"] == "Answer: A"
    assert dirty["messages"][-1]["content"] == "Answer: B"


def test_consistency_pair_generation_is_deterministic() -> None:
    kwargs = dict(
        names=ENTITY_NAMES,
        cycle_length=4,
        distractor_count=2,
        axis="x",
        seed=42,
    )
    assert generate_consistency_pair(**kwargs) == generate_consistency_pair(**kwargs)
