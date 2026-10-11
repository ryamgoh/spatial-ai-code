import pytest

from spatial.v2.study.metrics import holm_adjust, paired_macro_difference


def test_macro_does_not_overweight_large_cell():
    cells = {"small": ["a"], "large": ["b", "c", "d"]}
    first = dict.fromkeys(["a", "b", "c", "d"], False)
    second = dict(first, a=True)
    report = paired_macro_difference(cells, first, second, repetitions=100)
    assert report["difference"] == 0.5
    assert report["interval95"] == [0.5, 0.5]
    with pytest.raises(ValueError):
        paired_macro_difference(cells, first, {"a": True})


def test_holm_adjusts_primary_family():
    assert holm_adjust({"natural": 0.01, "symbolic": 0.04}) == {
        "natural": 0.02,
        "symbolic": 0.04,
    }


def test_zero_paired_difference_has_unit_permutation_pvalue():
    from spatial.v2.study.metrics import paired_macro_permutation

    assert (
        paired_macro_permutation(
            {"c": ["a", "b"]},
            {"a": True, "b": False},
            {"a": True, "b": False},
            repetitions=100,
        )
        == 1
    )
