from pathlib import Path

import pytest

from .utils import assert_only_these_changed_nan_safe, sig_equal

# -----------------------------------------------------------------------------#
# Assertions on signatures of written files, with NaN-stable equality
# This is required because in DLC h5s NaN means "unlabeled",
# so NaN to value changes are meaningful and should be detected,
# but NaN to NaN should be treated as unchanged  (remains unlabeled).
# Below tests are meant to avoid any future regressions in this logic,
# which is critical for correct writer behavior and testability.
# -----------------------------------------------------------------------------#


def test_sig_equal_treats_nan_as_equal():
    a = {"b2x": float("nan"), "b2y": float("nan")}
    b = {"b2x": float("nan"), "b2y": float("nan")}
    assert sig_equal(a, b)


def test_sig_equal_detects_nan_to_value_change():
    a = {"b2x": float("nan")}
    b = {"b2x": 77.0}
    assert not sig_equal(a, b)


def test_sig_equal_detects_value_change():
    a = {"b1x": 10.0}
    b = {"b1x": 11.0}
    assert not sig_equal(a, b)


def test_assert_only_these_changed_nan_safe_passes_expected_case(tmp_path: Path):
    p1 = tmp_path / "A.h5"
    p2 = tmp_path / "B.h5"

    before = {
        p1: {"b2x": float("nan")},
        p2: {"b2x": float("nan")},
    }
    after = {
        p1: {"b2x": float("nan")},  # unchanged
        p2: {"b2x": 77.0},  # changed
    }

    assert_only_these_changed_nan_safe(before, after, changed={p2})


def test_assert_only_these_changed_nan_safe_fails_when_unexpected_change(tmp_path: Path):
    p1 = tmp_path / "A.h5"
    before = {p1: {"b2x": float("nan")}}
    after = {p1: {"b2x": 1.0}}

    with pytest.raises(AssertionError):
        assert_only_these_changed_nan_safe(before, after, changed=set())
