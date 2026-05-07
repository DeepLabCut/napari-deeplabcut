# src/napari_deeplabcut/_tests/tracking/test_merge.py
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.tracking.core.merge import (
    LayerFingerprint,
    TrackingMergePolicy,
    apply_tracking_merge,
    fingerprint_points_layer,
    preview_tracking_merge,
)


def test_fingerprint_points_layer_uses_name_row_count_and_feature_columns(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 30, 40],
        ],
        labels=["nose", "tail"],
        ids=["", ""],
        extra_features={"likelihood": [0.9, 0.8]},
    )

    fp = fingerprint_points_layer(layer)

    assert fp == LayerFingerprint(
        layer_name="tracked",
        n_rows=2,
        feature_columns=("label", "id", "likelihood"),
    )


def test_preview_invalid_when_source_and_target_are_same_layer(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="same",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = preview_tracking_merge(layer, layer)

    assert preview.is_valid is False
    assert "different" in (preview.invalid_reason or "").lower()
    assert preview.n_appendable == 0
    assert preview.n_conflicts == 0
    assert preview.n_identical == 0
    assert preview.n_overwriteable == 0
    assert preview.conflicts == ()
    assert preview.overwrites == ()


def test_preview_classifies_append_identical_conflict_and_invalid_rows(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[
            [0, 10, 20],  # nose @ frame 0
            [0, 30, 40],  # tail @ frame 0
        ],
        labels=["nose", "tail"],
        ids=["", ""],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],  # identical to target nose
            [0, 35, 45],  # conflict with target tail
            [1, 50, 60],  # appendable new frame/keypoint
            [2, 70, 80],  # invalid because label is blank
        ],
        labels=["nose", "tail", "nose", ""],
        ids=["", "", "", ""],
    )

    preview = preview_tracking_merge(source, target)

    assert preview.is_valid is True
    assert preview.policy is TrackingMergePolicy.FILL_MISSING

    assert preview.n_source_rows == 4
    assert preview.n_identical == 1
    assert preview.n_conflicts == 1
    assert preview.n_appendable == 1
    assert preview.n_overwriteable == 0
    assert preview.n_invalid_source == 1

    assert preview.identical_source_indices == (0,)
    assert preview.conflict_source_indices == (1,)
    assert preview.overwrite_source_indices == ()
    assert preview.append_source_indices == (2,)
    assert preview.invalid_source_indices == (3,)

    assert preview.overwrites == ()
    assert len(preview.conflicts) == 1
    conflict = preview.conflicts[0]
    assert conflict.frame_label == "0"
    assert "tail" in conflict.keypoint_label
    assert "source:" not in conflict.source_coords_text  # stored as coords only
    assert conflict.source_coords_text == "(x=45.000, y=35.000)"
    assert conflict.target_coords_text == "(x=40.000, y=30.000)"


def test_preview_invalid_when_source_contains_duplicate_semantic_slots(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],
            [0, 11, 21],
        ],
        labels=["nose", "nose"],
        ids=["", ""],
    )

    preview = preview_tracking_merge(source, target)

    assert preview.is_valid is False
    assert preview.has_source_duplicates is True
    assert preview.has_target_duplicates is False
    assert "duplicate semantic slots" in (preview.invalid_reason or "").lower()


def test_preview_invalid_when_target_contains_duplicate_semantic_slots(fake_points_layer_factory):
    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    target = fake_points_layer_factory(
        name="target",
        data=[
            [0, 10, 20],
            [0, 11, 21],
        ],
        labels=["nose", "nose"],
        ids=["", ""],
    )

    preview = preview_tracking_merge(source, target)

    assert preview.is_valid is False
    assert preview.has_source_duplicates is False
    assert preview.has_target_duplicates is True
    assert "duplicate semantic slots" in (preview.invalid_reason or "").lower()


def test_apply_tracking_merge_appends_only_missing_rows_and_preserves_target_schema(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[
            [0, 10, 20],  # existing identical slot
        ],
        labels=["nose"],
        ids=[""],
        extra_features={
            "likelihood": [0.95],
            "valid": [True],
        },
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],  # identical -> skipped
            [1, 50, 60],  # appendable -> added
        ],
        labels=["nose", "tail"],
        ids=["", ""],
        extra_features={
            "likelihood": [0.95, 0.80],
            "tracking_visible": [True, True],  # must NOT leak into target schema
            "tracking_query_index": [0, 1],  # must NOT leak into target schema
        },
    )

    preview = preview_tracking_merge(source, target)
    assert preview.is_valid is True
    assert preview.n_identical == 1
    assert preview.n_appendable == 1
    assert preview.append_source_indices == (1,)
    assert preview.n_overwriteable == 0
    assert preview.overwrite_source_indices == ()

    new_data, new_features = apply_tracking_merge(
        source_layer=source,
        target_layer=target,
        preview=preview,
    )

    # Existing target row + one appended row
    assert new_data.shape == (2, 3)
    np.testing.assert_allclose(new_data[0], np.array([0, 10, 20], dtype=float))
    np.testing.assert_allclose(new_data[1], np.array([1, 50, 60], dtype=float))

    # Target feature schema is authoritative
    assert list(new_features.columns) == ["label", "id", "likelihood", "valid"]

    # Source-only tracking columns must not leak
    assert "tracking_visible" not in new_features.columns
    assert "tracking_query_index" not in new_features.columns

    # Shared columns copied for appended row
    assert new_features.loc[1, "label"] == "tail"
    assert new_features.loc[1, "id"] == ""
    assert new_features.loc[1, "likelihood"] == pytest.approx(0.80)

    # Non-shared / target-only column remains missing/default on appended row
    assert pd.isna(new_features.loc[1, "valid"])


def test_apply_tracking_merge_returns_target_copy_when_nothing_appendable(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
        extra_features={"likelihood": [0.9]},
    )

    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10, 20]],  # identical only
        labels=["nose"],
        ids=[""],
        extra_features={"likelihood": [0.9]},
    )

    preview = preview_tracking_merge(source, target)
    assert preview.is_valid is True
    assert preview.n_appendable == 0
    assert preview.n_overwriteable == 0
    assert preview.n_identical == 1

    new_data, new_features = apply_tracking_merge(
        source_layer=source,
        target_layer=target,
        preview=preview,
    )

    np.testing.assert_allclose(new_data, np.asarray(target.data, dtype=float))
    pd.testing.assert_frame_equal(
        new_features.reset_index(drop=True),
        pd.DataFrame(target.features).reset_index(drop=True),
        check_dtype=False,
    )

    # Should be copies, not the same objects
    assert new_data is not target.data
    assert new_features is not target.features


def test_apply_tracking_merge_raises_for_invalid_preview(fake_points_layer_factory):
    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = preview_tracking_merge(source, source)  # invalid by construction
    assert preview.is_valid is False

    with pytest.raises(ValueError, match="invalid tracking merge preview|Cannot apply an invalid"):
        apply_tracking_merge(
            source_layer=source,
            target_layer=target,
            preview=preview,
        )


def test_apply_tracking_merge_detects_stale_source_preview_when_layer_name_changes(fake_points_layer_factory):
    source = fake_points_layer_factory(
        name="source",
        data=[[1, 50, 60]],
        labels=["tail"],
        ids=[""],
        extra_features={"likelihood": [0.8]},
    )
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
        extra_features={"likelihood": [0.95]},
    )

    preview = preview_tracking_merge(source, target)
    assert preview.is_valid is True
    assert preview.n_appendable == 1

    # Mutate source fingerprint via layer name
    source.name = "source-renamed"

    with pytest.raises(ValueError, match="Source layer changed after preview was built"):
        apply_tracking_merge(
            source_layer=source,
            target_layer=target,
            preview=preview,
        )


def test_apply_tracking_merge_detects_stale_target_preview_when_feature_columns_change(fake_points_layer_factory):
    source = fake_points_layer_factory(
        name="source",
        data=[[1, 50, 60]],
        labels=["tail"],
        ids=[""],
        extra_features={"likelihood": [0.8]},
    )
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
        extra_features={"likelihood": [0.95]},
    )

    preview = preview_tracking_merge(source, target)
    assert preview.is_valid is True
    assert preview.n_appendable == 1

    # Mutate target fingerprint by changing feature columns
    target.features = pd.DataFrame(
        {
            "label": ["nose"],
            "id": [""],
            "likelihood": [0.95],
            "valid": [True],  # new column changes fingerprint
        }
    )

    with pytest.raises(ValueError, match="Target layer changed after preview was built"):
        apply_tracking_merge(
            source_layer=source,
            target_layer=target,
            preview=preview,
        )


def test_preview_uses_id_in_semantic_slot_identity(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=["animal-1"],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],  # same frame/label, but different id -> appendable, not identical/conflict
        ],
        labels=["nose"],
        ids=["animal-2"],
    )

    preview = preview_tracking_merge(source, target)

    assert preview.is_valid is True
    assert preview.n_appendable == 1
    assert preview.n_identical == 0
    assert preview.n_conflicts == 0
    assert preview.n_overwriteable == 0


def test_preview_tolerance_controls_identical_vs_conflict(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10.0, 20.0]],
        labels=["nose"],
        ids=[""],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10.0000004, 20.0000004]],
        labels=["nose"],
        ids=[""],
    )

    preview_loose = preview_tracking_merge(source, target, coord_tolerance=1e-3)
    assert preview_loose.n_identical == 1
    assert preview_loose.n_conflicts == 0
    assert preview_loose.n_overwriteable == 0

    preview_strict = preview_tracking_merge(source, target, coord_tolerance=1e-9)
    assert preview_strict.n_identical == 0
    assert preview_strict.n_conflicts == 1
    assert preview_strict.n_overwriteable == 0


def test_preview_overwrite_policy_classifies_mismatches_as_overwriteable_not_conflicts(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[
            [0, 10, 20],  # identical slot
            [0, 30, 40],  # differing slot -> overwriteable
        ],
        labels=["nose", "tail"],
        ids=["", ""],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],  # identical
            [0, 35, 45],  # overwriteable
            [1, 50, 60],  # appendable
            [2, 70, 80],  # invalid because label is blank
        ],
        labels=["nose", "tail", "nose", ""],
        ids=["", "", "", ""],
    )

    preview = preview_tracking_merge(
        source,
        target,
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
    )

    assert preview.is_valid is True
    assert preview.policy is TrackingMergePolicy.OVERWRITE_EXISTING

    assert preview.n_source_rows == 4
    assert preview.n_identical == 1
    assert preview.n_conflicts == 0
    assert preview.n_overwriteable == 1
    assert preview.n_appendable == 1
    assert preview.n_invalid_source == 1

    assert preview.identical_source_indices == (0,)
    assert preview.conflict_source_indices == ()
    assert preview.overwrite_source_indices == (1,)
    assert preview.append_source_indices == (2,)
    assert preview.invalid_source_indices == (3,)

    assert preview.conflicts == ()
    assert len(preview.overwrites) == 1

    overwrite = preview.overwrites[0]
    assert overwrite.frame_label == "0"
    assert "tail" in overwrite.keypoint_label
    assert overwrite.source_coords_text == "(x=45.000, y=35.000)"
    assert overwrite.target_coords_text == "(x=40.000, y=30.000)"


def test_apply_tracking_merge_overwrite_policy_overwrites_existing_and_appends_missing(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[
            [0, 10, 20],  # identical slot
            [0, 30, 40],  # overwriteable slot
        ],
        labels=["nose", "tail"],
        ids=["", ""],
        extra_features={
            "likelihood": [0.95, 0.50],
            "valid": [True, False],  # target-owned column
        },
    )

    source = fake_points_layer_factory(
        name="source",
        data=[
            [0, 10, 20],  # identical -> unchanged
            [0, 35, 45],  # overwrite existing tail
            [1, 50, 60],  # append new tail on frame 1
        ],
        labels=["nose", "tail", "tail"],
        ids=["", "", ""],
        extra_features={
            "likelihood": [0.95, 0.80, 0.77],
            "tracking_visible": [True, True, True],  # must not leak
        },
    )

    preview = preview_tracking_merge(
        source,
        target,
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
    )

    assert preview.is_valid is True
    assert preview.n_identical == 1
    assert preview.n_overwriteable == 1
    assert preview.n_appendable == 1
    assert preview.n_conflicts == 0

    new_data, new_features = apply_tracking_merge(
        source_layer=source,
        target_layer=target,
        preview=preview,
    )

    # Same number of target rows, plus one appended row
    assert new_data.shape == (3, 3)

    # Row 0 unchanged (identical)
    np.testing.assert_allclose(new_data[0], np.array([0, 10, 20], dtype=float))

    # Row 1 overwritten in place
    np.testing.assert_allclose(new_data[1], np.array([0, 35, 45], dtype=float))

    # Row 2 appended
    np.testing.assert_allclose(new_data[2], np.array([1, 50, 60], dtype=float))

    # Target schema remains authoritative
    assert list(new_features.columns) == ["label", "id", "likelihood", "valid"]
    assert "tracking_visible" not in new_features.columns

    # Overwritten row should take shared feature values from source
    assert new_features.loc[1, "label"] == "tail"
    assert new_features.loc[1, "id"] == ""
    assert new_features.loc[1, "likelihood"] == pytest.approx(0.80)

    # Target-only column should be preserved on overwritten row
    assert bool(new_features.loc[1, "valid"]) is False

    # Appended row should copy shared columns only
    assert new_features.loc[2, "label"] == "tail"
    assert new_features.loc[2, "id"] == ""
    assert new_features.loc[2, "likelihood"] == pytest.approx(0.77)

    # Target-only column remains NA/default on appended row
    assert pd.isna(new_features.loc[2, "valid"])


def test_apply_tracking_merge_overwrite_policy_can_overwrite_without_appending(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 30, 40]],
        labels=["tail"],
        ids=[""],
        extra_features={"likelihood": [0.50]},
    )

    source = fake_points_layer_factory(
        name="source",
        data=[[0, 35, 45]],
        labels=["tail"],
        ids=[""],
        extra_features={"likelihood": [0.80]},
    )

    preview = preview_tracking_merge(
        source,
        target,
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
    )

    assert preview.is_valid is True
    assert preview.n_appendable == 0
    assert preview.n_overwriteable == 1
    assert preview.n_conflicts == 0

    new_data, new_features = apply_tracking_merge(
        source_layer=source,
        target_layer=target,
        preview=preview,
    )

    assert new_data.shape == (1, 3)
    np.testing.assert_allclose(new_data[0], np.array([0, 35, 45], dtype=float))
    assert new_features.loc[0, "label"] == "tail"
    assert new_features.loc[0, "likelihood"] == pytest.approx(0.80)


def test_apply_tracking_merge_rejects_unknown_policy_even_if_preview_object_is_mutated(fake_points_layer_factory):
    source = fake_points_layer_factory(
        name="source",
        data=[[1, 50, 60]],
        labels=["tail"],
        ids=[""],
    )
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = preview_tracking_merge(source, target)
    assert preview.is_valid is True

    bad_preview = replace(preview, policy="bad-policy")

    with pytest.raises(ValueError, match="Unsupported merge policy"):
        apply_tracking_merge(
            source_layer=source,
            target_layer=target,
            preview=bad_preview,
        )


def test_preview_overwrite_policy_respects_tolerance_for_identical_vs_overwriteable(fake_points_layer_factory):
    target = fake_points_layer_factory(
        name="target",
        data=[[0, 10.0, 20.0]],
        labels=["nose"],
        ids=[""],
    )

    source = fake_points_layer_factory(
        name="source",
        data=[[0, 10.0000004, 20.0000004]],
        labels=["nose"],
        ids=[""],
    )

    preview_loose = preview_tracking_merge(
        source,
        target,
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        coord_tolerance=1e-3,
    )
    assert preview_loose.n_identical == 1
    assert preview_loose.n_overwriteable == 0
    assert preview_loose.n_conflicts == 0

    preview_strict = preview_tracking_merge(
        source,
        target,
        policy=TrackingMergePolicy.OVERWRITE_EXISTING,
        coord_tolerance=1e-9,
    )
    assert preview_strict.n_identical == 0
    assert preview_strict.n_overwriteable == 1
    assert preview_strict.n_conflicts == 0
