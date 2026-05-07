# src/napari_deeplabcut/_tests/tracking/test_refine.py
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.tracking.core.refine import (
    apply_delete_tracking_points_in_future,
    preview_delete_tracking_points_in_future,
)

def test_preview_delete_invalid_when_no_selection(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 11, 21],
        ],
        labels=["nose", "nose"],
        ids=["", ""],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices=set(),
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "select one or more" in (preview.invalid_reason or "").lower()
    assert preview.n_rows_to_delete == 0
    assert preview.delete_row_indices == ()
    
def test_preview_delete_invalid_when_selected_index_out_of_bounds(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={99},
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "no longer valid" in (preview.invalid_reason or "").lower()
    
def test_preview_delete_invalid_when_selected_row_is_not_semantically_valid(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 11, 21],
        ],
        labels=["", "nose"],  # row 0 invalid because blank label
        ids=["", ""],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "valid label/id identity" in (preview.invalid_reason or "").lower()
    
def test_preview_delete_invalid_when_selected_rows_are_not_on_anchor_frame(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 11, 21],
        ],
        labels=["nose", "nose"],
        ids=["", ""],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={1},
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "current frame" in (preview.invalid_reason or "").lower()
    

def test_preview_delete_invalid_when_selected_identity_is_not_unique_on_anchor_frame(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [0, 12, 22],
            [1, 15, 25],
        ],
        labels=["nose", "nose", "nose"],
        ids=["animal-a", "animal-a", "animal-a"],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0, 1},
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "not semantically unique" in (preview.invalid_reason or "").lower()
    

def test_preview_delete_valid_with_no_future_matches(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],  # selected
            [1, 11, 21],  # different identity
        ],
        labels=["nose", "tail"],
        ids=["animal-a", "animal-a"],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is True
    assert preview.n_selected_reference_rows == 1
    assert preview.n_selected_slot_keys == 1
    assert preview.n_rows_to_delete == 0
    assert preview.delete_row_indices == ()
    assert preview.ambiguous_slot_frames == ()
    
def test_preview_delete_matches_same_identity_across_future_frames(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],  # selected anchor row
            [1, 11, 21],  # same identity -> delete
            [2, 12, 22],  # same identity -> delete
            [1, 30, 40],  # different id -> keep
            [2, 50, 60],  # different label -> keep
        ],
        labels=["nose", "nose", "nose", "nose", "tail"],
        ids=["animal-a", "animal-a", "animal-a", "animal-b", "animal-a"],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is True
    assert preview.n_selected_reference_rows == 1
    assert preview.n_selected_slot_keys == 1
    assert preview.n_rows_to_delete == 2
    assert preview.delete_row_indices == (1, 2)
    assert preview.ambiguous_slot_frames == ()
    
def test_preview_delete_invalid_when_future_identity_is_ambiguous(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],  # selected anchor row
            [1, 11, 21],  # duplicate future identity
            [1, 12, 22],  # duplicate future identity
            [2, 13, 23],  # same identity future row
        ],
        labels=["nose", "nose", "nose", "nose"],
        ids=["animal-a", "animal-a", "animal-a", "animal-a"],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is False
    assert "ambiguous" in (preview.invalid_reason or "").lower()
    assert preview.n_rows_to_delete == 0
    assert preview.delete_row_indices == ()
    assert preview.ambiguous_slot_frames == ((1, "animal-a", "nose"),)
    
    
def test_apply_delete_tracking_points_in_future_removes_rows_and_preserves_order(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],  # keep (anchor row)
            [1, 11, 21],  # delete
            [2, 12, 22],  # delete
            [1, 30, 40],  # keep
        ],
        labels=["nose", "nose", "nose", "tail"],
        ids=["animal-a", "animal-a", "animal-a", "animal-a"],
        extra_features={"likelihood": [0.9, 0.8, 0.7, 0.6]},
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is True
    assert preview.delete_row_indices == (1, 2)

    new_data, new_features = apply_delete_tracking_points_in_future(
        layer,
        preview=preview,
    )

    assert new_data.shape == (2, 3)
    np.testing.assert_allclose(
        new_data,
        np.array(
            [
                [0, 10, 20],
                [1, 30, 40],
            ],
            dtype=float,
        ),
    )

    assert list(new_features.columns) == ["label", "id", "likelihood"]
    assert new_features["label"].tolist() == ["nose", "tail"]
    assert new_features["id"].tolist() == ["animal-a", "animal-a"]
    assert new_features["likelihood"].tolist() == [0.9, 0.6]
    
def test_apply_delete_tracking_points_in_future_returns_copies_when_nothing_to_delete(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 30, 40],
        ],
        labels=["nose", "tail"],
        ids=["animal-a", "animal-a"],
        extra_features={"likelihood": [0.9, 0.6]},
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    assert preview.is_valid is True
    assert preview.n_rows_to_delete == 0

    new_data, new_features = apply_delete_tracking_points_in_future(
        layer,
        preview=preview,
    )

    np.testing.assert_allclose(new_data, np.asarray(layer.data, dtype=float))
    pd.testing.assert_frame_equal(
        new_features.reset_index(drop=True),
        pd.DataFrame(layer.features).reset_index(drop=True),
        check_dtype=False,
    )

    assert new_data is not layer.data
    assert new_features is not layer.features
    

def test_apply_delete_tracking_points_in_future_raises_for_invalid_preview(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[[0, 10, 20]],
        labels=["nose"],
        ids=[""],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices=set(),
        anchor_frame=0,
    )

    assert preview.is_valid is False

    with pytest.raises(ValueError, match="invalid future-delete preview|Cannot apply an invalid"):
        apply_delete_tracking_points_in_future(
            layer,
            preview=preview,
        )
        
    
def test_apply_delete_tracking_points_in_future_raises_when_delete_indices_are_out_of_bounds(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        name="tracked",
        data=[
            [0, 10, 20],
            [1, 11, 21],
        ],
        labels=["nose", "nose"],
        ids=["animal-a", "animal-a"],
    )

    preview = preview_delete_tracking_points_in_future(
        layer,
        selected_indices={0},
        anchor_frame=0,
    )

    # Simulate stale preview after layer changed
    bad_preview = replace(
        preview,
        delete_row_indices=(99,),
        n_rows_to_delete=1,
    )

    with pytest.raises(ValueError, match="out of bounds"):
        apply_delete_tracking_points_in_future(
            layer,
            preview=bad_preview,
        )