from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.tracking.core import utils as tracking_utils

# -----------------------------------------------------------------------------#
# extract_layer_data_and_features
# -----------------------------------------------------------------------------#


def test_extract_layer_data_and_features_prefers_aligned_features(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        data=[[0, 10, 20], [1, 11, 21]],
        features=pd.DataFrame({"label": ["nose", "tail"], "id": ["a", "a"]}),
        properties={"label": np.array(["wrong1", "wrong2"], dtype=object)},
    )

    data, feat_df = tracking_utils.extract_layer_data_and_features(layer)

    assert data.shape == (2, 3)
    assert list(feat_df.columns) == ["label", "id"]
    assert feat_df["label"].tolist() == ["nose", "tail"]


def test_extract_layer_data_and_features_falls_back_to_properties_if_features_length_mismatch(
    fake_points_layer_factory,
):
    layer = fake_points_layer_factory(
        data=[[0, 10, 20], [1, 11, 21]],
        features=pd.DataFrame({"label": ["nose"]}),
        properties={
            "label": np.array(["nose", "tail"], dtype=object),
            "id": np.array(["a", "b"], dtype=object),
        },
    )

    data, feat_df = tracking_utils.extract_layer_data_and_features(layer)

    assert data.shape == (2, 3)
    assert feat_df["label"].tolist() == ["nose", "tail"]
    assert feat_df["id"].tolist() == ["a", "b"]


def test_extract_layer_data_and_features_raises_if_neither_features_nor_properties_align(
    fake_points_layer_factory,
):
    layer = fake_points_layer_factory(
        data=[[0, 10, 20], [1, 11, 21]],
        features=pd.DataFrame({"label": ["nose"]}),
        properties={"label": np.array(["nose"], dtype=object)},
        name="bad_layer",
    )

    with pytest.raises(ValueError, match="mismatched data/features lengths"):
        tracking_utils.extract_layer_data_and_features(layer)


# -----------------------------------------------------------------------------#
# coord / semantic helpers
# -----------------------------------------------------------------------------#


def test_coord_columns_for_data_handles_3_and_extra_dimensions():
    arr3 = np.zeros((2, 3), dtype=float)
    arr5 = np.zeros((2, 5), dtype=float)

    assert tracking_utils.coord_columns_for_data(arr3) == ["frame", "y", "x"]
    assert tracking_utils.coord_columns_for_data(arr5) == ["frame", "y", "x", "coord_3", "coord_4"]


def test_coord_columns_for_data_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least 3 columns"):
        tracking_utils.coord_columns_for_data(np.zeros((2, 2), dtype=float))

    with pytest.raises(ValueError, match="at least 3 columns"):
        tracking_utils.coord_columns_for_data(np.zeros((3,), dtype=float))


def test_pick_semantic_series_prefers_features_over_properties(fake_points_layer_factory):
    df = pd.DataFrame({"label": ["nose", "tail"]})
    layer = fake_points_layer_factory(
        data=[[0, 0, 0], [1, 1, 1]],
        properties={"label": np.array(["wrong1", "wrong2"], dtype=object)},
    )

    out = tracking_utils.pick_semantic_series(
        df,
        layer,
        primary="label",
        fallback_prop="label",
    )

    assert out.tolist() == ["nose", "tail"]


def test_pick_semantic_series_falls_back_to_properties(fake_points_layer_factory):
    df = pd.DataFrame(index=range(2))
    layer = fake_points_layer_factory(
        data=[[0, 0, 0], [1, 1, 1]],
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    out = tracking_utils.pick_semantic_series(
        df,
        layer,
        primary="label",
        fallback_prop="label",
    )

    assert out.tolist() == ["nose", "tail"]


def test_pick_semantic_series_returns_blank_series_if_property_missing_or_wrong_length(fake_points_layer_factory):
    df = pd.DataFrame(index=range(2))

    missing_layer = fake_points_layer_factory(data=[[0, 0, 0], [1, 1, 1]], properties={})
    out_missing = tracking_utils.pick_semantic_series(
        df,
        missing_layer,
        primary="label",
        fallback_prop="label",
    )
    assert out_missing.tolist() == ["", ""]

    wrong_len_layer = fake_points_layer_factory(
        data=[[0, 0, 0], [1, 1, 1]],
        properties={"label": np.array(["only_one"], dtype=object)},
    )
    out_wrong = tracking_utils.pick_semantic_series(
        df,
        wrong_len_layer,
        primary="label",
        fallback_prop="label",
    )
    assert out_wrong.tolist() == ["", ""]


def test_coerce_frame_series_rounds_and_coerces_invalid_to_na():
    out = tracking_utils.coerce_frame_series(pd.Series([0.2, 1.8, "bad", None]))

    assert out.dtype.name == "Float64"
    assert out.iloc[0] == 0
    assert out.iloc[1] == 2
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        (None, ""),
        (np.nan, ""),
        ("nan", ""),
        ("  cat_a  ", "cat_a"),
        (3, "3"),
    ],
)
def test_normalize_slot_id(value, expected):
    assert tracking_utils.normalize_slot_id(value) == expected


def test_build_slot_key_normalizes_id_and_strips_label():
    key = tracking_utils.build_slot_key(frame=5.0, slot_id="  cat_a  ", label=" nose ")
    assert key == (5, "cat_a", "nose")


def test_duplicate_slot_row_indices_returns_original_indices_for_any_duplicate():
    df = pd.DataFrame(
        {
            "_source_row_index": [10, 11, 12, 13],
            "_slot_key": [
                (0, "a", "nose"),
                (0, "a", "nose"),
                (1, "b", "tail"),
                (2, "c", "ear"),
            ],
        }
    )

    out = tracking_utils.duplicate_slot_row_indices(df)
    assert out == (10, 11)


def test_sorted_int_tuple_normalizes_and_sorts():
    assert tracking_utils.sorted_int_tuple([3, "1", 2]) == (1, 2, 3)


def test_format_helpers():
    assert tracking_utils.format_frame_label(5.0) == "5"
    assert tracking_utils.format_slot_label("nose", "cat_a") == "nose (id: cat_a)"
    assert tracking_utils.format_slot_label("nose", "") == "nose"
    assert tracking_utils.format_coords_text({"x": 1.23456, "y": 9.87654}) == "(x=1.235, y=9.877)"


# -----------------------------------------------------------------------------#
# normalize_points_layer_for_tracking
# -----------------------------------------------------------------------------#


def test_normalize_points_layer_for_tracking_builds_expected_columns(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        data=[[0, 10, 20], [1, 11, 21]],
        features=pd.DataFrame({"label": ["nose", "tail"], "id": ["cat_a", "cat_b"]}),
    )

    df = tracking_utils.normalize_points_layer_for_tracking(
        layer,
        valid_flag_column="_is_valid_merge_row",
    )

    assert list(df["frame"].astype("Int64")) == [0, 1]
    assert df["y"].tolist() == [10.0, 11.0]
    assert df["x"].tolist() == [20.0, 21.0]
    assert df["label"].tolist() == ["nose", "tail"]
    assert df["id"].tolist() == ["cat_a", "cat_b"]
    assert df["_source_row_index"].tolist() == [0, 1]
    assert df["_is_valid_merge_row"].tolist() == [True, True]
    assert df["_slot_key"].tolist() == [
        (0, "cat_a", "nose"),
        (1, "cat_b", "tail"),
    ]


def test_normalize_points_layer_for_tracking_uses_properties_fallback_for_label_and_id(
    fake_points_layer_factory,
):
    layer = fake_points_layer_factory(
        data=[[0, 10, 20], [1, 11, 21]],
        features=pd.DataFrame(index=range(2)),
        properties={
            "label": np.array(["nose", "tail"], dtype=object),
            "id": np.array(["cat_a", "cat_b"], dtype=object),
        },
    )

    df = tracking_utils.normalize_points_layer_for_tracking(layer)

    assert df["label"].tolist() == ["nose", "tail"]
    assert df["id"].tolist() == ["cat_a", "cat_b"]
    assert df["_slot_key"].tolist() == [
        (0, "cat_a", "nose"),
        (1, "cat_b", "tail"),
    ]


def test_normalize_points_layer_for_tracking_marks_invalid_rows(fake_points_layer_factory):
    layer = fake_points_layer_factory(
        data=[
            [0, 10, 20],
            [1, np.nan, 21],
            [2, 12, 22],
        ],
        features=pd.DataFrame(
            {
                "label": ["nose", "tail", ""],
                "id": ["cat_a", "cat_a", "cat_a"],
            }
        ),
    )

    df = tracking_utils.normalize_points_layer_for_tracking(layer)

    assert df["_is_valid_slot_row"].tolist() == [True, False, False]
    assert df["_slot_key"].tolist() == [
        (0, "cat_a", "nose"),
        None,
        None,
    ]


# -----------------------------------------------------------------------------#
# tracked naming helpers
# -----------------------------------------------------------------------------#


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CollectedData_me", "CollectedData_me"),
        ("[Tracked v1] CollectedData_me", "CollectedData_me"),
        ("[Tracked] [Tracked v2] foo", "foo"),
        ("   [Tracked v12]   bar   ", "bar"),
    ],
)
def test_strip_tracked_prefix(name, expected):
    assert tracking_utils._strip_tracked_prefix(name) == expected


def test_base_tracking_source_name_prefers_recorded_original_name_for_tracking_layers(
    fake_points_layer_factory,
    tracking_manager_factory,
):
    tracked_source = fake_points_layer_factory(
        data=[[0, 0, 0]],
        name="[Tracked v2] Something noisy",
    )

    manager = tracking_manager_factory(
        tracking_layers=(tracked_source,),
        source_name_by_layer_id={id(tracked_source): "CollectedData_me"},
    )

    out = tracking_utils._base_tracking_source_name(tracked_source, manager)
    assert out == "CollectedData_me"


def test_base_tracking_source_name_falls_back_to_cleaned_layer_name_for_regular_layers(
    fake_points_layer_factory,
    tracking_manager_factory,
):
    source = fake_points_layer_factory(data=[[0, 0, 0]], name="[Tracked v1] CollectedData_me")

    manager = tracking_manager_factory()

    out = tracking_utils._base_tracking_source_name(source, manager)
    assert out == "CollectedData_me"


def test_make_tracking_iteration_name_starts_at_v1(
    fake_points_layer_factory,
    dummy_viewer_factory,
    tracking_manager_factory,
    patch_tracking_manager,
):
    source = fake_points_layer_factory(data=[[0, 0, 0]], name="CollectedData_me")
    viewer = dummy_viewer_factory(layers=[])

    manager = tracking_manager_factory()
    patch_tracking_manager(manager)

    name = tracking_utils.make_tracking_iteration_name(
        viewer=viewer,
        tracker_name="Cotracker3",
        ref_frame_idx=0,
        source=source,
    )

    assert name == "[Tracked v1] CollectedData_me - t0 - Cotracker3"


def test_make_tracking_iteration_name_increments_across_frames_for_same_source_and_tracker(
    fake_points_layer_factory,
    dummy_viewer_factory,
    tracking_manager_factory,
    patch_tracking_manager,
):
    source = fake_points_layer_factory(data=[[0, 0, 0]], name="CollectedData_me")
    viewer = dummy_viewer_factory(
        layers=[
            SimpleNamespace(name="[Tracked v1] CollectedData_me - t0 - Cotracker3"),
            SimpleNamespace(name="[Tracked v2] CollectedData_me - t5 - Cotracker3"),
        ]
    )

    manager = tracking_manager_factory()
    patch_tracking_manager(manager)

    name = tracking_utils.make_tracking_iteration_name(
        viewer=viewer,
        tracker_name="Cotracker3",
        ref_frame_idx=12,
        source=source,
    )

    assert name == "[Tracked v3] CollectedData_me - t12 - Cotracker3"


def test_make_tracking_iteration_name_uses_independent_version_families_per_tracker(
    fake_points_layer_factory,
    dummy_viewer_factory,
    tracking_manager_factory,
    patch_tracking_manager,
):
    source = fake_points_layer_factory(data=[[0, 0, 0]], name="CollectedData_me")
    viewer = dummy_viewer_factory(
        layers=[
            SimpleNamespace(name="[Tracked v1] CollectedData_me - t0 - Cotracker3"),
            SimpleNamespace(name="[Tracked v2] CollectedData_me - t5 - Cotracker3"),
            SimpleNamespace(name="[Tracked v1] CollectedData_me - t3 - SomeOtherTracker"),
        ]
    )

    manager = tracking_manager_factory()
    patch_tracking_manager(manager)

    name = tracking_utils.make_tracking_iteration_name(
        viewer=viewer,
        tracker_name="SomeOtherTracker",
        ref_frame_idx=99,
        source=source,
    )

    assert name == "[Tracked v2] CollectedData_me - t99 - SomeOtherTracker"
