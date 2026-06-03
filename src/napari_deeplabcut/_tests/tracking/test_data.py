from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.core.schemas.layer_identity import (
    DLC_LAYER_ROLE_KEY,
    DLC_SAVE_BEHAVIOR_KEY,
    LayerRole,
)
from napari_deeplabcut.tracking.core.data import (
    TRACKING_LAYER_METADATA_KEY,
    TRACKING_RESULT_TYPE,
    TRACKING_SCHEMA_VERSION,
    add_query_identity_columns,
    build_tracking_result_metadata,
    coerce_features_df,
    expand_query_features_over_time,
    is_tracking_result_points_layer,
)

# -----------------------------------------------------------------------------#
# coerce_features_df
# -----------------------------------------------------------------------------#


def test_coerce_features_df_returns_defensive_copy_for_dataframe():
    src = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["a", "b"],
        },
        index=[10, 20],
    )

    out = coerce_features_df(src)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["label", "id"]
    assert out.index.tolist() == [0, 1]
    assert out["label"].tolist() == ["nose", "tail"]
    assert out["id"].tolist() == ["a", "b"]

    # Defensive copy
    assert out is not src
    out.loc[0, "label"] = "changed"
    assert src.loc[10, "label"] == "nose"


def test_coerce_features_df_builds_dataframe_from_mapping():
    out = coerce_features_df(
        {
            "label": ["nose", "tail"],
            "id": ["a", "b"],
        }
    )

    assert isinstance(out, pd.DataFrame)
    assert out.index.tolist() == [0, 1]
    assert out["label"].tolist() == ["nose", "tail"]
    assert out["id"].tolist() == ["a", "b"]


# -----------------------------------------------------------------------------#
# add_query_identity_columns
# -----------------------------------------------------------------------------#


def test_add_query_identity_columns_preserves_existing_columns_and_adds_tracking_columns():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )

    out = add_query_identity_columns(
        seed,
        query_frame=12,
        source_layer_name="CollectedData_me",
    )

    assert list(out.columns) == [
        "label",
        "id",
        "tracking_query_index",
        "tracking_query_frame",
        "tracking_source_layer_name",
    ]
    assert out["label"].tolist() == ["nose", "tail"]
    assert out["id"].tolist() == ["animal-a", "animal-a"]
    assert out["tracking_query_index"].tolist() == [0, 1]
    assert out["tracking_query_frame"].tolist() == [12, 12]
    assert out["tracking_source_layer_name"].tolist() == ["CollectedData_me", "CollectedData_me"]


def test_add_query_identity_columns_returns_defensive_copy():
    seed = pd.DataFrame({"label": ["nose"], "id": ["animal-a"]})

    out = add_query_identity_columns(
        seed,
        query_frame=0,
        source_layer_name="source",
    )

    assert out is not seed
    out.loc[0, "label"] = "changed"
    assert seed.loc[0, "label"] == "nose"


# -----------------------------------------------------------------------------#
# expand_query_features_over_time
# -----------------------------------------------------------------------------#


def test_expand_query_features_over_time_repeats_seed_features_and_adds_tracking_fields():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
            "tracking_query_index": [0, 1],
        }
    )
    frame_ids = np.array([5, 6, 7], dtype=int)

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=None,
        tracker_name="Cotracker 3",
    )

    # T=3, K=2 => 6 rows
    assert len(out) == 6

    # Original semantic columns repeated in query order for each frame
    assert out["label"].tolist() == ["nose", "tail", "nose", "tail", "nose", "tail"]
    assert out["id"].tolist() == ["animal-a"] * 6
    assert out["tracking_query_index"].tolist() == [0, 1, 0, 1, 0, 1]

    # Tracking-specific columns
    assert out["tracking_tracker_name"].tolist() == ["Cotracker 3"] * 6
    assert out["tracking_frame"].tolist() == [5, 5, 6, 6, 7, 7]
    assert out["tracking_is_prediction"].tolist() == [True] * 6
    assert out["tracking_visible"].tolist() == [True] * 6


def test_expand_query_features_over_time_uses_visibility_matrix_of_shape_t_k():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )
    frame_ids = np.array([0, 1], dtype=int)
    visibility = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    assert len(out) == 4
    assert out["tracking_visible"].tolist() == [True, False, False, True]


def test_expand_query_features_over_time_accepts_visibility_shape_t_k_1():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )
    frame_ids = np.array([0, 1], dtype=int)
    visibility = np.array(
        [
            [[True], [False]],
            [[False], [True]],
        ],
        dtype=bool,
    )  # shape (T, K, 1)

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    assert out["tracking_visible"].tolist() == [True, False, False, True]


def test_expand_query_features_over_time_accepts_visibility_shape_1_t_k():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )
    frame_ids = np.array([0, 1], dtype=int)
    visibility = np.array(
        [
            [
                [True, False],
                [False, True],
            ]
        ],
        dtype=bool,
    )  # shape (1, T, K)

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    assert out["tracking_visible"].tolist() == [True, False, False, True]


def test_expand_query_features_over_time_raises_for_visibility_shape_mismatch():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )
    frame_ids = np.array([0, 1], dtype=int)

    bad_visibility = np.array([True, False, True], dtype=bool)

    with pytest.raises(ValueError, match="Visibility shape mismatch"):
        expand_query_features_over_time(
            seed,
            frame_ids=frame_ids,
            visibility=bad_visibility,
            tracker_name="Cotracker 3",
        )


def test_expand_query_features_over_time_accepts_single_query_visibility_shape_k_t_regression():
    seed = pd.DataFrame(
        {
            "label": ["nose"],
            "id": ["animal-a"],
        }
    )
    frame_ids = np.array([10, 11, 12], dtype=int)

    # Regression case:
    # T=3, K=1
    # Expected canonical shape is (T, K) == (3, 1),
    # but some model outputs can arrive as (K, T) == (1, 3).
    visibility = np.array(
        [[True, False, True]],
        dtype=bool,
    )  # shape (1, T)

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    assert len(out) == 3
    assert out["tracking_frame"].tolist() == [10, 11, 12]
    assert out["tracking_visible"].tolist() == [True, False, True]


def test_expand_query_features_over_time_accepts_visibility_shape_k_t_transposed():
    seed = pd.DataFrame(
        {
            "label": ["nose", "tail"],
            "id": ["animal-a", "animal-a"],
        }
    )
    frame_ids = np.array([0, 1, 2], dtype=int)

    # Shape is (K, T) == (2, 3), not canonical (T, K) == (3, 2).
    visibility = np.array(
        [
            [True, False, True],  # nose over time
            [False, True, False],  # tail over time
        ],
        dtype=bool,
    )

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    # Output is frame-major:
    # frame 0: nose, tail
    # frame 1: nose, tail
    # frame 2: nose, tail
    assert out["tracking_visible"].tolist() == [
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_expand_query_features_over_time_accepts_single_frame_single_query_visibility_vector():
    seed = pd.DataFrame(
        {
            "label": ["nose"],
            "id": ["animal-a"],
        }
    )
    frame_ids = np.array([42], dtype=int)

    visibility = np.array([True], dtype=bool)  # shape (1,)

    out = expand_query_features_over_time(
        seed,
        frame_ids=frame_ids,
        visibility=visibility,
        tracker_name="Cotracker 3",
    )

    assert len(out) == 1
    assert out["tracking_frame"].tolist() == [42]
    assert out["tracking_visible"].tolist() == [True]


# -----------------------------------------------------------------------------#
# build_tracking_result_metadata
# -----------------------------------------------------------------------------#


def test_build_tracking_result_metadata_preserves_existing_metadata_and_adds_tracking_payload():
    source_metadata = {
        "name": "CollectedData_me",
        "header": {"columns": ["dummy"]},
    }

    out = build_tracking_result_metadata(
        source_metadata,
        tracker_name="Cotracker 3",
        source_layer_name="CollectedData_me",
        query_frame=5,
    )

    assert out["name"] == "CollectedData_me"
    assert out["header"] == {"columns": ["dummy"]}

    info = out[TRACKING_LAYER_METADATA_KEY]
    tracker = "Cotracker 3"
    assert info == {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "kind": TRACKING_RESULT_TYPE,
        "tracker_name": tracker,
        "source_layer_name": "CollectedData_me",
        "query_frame": 5,
    }


def test_build_tracking_result_metadata_returns_deep_copy():
    source_metadata = {
        "nested": {"a": 1},
    }

    out = build_tracking_result_metadata(
        source_metadata,
        tracker_name="Cotracker 3",
        source_layer_name="CollectedData_me",
        query_frame=0,
    )

    assert out is not source_metadata
    assert out["nested"] is not source_metadata["nested"]

    out["nested"]["a"] = 999
    assert source_metadata["nested"]["a"] == 1


def test_build_tracking_result_metadata_handles_none_source_metadata():
    tracker = "cotracker"
    out = build_tracking_result_metadata(
        None,
        tracker_name=tracker,
        source_layer_name="CollectedData_me",
        query_frame=0,
    )

    assert isinstance(out, dict)
    assert TRACKING_LAYER_METADATA_KEY in out
    assert out[TRACKING_LAYER_METADATA_KEY]["kind"] == TRACKING_RESULT_TYPE


# -----------------------------------------------------------------------------#
# is_tracking_result_points_layer
# -----------------------------------------------------------------------------#


def test_is_tracking_result_points_layer_returns_true_for_expected_metadata():
    tracker = "cotracker"
    layer = SimpleNamespace(
        metadata={
            TRACKING_LAYER_METADATA_KEY: {
                "kind": TRACKING_RESULT_TYPE,
                "tracker_name": tracker,
            }
        }
    )

    assert is_tracking_result_points_layer(layer) is True


def test_is_tracking_result_points_layer_returns_false_when_metadata_missing_or_wrong():
    missing = SimpleNamespace(metadata={})
    wrong_kind = SimpleNamespace(metadata={TRACKING_LAYER_METADATA_KEY: {"kind": "something-else"}})
    wrong_type = SimpleNamespace(metadata={TRACKING_LAYER_METADATA_KEY: "not-a-dict"})
    no_metadata = SimpleNamespace()

    assert is_tracking_result_points_layer(missing) is False
    assert is_tracking_result_points_layer(wrong_kind) is False
    assert is_tracking_result_points_layer(wrong_type) is False
    assert is_tracking_result_points_layer(no_metadata) is False


def test_build_tracking_result_metadata_tags_tracking_identity():
    tracker = "cotracker"
    md = build_tracking_result_metadata(
        {
            DLC_LAYER_ROLE_KEY: LayerRole.DLC_ANNOTATION.value,
            "source_field": "kept",
        },
        tracker_name=tracker,
        source_layer_name="CollectedData_scorer",
        query_frame=5,
    )

    assert md["source_field"] == "kept"
    assert md[DLC_LAYER_ROLE_KEY] == LayerRole.TRACKING_RESULT.value
    assert DLC_SAVE_BEHAVIOR_KEY not in md

    info = md[TRACKING_LAYER_METADATA_KEY]
    assert info["kind"] == TRACKING_RESULT_TYPE
    assert info["tracker_name"] == tracker
    assert info["source_layer_name"] == "CollectedData_scorer"
    assert info["query_frame"] == 5
