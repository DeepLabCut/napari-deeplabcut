import logging

import numpy as np

from napari_deeplabcut.core.project_paths import PathMatchPolicy
from napari_deeplabcut.core.remap import (
    RemapOutcome,
    RemapReason,
    build_frame_index_map,
    remap_layer_data_by_paths,
    remap_time_indices,
)


def test_build_frame_index_map_depth3_overlap_and_reorder():
    """
    If canonicalization depth=3 yields overlap, and new_paths order differs,
    build_frame_index_map must produce correct old_idx -> new_idx mapping.
    """
    old_paths = [
        "projA/labeled-data/test/img000.png",
        "projA/labeled-data/test/img001.png",
    ]
    new_paths = [
        "projB/labeled-data/test/img001.png",
        "projB/labeled-data/test/img000.png",
    ]

    idx_map, depth = build_frame_index_map(
        old_paths=old_paths,
        new_paths=new_paths,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert depth == 3
    assert idx_map == {0: 1, 1: 0}


def test_build_frame_index_map_falls_back_to_depth2_when_depth3_has_no_overlap():
    """
    Construct paths such that:
      - last 3 components differ, so no overlap at depth=3
      - last 2 components match, so overlap exists at depth=2
    """
    old_paths = [
        "A/one/img000.png",
        "A/one/img001.png",
    ]
    new_paths = [
        "B/one/img000.png",
        "B/one/img001.png",
    ]

    idx_map, depth = build_frame_index_map(
        old_paths=old_paths,
        new_paths=new_paths,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert depth == 2
    assert idx_map == {0: 0, 1: 1}


def test_remap_layer_data_by_paths_array_like_points_time_col0():
    """
    Array-like data, such as Points data, should have its time column remapped.
    """
    old_paths = ["x/y/img0.png", "x/y/img1.png"]
    new_paths = ["x/y/img1.png", "x/y/img0.png"]

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.APPLIED_FULL
    assert res.reason is RemapReason.REMAPPED
    assert res.applied is True
    assert res.paths_updated is True
    assert res.depth_used in (3, 2, 1)
    assert res.data is not None
    assert np.array_equal(res.data[:, 0].astype(int), np.array([1, 0]))


def test_remap_layer_data_by_paths_array_like_tracks_time_col1():
    """
    For Tracks layers, time column is typically column 1.
    Ensure remap works for arbitrary time_col.
    """
    old_paths = ["root/seq/img0.png", "root/seq/img1.png"]
    new_paths = ["root/seq/img1.png", "root/seq/img0.png"]

    data = np.array(
        [
            [5.0, 0.0, 10.0, 20.0],
            [5.0, 1.0, 11.0, 21.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=1,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.APPLIED_FULL
    assert res.reason is RemapReason.REMAPPED
    assert res.applied is True
    assert res.paths_updated is True
    assert res.data is not None
    assert np.array_equal(res.data[:, 1].astype(int), np.array([1, 0]))
    assert np.array_equal(res.data[:, 0].astype(int), np.array([5, 5]))


def test_remap_layer_data_by_paths_list_like_shapes():
    """
    Shapes-like data is a list of arrays; remap should apply per-vertex array.
    """
    old_paths = ["p/q/img0.png", "p/q/img1.png"]
    new_paths = ["p/q/img1.png", "p/q/img0.png"]

    shapes = [
        np.array(
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
            ],
            dtype=float,
        )
    ]

    res = remap_layer_data_by_paths(
        data=shapes,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.APPLIED_FULL
    assert res.reason is RemapReason.REMAPPED
    assert res.applied is True
    assert res.paths_updated is True
    assert isinstance(res.data, list)
    assert np.array_equal(np.asarray(res.data[0])[:, 0].astype(int), np.array([1, 0]))


def test_no_overlap_returns_skipped_and_no_data():
    old_paths = ["a/b/c/img0.png"]
    new_paths = ["x/y/z/other.png"]

    data = np.array([[0.0, 1.0, 2.0]], dtype=float)

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.SKIPPED
    assert res.reason is RemapReason.NO_OVERLAP
    assert res.applied is False
    assert res.paths_updated is False
    assert res.data is None
    assert res.depth_used is None
    assert "No overlap" in res.message or "skipping" in res.message.lower()


def test_already_aligned_paths_returns_noop():
    """
    If canonicalized old_keys == new_keys, remap_layer_data_by_paths should no-op
    while allowing callers to update/sync path metadata.
    """
    paths = ["labeled-data/test/img0.png", "labeled-data/test/img1.png"]

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=paths,
        new_paths=paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.NOOP
    assert res.reason is RemapReason.ALREADY_ALIGNED
    assert res.applied is False
    assert res.paths_updated is True
    assert res.data is None
    assert res.depth_used is not None
    assert "already aligned" in res.message.lower() or "no remap needed" in res.message.lower()


def test_missing_old_or_new_paths_skips():
    data = np.array([[0.0, 1.0, 2.0]], dtype=float)

    res1 = remap_layer_data_by_paths(
        data=data,
        old_paths=[],
        new_paths=["a/b/c.png"],
        time_col=0,
    )
    assert res1.outcome is RemapOutcome.SKIPPED
    assert res1.reason is RemapReason.NO_OLD_PATHS
    assert res1.applied is False
    assert res1.paths_updated is False
    assert res1.data is None

    res2 = remap_layer_data_by_paths(
        data=data,
        old_paths=["a/b/c.png"],
        new_paths=[],
        time_col=0,
    )
    assert res2.outcome is RemapOutcome.SKIPPED
    assert res2.reason is RemapReason.NO_NEW_PATHS
    assert res2.applied is False
    assert res2.paths_updated is False
    assert res2.data is None


def test_remap_time_indices_rejects_unmapped_indices_in_strict_mode():
    """
    Strict remap no longer leaves unmapped indices unchanged.

    If idx_map does not contain a used frame index, remap_time_indices rejects
    rather than mixing remapped and stale indices.
    """
    idx_map = {0: 10}
    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
        ],
        dtype=float,
    )

    res = remap_time_indices(data=data, time_col=0, idx_map=idx_map)

    assert res.outcome is RemapOutcome.REJECTED
    assert res.reason is RemapReason.REMAP_FAILED
    assert res.applied is False
    assert res.paths_updated is False
    assert res.data is None
    assert "Failed to remap time column" in res.message


def test_remap_time_indices_allows_partial_array_remap_when_requested():
    """
    With allow_partial=True, array-like data should drop rows whose frame index
    cannot be mapped and report APPLIED_PARTIAL.
    """
    idx_map = {0: 10}
    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [1.0, 12.0, 22.0],
        ],
        dtype=float,
    )

    res = remap_time_indices(
        data=data,
        time_col=0,
        idx_map=idx_map,
        allow_partial=True,
    )

    assert res.outcome is RemapOutcome.APPLIED_PARTIAL
    assert res.reason is RemapReason.PARTIAL_ROWS_DROPPED
    assert res.applied is True
    assert res.paths_updated is True
    assert res.data is not None
    assert res.dropped_row_count == 2
    assert res.dropped_frame_indices == (1,)
    assert np.array_equal(res.data[:, 0].astype(int), np.array([10]))
    assert np.array_equal(res.data[:, 1:].astype(int), np.array([[10, 20]]))


def test_remap_time_indices_partial_rejects_when_no_rows_are_mappable():
    idx_map = {0: 10}
    data = np.array(
        [
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
        ],
        dtype=float,
    )

    res = remap_time_indices(
        data=data,
        time_col=0,
        idx_map=idx_map,
        allow_partial=True,
    )

    assert res.outcome is RemapOutcome.REJECTED
    assert res.reason is RemapReason.NO_MAPPABLE_ROWS
    assert res.applied is False
    assert res.paths_updated is False
    assert res.data is None
    assert res.dropped_row_count == 2
    assert res.dropped_frame_indices == (1, 2)


def test_remap_time_indices_gracefully_handles_empty_and_none():
    res_none = remap_time_indices(data=None, time_col=0, idx_map={0: 1})
    assert res_none.outcome is RemapOutcome.SKIPPED
    assert res_none.reason is RemapReason.NO_DATA
    assert res_none.applied is False
    assert res_none.paths_updated is False
    assert res_none.data is None

    res_empty = remap_time_indices(data=np.array([]), time_col=0, idx_map={0: 1})
    assert res_empty.outcome is RemapOutcome.SKIPPED
    assert res_empty.reason is RemapReason.NO_DATA
    assert res_empty.applied is False
    assert res_empty.paths_updated is False
    assert res_empty.data is None


def test_remap_rejects_on_duplicate_canonical_keys(caplog):
    caplog.set_level(logging.WARNING, logger="napari_deeplabcut.core.remap")

    old_paths = [
        "A/dup/img0.png",
        "B/dup/img0.png",
        "C/dup/img1.png",
    ]
    new_paths = [
        "X/dup/img1.png",
        "Y/dup/img0.png",
        "Z/dup/img0.png",
    ]

    data = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [2.0, 5.0, 6.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome is RemapOutcome.REJECTED
    assert res.reason is RemapReason.AMBIGUOUS_MATCH
    assert res.applied is False
    assert res.paths_updated is False
    assert res.data is None
    assert "Remap may be ambiguous/risky" in caplog.text
    assert "Duplicate canonical keys" in caplog.text
    assert any("Duplicate canonical keys" in w for w in res.warnings)


def test_remap_warns_on_low_old_path_coverage_in_strict_mode(caplog):
    caplog.set_level(logging.WARNING, logger="napari_deeplabcut.core.remap")

    old_paths = [
        "x/y/img0.png",
        "x/y/img1.png",
        "x/y/img2.png",
        "x/y/img3.png",
    ]
    new_paths = [
        "x/y/img0.png",
        "x/y/img9.png",
        "x/y/img8.png",
        "x/y/img7.png",
    ]

    data = np.array([[0.0, 1.0, 2.0]], dtype=float)

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
    )

    assert res.outcome in {RemapOutcome.NOOP, RemapOutcome.APPLIED_FULL}
    assert res.paths_updated is True
    assert "Low old-path coverage" in caplog.text
    assert any("Low old-path coverage" in w for w in res.warnings)


def test_remap_layer_data_by_paths_rejects_unmapped_used_indices_in_strict_mode():
    """
    In strict mode, if any actually-used frame index has no path mapping,
    the high-level remap is rejected.
    """
    old_paths = [
        "old/session/img0.png",
        "old/session/img1.png",
        "old/session/img2.png",
        "old/session/img3.png",
    ]
    new_paths = [
        "new/session/img0.png",
        "new/session/img1.png",
    ]

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
        allow_partial=False,
    )

    assert res.outcome is RemapOutcome.REJECTED
    assert res.reason is RemapReason.USED_INDICES_UNMAPPED
    assert res.applied is False
    assert res.paths_updated is False
    assert res.data is None
    assert res.mapped_count == 2
    assert any("Used frame indices without mapping" in w for w in res.warnings)


def test_remap_layer_data_by_paths_applies_partial_when_allowed():
    """
    DLC-style partial remap case.

    Some old frame paths map to the new image stack, others do not. With
    allow_partial=True, rows on unmappable frames are dropped and the remaining
    rows are remapped.
    """
    old_paths = [
        "old/session/img0.png",
        "old/session/img1.png",
        "old/session/img2.png",
        "old/session/img3.png",
    ]
    new_paths = [
        "new/session/img1.png",
        "new/session/img0.png",
    ]

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
            [3.0, 14.0, 24.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
        allow_partial=True,
    )

    assert res.outcome is RemapOutcome.APPLIED_PARTIAL
    assert res.reason is RemapReason.PARTIAL_ROWS_DROPPED
    assert res.applied is True
    assert res.paths_updated is True
    assert res.data is not None
    assert res.mapped_count == 2
    assert res.dropped_row_count == 3
    assert res.dropped_frame_indices == (2, 3)
    assert np.array_equal(res.data[:, 0].astype(int), np.array([1, 0]))
    assert np.array_equal(
        res.data[:, 1:].astype(int),
        np.array(
            [
                [10, 20],
                [11, 21],
            ]
        ),
    )
    assert any("Partial remap" in w for w in res.warnings)


def test_remap_layer_data_by_paths_partial_suppresses_low_coverage_warning(caplog):
    """
    When allow_partial=True and unmapped used frames exist, low old-path coverage
    is expected and should be debug-only rather than a user-facing warning.
    """
    caplog.set_level(logging.DEBUG, logger="napari_deeplabcut.core.remap")

    old_paths = [
        "old/session/img0.png",
        "old/session/img1.png",
        "old/session/img2.png",
        "old/session/img3.png",
    ]
    new_paths = [
        "new/session/img0.png",
        "new/session/img1.png",
    ]

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
        ],
        dtype=float,
    )

    res = remap_layer_data_by_paths(
        data=data,
        old_paths=old_paths,
        new_paths=new_paths,
        time_col=0,
        policy=PathMatchPolicy.ORDERED_DEPTHS,
        allow_partial=True,
    )

    assert res.outcome is RemapOutcome.APPLIED_PARTIAL
    assert res.reason is RemapReason.PARTIAL_ROWS_DROPPED
    assert "Low old-path coverage during partial remap" in caplog.text
    assert not any("Low old-path coverage" in w for w in res.warnings)
    assert any("Partial remap" in w for w in res.warnings)
