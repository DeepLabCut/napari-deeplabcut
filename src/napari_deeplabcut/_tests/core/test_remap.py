import numpy as np

from napari_deeplabcut.core.paths import PathMatchPolicy
from napari_deeplabcut.core.remap import (
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
    # reverse order in new paths
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
      - last 3 components differ (no overlap at n=3)
      - last 2 components match (overlap at n=2)
    """
    old_paths = [
        "A/one/img000.png",  # last3 => A/one/img000.png
        "A/one/img001.png",
    ]
    new_paths = [
        "B/one/img000.png",  # last3 => B/one/img000.png (no overlap)
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
    Array-like data (Points/Tracks) should have its time column remapped.
    """
    old_paths = ["x/y/img0.png", "x/y/img1.png"]
    new_paths = ["x/y/img1.png", "x/y/img0.png"]  # swap order -> mapping {0:1,1:0}

    # data columns: (time, y, x)
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

    assert res.depth_used in (3, 2, 1)  # depends on how many components are present
    assert res.changed is True
    assert res.data is not None
    assert np.array_equal(res.data[:, 0].astype(int), np.array([1, 0]))


def test_remap_layer_data_by_paths_array_like_tracks_time_col1():
    """
    For Tracks layers, time column is typically column 1.
    Ensure remap works for arbitrary time_col.
    """
    old_paths = ["root/seq/img0.png", "root/seq/img1.png"]
    new_paths = ["root/seq/img1.png", "root/seq/img0.png"]

    # tracks-like: (track_id, time, y, x)
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

    assert res.changed is True
    assert res.data is not None
    assert np.array_equal(res.data[:, 1].astype(int), np.array([1, 0]))
    # track_id must remain unchanged
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

    assert res.changed is True
    assert isinstance(res.data, list)
    assert np.array_equal(np.asarray(res.data[0])[:, 0].astype(int), np.array([1, 0]))


def test_no_overlap_returns_no_change_and_no_data():
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

    assert res.changed is False
    assert res.data is None
    assert res.depth_used is None
    assert "No overlap" in res.message or "skipping" in res.message.lower()


def test_already_aligned_paths_skips_remap():
    """
    If canonicalized old_keys == new_keys, remap_layer_data_by_paths should no-op.
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

    assert res.changed is False
    assert res.data is None
    assert res.depth_used is not None  # a matching depth exists
    assert "already aligned" in res.message.lower() or "no remap needed" in res.message.lower()


def test_missing_old_or_new_paths_skips():
    data = np.array([[0.0, 1.0, 2.0]], dtype=float)

    res1 = remap_layer_data_by_paths(data=data, old_paths=[], new_paths=["a/b/c.png"], time_col=0)
    assert res1.changed is False
    assert res1.data is None

    res2 = remap_layer_data_by_paths(data=data, old_paths=["a/b/c.png"], new_paths=[], time_col=0)
    assert res2.changed is False
    assert res2.data is None


def test_remap_time_indices_leaves_unmapped_indices_unchanged():
    """
    If idx_map doesn't contain a value, it should remain unchanged.
    """
    idx_map = {0: 10}  # only frame 0 remaps
    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],  # frame 1 unmapped -> should stay 1
        ],
        dtype=float,
    )

    res = remap_time_indices(data=data, time_col=0, idx_map=idx_map)

    assert res.changed is True
    assert res.data is not None
    assert np.array_equal(res.data[:, 0].astype(int), np.array([10, 1]))


def test_remap_time_indices_gracefully_handles_empty_and_none():
    res_none = remap_time_indices(data=None, time_col=0, idx_map={0: 1})
    assert res_none.changed is False
    assert res_none.data is None

    res_empty = remap_time_indices(data=np.array([]), time_col=0, idx_map={0: 1})
    assert res_empty.changed is False
    assert res_empty.data is None
