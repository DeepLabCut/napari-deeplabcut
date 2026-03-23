from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.io import read_hdf_single


def _write_minimal_h5(path: Path, scorer: str, with_likelihood: bool = False, all_nan: bool = True):
    # Create a single-row DLC-style dataframe with x/y(/likelihood)
    coords = ["x", "y"] + (["likelihood"] if with_likelihood else [])
    cols = pd.MultiIndex.from_product([[scorer], ["bp1"], coords], names=["scorer", "bodyparts", "coords"])
    if all_nan:
        row = [np.nan] * len(coords)
    else:
        row = [10.0, 20.0] + ([0.9] if with_likelihood else [])
    df = pd.DataFrame([row], index=["img000.png"], columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key="keypoints", mode="w")
    return df


def _assert_properties_match_data_len(meta: dict, n: int):
    props = meta.get("properties") or {}
    assert isinstance(props, dict)
    for k, v in props.items():
        # v can be list, np array, pandas series
        ln = len(v)
        assert ln == n, f"Property '{k}' has len={ln} but data has n={n}"


def test_read_hdf_single_machine_all_nan_returns_empty_points_layer(tmp_path: Path):
    h5 = tmp_path / "machinelabels-iter0.h5"
    _write_minimal_h5(h5, scorer="machine", with_likelihood=False, all_nan=True)

    layers = read_hdf_single(h5, kind=AnnotationKind.MACHINE)
    assert len(layers) == 1

    data, meta, layer_type = layers[0]
    assert layer_type == "points"
    assert np.asarray(data).shape[1] == 3

    # Expect: no finite coords => empty points
    assert np.asarray(data).shape[0] == 0

    # Properties must match data length (0)
    _assert_properties_match_data_len(meta, 0)


def test_read_hdf_single_gt_with_point_produces_one_point_and_matching_properties(tmp_path: Path):
    h5 = tmp_path / "CollectedData_John.h5"
    _write_minimal_h5(h5, scorer="John", with_likelihood=False, all_nan=False)

    layers = read_hdf_single(h5, kind=AnnotationKind.GT)
    data, meta, layer_type = layers[0]

    assert np.asarray(data).shape[0] == 1
    _assert_properties_match_data_len(meta, 1)
    assert meta["properties"]["label"][0] == "bp1"


def test_read_hdf_single_filters_data_and_properties_consistently(tmp_path: Path):
    # Two "rows" will appear after stack: simulate one finite, one non-finite in the stacked frame.
    # Easiest: two bodyparts, one is NaN, one is finite.
    h5 = tmp_path / "CollectedData_John.h5"
    cols = pd.MultiIndex.from_product([["John"], ["bp1", "bp2"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame([[10.0, 20.0, np.nan, np.nan]], index=["img000.png"], columns=cols)
    df.to_hdf(h5, key="keypoints", mode="w")

    layers = read_hdf_single(h5, kind=AnnotationKind.GT)
    data, meta, _ = layers[0]

    # Only bp1 is finite => one point
    assert np.asarray(data).shape[0] == 1
    assert meta["properties"]["label"] == ["bp1"]
    assert len(meta["properties"]["id"]) == 1


def test_reader_never_returns_nonempty_data_with_empty_properties(tmp_path: Path):
    h5 = tmp_path / "machinelabels-iter0.h5"
    _write_minimal_h5(h5, scorer="machine", all_nan=True)

    data, meta, _ = read_hdf_single(h5, kind=AnnotationKind.MACHINE)[0]
    n = np.asarray(data).shape[0]
    props = meta.get("properties") or {}
    if n > 0:
        assert all(len(v) == n for v in props.values())
