from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.io import read_hdf_single

# -----------------------------
# Helpers to write minimal DLC-like H5 files
# -----------------------------


def _write_h5_single_animal(
    path: Path,
    *,
    scorer: str = "John",
    bodyparts=("bp1", "bp2"),
    values=None,  # [bp1x,bp1y,bp2x,bp2y] for one image
    index=("img000.png",),
):
    if values is None:
        values = [np.nan, np.nan, np.nan, np.nan]
    cols = pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame([values], index=list(index), columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key="keypoints", mode="w")
    return df


def _assert_layerdata_invariants(layerdata):
    data, meta, layer_type = layerdata
    assert layer_type == "points"
    arr = np.asarray(data)
    assert arr.ndim == 2 and arr.shape[1] == 3, "Points data must be (N,3): [frame,y,x] for napari"
    n = arr.shape[0]

    # Napari expects per-point properties lengths to match n
    props = meta.get("properties") or {}
    assert isinstance(props, dict)
    for k, v in props.items():
        assert len(v) == n, f"property '{k}' length {len(v)} != N points {n}"


# -----------------------------
# Tests
# -----------------------------


def test_read_hdf_single_all_nan_machine_is_empty_points_and_properties(tmp_path: Path):
    """
    Contract: empty Points layers are valid, but must have empty properties too.
    This test directly prevents the napari ValueError you saw in E2Es.
    """
    h5 = tmp_path / "machinelabels-iter0.h5"
    _write_h5_single_animal(h5, scorer="machine", values=[np.nan, np.nan, np.nan, np.nan])

    layers = read_hdf_single(h5, kind=AnnotationKind.MACHINE)
    assert len(layers) == 1
    data, meta, _ = layers[0]

    assert np.asarray(data).shape[0] == 0
    _assert_layerdata_invariants(layers[0])


def test_read_hdf_single_one_finite_point_produces_one_point_and_properties(tmp_path: Path):
    h5 = tmp_path / "CollectedData_John.h5"
    _write_h5_single_animal(h5, scorer="John", values=[10.0, 20.0, np.nan, np.nan])

    layers = read_hdf_single(h5, kind=AnnotationKind.GT)
    data, meta, _ = layers[0]

    assert np.asarray(data).shape[0] == 1
    _assert_layerdata_invariants(layers[0])

    # Ensure the point corresponds to bp1 (since bp2 is NaN)
    assert meta["properties"]["label"] == ["bp1"]


def test_read_hdf_single_filters_data_and_properties_consistently(tmp_path: Path):
    """
    Regression guard: if finite mask filters df, it must also filter data.
    This is the exact failure mode that triggers napari's 'length mismatch' error.
    """
    h5 = tmp_path / "CollectedData_John.h5"
    _write_h5_single_animal(h5, scorer="John", values=[10.0, 20.0, np.nan, np.nan])

    layers = read_hdf_single(h5)
    data, meta, _ = layers[0]

    n = np.asarray(data).shape[0]
    assert len(meta["properties"]["label"]) == n
    assert len(meta["properties"]["id"]) == n
    assert len(meta["properties"]["likelihood"]) == n


def test_read_hdf_single_accepts_3level_header_and_inserts_individuals(tmp_path: Path):
    """
    Reader must accept classic 3-level single-animal DLC tables and normalize internally.
    """
    h5 = tmp_path / "CollectedData_John.h5"
    _write_h5_single_animal(h5, scorer="John", values=[10.0, 20.0, np.nan, np.nan])

    layers = read_hdf_single(h5)
    _, meta, _ = layers[0]

    # 'id' must exist and match data length; for single animal, values are expected to be empty string
    assert "id" in meta["properties"]
    assert meta["properties"]["id"] == [""]


def test_read_hdf_single_metadata_contains_root_and_name(tmp_path: Path):
    """
    Reader contract used by multiple E2E tests: metadata.root/name should be present.
    """
    h5 = tmp_path / "CollectedData_John.h5"
    _write_h5_single_animal(h5, scorer="John", values=[10.0, 20.0, np.nan, np.nan])

    layers = read_hdf_single(h5)
    _, meta, _ = layers[0]
    assert meta["name"] == "CollectedData_John"
    assert meta["metadata"]["root"] == str(h5.parent)
    assert meta["metadata"]["name"] == "CollectedData_John"
