# src/napari_deeplabcut/_tests/e2e/_helpers.py
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.config.settings import (
    DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP,
    DEFAULT_SINGLE_ANIMAL_CMAP,
)
from napari_deeplabcut.keypoints import build_color_cycles
from napari_deeplabcut.ui.color_scheme_display import _to_hex


def file_sig(p: Path):
    b = p.read_bytes()
    return {
        "mtime": os.path.getmtime(p),
        "size": len(b),
        "sha256": hashlib.sha256(b).hexdigest()[:16],
    }


def _write_minimal_png(path: Path, *, shape=(64, 64, 3)) -> None:
    """Write a tiny RGB image to satisfy the folder reader."""
    from skimage.io import imsave

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros(shape, dtype=np.uint8)
    img[8:24, 8:24, 0] = 255
    imsave(str(path), img, check_contrast=False)


def _make_minimal_dlc_project(tmp_path: Path):
    """
    Build a minimal DLC-like folder:
      project/
        config.yaml
        labeled-data/test/img000.png
        labeled-data/test/CollectedData_John.h5 (bodypart1 labeled, bodypart2 NaN)
    """
    import yaml

    project = tmp_path / "project"
    labeled = project / "labeled-data" / "test"
    labeled.mkdir(parents=True, exist_ok=True)

    img_rel = ("labeled-data", "test", "img000.png")
    img_path = project / Path(*img_rel)
    _write_minimal_png(img_path)

    cfg = {
        "scorer": "John",
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "viridis",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cols = pd.MultiIndex.from_product(
        [["John"], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    idx = pd.MultiIndex.from_tuples([img_rel])
    df0 = pd.DataFrame([[10.0, 20.0, np.nan, np.nan]], index=idx, columns=cols)

    h5_path = labeled / "CollectedData_John.h5"
    df0.to_hdf(h5_path, key="keypoints", mode="w")
    df0.to_csv(str(h5_path).replace(".h5", ".csv"))

    return project, config_path, labeled, h5_path


def _make_labeled_folder_with_machine_only(tmp_path: Path) -> Path:
    """
    Folder contains:
      - images
      - machinelabels-iter0.h5 (no CollectedData*, no config.yaml)
    """
    folder = tmp_path / "shared" / "labeled-data" / "test"
    folder.mkdir(parents=True, exist_ok=True)

    _write_minimal_png(folder / "img000.png")

    cols = pd.MultiIndex.from_product(
        [["machine"], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df0 = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], index=["img000.png"], columns=cols)
    (folder / "machinelabels-iter0.h5").unlink(missing_ok=True)
    df0.to_hdf(folder / "machinelabels-iter0.h5", key="keypoints", mode="w")
    df0.to_csv(str(folder / "machinelabels-iter0.csv"))

    return folder


def _write_keypoints_h5(
    path: Path,
    *,
    scorer: str,
    img_rel: tuple[str, ...],
    bodyparts=("bodypart1", "bodypart2"),
    values=None,
) -> Path:
    """
    Write a single-row DLC keypoints H5 in the same format used by _make_minimal_dlc_project.
    `values` should be [b1x, b1y, b2x, b2y] where some can be NaN.
    """
    if values is None:
        values = [10.0, 20.0, np.nan, np.nan]

    cols = pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    idx = pd.MultiIndex.from_tuples([img_rel])
    df = pd.DataFrame([values], index=idx, columns=cols)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key="keypoints", mode="w")
    df.to_csv(str(path).replace(".h5", ".csv"))
    return path


def _make_dlc_project_with_multiple_gt(
    tmp_path: Path,
    *,
    scorers=("John", "Jane"),
    with_machine: bool = False,
):
    """
    Build a minimal DLC-like labeled-data folder with multiple GT files.
    """
    import yaml

    project = tmp_path / "project"
    labeled = project / "labeled-data" / "test"
    labeled.mkdir(parents=True, exist_ok=True)

    img_rel = ("labeled-data", "test", "img000.png")
    img_path = project / Path(*img_rel)
    _write_minimal_png(img_path)

    cfg = {
        "scorer": scorers[0],
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "viridis",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    gt_paths = []
    base = 10.0
    for i, scorer in enumerate(scorers):
        vals = [base + i * 100.0, base + i * 100.0 + 10.0, np.nan, np.nan]
        gt_path = labeled / f"CollectedData_{scorer}.h5"
        _write_keypoints_h5(gt_path, scorer=scorer, img_rel=img_rel, values=vals)
        gt_paths.append(gt_path)

    machine_path = None
    if with_machine:
        machine_path = labeled / "machinelabels-iter0.h5"
        _write_keypoints_h5(
            machine_path,
            scorer="machine",
            img_rel=img_rel,
            values=[np.nan, np.nan, np.nan, np.nan],
        )

    return project, config_path, labeled, gt_paths, machine_path


def _make_project_config_and_frames_no_gt(tmp_path: Path):
    """
    Project with:
      project/config.yaml
      project/labeled-data/test/img000.png
    No CollectedData*.h5 initially.
    """
    import yaml

    project = tmp_path / "project"
    labeled = project / "labeled-data" / "test"
    labeled.mkdir(parents=True, exist_ok=True)

    img_rel = ("labeled-data", "test", "img000.png")
    _write_minimal_png(project / Path(*img_rel))

    cfg = {
        "scorer": "John",
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "magma",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return project, config_path, labeled


def _read_h5_keypoints(path: Path) -> pd.DataFrame:
    return pd.read_hdf(path, key="keypoints")


def _index_mask_for_img(df: pd.DataFrame, basename: str) -> np.ndarray:
    """Return boolean mask selecting rows that correspond to a given image basename."""
    if isinstance(df.index, pd.MultiIndex):
        return np.array([str(Path(*t)).endswith(basename) for t in df.index])
    return df.index.astype(str).str.endswith(basename).to_numpy()


def _get_coord_from_df(df: pd.DataFrame, bodypart: str, coord: str, basename: str = "img000.png") -> float:
    """Extract the single value for (bodypart, coord) in the row matching basename."""
    series = df.xs((bodypart, coord), axis=1, level=["bodyparts", "coords"])
    mask = _index_mask_for_img(series, basename)
    assert mask.any(), f"Could not find row for {basename} in saved dataframe index: {df.index!r}"
    return float(series.loc[series.index[mask]].iloc[0, 0])


def _snapshot_coords(path: Path) -> dict[str, float]:
    df = _read_h5_keypoints(path)
    return {
        "b1x": _get_coord_from_df(df, "bodypart1", "x"),
        "b1y": _get_coord_from_df(df, "bodypart1", "y"),
        "b2x": _get_coord_from_df(df, "bodypart2", "x"),
        "b2y": _get_coord_from_df(df, "bodypart2", "y"),
    }


def sig_equal(a: dict, b: dict) -> bool:
    """NaN-stable signature equality for test signatures."""
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
        if va != vb:
            return False
    return True


def assert_only_these_changed_nan_safe(before: dict[Path, dict], after: dict[Path, dict], changed: set[Path]):
    for p in before:
        if p in changed:
            assert not sig_equal(before[p], after[p]), f"Expected {p.name} to change, but signature did not."
        else:
            assert sig_equal(before[p], after[p]), f"Expected {p.name} NOT to change, but signature changed."


def _assert_only_these_files_changed(before: dict[Path, dict], after: dict[Path, dict], changed: set[Path]):
    return assert_only_these_changed_nan_safe(before, after, changed)


def _get_points_layer_with_data(viewer) -> Points:
    """Return the first Points layer with actual data; fallback to first Points layer."""
    pts = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts, "Expected at least one Points layer in viewer."
    return next((ly for ly in pts if ly.data is not None and np.isfinite(np.asarray(ly.data)[:, 1:3]).any()), pts[0])


def _set_or_add_bodypart_xy(points_layer: Points, store, bodypart: str, *, x: float, y: float, frame: int = 0):
    """
    Cross-version helper:
    - If the bodypart already exists as a row (possibly NaN placeholder), update it.
    - Otherwise, add a new point for that bodypart via the store/Points.add.
    """
    labels = np.asarray(points_layer.properties.get("label", []), dtype=object)
    mask = labels == bodypart

    if mask.any():
        data = np.array(points_layer.data, copy=True)  # (frame, y, x)
        data[mask, 1] = y
        data[mask, 2] = x
        points_layer.data = data
        return

    from napari_deeplabcut import keypoints

    store.current_keypoint = keypoints.Keypoint(bodypart, "")
    points_layer.add(np.array([float(frame), float(y), float(x)], dtype=float))


def _header_model_from_layer(layer) -> DLCHeaderModel:
    hdr = layer.metadata.get("header")
    if hdr is None:
        raise AssertionError("Expected header in layer metadata")
    return hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)


def _is_multianimal_header(header: DLCHeaderModel) -> bool:
    inds = list(getattr(header, "individuals", []) or [])
    return bool(inds and str(inds[0]) != "")


def _config_colormap_from_layer(layer) -> str:
    md = layer.metadata or {}
    cmap = md.get("config_colormap")
    if isinstance(cmap, str) and cmap:
        return cmap
    return DEFAULT_SINGLE_ANIMAL_CMAP


def _cycles_from_policy(layer) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute expected cycles from the new centralized color policy.

    Source of truth:
    - layer header
    - metadata['config_colormap']
    - multi-animal id coloring uses DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP
    """
    header = _header_model_from_layer(layer)
    config_cmap = _config_colormap_from_layer(layer)

    config_cycles = build_color_cycles(header, config_cmap) or {}

    if _is_multianimal_header(header):
        individual_cycles = build_color_cycles(header, DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP) or {}
    else:
        individual_cycles = config_cycles

    return {
        "label": config_cycles.get("label", {}),
        "id": individual_cycles.get("id", {}),
    }


def _scheme_from_policy(layer, prop: str, names: list[str]) -> dict[str, str]:
    cycles = _cycles_from_policy(layer)
    mapping = cycles.get(prop, {})
    return {name: _to_hex(mapping[name]) for name in names if name in mapping}
