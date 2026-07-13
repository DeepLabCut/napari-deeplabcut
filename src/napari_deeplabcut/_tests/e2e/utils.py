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
from napari_deeplabcut.core.keypoints import build_color_cycles
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
    df0.to_hdf(h5_path, key="df_with_missing", mode="w")
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
    df0.to_hdf(folder / "machinelabels-iter0.h5", key="df_with_missing", mode="w")
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
    df.to_hdf(path, key="df_with_missing", mode="w")
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
    return pd.read_hdf(path, key="df_with_missing")


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

    from napari_deeplabcut.core import keypoints

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


def _row_key_to_posix(row_key) -> str:
    """Normalize a DLC dataframe row key to a POSIX-style string."""
    if isinstance(row_key, tuple):
        return "/".join(
            str(part).replace("\\", "/").strip("/") for part in row_key if part is not None and str(part) != ""
        )

    return str(row_key).replace("\\", "/").strip("/")


def _dataframe_rows_by_path(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Map normalized DLC row paths to their original pandas index values.

    The original values are retained so callers can safely use them with
    df.loc regardless of whether the dataframe has an Index or MultiIndex.
    """
    rows: dict[str, object] = {}

    for row_key in df.index:
        normalized = _row_key_to_posix(row_key)

        if normalized in rows:
            raise AssertionError(f"Duplicate normalized DLC row key: {normalized}")

        rows[normalized] = row_key

    return rows


def _seed_gt_and_machine_outlier_dataset(
    labeled_folder: Path,
    *,
    scorer: str = "John",
    model_scorer: str = "DLC_model",
    bodypart: str = "bodypart1",
    n_initial_frames: int = 30,
    n_outlier_frames: int = 20,
) -> tuple[
    Path,
    Path,
    list[str],
    list[str],
    dict[str, tuple[float, float]],
]:
    """
    Seed the exact disk layout involved in machine-to-GT refinement.

    Creates
    -------
    CollectedData_<scorer>.h5
        Finite manual GT annotations on the first ``n_initial_frames``.

    machinelabels-iter0.h5
        Finite machine annotations on the following ``n_outlier_frames``.

    PNG files
        One shared image set containing both groups of frames.

    Returns
    -------
    gt_path
    machine_path
    initial_paths
        Canonical DLC row paths for the initial GT frames.
    outlier_paths
        Canonical DLC row paths for the machine frames.
    expected_outlier_xy
        Expected promoted coordinates keyed by canonical DLC row path.
    """
    if n_initial_frames <= 0:
        raise ValueError("n_initial_frames must be positive.")

    if n_outlier_frames <= 0:
        raise ValueError("n_outlier_frames must be positive.")

    total_frames = n_initial_frames + n_outlier_frames

    existing_images = sorted(labeled_folder.glob("*.png"))
    assert existing_images, f"The project helper must create at least one readable PNG in {labeled_folder}."

    # Reuse the bytes of an image created by the existing project fixture.
    # This gives us 50 valid image files without introducing an image-writing
    # dependency into this test.
    template_image_bytes = existing_images[0].read_bytes()

    for image in existing_images:
        image.unlink()

    image_names = [f"img{i:03d}.png" for i in range(total_frames)]

    for image_name in image_names:
        destination = labeled_folder / image_name
        destination.write_bytes(template_image_bytes)

    dataset_name = labeled_folder.name

    initial_paths = [f"labeled-data/{dataset_name}/{image_name}" for image_name in image_names[:n_initial_frames]]
    outlier_paths = [f"labeled-data/{dataset_name}/{image_name}" for image_name in image_names[n_initial_frames:]]

    # ------------------------------------------------------------------
    # Existing human ground truth: canonical single-animal DLC format.
    # ------------------------------------------------------------------

    gt_columns = pd.MultiIndex.from_product(
        [
            [scorer],
            [bodypart],
            ["x", "y"],
        ],
        names=[
            "scorer",
            "bodyparts",
            "coords",
        ],
    )

    gt_index = pd.MultiIndex.from_tuples([tuple(path.split("/")) for path in initial_paths])

    gt_values = np.empty(
        (n_initial_frames, len(gt_columns)),
        dtype=float,
    )

    for frame_index in range(n_initial_frames):
        gt_values[frame_index, 0] = 1000.0 + frame_index
        gt_values[frame_index, 1] = 2000.0 + frame_index

    gt_df = pd.DataFrame(
        gt_values,
        index=gt_index,
        columns=gt_columns,
    )

    gt_path = labeled_folder / f"CollectedData_{scorer}.h5"

    gt_df.to_hdf(
        gt_path,
        key="df_with_missing",
        mode="w",
    )
    gt_df.to_csv(gt_path.with_suffix(".csv"))

    # ------------------------------------------------------------------
    # Machine labels: x/y/likelihood on the 20 outlier frames.
    # ------------------------------------------------------------------

    machine_columns = pd.MultiIndex.from_product(
        [
            [model_scorer],
            [bodypart],
            ["x", "y", "likelihood"],
        ],
        names=[
            "scorer",
            "bodyparts",
            "coords",
        ],
    )

    machine_index = pd.MultiIndex.from_tuples([tuple(path.split("/")) for path in outlier_paths])

    machine_values = np.empty(
        (n_outlier_frames, len(machine_columns)),
        dtype=float,
    )

    expected_outlier_xy: dict[
        str,
        tuple[float, float],
    ] = {}

    for local_index, path in enumerate(outlier_paths):
        x = 5000.0 + local_index
        y = 6000.0 + local_index
        likelihood = 0.95

        machine_values[local_index, 0] = x
        machine_values[local_index, 1] = y
        machine_values[local_index, 2] = likelihood

        expected_outlier_xy[path] = (x, y)

    machine_df = pd.DataFrame(
        machine_values,
        index=machine_index,
        columns=machine_columns,
    )

    machine_path = labeled_folder / "machinelabels-iter0.h5"

    machine_df.to_hdf(
        machine_path,
        key="df_with_missing",
        mode="w",
    )

    return (
        gt_path,
        machine_path,
        initial_paths,
        outlier_paths,
        expected_outlier_xy,
    )
