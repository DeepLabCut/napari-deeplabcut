# src/napari_deeplabcut/_tests/test_e2e.py
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points
from qtpy.QtWidgets import QMessageBox

# -----------------------------------------------------------------------------
# Global fixtures: avoid modal hangs + control overwrite confirmation path
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _auto_accept_qmessagebox(monkeypatch):
    """Prevent any QMessageBox modal dialogs from blocking tests."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)


@pytest.fixture(autouse=True)
def overwrite_confirm(monkeypatch):
    """
    Control the overwrite-confirmation path used by the writer.

    NOTE: write_hdf imports _maybe_confirm_overwrite directly:
        from napari_deeplabcut.ui.dialogs import _maybe_confirm_overwrite
    so we must patch napari_deeplabcut._writer._maybe_confirm_overwrite.

    API:
      - forbid(): fail test if confirmation is requested
      - cancel(): return False (simulate user cancel)
      - capture(): record calls and return True
      - set_result(bool): return chosen bool
      - reset_calls(): clear recorded calls
    """
    calls = []
    state = {"mode": "always_true", "result": True}

    def _patched_maybe_confirm_overwrite(metadata, key_conflict):
        # record minimal info about the call
        calls.append(
            {
                "metadata_keys": sorted(list(metadata.keys())) if isinstance(metadata, dict) else None,
                "n_pairs": int(key_conflict.to_numpy().sum()) if hasattr(key_conflict, "to_numpy") else None,
                "n_images": int(key_conflict.any(axis=1).to_numpy().sum()) if hasattr(key_conflict, "any") else None,
            }
        )
        n_pairs = int(key_conflict.to_numpy().sum()) if hasattr(key_conflict, "to_numpy") else 0
        # In "forbid" mode: allow calls that represent "no conflict" (n_pairs == 0),
        # but fail if we would actually overwrite something.
        if state["mode"] == "forbid" and n_pairs > 0:
            raise AssertionError("_maybe_confirm_overwrite was called unexpectedly for a real overwrite (n_pairs>0).")

        return state["result"]

    # Patch at import site used by write_hdf
    import napari_deeplabcut._writer as writer_mod

    monkeypatch.setattr(writer_mod, "_maybe_confirm_overwrite", _patched_maybe_confirm_overwrite)

    class Controller:
        @property
        def calls(self):
            return calls

        def forbid(self):
            state["mode"] = "forbid"
            state["result"] = True
            return self

        def cancel(self):
            state["mode"] = "capture"
            state["result"] = False
            return self

        def capture(self):
            state["mode"] = "capture"
            state["result"] = True
            return self

        def set_result(self, value: bool):
            state["mode"] = "capture"
            state["result"] = bool(value)
            return self

        def reset_calls(self):
            calls.clear()
            return self

    return Controller()


# -----------------------------------------------------------------------------
# Helpers: minimal DLC project + stable assertions on written H5 content
# -----------------------------------------------------------------------------


def _ensure_last_point_has_label(points_layer: Points, label: str, id_: str = "", likelihood: float = 1.0):
    """Force properties for the last added point so writer can map it to the correct keypoint."""
    props = points_layer.properties.copy()
    n = len(points_layer.data)
    # Ensure arrays exist and have the right length
    for key, default in (("label", label), ("id", id_), ("likelihood", likelihood)):
        arr = np.asarray(props.get(key, []))
        if arr.size < n:
            # pad to length n with default
            pad = np.array([default] * (n - arr.size), dtype=object if key in ("label", "id") else float)
            arr = np.concatenate([arr.astype(pad.dtype, copy=False), pad]) if arr.size else pad
        arr[-1] = default
        props[key] = arr
    points_layer.properties = props


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

    # Image
    img_rel = ("labeled-data", "test", "img000.png")
    img_path = project / Path(*img_rel)
    _write_minimal_png(img_path)

    # config.yaml
    cfg = {
        "scorer": "John",
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "viridis",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Existing GT H5
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


def _get_points_layer_with_data(viewer) -> Points:
    """Return the first Points layer with actual data; fallback to first Points layer."""
    pts = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts, "Expected at least one Points layer in viewer."
    return next((ly for ly in pts if ly.data is not None and np.any(ly.data)), pts[0])


def _index_mask_for_img(df: pd.DataFrame, basename: str) -> np.ndarray:
    """Return boolean mask selecting rows that correspond to a given image basename."""
    if isinstance(df.index, pd.MultiIndex):
        # rows are tuples like ('labeled-data','test','img000.png')
        return np.array([str(Path(*t)).endswith(basename) for t in df.index])
    else:
        return df.index.astype(str).str.endswith(basename).to_numpy()


def _get_coord_from_df(df: pd.DataFrame, bodypart: str, coord: str, basename: str = "img000.png") -> float:
    """Extract the single value for (bodypart, coord) in the row matching basename."""
    series = df.xs((bodypart, coord), axis=1, level=["bodyparts", "coords"])
    mask = _index_mask_for_img(series, basename)
    assert mask.any(), f"Could not find row for {basename} in saved dataframe index: {df.index!r}"
    return float(series.loc[series.index[mask]].iloc[0, 0])


def _set_bodypart_xy(points_layer: Points, bodypart: str, *, x: float, y: float):
    """
    Modify an existing bodypart point in the napari Points layer (conflict generator).
    """
    labels = np.asarray(points_layer.properties.get("label"))
    mask = labels == bodypart
    assert mask.any(), f"Could not find {bodypart} in Points layer properties."

    data = np.array(points_layer.data, copy=True)  # (frame, y, x)
    data[mask, 1] = y
    data[mask, 2] = x
    points_layer.data = data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("qtbot")
def test_config_first_hazard_regression_no_silent_deletion(make_napari_viewer, qtbot, tmp_path, caplog):
    """
    Regression for the original report:
    Save the WRONG (placeholder) layer and still preserve previous labels due to merge-on-save.
    """
    logging.getLogger("napari_deeplabcut.tests.overwrite")
    caplog.set_level(logging.DEBUG, logger="napari_deeplabcut.tests.overwrite")

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    pre = pd.read_hdf(h5_path, key="keypoints")
    assert np.isfinite(_get_coord_from_df(pre, "bodypart1", "x"))
    assert np.isnan(_get_coord_from_df(pre, "bodypart2", "x"))

    viewer = make_napari_viewer()
    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open config first -> placeholder Points layer exists
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 1, timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # Open folder -> images + GT points layer
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    assert placeholder in viewer.layers
    store = controls._stores.get(placeholder)
    assert store is not None

    # Add a new bodypart2 point *to placeholder* using 2D coords (y,x); store injects frame index
    store.current_keypoint = keypoints.Keypoint("bodypart2", "")
    placeholder.add(np.array([0.0, 33.0, 44.0], dtype=float))

    viewer.layers.selection.active = placeholder
    viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(100)

    post = pd.read_hdf(h5_path, key="keypoints")
    b1x_post = _get_coord_from_df(post, "bodypart1", "x")
    b2x_post = _get_coord_from_df(post, "bodypart2", "x")

    assert np.isfinite(b1x_post), "bodypart1 must be preserved (no silent deletion)."
    assert np.isfinite(b2x_post), "bodypart2 must be saved."
    assert b2x_post == 44.0


@pytest.mark.usefixtures("qtbot")
def test_no_overwrite_warning_when_only_filling_nans(make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm):
    """
    Adding new labels (filling NaNs) must not prompt overwrite confirmation.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open folder only -> real points layer loaded
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    store = controls._stores.get(points)
    assert store is not None

    # bodypart2 exists as a placeholder row with NaN coords in the loaded Points layer
    _set_bodypart_xy(points, "bodypart2", x=44.0, y=33.0)

    viewer.layers.selection.active = points
    viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(100)

    post = pd.read_hdf(h5_path, key="keypoints")
    assert np.isfinite(_get_coord_from_df(post, "bodypart1", "x"))
    assert np.isfinite(_get_coord_from_df(post, "bodypart2", "x"))


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_triggers_on_conflict(make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm):
    """
    Modifying an existing non-NaN label must trigger overwrite confirmation.
    """
    overwrite_confirm.capture().reset_calls()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)

    # Create conflict: overwrite bodypart1
    _set_bodypart_xy(points, "bodypart1", x=99.0, y=88.0)

    viewer.layers.selection.active = points
    viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(100)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."
    assert overwrite_confirm.calls[0]["n_pairs"] is not None
    assert overwrite_confirm.calls[0]["n_pairs"] >= 1

    post = pd.read_hdf(h5_path, key="keypoints")
    assert _get_coord_from_df(post, "bodypart1", "x") == 99.0


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_cancel_aborts_write(make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm):
    """
    If overwrite confirmation is requested and user cancels, file must remain unchanged.
    """
    overwrite_confirm.cancel().reset_calls()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    pre = pd.read_hdf(h5_path, key="keypoints")
    b1x_pre = _get_coord_from_df(pre, "bodypart1", "x")
    b1y_pre = _get_coord_from_df(pre, "bodypart1", "y")

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    _set_bodypart_xy(points, "bodypart1", x=456.0, y=123.0)

    viewer.layers.selection.active = points
    try:
        viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    except Exception:
        # Some napari/npe2 versions may raise when writer aborts; file integrity is what matters.
        pass

    qtbot.wait(100)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."

    post = pd.read_hdf(h5_path, key="keypoints")
    assert _get_coord_from_df(post, "bodypart1", "x") == b1x_pre
    assert _get_coord_from_df(post, "bodypart1", "y") == b1y_pre
