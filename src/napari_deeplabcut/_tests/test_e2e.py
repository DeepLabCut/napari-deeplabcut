# src/napari_deeplabcut/_tests/test_e2e.py
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points
from qtpy.QtWidgets import QInputDialog, QMessageBox

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.errors import UnresolvablePathError

# -----------------------------------------------------------------------------
# Fixtures: avoid modal hangs + control overwrite confirmation path
# -----------------------------------------------------------------------------
# TODO @C-Achard 2026-02-17: Many of these can be moved to conftest
# as they are useful for multiple test modules, and some can be made more generic.


@pytest.fixture(autouse=True)
def _auto_accept_qmessagebox(monkeypatch):
    """Prevent any QMessageBox modal dialogs from blocking tests."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)


@pytest.fixture
def inputdialog(monkeypatch):
    """
    Controller for QInputDialog.getText used by promotion-to-GT first save prompt.
    """
    state = {"value": "Alice", "ok": True, "calls": 0, "forbid": False}

    def _fake_getText(*args, **kwargs):
        state["calls"] += 1
        if state["forbid"]:
            raise AssertionError("QInputDialog.getText was called but forbid=True")
        return state["value"], state["ok"]

    monkeypatch.setattr(QInputDialog, "getText", _fake_getText)

    class Controller:
        @property
        def calls(self):
            return state["calls"]

        def set(self, value: str, ok: bool = True):
            state["value"] = value
            state["ok"] = ok
            return self

        def forbid(self):
            state["forbid"] = True
            return self

    return Controller()


@pytest.fixture(autouse=True)
def overwrite_confirm(monkeypatch):
    """
    Control the overwrite-confirmation path used by the writer.

    NOTE: the io module imports/uses maybe_confirm_overwrite at module scope,
    so we patch napari_deeplabcut.core.io.maybe_confirm_overwrite.

    API:
      - forbid(): fail test if confirmation is requested for a real overwrite
      - cancel(): return False (simulate user cancel)
      - capture(): record calls and return True
      - set_result(bool): return chosen bool
      - reset_calls(): clear recorded calls
    """
    calls = []
    state = {"mode": "always_true", "result": True}

    def _metadata_keys(meta: Any) -> list[str] | None:
        """Support dict metadata or pydantic models with model_dump()."""
        if meta is None:
            return None
        if isinstance(meta, dict):
            return sorted(list(meta.keys()))
        dump = getattr(meta, "model_dump", None)
        if callable(dump):
            try:
                d = dump()
                if isinstance(d, dict):
                    return sorted(list(d.keys()))
            except Exception:
                return None
        return None

    def _patched_maybe_confirm_overwrite(metadata, key_conflict):
        # record minimal info about the call
        n_pairs = int(key_conflict.to_numpy().sum()) if hasattr(key_conflict, "to_numpy") else 0
        n_images = int(key_conflict.any(axis=1).to_numpy().sum()) if hasattr(key_conflict, "any") else None

        calls.append(
            {
                "metadata_keys": _metadata_keys(metadata),
                "n_pairs": n_pairs,
                "n_images": n_images,
            }
        )

        # In "forbid" mode: allow calls only when there is no actual overwrite.
        if state["mode"] == "forbid" and n_pairs > 0:
            raise AssertionError("maybe_confirm_overwrite was called unexpectedly for a real overwrite (n_pairs > 0).")

        return state["result"]

    import napari_deeplabcut.core.io as io_mod

    monkeypatch.setattr(io_mod, "maybe_confirm_overwrite", _patched_maybe_confirm_overwrite)

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


def _make_labeled_folder_with_machine_only(tmp_path: Path) -> Path:
    """
    Folder contains:
      - images
      - machinelabels-iter0.h5 (no CollectedData*, no config.yaml)
    """
    folder = tmp_path / "shared" / "labeled-data" / "test"
    folder.mkdir(parents=True, exist_ok=True)

    # image
    _write_minimal_png(folder / "img000.png")

    # machine h5 (minimal DLC-like structure)
    cols = pd.MultiIndex.from_product(
        [["machine"], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df0 = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], index=["img000.png"], columns=cols)
    (folder / "machinelabels-iter0.h5").unlink(missing_ok=True)
    df0.to_hdf(folder / "machinelabels-iter0.h5", key="keypoints", mode="w")
    df0.to_csv(str(folder / "machinelabels-iter0.csv"))

    return folder


# -----------------------------------------------------------------------------
# Helpers: multi-file DLC folders (multiple GT + optional machine file)
# -----------------------------------------------------------------------------


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

      project/
        config.yaml
        labeled-data/test/img000.png
        labeled-data/test/CollectedData_<scorer1>.h5
        labeled-data/test/CollectedData_<scorer2>.h5
        (optional) labeled-data/test/machinelabels-iter0.h5

    Returns:
      project, config_path, labeled_folder, gt_paths(list), machine_path(optional)
    """
    import yaml

    project = tmp_path / "project"
    labeled = project / "labeled-data" / "test"
    labeled.mkdir(parents=True, exist_ok=True)

    img_rel = ("labeled-data", "test", "img000.png")
    img_path = project / Path(*img_rel)
    _write_minimal_png(img_path)

    cfg = {
        "scorer": scorers[0],  # config scorer — not necessarily unique in folder
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "viridis",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    gt_paths = []
    # Make GT files distinct so we can tell which was modified
    # Each scorer gets a different bodypart1 x,y and NaNs for bodypart2.
    base = 10.0
    for i, scorer in enumerate(scorers):
        vals = [base + i * 100.0, base + i * 100.0 + 10.0, np.nan, np.nan]  # b1x, b1y, b2x, b2y
        gt_path = labeled / f"CollectedData_{scorer}.h5"
        _write_keypoints_h5(gt_path, scorer=scorer, img_rel=img_rel, values=vals)
        gt_paths.append(gt_path)

    machine_path = None
    if with_machine:
        # Machine file should look like a normal keypoints table for now.
        # We'll edit bodypart2 there and ensure it doesn't touch GT.
        machine_path = labeled / "machinelabels-iter0.h5"
        _write_keypoints_h5(
            machine_path,
            scorer="machine",
            img_rel=img_rel,
            values=[np.nan, np.nan, np.nan, np.nan],
        )

    return project, config_path, labeled, gt_paths, machine_path


def _read_h5_keypoints(path: Path) -> pd.DataFrame:
    return pd.read_hdf(path, key="keypoints")


def _snapshot_coords(path: Path) -> dict[str, float]:
    """
    Small “signature” of a keypoints file for stable comparisons:
    b1x,b1y,b2x,b2y for img000.png.
    """
    df = _read_h5_keypoints(path)
    return {
        "b1x": _get_coord_from_df(df, "bodypart1", "x"),
        "b1y": _get_coord_from_df(df, "bodypart1", "y"),
        "b2x": _get_coord_from_df(df, "bodypart2", "x"),
        "b2y": _get_coord_from_df(df, "bodypart2", "y"),
    }


def _assert_only_these_files_changed(before: dict[Path, dict], after: dict[Path, dict], changed: set[Path]):
    """
    Assert that only the files in `changed` have different signatures.
    """
    for p in before:
        if p in changed:
            assert before[p] != after[p], f"Expected {p.name} to change, but signature did not."
        else:
            assert before[p] == after[p], f"Expected {p.name} NOT to change, but signature changed."


def _get_points_layer_with_data(viewer) -> Points:
    """Return the first Points layer with actual data; fallback to first Points layer."""
    pts = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts, "Expected at least one Points layer in viewer."
    return next((ly for ly in pts if ly.data is not None and np.any(ly.data)), pts[0])


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


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("qtbot")
def test_config_first_hazard_regression_no_silent_deletion(make_napari_viewer, qtbot, tmp_path, caplog):
    """
    Regression for the original report:
    Save the WRONG (placeholder) layer and still preserve previous labels due to merge-on-save.
    """
    caplog.set_level(logging.DEBUG)

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
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 1, timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # Open folder -> images + GT points layer
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    # Placeholder should still be present for this regression to apply
    assert placeholder in viewer.layers

    store = controls._stores.get(placeholder)
    assert store is not None

    # Add a new bodypart2 point to placeholder using (frame, y, x)
    store.current_keypoint = keypoints.Keypoint("bodypart2", "")
    placeholder.add(np.array([0.0, 33.0, 44.0], dtype=float))

    viewer.layers.selection.active = placeholder
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(200)

    post = pd.read_hdf(h5_path, key="keypoints")
    b1x_post = _get_coord_from_df(post, "bodypart1", "x")
    b2x_post = _get_coord_from_df(post, "bodypart2", "x")

    assert np.isfinite(b1x_post), "bodypart1 must be preserved (no silent deletion)."
    assert np.isfinite(b2x_post), "bodypart2 must be saved."
    assert b2x_post == 44.0


@pytest.mark.usefixtures("qtbot")
def test_no_overwrite_warning_when_only_filling_nans(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Adding new labels (filling NaNs) must not prompt overwrite confirmation.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    store = controls._stores.get(points)
    assert store is not None

    # Fill NaNs for bodypart2
    _set_or_add_bodypart_xy(points, store, "bodypart2", x=44.0, y=33.0)

    viewer.layers.selection.active = points
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(200)

    post = pd.read_hdf(h5_path, key="keypoints")
    assert np.isfinite(_get_coord_from_df(post, "bodypart1", "x"))
    assert np.isfinite(_get_coord_from_df(post, "bodypart2", "x"))


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_triggers_on_conflict(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
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
    store = controls._stores.get(points)
    assert store is not None

    # Create conflict: overwrite bodypart1 from (10,20) -> (99,88)
    _set_or_add_bodypart_xy(points, store, "bodypart1", x=99.0, y=88.0)

    viewer.layers.selection.active = points
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(200)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."
    assert overwrite_confirm.calls[0]["n_pairs"] is not None
    assert overwrite_confirm.calls[0]["n_pairs"] >= 1

    post = pd.read_hdf(h5_path, key="keypoints")
    assert _get_coord_from_df(post, "bodypart1", "x") == 99.0


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_cancel_aborts_write(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
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
    store = controls._stores.get(points)
    assert store is not None

    _set_or_add_bodypart_xy(points, store, "bodypart1", x=456.0, y=123.0)

    viewer.layers.selection.active = points
    try:
        viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    except Exception:
        # Some napari/npe2 versions may raise when writer aborts; file integrity is what matters.
        pass

    qtbot.wait(200)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."

    post = pd.read_hdf(h5_path, key="keypoints")
    assert _get_coord_from_df(post, "bodypart1", "x") == b1x_pre
    assert _get_coord_from_df(post, "bodypart1", "y") == b1y_pre


@pytest.mark.usefixtures("qtbot")
def test_save_routes_to_correct_gt_when_multiple_gt_exist(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: Saving a Points layer must write back ONLY to the file it came from.
    No 'first CollectedData*.h5' selection when multiple exist.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, _ = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=False
    )
    gt_a, gt_b = gt_paths

    before = {p: _snapshot_coords(p) for p in gt_paths}

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open both GT files explicitly so we get two Points layers
    viewer.open(str(gt_a), plugin="napari-deeplabcut")
    viewer.open(str(gt_b), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 2, timeout=10_000)
    qtbot.wait(200)

    # Select the layer corresponding to gt_b
    points_b = next((ly for ly in viewer.layers if isinstance(ly, Points) and ly.name == gt_b.stem), None)
    assert points_b is not None, f"Expected a Points layer named {gt_b.stem}"

    store_b = controls._stores.get(points_b)
    assert store_b is not None

    # Fill NaNs for bodypart2 in B only (no overwrite dialog)
    _set_or_add_bodypart_xy(points_b, store_b, "bodypart2", x=77.0, y=66.0)

    viewer.layers.selection.active = points_b
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(200)

    after = {p: _snapshot_coords(p) for p in gt_paths}

    _assert_only_these_files_changed(before, after, changed={gt_b})
    assert after[gt_b]["b2x"] == 77.0


@pytest.mark.usefixtures("qtbot")
def test_machine_layer_does_not_modify_gt_on_save(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: machine outputs must never save to their own file.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, machine_path = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=True
    )
    assert machine_path is not None

    before = {p: _snapshot_coords(p) for p in gt_paths + [machine_path]}

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(machine_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 1, timeout=10_000)
    qtbot.wait(200)

    machine_layer = next((ly for ly in viewer.layers if isinstance(ly, Points) and ly.name == machine_path.stem), None)
    assert machine_layer is not None

    store = controls._stores.get(machine_layer)
    assert store is not None

    # Fill NaNs in machine file (no overwrite prompt)
    _set_or_add_bodypart_xy(machine_layer, store, "bodypart2", x=55.0, y=44.0)

    viewer.layers.selection.active = machine_layer

    # FIXME exception type
    with pytest.raises(UnresolvablePathError):
        viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")

    qtbot.wait(200)

    after = {p: _snapshot_coords(p) for p in gt_paths + [machine_path]}

    # Only machine file should change
    _assert_only_these_files_changed(before, after, changed=set())
    # assert after[machine_path]["b2x"] == 55.0


@pytest.mark.usefixtures("qtbot")
def test_layer_rename_does_not_change_save_target(make_napari_viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: layer renaming must not redirect output or create new file.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, _ = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=False
    )
    gt_a = gt_paths[0]

    before = {p: _snapshot_coords(p) for p in gt_paths}

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(gt_a), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 1, timeout=10_000)
    qtbot.wait(200)

    layer = next((ly for ly in viewer.layers if isinstance(ly, Points) and ly.name == gt_a.stem), None)
    assert layer is not None
    store = controls._stores.get(layer)
    assert store is not None

    # Rename in UI
    layer.name = "foo"

    # Fill NaNs so no overwrite dialog
    _set_or_add_bodypart_xy(layer, store, "bodypart2", x=12.0, y=34.0)

    viewer.layers.selection.active = layer
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(200)

    # Must not create foo.h5 in the folder
    assert not (gt_a.parent / "foo.h5").exists(), "Renaming must not create foo.h5"

    after = {p: _snapshot_coords(p) for p in gt_paths}
    _assert_only_these_files_changed(before, after, changed={gt_a})


@pytest.mark.usefixtures("qtbot")
def test_ambiguous_placeholder_save_aborts_when_multiple_gt_exist(
    make_napari_viewer, qtbot, tmp_path, overwrite_confirm
):
    """
    Contract: If provenance is missing and multiple candidate GT files exist,
    save must refuse (deterministic) rather than silently choosing.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, _ = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=False
    )

    before = {p: _snapshot_coords(p) for p in gt_paths}

    viewer = make_napari_viewer()
    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open config first => placeholder points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 1, timeout=5_000)
    qtbot.wait(200)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None

    # Ensure it's a placeholder (no actual data)
    assert placeholder.data is None or len(placeholder.data) == 0

    # Open labeled folder (images) so root/paths are present for saving attempt
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.wait(200)

    store = controls._stores.get(placeholder)
    assert store is not None

    # Add a point to placeholder
    store.current_keypoint = keypoints.Keypoint("bodypart2", "")
    placeholder.add(np.array([0.0, 33.0, 44.0], dtype=float))

    viewer.layers.selection.active = placeholder

    # Expect save to abort deterministically
    try:
        viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    except Exception:
        pass  # acceptable in headless/test mode

    qtbot.wait(200)

    after = {p: _snapshot_coords(p) for p in gt_paths}
    _assert_only_these_files_changed(before, after, changed=set())


@pytest.mark.usefixtures("qtbot")
def test_folder_open_loads_all_h5_when_multiple_exist(make_napari_viewer, qtbot, tmp_path):
    """
    Contract: Opening a labeled-data folder with multiple H5 files should not
    silently pick the first one. Preferred policy: load all as separate Points layers.
    """
    project, config_path, labeled_folder, gt_paths, machine_path = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=True
    )

    viewer = make_napari_viewer()

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)  # images + points at least
    qtbot.wait(200)

    pts = [ly for ly in viewer.layers if isinstance(ly, Points)]
    # Expected: one points layer per H5 file (2 GT + 1 machine)
    assert len(pts) == 3, f"Expected 3 Points layers (2 GT + 1 machine), got {len(pts)}: {[p.name for p in pts]}"
    # ------------------------------------------------------------------
    # New assertion: each Points layer must carry authoritative source_h5
    # matching the file it originated from (stable across layer renames).
    # ------------------------------------------------------------------
    all_expected = list(gt_paths) + ([machine_path] if machine_path is not None else [])
    expected_by_stem = {p.stem: str(p.expanduser().resolve()) for p in all_expected}

    for ly in pts:
        assert "source_h5" in ly.metadata, f"Missing source_h5 in layer.metadata for {ly.name}"
        # Ensure it points to the actual file for that layer stem
        assert ly.metadata["source_h5"] == expected_by_stem[ly.name], (
            f"Layer {ly.name} has wrong source_h5:\n"
            f"  got: {ly.metadata['source_h5']}\n"
            f"  expected: {expected_by_stem[ly.name]}"
        )

        assert "io" in (ly.metadata or {}), f"Missing io provenance dict in layer.metadata for {ly.name}"
        assert ly.metadata["io"].get("source_relpath_posix"), f"io.source_relpath_posix missing for {ly.name}"


@pytest.mark.usefixtures("qtbot")
def test_promotion_first_save_prompts_and_creates_sidecar(make_napari_viewer, qtbot, tmp_path, inputdialog):
    """
    First save on a machine/prediction layer (no config.yaml, no sidecar):
    - prompts for scorer
    - writes .napari-deeplabcut.json sidecar
    - creates CollectedData_<scorer>.h5
    - does NOT modify machinelabels-iter0.h5
    """
    labeled_folder = _make_labeled_folder_with_machine_only(tmp_path)

    machine_path = labeled_folder / "machinelabels-iter0.h5"
    machine_pre = pd.read_hdf(machine_path, key="keypoints")

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open folder
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    # Find machine points layer
    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert any(p.name == "machinelabels-iter0" for p in pts_layers)
    machine_layer = next(p for p in pts_layers if p.name == "machinelabels-iter0")

    # Edit: add bodypart2 (use helper that works across versions)
    store = controls._stores.get(machine_layer)
    assert store is not None
    _set_or_add_bodypart_xy(machine_layer, store, "bodypart2", x=44.0, y=33.0)

    # Set user input for scorer
    inputdialog.set("Alice", ok=True)

    # Save via the widget path (ensures prompt runs)
    viewer.layers.selection.active = machine_layer
    controls.viewer.layers.selection.active = machine_layer
    controls.viewer.layers.selection.select_only(machine_layer)

    assert "io" in machine_layer.metadata
    assert machine_layer.metadata["io"].get("kind") in ("machine", AnnotationKind.MACHINE)

    # Call your menu-hooked save action (this hits promotion logic)
    controls._save_layers_dialog(selected=True)
    qtbot.wait(200)
    assert "save_target" in machine_layer.metadata, machine_layer.metadata.keys()

    # Sidecar created
    sidecar = labeled_folder / ".napari-deeplabcut.json"
    assert sidecar.exists()
    assert "Alice" in sidecar.read_text(encoding="utf-8")

    # GT created
    gt_path = labeled_folder / "CollectedData_Alice.h5"
    assert gt_path.exists()

    # Machine file unchanged
    machine_post = pd.read_hdf(machine_path, key="keypoints")
    pd.testing.assert_frame_equal(machine_pre, machine_post)


@pytest.mark.usefixtures("qtbot")
def test_promotion_second_save_uses_sidecar_no_prompt(make_napari_viewer, qtbot, tmp_path, inputdialog):
    """
    After sidecar exists, saving again must not prompt:
    - QInputDialog.getText not called
    - writes/updates same CollectedData_<scorer>.h5
    - machine file unchanged
    """
    labeled_folder = _make_labeled_folder_with_machine_only(tmp_path)

    # Pre-create sidecar (as if first run already happened)
    sidecar = labeled_folder / ".napari-deeplabcut.json"
    sidecar.write_text('{"schema_version": 1, "default_scorer": "Alice"}', encoding="utf-8")

    machine_path = labeled_folder / "machinelabels-iter0.h5"
    machine_pre = pd.read_hdf(machine_path, key="keypoints")

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    machine_layer = next(p for p in pts_layers if p.name == "machinelabels-iter0")

    store = controls._stores.get(machine_layer)
    assert store is not None
    _set_or_add_bodypart_xy(machine_layer, store, "bodypart1", x=99.0, y=88.0)

    # No prompt expected
    inputdialog.forbid()

    # Save via widget path
    controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    assert inputdialog.calls == 0

    gt_path = labeled_folder / "CollectedData_Alice.h5"
    assert gt_path.exists()

    machine_post = pd.read_hdf(machine_path, key="keypoints")
    pd.testing.assert_frame_equal(machine_pre, machine_post)
