# src/napari_deeplabcut/_tests/test_e2e.py
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points
from qtpy.QtWidgets import QMessageBox

from napari_deeplabcut.ui.dialogs import OverwriteConflictsDialog


@pytest.fixture(autouse=True)
def _auto_accept_qmessagebox(monkeypatch):
    """
    Prevent modal dialogs from blocking tests.
    - Always accept the 'Data were not saved' close warning.
    - Also auto-accept generic question dialogs if they appear.
    """
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)


@pytest.fixture(autouse=True)
def overwrite_confirm(monkeypatch):
    """
    Autouse: by default, auto-confirm overwrite warnings so tests can save.
    Provides helpers to:
      - forbid()  -> fail test if confirm is called
      - cancel()  -> confirm called, return False
      - capture() -> record calls, return True (default)
      - set_result(value) -> custom return
    """
    calls = []

    # internal state
    state = {"mode": "always_true", "result": True}

    def _confirm(parent, *, summary=None, details=None, **kwargs):
        calls.append({"summary": summary, "details": details, "kwargs": kwargs})
        if state["mode"] == "forbid":
            raise AssertionError("OverwriteConflictsDialog.confirm was called unexpectedly.")
        return state["result"]

    # install default behavior: always return True
    monkeypatch.setattr(OverwriteConflictsDialog, "confirm", staticmethod(_confirm))

    # exposed “controller”
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


def _write_minimal_png(path: Path, *, shape=(64, 64, 3)) -> None:
    """Write a tiny RGB image to satisfy the folder reader."""
    from skimage.io import imsave

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros(shape, dtype=np.uint8)
    # add a simple pattern so it isn't fully empty
    img[8:24, 8:24, 0] = 255
    imsave(str(path), img, check_contrast=False)


def _set_bodypart_xy(points_layer: Points, bodypart: str, *, x: float, y: float):
    labels = np.asarray(points_layer.properties.get("label"))
    mask = labels == bodypart
    assert mask.any(), f"Could not find {bodypart} in Points layer properties."

    data = np.array(points_layer.data, copy=True)
    data[mask, 1] = y  # frame,y,x convention
    data[mask, 2] = x
    points_layer.data = data


def _make_minimal_dlc_project(tmp_path: Path):
    """
    Build a minimal DLC-like folder:
      project/
        config.yaml
        labeled-data/test/img000.png
        labeled-data/test/CollectedData_John.h5 (with bodypart1 labeled)
    """
    import yaml

    project = tmp_path / "project"
    labeled = project / "labeled-data" / "test"
    labeled.mkdir(parents=True, exist_ok=True)

    # 1) Image
    img_rel = ("labeled-data", "test", "img000.png")
    img_path = project / Path(*img_rel)
    _write_minimal_png(img_path)

    # 2) config.yaml (minimal keys used by read_config / DLCHeader.from_config)
    cfg = {
        "scorer": "John",
        "bodyparts": ["bodypart1", "bodypart2"],
        "dotsize": 8,
        "pcutoff": 0.6,
        "colormap": "viridis",
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 3) Existing ground-truth labels: only bodypart1 is labeled
    # DLC-style columns: scorer/bodyparts/coords
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


def _dump_layers(viewer, log: logging.Logger, label: str):
    from napari.layers import Image, Points

    log.debug("==== %s ====", label)
    for i, layer in enumerate(viewer.layers):
        kind = type(layer).__name__
        selected = viewer.layers.selection.active is layer
        md_keys = sorted(list(getattr(layer, "metadata", {}).keys()))
        if isinstance(layer, Points):
            npts = 0 if layer.data is None else len(layer.data)
            any_data = False if layer.data is None else bool(np.any(layer.data))
            log.debug(
                "[%02d] %s name=%r selected=%s npts=%s any_data=%s md_keys=%s md.project=%r md.root=%r",
                i,
                kind,
                layer.name,
                selected,
                npts,
                any_data,
                md_keys,
                layer.metadata.get("project"),
                layer.metadata.get("root"),
            )
        elif isinstance(layer, Image):
            shape = getattr(layer.data, "shape", None)
            log.debug(
                "[%02d] %s name=%r selected=%s shape=%s md.root=%r md.paths=%s",
                i,
                kind,
                layer.name,
                selected,
                shape,
                layer.metadata.get("root"),
                "yes" if layer.metadata.get("paths") else "no",
            )
        else:
            log.debug(
                "[%02d] %s name=%r selected=%s md_keys=%s",
                i,
                kind,
                layer.name,
                selected,
                md_keys,
            )


@pytest.mark.usefixtures("qtbot")
def test_merge_on_save_preserves_existing_labels_and_adds_new_no_warning(
    make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm
):
    """
    Regression: saving after config-first + folder load must NEVER erase old labels.
    Specifically:
      - existing bodypart1 stays finite
      - new bodypart2 becomes finite
      - overwrite warning must NOT be shown when only filling NaNs (no conflicts)
    """
    logging.getLogger("napari_deeplabcut.tests.overwrite")
    caplog.set_level(logging.DEBUG, logger="napari_deeplabcut.tests.overwrite")

    overwrite_confirm.forbid()  # no warnings expected in this test

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    # Precondition
    pre = pd.read_hdf(h5_path, key="keypoints")
    b1x_pre = pre.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    b2x_pre = pre.xs(("bodypart2", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    assert np.isfinite(b1x_pre)
    assert np.isnan(b2x_pre)

    viewer = make_napari_viewer()

    from napari.layers import Points

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open config first
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 1, timeout=5_000)

    # Capture placeholder early (may lose md.project later)
    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None

    # Open folder (loads images + existing labels)
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert points_layers, "Expected at least one Points layer after opening folder."

    # Choose target layer to add to: prefer placeholder if it still exists (reproduces real hazard),
    # otherwise fall back to an active Points layer.
    target = placeholder if placeholder in viewer.layers else viewer.layers.selection.active
    if not isinstance(target, Points):
        target = points_layers[0]

    # Ensure store exists (patched add method depends on KeypointControls.on_insert wiring)
    store = controls._stores.get(target)
    assert store is not None, "No KeypointStore registered for target Points layer."

    from napari_deeplabcut import keypoints

    store.current_keypoint = keypoints.Keypoint("bodypart2", "")

    # Add a bodypart2 point
    target.add(np.array([0, 33.0, 44.0], dtype=float))

    # Save selected layer via plugin writer
    viewer.layers.selection.active = target
    viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(100)

    # Postcondition: bodypart1 preserved, bodypart2 added
    post = pd.read_hdf(h5_path, key="keypoints")
    b1x_post = post.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    b2x_post = post.xs(("bodypart2", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]

    assert np.isfinite(b1x_post), "Regression: bodypart1 label must be preserved (no silent deletion)."
    assert np.isfinite(b2x_post), "Regression: bodypart2 label must be saved."
    assert b2x_post == 44.0


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_triggers_on_conflict_and_mentions_image_and_keypoint(
    make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm
):
    """
    Integration: If user overwrites an existing non-NaN label (conflict),
    we must show the overwrite warning with image + keypoint details.
    """
    logging.getLogger("napari_deeplabcut.tests.overwrite")
    caplog.set_level(logging.DEBUG, logger="napari_deeplabcut.tests.overwrite")
    overwrite_confirm.capture().reset_calls()  # prepare to capture calls in this test

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    # Precondition: bodypart1 exists
    pre = pd.read_hdf(h5_path, key="keypoints")
    b1x_pre = pre.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    assert np.isfinite(b1x_pre)

    viewer = make_napari_viewer()

    from napari.layers import Points

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open folder only: loads images + existing labels into a Points layer
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    # Identify the Points layer with actual data
    points_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert points_layers, "Expected at least one Points layer."
    points = next((ly for ly in points_layers if np.any(ly.data)), points_layers[0])

    # Create a true conflict by changing bodypart1 coords in-memory
    # The H5 contains bodypart1 (10,20). We change it to something else.
    labels = np.asarray(points.properties.get("label"))
    assert labels is not None and labels.size > 0

    mask = labels == "bodypart1"
    assert mask.any(), "Could not find bodypart1 row in Points layer properties."

    _set_bodypart_xy(points, "bodypart1", x=99.0, y=88.0)

    # Save via plugin writer
    viewer.layers.selection.active = points
    viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(100)

    # Assert warning was shown and contains image+keypoint
    assert overwrite_confirm.calls == 1, "Expected overwrite warning to be shown exactly once."
    assert overwrite_confirm.summary is not None
    assert "overwritten" in overwrite_confirm.summary.lower()

    assert overwrite_confirm.details is not None
    # We want image + keypoint in details
    assert "img000.png" in overwrite_confirm.details, "Expected image name in overwrite warning details."
    assert "bodypart1" in overwrite_confirm.details, "Expected keypoint name in overwrite warning details."

    # Postcondition: file updated (since user accepted)
    post = pd.read_hdf(h5_path, key="keypoints")
    b1x_post = post.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    assert b1x_post == 99.0, "Expected bodypart1.x to be updated to new value after accepting overwrite warning."


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_cancel_aborts_write_and_file_unchanged(
    make_napari_viewer, qtbot, tmp_path, caplog, overwrite_confirm
):
    """
    Integration: If overwrite warning is shown and user cancels,
    the save must be aborted and the on-disk file must remain unchanged.
    """
    logging.getLogger("napari_deeplabcut.tests.overwrite")
    caplog.set_level(logging.DEBUG, logger="napari_deeplabcut.tests.overwrite")
    overwrite_confirm.cancel().reset_calls()  # prepare to capture calls in this test

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    pre = pd.read_hdf(h5_path, key="keypoints")
    b1x_pre = pre.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    b1y_pre = pre.xs(("bodypart1", "y"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    assert np.isfinite(b1x_pre) and np.isfinite(b1y_pre)

    viewer = make_napari_viewer()

    from napari.layers import Points

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    points = next((ly for ly in points_layers if np.any(ly.data)), points_layers[0])

    # Introduce conflict (change bodypart1)
    labels = np.asarray(points.properties.get("label"))
    mask = labels == "bodypart1"
    assert mask.any()

    _set_bodypart_xy(points, "bodypart1", x=99.0, y=88.0)

    # Attempt save. Depending on napari/npe2 behavior, writer returning None may raise.
    viewer.layers.selection.active = points
    try:
        viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
    except Exception:
        # Accept any exception here: the key requirement is "file not modified"
        pass

    qtbot.wait(100)

    assert overwrite_confirm.calls == 1, "Expected overwrite warning to be shown once."

    # File should be unchanged
    post = pd.read_hdf(h5_path, key="keypoints")
    b1x_post = post.xs(("bodypart1", "x"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]
    b1y_post = post.xs(("bodypart1", "y"), axis=1, level=["bodyparts", "coords"]).iloc[0, 0]

    assert b1x_post == b1x_pre, "Cancel must prevent overwriting existing bodypart1.x"
    assert b1y_post == b1y_pre, "Cancel must prevent overwriting existing bodypart1.y"
