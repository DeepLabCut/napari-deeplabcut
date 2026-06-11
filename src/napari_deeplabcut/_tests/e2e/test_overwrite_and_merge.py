import logging

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel

from .utils import _get_coord_from_df, _get_points_layer_with_data, _make_minimal_dlc_project, _set_or_add_bodypart_xy

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("qtbot")
def test_config_layer_then_h5_then_config_again_uses_plugin_save_checks(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
    monkeypatch,
):
    """
    Regression workflow:

    1. Load config.yaml and use the config-created Points layer as a DLC
       annotation layer once frame/save context exists.
    2. Save annotations through the plugin workflow.
    3. Reload the resulting H5/folder.
    4. Add config.yaml again.
    5. Saving the real DLC annotation layer must go through normal
       overwrite/deletion preflight, not generic napari save and not
       "Nothing to save".
    """
    from napari_deeplabcut.core import keypoints

    def fail_information(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox.information; save should route through plugin DLC workflow.")

    monkeypatch.setattr(
        "napari_deeplabcut.ui.ui_dialogs.save.QMessageBox.information",
        fail_information,
    )

    class FailFileDialog:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "Unexpected QFileDialog; DLC config/annotation layers must not use generic napari save."
            )

    monkeypatch.setattr(
        "napari_deeplabcut.ui.ui_dialogs.save.QFileDialog",
        FailFileDialog,
    )

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer.window.add_dock_widget(keypoint_controls, name="Keypoint controls", area="right")

    # ------------------------------------------------------------------
    # Phase 1: config-created layer becomes a real annotation layer.
    # ------------------------------------------------------------------

    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(
        lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) >= 1,
        timeout=5_000,
    )

    config_layer = next(ly for ly in viewer.layers if isinstance(ly, Points))
    assert keypoint_controls.layer_manager.is_config_placeholder_points_layer(config_layer)

    # Load frames/folder so the config layer can acquire save context.
    # Depending on reader behavior this may also load an existing H5 annotation
    # layer. The important thing is that the config-created layer should now
    # be promotable once it has annotations/save context.
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(300)

    store = keypoint_controls.get_layer_store(config_layer)
    assert store is not None

    # Add concrete annotations to the config-created layer.
    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    config_layer.add(np.array([0.0, 10.0, 20.0], dtype=float))

    store.current_keypoint = keypoints.Keypoint("bodypart2", "")
    config_layer.add(np.array([0.0, 33.0, 44.0], dtype=float))

    # NOTE: this kind of thing is why the layer identity system is useful
    assert not keypoint_controls.layer_manager.is_config_placeholder_points_layer(config_layer)

    viewer.layers.selection.clear()
    viewer.layers.selection.add(config_layer)
    viewer.layers.selection.active = config_layer

    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(300)

    assert h5_path.exists()

    # ------------------------------------------------------------------
    # Phase 2: reload saved H5/folder, add config.yaml again, then save
    # through overwrite/deletion-aware plugin workflow.
    # ------------------------------------------------------------------

    viewer.layers.clear()
    qtbot.wait(300)

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(300)

    points = _get_points_layer_with_data(viewer)

    # Add config.yaml again. This should not create an invalid generic-save path.
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.wait(300)

    store = keypoint_controls.get_layer_store(points)
    assert store is not None

    # Create an overwrite conflict to prove the save path runs preflight.
    overwrite_confirm.capture().reset_calls()

    _set_or_add_bodypart_xy(points, store, "bodypart1", x=99.0, y=88.0)

    viewer.layers.selection.clear()
    viewer.layers.selection.add(points)
    viewer.layers.selection.active = points

    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(300)

    assert len(overwrite_confirm.calls) == 1

    post = pd.read_hdf(h5_path, key="df_with_missing")
    assert _get_coord_from_df(post, "bodypart1", "x") == 99.0


@pytest.mark.usefixtures("qtbot")
def test_no_overwrite_warning_when_only_filling_nans(viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm):
    """
    Adding new labels (filling NaNs) must not prompt overwrite confirmation.
    """
    overwrite_confirm.forbid()

    _, _, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer.window.add_dock_widget(keypoint_controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)

    logger.debug("points.name: %s", points.name)
    logger.debug("points.data shape: %s", None if points.data is None else np.asarray(points.data).shape)
    logger.debug("points.data[:5]: %s", None if points.data is None else np.asarray(points.data)[:5])
    logger.debug(
        "any NaNs in points.data: %s", False if points.data is None else np.isnan(np.asarray(points.data)).any()
    )
    logger.debug(
        "any finite xy: %s", False if points.data is None else np.isfinite(np.asarray(points.data)[:, 1:3]).any()
    )

    logger.debug("len(label): %s", len(points.properties.get("label", [])))
    logger.debug("len(id): %s", len(points.properties.get("id", [])))
    logger.debug("label[:10]: %s", points.properties.get("label", [])[:10])
    logger.debug("id[:10]: %s", points.properties.get("id", [])[:10])

    hdr = points.metadata.get("header")
    logger.debug("header type: %s", type(hdr))
    if hdr is not None:
        if isinstance(hdr, DLCHeaderModel):
            header_model = hdr
        elif isinstance(hdr, dict):
            header_model = DLCHeaderModel.model_validate(hdr)
        else:
            header_model = DLCHeaderModel(columns=hdr)

    # Prefer portable inspection: tuple columns (pandas optional)
    logger.debug("header ncols=%s", len(header_model.columns))
    logger.debug("header scorer=%s", header_model.scorer)
    logger.debug("header individuals=%s", header_model.individuals)
    logger.debug("header bodyparts=%s", header_model.bodyparts)
    logger.debug("header coords=%s", header_model.coords)

    logger.info("points.data[:5] = %s", points.data[:5])
    logger.info("any NaNs in points.data = %s", np.isnan(points.data).any())
    logger.info("labels[:10] = %s", points.properties.get("label")[:10])
    logger.info("ids[:10] = %s", points.properties.get("id")[:10] if "id" in points.properties else None)
    store = keypoint_controls.get_layer_store(points)
    assert store is not None

    # Fill NaNs for bodypart2
    _set_or_add_bodypart_xy(points, store, "bodypart2", x=44.0, y=33.0)

    viewer.layers.selection.active = points
    # viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    post = pd.read_hdf(h5_path, key="df_with_missing")
    assert np.isfinite(_get_coord_from_df(post, "bodypart1", "x"))
    assert np.isfinite(_get_coord_from_df(post, "bodypart2", "x"))


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_triggers_on_conflict(viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm):
    """
    Modifying an existing non-NaN label must trigger overwrite confirmation.
    """
    overwrite_confirm.capture().reset_calls()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)
    viewer.window.add_dock_widget(keypoint_controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    store = keypoint_controls.get_layer_store(points)
    assert store is not None

    # Create conflict: overwrite bodypart1 from (10,20) -> (99,88)
    _set_or_add_bodypart_xy(points, store, "bodypart1", x=99.0, y=88.0)

    viewer.layers.selection.active = points
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."
    assert overwrite_confirm.calls[0]["n_pairs"] is not None
    assert overwrite_confirm.calls[0]["n_pairs"] >= 1

    post = pd.read_hdf(h5_path, key="df_with_missing")
    assert _get_coord_from_df(post, "bodypart1", "x") == 99.0


@pytest.mark.usefixtures("qtbot")
def test_deletion_warning_triggers_when_existing_keypoint_is_removed(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
    monkeypatch,
):
    """
    Removing an existing saved keypoint from a real loaded DLC annotation layer
    must trigger confirmation before the saved coordinate is cleared.
    """
    overwrite_confirm.capture().reset_calls()
    monkeypatch.setattr(
        "napari_deeplabcut.ui.ui_dialogs.save.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("Unexpected warning dialog; deletion preflight should run."),
    )

    _, _, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    pre = pd.read_hdf(h5_path, key="df_with_missing")
    assert np.isfinite(_get_coord_from_df(pre, "bodypart1", "x"))

    viewer.window.add_dock_widget(keypoint_controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    labels = np.asarray(points.properties.get("label"))
    mask = labels != "bodypart1"

    points.data = np.asarray(points.data)[mask]
    for key, values in list(points.properties.items()):
        try:
            arr = np.asarray(values)
            if len(arr) == len(mask):
                points.properties[key] = arr[mask]
        except Exception:
            pass

    viewer.layers.selection.clear()
    viewer.layers.selection.add(points)
    viewer.layers.selection.active = points

    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    call = overwrite_confirm.calls[0]
    assert call["n_deletions"] >= 1

    deleted = []
    for entry in call["entries"]:
        deleted.extend(getattr(entry, "deleted_keypoints", ()) or ())
    assert "bodypart1" in deleted

    post = pd.read_hdf(h5_path, key="df_with_missing")
    assert np.isnan(_get_coord_from_df(post, "bodypart1", "x"))


@pytest.mark.usefixtures("qtbot")
def test_overwrite_warning_cancel_aborts_write(viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm):
    """
    If overwrite confirmation is requested and user cancels, file must remain unchanged.
    """
    overwrite_confirm.cancel().reset_calls()

    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    pre = pd.read_hdf(h5_path, key="df_with_missing")
    b1x_pre = _get_coord_from_df(pre, "bodypart1", "x")
    b1y_pre = _get_coord_from_df(pre, "bodypart1", "y")

    viewer.window.add_dock_widget(keypoint_controls, name="Keypoint controls", area="right")

    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    points = _get_points_layer_with_data(viewer)
    store = keypoint_controls.get_layer_store(points)
    assert store is not None

    _set_or_add_bodypart_xy(points, store, "bodypart1", x=456.0, y=123.0)

    viewer.layers.selection.active = points
    try:
        keypoint_controls._save_layers_dialog(selected=True)
    except Exception:
        # Some napari/npe2 versions may raise when writer aborts; file integrity is what matters.
        pass

    qtbot.wait(200)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."

    post = pd.read_hdf(h5_path, key="df_with_missing")
    assert _get_coord_from_df(post, "bodypart1", "x") == b1x_pre
    assert _get_coord_from_df(post, "bodypart1", "y") == b1y_pre
