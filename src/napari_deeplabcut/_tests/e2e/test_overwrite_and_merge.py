import logging

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel

from .utils import _get_coord_from_df, _get_points_layer_with_data, _make_minimal_dlc_project, _set_or_add_bodypart_xy

logger = logging.getLogger(__name__)


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

    _, _, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

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
    controls._save_layers_dialog(selected=True)
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
        controls._save_layers_dialog(selected=True)
    except Exception:
        # Some napari/npe2 versions may raise when writer aborts; file integrity is what matters.
        pass

    qtbot.wait(200)

    assert len(overwrite_confirm.calls) == 1, "Expected overwrite confirmation to be requested once."

    post = pd.read_hdf(h5_path, key="keypoints")
    assert _get_coord_from_df(post, "bodypart1", "x") == b1x_pre
    assert _get_coord_from_df(post, "bodypart1", "y") == b1y_pre
