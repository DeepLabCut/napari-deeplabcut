import logging

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel
from napari_deeplabcut.core.io import _read_hdf_any_key
from napari_deeplabcut.core.layers import is_machine_layer

from .utils import (
    _dataframe_rows_by_path,
    _get_coord_from_df,
    _get_points_layer_with_data,
    _make_minimal_dlc_project,
    _make_project_config_and_frames_no_gt,
    _seed_gt_and_machine_outlier_dataset,
    _set_or_add_bodypart_xy,
)

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


@pytest.mark.usefixtures("qtbot")
def test_machine_label_promotion_preserves_existing_gt_after_frame_remap(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
):
    """
    End-to-end regression for destructive machine-to-GT promotion.

    Reproduced workflow
    -------------------
    1. A labeled-data folder contains 50 extracted images.
    2. CollectedData_John.h5 contains manually labeled GT for the first
       30 frames.
    3. machinelabels-iter0.h5 contains machine annotations for the final
       20 outlier frames.
    4. Opening the folder aligns both annotation layers to one 50-frame
       image stack.
    5. The machine layer is selected and saved through the plugin UI.
    6. The save workflow promotes the machine source into the existing
       CollectedData_John.h5 target.

    Regression assertions
    ---------------------
    - the original 30 GT rows remain unchanged;
    - the 20 machine rows are promoted into GT;
    - the resulting GT file contains 50 rows;
    - machine likelihood values are not written to GT;
    - the machine source file is not overwritten;
    - the on-disk GT file remains canonical single-animal DLC data.
    """
    overwrite_confirm.capture()

    (
        _project,
        _config_path,
        labeled_folder,
    ) = _make_project_config_and_frames_no_gt(tmp_path)

    (
        gt_path,
        machine_path,
        initial_paths,
        outlier_paths,
        expected_outlier_xy,
    ) = _seed_gt_and_machine_outlier_dataset(
        labeled_folder,
        scorer="John",
        bodypart="bodypart1",
        n_initial_frames=30,
        n_outlier_frames=20,
    )

    # Save exact copies for post-operation comparisons.
    gt_before = _read_hdf_any_key(gt_path).sort_index()
    machine_before = _read_hdf_any_key(machine_path).sort_index()

    assert len(gt_before.index) == 30
    assert len(machine_before.index) == 20

    assert gt_before.notna().all().all()
    assert machine_before.notna().all().all()

    # Canonical disk-state sanity checks before loading the plugin.
    assert gt_before.columns.nlevels == 3
    assert list(gt_before.columns.names) == [
        "scorer",
        "bodyparts",
        "coords",
    ]

    assert machine_before.columns.nlevels == 3
    assert "likelihood" in set(machine_before.columns.get_level_values("coords"))

    # Open the complete labeled-data folder through the real reader and
    # lifecycle manager.
    viewer.open(
        str(labeled_folder),
        plugin="napari-deeplabcut",
    )

    qtbot.waitUntil(
        lambda: len([layer for layer in viewer.layers if isinstance(layer, Points)]) >= 2,
        timeout=10_000,
    )

    qtbot.waitUntil(
        lambda: any(isinstance(layer, Points) and layer.name == "machinelabels-iter0" for layer in viewer.layers),
        timeout=10_000,
    )

    points_layers = [layer for layer in viewer.layers if isinstance(layer, Points)]

    machine_layers = [layer for layer in points_layers if is_machine_layer(layer)]

    assert len(machine_layers) == 1, (
        "Expected exactly one machine annotation layer. "
        "Loaded Points layers were: "
        f"{[(layer.name, layer.metadata) for layer in points_layers]}"
    )

    machine_layer = machine_layers[0]

    assert machine_layer.name == ("machinelabels-iter0")

    machine_io = (machine_layer.metadata or {}).get("io")
    machine_kind = machine_io.get("kind") if isinstance(machine_io, dict) else getattr(machine_io, "kind", None)
    assert machine_kind in ("machine", "MACHINE", AnnotationKind.MACHINE)

    # The source machine HDF has 20 annotation rows. After remapping, finite
    # machine points should still occupy exactly 20 frame positions.
    machine_data = np.asarray(machine_layer.data)

    assert machine_data.ndim == 2
    assert machine_data.shape[1] == 3

    machine_frame_indices = {int(frame_index) for frame_index in machine_data[:, 0]}

    assert len(machine_frame_indices) == 20

    # Reproduce and document the exact dangerous state:
    #
    # - finite machine annotations: 20 frames
    # - machine metadata paths:     combined 50-frame viewer context
    remapped_paths = list((machine_layer.metadata or {}).get("paths") or [])

    assert len(remapped_paths) == 50, (
        "The regression setup did not reproduce the post-remap state. "
        "Expected the machine layer to have 50 shared viewer paths, "
        f"but got {len(remapped_paths)}."
    )

    normalized_remapped_paths = {str(path).replace("\\", "/") for path in remapped_paths}

    assert set(initial_paths).issubset(normalized_remapped_paths)
    assert set(outlier_paths).issubset(normalized_remapped_paths)

    # Before saving, the machine source should not yet have a promotion target.
    assert (machine_layer.metadata or {}).get("save_target") is None

    # Exercise the real selected-layer save workflow. This should:
    #
    # 1. detect the MACHINE source;
    # 2. discover config.yaml and scorer John;
    # 3. attach a GT save target;
    # 4. preflight against CollectedData_John.h5;
    # 5. write through write_hdf();
    # 6. use allow_deletions=False for the MACHINE source.
    viewer.layers.selection.active = machine_layer
    keypoint_controls.viewer.layers.selection.select_only(machine_layer)

    keypoint_controls._save_layers_dialog(selected=True)

    # Wait for the final combined file rather than relying on a fixed sleep.
    def _gt_has_expected_rows() -> bool:
        try:
            saved = _read_hdf_any_key(gt_path)
            return len(saved.index) == 50
        except Exception:
            return False

    qtbot.waitUntil(
        _gt_has_expected_rows,
        timeout=10_000,
    )

    # Promotion target should now be attached to the live machine layer.
    save_target = (machine_layer.metadata or {}).get("save_target")

    assert save_target is not None

    if isinstance(save_target, dict):
        assert save_target.get("kind") is (AnnotationKind.GT)
        assert save_target.get("scorer") == ("John")
        assert save_target.get("source_relpath_posix") == "CollectedData_John.h5"
    else:
        assert (
            getattr(
                save_target,
                "kind",
                None,
            )
            is AnnotationKind.GT
        )
        assert (
            getattr(
                save_target,
                "scorer",
                None,
            )
            == "John"
        )
        assert (
            getattr(
                save_target,
                "source_relpath_posix",
                None,
            )
            == "CollectedData_John.h5"
        )

    assert gt_path.exists()
    assert gt_path.with_suffix(".csv").exists()

    gt_after = _read_hdf_any_key(gt_path).sort_index()

    # ------------------------------------------------------------------
    # Validate final GT schema.
    # ------------------------------------------------------------------

    assert isinstance(
        gt_after.columns,
        pd.MultiIndex,
    )
    assert gt_after.columns.nlevels == 3
    assert list(gt_after.columns.names) == [
        "scorer",
        "bodyparts",
        "coords",
    ]

    assert set(gt_after.columns.get_level_values("scorer")) == {"John"}

    assert tuple(dict.fromkeys(gt_after.columns.get_level_values("bodyparts"))) == ("bodypart1",)

    assert set(gt_after.columns.get_level_values("coords")) == {"x", "y"}

    assert "likelihood" not in set(gt_after.columns.get_level_values("coords"))

    # Expected final dataset:
    # 30 manual GT frames + 20 promoted machine frames.
    assert len(gt_after.index) == 50

    before_rows = _dataframe_rows_by_path(gt_before)
    after_rows = _dataframe_rows_by_path(gt_after)

    assert set(initial_paths).issubset(after_rows)
    assert set(outlier_paths).issubset(after_rows)

    # ------------------------------------------------------------------
    # Core regression assertion: original GT is unchanged.
    # ------------------------------------------------------------------

    for path in initial_paths:
        before_row = gt_before.loc[before_rows[path]]
        after_row = gt_after.loc[after_rows[path]]

        pd.testing.assert_series_equal(
            after_row,
            before_row,
            check_dtype=False,
            check_names=False,
        )

    # ------------------------------------------------------------------
    # Promoted machine values were added with the target GT scorer.
    # ------------------------------------------------------------------

    for path, (
        expected_x,
        expected_y,
    ) in expected_outlier_xy.items():
        row_key = after_rows[path]

        actual_x = gt_after.loc[
            row_key,
            (
                "John",
                "bodypart1",
                "x",
            ),
        ]
        actual_y = gt_after.loc[
            row_key,
            (
                "John",
                "bodypart1",
                "y",
            ),
        ]

        assert actual_x == pytest.approx(expected_x)
        assert actual_y == pytest.approx(expected_y)

    # There should be no fully empty rows in the final combined dataset.
    assert not gt_after.isna().all(axis=1).any()

    # ------------------------------------------------------------------
    # Saving refined annotations must not rewrite the machine source HDF.
    # ------------------------------------------------------------------

    machine_after = _read_hdf_any_key(machine_path).sort_index()

    pd.testing.assert_frame_equal(
        machine_after,
        machine_before,
        check_dtype=False,
    )
