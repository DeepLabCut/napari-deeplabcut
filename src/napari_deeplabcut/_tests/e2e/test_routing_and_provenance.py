import logging

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.core.io import AnnotationKind, MissingProvenanceError

from .utils import (
    _assert_only_these_files_changed,
    _make_dlc_project_with_multiple_gt,
    _make_labeled_folder_with_machine_only,
    _make_project_config_and_frames_no_gt,
    _set_or_add_bodypart_xy,
    _snapshot_coords,
    file_sig,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def forbid_project_config_dialog(monkeypatch):
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.prompt_for_project_config_for_save",
        lambda *args, **kwargs: pytest.fail("Unexpected project-config dialog."),
    )
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.maybe_confirm_dataset_path_rewrite",
        lambda *args, **kwargs: pytest.fail("Unexpected dataset path rewrite confirmation."),
    )
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.warn_existing_dataset_folder_conflict",
        lambda *args, **kwargs: pytest.fail("Unexpected dataset-folder conflict warning."),
    )


@pytest.mark.usefixtures("qtbot")
def test_save_routes_to_correct_gt_when_multiple_gt_exist(viewer, qtbot, tmp_path, overwrite_confirm):
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

    logger.info("BEFORE SAVE : name=%s, sig=%s", gt_paths[0].name, file_sig(gt_paths[0]))
    logger.info("BEFORE SAVE : name=%s, sig=%s", gt_paths[1].name, file_sig(gt_paths[1]))

    viewer.layers.selection.active = points_b
    logger.info("Layer selected for save: %s", points_b.name)
    # logger.info("Layer metadata: %s", points_b.metadata)
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")

    logger.info("AFTER SAVE : name=%s, sig=%s", gt_paths[0].name, file_sig(gt_paths[0]))
    logger.info("AFTER SAVE : name=%s, sig=%s", gt_paths[1].name, file_sig(gt_paths[1]))

    qtbot.wait(200)

    after = {p: _snapshot_coords(p) for p in gt_paths}

    _assert_only_these_files_changed(before, after, changed={gt_b})
    assert after[gt_b]["b2x"] == 77.0


@pytest.mark.usefixtures("qtbot")
def test_machine_layer_does_not_modify_gt_on_save(viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: machine outputs must never save to their own file.
    Users must explicitly provide a scorer name that is then used to save the h5.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, machine_path = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=True
    )
    assert machine_path is not None

    before = {p: _snapshot_coords(p) for p in gt_paths + [machine_path]}

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

    with pytest.raises(MissingProvenanceError):
        viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")

    qtbot.wait(200)

    after = {p: _snapshot_coords(p) for p in gt_paths + [machine_path]}

    # Machine file should be unchanged (no save path),
    # and GT files should be unchanged (machine edits must not touch GT).
    _assert_only_these_files_changed(before, after, changed=set())
    # assert after[machine_path]["b2x"] == 55.0


@pytest.mark.usefixtures("qtbot")
def test_layer_rename_does_not_change_save_target(viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: layer renaming must not redirect output or create new file.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, _ = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=False
    )
    gt_a = gt_paths[0]

    before = {p: _snapshot_coords(p) for p in gt_paths}

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
def test_ambiguous_placeholder_save_aborts_when_multiple_gt_exist(viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Contract: If provenance is missing and multiple candidate GT files exist,
    save must refuse (deterministic) rather than silently choosing.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder, gt_paths, _ = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=False
    )

    before = {p: _snapshot_coords(p) for p in gt_paths}

    from napari_deeplabcut._widgets import KeypointControls
    from napari_deeplabcut.core import keypoints

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
def test_folder_open_loads_all_h5_when_multiple_exist(viewer, qtbot, tmp_path):
    """
    Contract: Opening a labeled-data folder with multiple H5 files should not
    silently pick the first one. Preferred policy: load all as separate Points layers.
    """
    project, config_path, labeled_folder, gt_paths, machine_path = _make_dlc_project_with_multiple_gt(
        tmp_path, scorers=("John", "Jane"), with_machine=True
    )

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
def test_config_first_save_writes_gt_into_dataset_folder(viewer, qtbot, tmp_path, overwrite_confirm):
    """
    Regression: config-first workflow must save CollectedData_<scorer>.h5 inside
    project/labeled-data/<dataset>/, not next to config.yaml.
    """
    overwrite_confirm.forbid()

    project, config_path, labeled_folder = _make_project_config_and_frames_no_gt(tmp_path)

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open config first -> placeholder points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    # Open dataset folder -> provides dataset context
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts_layers, "Expected a Points layer from config.yaml"

    points = pts_layers[0]
    store = controls._stores.get(points)
    assert store is not None

    # Add a point and save
    _set_or_add_bodypart_xy(points, store, "bodypart1", x=11.0, y=22.0)

    viewer.layers.selection.active = points
    viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
    qtbot.wait(300)

    expected = labeled_folder / "CollectedData_John.h5"
    assert expected.exists(), f"Expected GT to be created in dataset folder: {expected}"

    wrong = project / "CollectedData_John.h5"
    assert not wrong.exists(), f"Must not save next to config.yaml: {wrong}"


@pytest.mark.usefixtures("qtbot")
def test_promotion_first_save_prompts_and_creates_sidecar(
    viewer, qtbot, tmp_path, inputdialog, forbid_project_config_dialog
):
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
def test_promotion_second_save_uses_sidecar_no_prompt(
    viewer, qtbot, tmp_path, inputdialog, forbid_project_config_dialog
):
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


@pytest.mark.usefixtures("qtbot")
def test_projectless_folder_save_can_associate_with_config_and_coerce_paths_to_dlc_row_keys(
    viewer,
    qtbot,
    tmp_path,
    monkeypatch,
    overwrite_confirm,
):
    """
    Contract: a project-less labeled folder can be associated with a chosen DLC
    project at save time by rewriting safe paths to canonical DLC row keys.

    Goals
    -----
    - Use current external folder name as the target dataset name.
    - Save safe paths as labeled-data/<dataset>/<image>.
    - Use the same normalized metadata for overwrite preflight and actual write.
    - Persist the improved metadata on the live layer after successful save.

    Non-goals
    ---------
    - Do NOT require the current files to already be inside the selected project.
    - Do NOT coerce nested/multi-folder layouts into DLC row keys.
    - Do NOT rewrite unrelated outside paths.
    """
    overwrite_confirm.forbid()

    project, config_path, _project_dataset_folder = _make_project_config_and_frames_no_gt(tmp_path)

    # External project-less folder that the user labeled outside the project.
    external_folder = tmp_path / "session_external"
    external_folder.mkdir()

    inside_abs = external_folder / "img001.png"
    inside_abs.write_bytes(b"placeholder")
    dataset = external_folder.name

    outside_dir = tmp_path / "external-images"
    outside_dir.mkdir()
    outside_img = outside_dir / "img999.png"
    outside_img.write_bytes(b"placeholder")

    from napari_deeplabcut._widgets import KeypointControls
    from napari_deeplabcut.core import keypoints

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Open config first -> placeholder points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    points = next(ly for ly in viewer.layers if isinstance(ly, Points))
    store = controls._stores.get(points)
    assert store is not None

    # Simulate project-less folder metadata:
    points.metadata = dict(points.metadata or {})
    points.metadata["root"] = str(external_folder)
    points.metadata["paths"] = [
        str(inside_abs),  # direct child of source_root -> should coerce
        "img002.png",  # basename -> should coerce
        f"labeled-data/{dataset}/img003.png",  # already canonical -> preserve
        str(outside_img),  # unrelated absolute path -> preserve unchanged
    ]
    points.metadata.pop("project", None)

    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    points.add(np.array([0.0, 11.0, 22.0], dtype=float))

    from napari_deeplabcut.ui import dialogs as ui_dialogs

    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.prompt_for_project_config_for_save",
        lambda *args, **kwargs: ui_dialogs.ProjectConfigPromptResult(
            action=ui_dialogs.ProjectConfigPromptAction.ASSOCIATE,
            config_path=str(config_path),
        ),
    )
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.maybe_confirm_dataset_path_rewrite",
        lambda *args, **kwargs: True,
    )

    import napari_deeplabcut.core.conflicts as conflicts

    real_compute = conflicts.compute_overwrite_report_for_points_save
    captured = {}

    def _wrapped_compute(data, attributes):
        captured["attributes"] = attributes
        return real_compute(data, attributes)

    monkeypatch.setattr(
        "napari_deeplabcut._widgets.compute_overwrite_report_for_points_save",
        _wrapped_compute,
    )

    viewer.layers.selection.active = points
    controls.viewer.layers.selection.active = points
    controls.viewer.layers.selection.select_only(points)

    controls._save_layers_dialog(selected=True)
    qtbot.wait(300)

    # After project association, save should route into the chosen project's
    # labeled-data/<dataset>/ folder inferred from the rewritten metadata.
    expected_dataset_dir = project / "labeled-data" / dataset
    expected_h5 = expected_dataset_dir / "CollectedData_John.h5"
    expected_csv = expected_dataset_dir / "CollectedData_John.csv"

    assert expected_h5.exists()
    assert expected_csv.exists()

    # And it should NOT create a GT file next to the external source folder.
    assert not (external_folder / "CollectedData_John.h5").exists()
    assert not (external_folder / "CollectedData_John.csv").exists()

    expected_paths = [
        f"labeled-data/{dataset}/{inside_abs.name}",
        f"labeled-data/{dataset}/img002.png",
        f"labeled-data/{dataset}/img003.png",
        outside_img.as_posix(),
    ]

    # Preflight saw normalized metadata
    assert captured["attributes"]["metadata"]["project"] == str(project)
    assert captured["attributes"]["metadata"]["paths"] == expected_paths

    # Live layer metadata persisted the successful normalization
    assert points.metadata["project"] == str(project)
    assert points.metadata["paths"] == expected_paths

    # H5 row index contains canonical DLC row keys for the safe cases
    df = pd.read_hdf(expected_h5, key="keypoints")
    if isinstance(df.index, pd.MultiIndex):
        observed_rows = ["/".join(map(str, idx)) for idx in df.index]
    else:
        observed_rows = [str(idx).replace("\\", "/") for idx in df.index]

    assert f"labeled-data/{dataset}/{inside_abs.name}" in observed_rows
    assert f"labeled-data/{dataset}/img002.png" not in observed_rows
    assert f"labeled-data/{dataset}/img003.png" not in observed_rows
    assert outside_img.as_posix() not in observed_rows


@pytest.mark.usefixtures("qtbot")
def test_projectless_folder_save_refuses_when_target_dataset_folder_already_contains_files(
    viewer,
    qtbot,
    tmp_path,
    monkeypatch,
    overwrite_confirm,
):
    """
    Contract: project-association save must refuse if the target dataset folder
    already exists in the chosen project and contains files.
    """
    overwrite_confirm.forbid()

    project, config_path, existing_project_dataset = _make_project_config_and_frames_no_gt(tmp_path)
    dataset = existing_project_dataset.name

    # Existing populated target dataset folder inside project -> must refuse
    assert existing_project_dataset.exists()
    assert any(existing_project_dataset.iterdir()), "Expected existing project dataset folder to already contain files."

    # External project-less folder with the SAME dataset name
    external_parent = tmp_path / "external-root"
    external_parent.mkdir()
    external_folder = external_parent / dataset
    external_folder.mkdir()

    external_img = external_folder / "img_external.png"
    external_img.write_bytes(b"placeholder")

    from napari_deeplabcut._widgets import KeypointControls
    from napari_deeplabcut.core import keypoints

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    points = next(ly for ly in viewer.layers if isinstance(ly, Points))
    store = controls._stores.get(points)
    assert store is not None

    points.metadata = dict(points.metadata or {})
    points.metadata["root"] = str(external_folder)
    points.metadata["paths"] = [str(external_img)]
    points.metadata.pop("project", None)

    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    points.add(np.array([0.0, 11.0, 22.0], dtype=float))

    warned = {}

    from napari_deeplabcut.ui import dialogs as ui_dialogs

    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.prompt_for_project_config_for_save",
        lambda *args, **kwargs: ui_dialogs.ProjectConfigPromptResult(
            action=ui_dialogs.ProjectConfigPromptAction.ASSOCIATE,
            config_path=str(config_path),
        ),
    )
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.warn_existing_dataset_folder_conflict",
        lambda *args, **kwargs: warned.setdefault("called", True),
    )
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.ui_dialogs.maybe_confirm_dataset_path_rewrite",
        lambda *args, **kwargs: True,
    )

    viewer.layers.selection.active = points
    controls.viewer.layers.selection.select_only(points)

    controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    assert warned.get("called", False), "Expected conflict warning for populated target dataset folder."

    # No GT should be created in the external folder because association was refused.
    assert not (external_folder / "CollectedData_John.h5").exists()
