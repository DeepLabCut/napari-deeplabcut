from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.widget_factory import get_existing_keypoint_controls

from .utils import _cycles_from_policy, _make_minimal_dlc_project, _scheme_from_policy


@pytest.mark.usefixtures("qtbot")
def test_config_placeholder_points_layer_colors_after_first_keypoint_added(viewer, qtbot, tmp_path):
    """
    E2E regression: a Points layer created from config.yaml starts empty (placeholder).
    When the user begins adding keypoints, the layer must switch into categorical
    coloring (cycle mode) and colors must follow the derived bodypart policy.
    """
    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    from napari_deeplabcut.core import keypoints

    controls = get_existing_keypoint_controls(viewer)

    # 1) Open config.yaml -> creates placeholder Points layer (empty)
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # 2) Placeholder should carry the policy input we rely on
    md = placeholder.metadata or {}
    assert "config_colormap" in md, "Expected config_colormap in metadata for config.yaml placeholder layer"
    assert "header" in md, "Expected header in metadata for config.yaml placeholder layer"

    # 3) Begin editing: add bodypart1 then bodypart2
    store = controls.get_layer_store(placeholder)
    assert store is not None, "Expected KeypointStore to be registered for placeholder Points layer"

    # Add first point
    placeholder.add(np.array([0.0, 20.0, 10.0], dtype=float))
    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2000)

    # Wait for recolor to activate cycle mode
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5000)

    # Validate color matches derived policy for the actual stored label
    label0 = str(placeholder.properties["label"][0])
    expected_cycles = _cycles_from_policy(placeholder)
    expected0 = np.asarray(expected_cycles["label"][label0], dtype=float)
    c0 = np.asarray(placeholder._face.colors[0], dtype=float)
    assert np.allclose(c0, expected0, atol=1e-6), f"color mismatch for {label0!r}: got={c0}, expected={expected0}"

    # Ensure the second add targets a different bodypart.
    hdr = placeholder.metadata.get("header")
    assert hdr is not None, "Expected header in placeholder metadata"

    header_model = hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)
    all_bodyparts = list(header_model.bodyparts)
    assert all_bodyparts, "Header has no bodyparts; cannot drive second add deterministically."

    label_alt = next((bp for bp in all_bodyparts if str(bp) != label0), None)
    assert label_alt is not None, f"Only one bodypart present; cannot add a second distinct keypoint. label0={label0!r}"

    placeholder.selected_data = set()
    store.current_keypoint = keypoints.Keypoint(str(label_alt), "")

    placeholder.add(np.array([0.0, 33.0, 44.0], dtype=float))
    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 2, timeout=2000)

    label1 = str(placeholder.properties["label"][1])
    expected1 = np.asarray(expected_cycles["label"][label1], dtype=float)
    c1 = np.asarray(placeholder._face.colors[1], dtype=float)
    assert np.allclose(c1, expected1, atol=1e-6), f"color mismatch for {label1!r}: got={c1}, expected={expected1}"

    assert label0 != label1, f"Expected successive adds to label different keypoints, got {label0!r} then {label1!r}"
    assert not np.allclose(c0, c1, atol=1e-6), "Expected distinct colors for different labels in cycle mode"


@pytest.mark.usefixtures("qtbot")
def test_config_placeholder_multianimal_colors_by_id_after_first_keypoint_added(
    viewer,
    qtbot,
    multianimal_config_project,
):
    """
    E2E regression: a Points layer created from a multi-animal config.yaml starts empty.
    When the user adds keypoints, the layer must switch into categorical coloring
    (cycle mode) and, in multi-animal mode, color by id according to the derived policy.
    """
    _, config_path = multianimal_config_project

    from napari_deeplabcut.core import keypoints

    controls = get_existing_keypoint_controls(viewer)

    # 1) Open config.yaml -> empty placeholder Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # 2) Must carry policy inputs; exact cycles are derived, not metadata-authored
    md = placeholder.metadata or {}
    assert "config_colormap" in md, "Expected config_colormap in metadata"
    assert "header" in md, "Expected header in metadata"

    expected_cycles = _cycles_from_policy(placeholder)
    id_cycles = expected_cycles["id"]
    assert "animal1" in id_cycles, f"Expected 'animal1' in derived id cycles; got keys={list(id_cycles)[:10]}"
    assert "animal2" in id_cycles, f"Expected 'animal2' in derived id cycles; got keys={list(id_cycles)[:10]}"

    # 3) Begin editing: add a point for animal1, then animal2
    store = controls.get_layer_store(placeholder)
    assert store is not None, "Expected KeypointStore for placeholder Points layer"

    # Add first point: (frame, y, x)
    store.current_keypoint = keypoints.Keypoint("bodypart1", "animal1")
    placeholder.add(np.array([0.0, 12.0, 34.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)

    # Wait for recolor timer to switch to cycle mode
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    # Must be coloring by id in multi-animal case
    assert placeholder._face.color_properties.name == "id"

    got0 = np.asarray(placeholder._face.colors[0], dtype=float)
    exp0 = np.asarray(id_cycles["animal1"], dtype=float)
    assert np.allclose(got0, exp0, atol=1e-6), f"animal1 color mismatch: got={got0}, expected={exp0}"

    # Add second point for animal2
    store.current_keypoint = keypoints.Keypoint("bodypart2", "animal2")
    placeholder.add(np.array([0.0, 56.0, 78.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 2, timeout=2_000)
    qtbot.wait(50)  # small buffer for color refresh

    assert placeholder.face_color_mode == "cycle"
    assert placeholder._face.color_properties.name == "id"

    got1 = np.asarray(placeholder._face.colors[1], dtype=float)
    exp1 = np.asarray(id_cycles["animal2"], dtype=float)
    assert np.allclose(got1, exp1, atol=1e-6), f"animal2 color mismatch: got={got1}, expected={exp1}"

    assert not np.allclose(got0, got1, atol=1e-6), "Expected distinct colors for animal1 vs animal2"


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_resolver_single_animal_active_then_config_bodyparts(
    viewer,
    qtbot,
    tmp_path,
):
    """
    Integration test at the resolver level (not widget visibility/lifecycle):

    - open config first -> placeholder points layer
    - add one visible keypoint to placeholder
    - resolver in ACTIVE mode should show only the visible/current keypoint(s)
    - resolver in CONFIG mode should show all bodyparts from config/header policy
    """
    project, config_path, labeled_folder, _h5_path = _make_minimal_dlc_project(tmp_path)

    from napari_deeplabcut.core import keypoints

    controls = get_existing_keypoint_controls(viewer)

    # Open config -> placeholder Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # Wait until the controls instance has wired the layer store
    qtbot.waitUntil(lambda: controls.get_layer_store(placeholder) is not None, timeout=5_000)
    store = controls.get_layer_store(placeholder)
    assert store is not None

    # Make the placeholder the active target layer
    viewer.layers.selection.active = placeholder

    # Add one visible keypoint
    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    placeholder.add(np.array([0.0, 20.0, 10.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    panel = controls._color_scheme_panel
    resolver = panel._resolver

    # ACTIVE mode: assert against the actual visible category on the layer,
    # not the category we attempted to set before add().
    active_prop = resolver.get_active_color_property(placeholder)
    assert active_prop == "label"

    active_name = str(np.asarray(placeholder.properties[active_prop], dtype=object).ravel()[0])
    expected_active = _scheme_from_policy(placeholder, active_prop, [active_name])
    actual_active = resolver.resolve(show_config_keypoints=False)
    assert actual_active == expected_active

    # CONFIG mode: assert against all configured bodyparts from the resolved header policy.
    hdr = placeholder.metadata.get("header")
    assert hdr is not None, "Expected header in placeholder metadata"
    header_model = hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)
    config_names = [str(x) for x in header_model.bodyparts]

    expected_config = _scheme_from_policy(placeholder, "label", config_names)
    actual_config = resolver.resolve(show_config_keypoints=True)
    assert actual_config == expected_config


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_resolver_multianimal_active_then_config_individuals(
    viewer,
    qtbot,
    multianimal_config_project,
):
    """
    Integration test at the resolver level (not widget visibility/lifecycle):

    - open multi-animal config first -> placeholder points layer
    - add one keypoint for animal1
    - resolver in ACTIVE mode should show only currently visible active individual(s)
    - resolver in CONFIG mode should show all configured individuals from config/header policy
    """
    _, config_path = multianimal_config_project

    from napari_deeplabcut.core import keypoints

    controls = get_existing_keypoint_controls(viewer)

    # Open config -> placeholder Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(
        lambda: any(isinstance(ly, Points) for ly in viewer.layers),
        timeout=5_000,
    )

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # Wait until the existing controls instance has wired the layer
    qtbot.waitUntil(lambda: controls.get_layer_store(placeholder) is not None, timeout=5_000)
    store = controls.get_layer_store(placeholder)
    assert store is not None

    # Multi-animal setup should flip the controls to individual mode
    qtbot.waitUntil(lambda: controls.color_mode == "individual", timeout=5_000)
    assert controls.color_mode == "individual"

    # Make the placeholder the active target layer
    viewer.layers.selection.active = placeholder

    # Add one keypoint for animal1
    store.current_keypoint = keypoints.Keypoint("bodypart1", "animal1")
    placeholder.add(np.array([0.0, 12.0, 34.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)
    qtbot.waitUntil(lambda: placeholder._face.color_properties.name == "id", timeout=5_000)

    panel = controls._color_scheme_panel
    resolver = panel._resolver

    expected_active = _scheme_from_policy(placeholder, "id", ["animal1"])
    actual_active = resolver.resolve(show_config_keypoints=False)
    assert actual_active == expected_active

    expected_config = _scheme_from_policy(placeholder, "id", ["animal1", "animal2"])
    actual_config = resolver.resolve(show_config_keypoints=True)
    assert actual_config == expected_config


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_panel_update_scheme_pushes_resolver_to_display_without_force_show(
    viewer,
    qtbot,
    tmp_path,
    monkeypatch,
):
    """
    Small smoke test for the panel glue:

    - do not show/force-show the widget tree
    - bypass visibility gating
    - call update_scheme() directly
    - assert the display receives the resolver output

    This keeps a tiny amount of panel coverage without depending on dock/widget
    visibility or QTimer-driven lifecycle churn.
    """
    project, config_path, labeled_folder, _h5_path = _make_minimal_dlc_project(tmp_path)

    from napari_deeplabcut.core import keypoints

    controls = get_existing_keypoint_controls(viewer)

    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None

    qtbot.waitUntil(lambda: controls.get_layer_store(placeholder) is not None, timeout=5_000)
    store = controls.get_layer_store(placeholder)
    assert store is not None

    viewer.layers.selection.active = placeholder

    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    placeholder.add(np.array([0.0, 20.0, 10.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    panel = controls._color_scheme_panel

    # Avoid real widget visibility/show churn in CI.
    monkeypatch.setattr(panel, "_is_effectively_visible", lambda: True)

    resolver = panel._resolver
    active_prop = resolver.get_active_color_property(placeholder)
    assert active_prop == "label"

    active_name = str(np.asarray(placeholder.properties[active_prop], dtype=object).ravel()[0])
    expected_active = _scheme_from_policy(placeholder, active_prop, [active_name])

    panel.update_scheme()
    assert panel.display.scheme_dict == expected_active

    # Flip to config preview and apply directly (no need to rely on the timer path)
    panel._toggle.blockSignals(True)
    panel._toggle.setChecked(True)
    panel._toggle.blockSignals(False)

    hdr = placeholder.metadata.get("header")
    assert hdr is not None, "Expected header in placeholder metadata"
    header_model = hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)
    expected_config_names = [str(x) for x in header_model.bodyparts]
    expected_config = _scheme_from_policy(placeholder, active_prop, expected_config_names)
    panel.update_scheme()
    assert panel.display.scheme_dict == expected_config
