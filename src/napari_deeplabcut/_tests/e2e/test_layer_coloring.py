from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel

from .utils import _make_minimal_dlc_project, _scheme_from_cycle


@pytest.mark.usefixtures("qtbot")
def test_config_placeholder_points_layer_colors_after_first_keypoint_added(make_napari_viewer, qtbot, tmp_path):
    """
    E2E regression: a Points layer created from config.yaml starts empty (placeholder).
    When the user begins adding keypoints, the layer must switch into categorical
    coloring (cycle mode) and colors must match face_color_cycles from metadata.
    """
    project, config_path, labeled_folder, h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()
    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # 1) Open config.yaml -> creates placeholder Points layer (empty)
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # 2) Must carry the cycles we rely on
    md = placeholder.metadata or {}
    assert "face_color_cycles" in md, "Expected face_color_cycles in metadata for config.yaml placeholder layer"
    assert "colormap_name" in md or "colormap" in md, "Expected a colormap name in metadata"
    assert "label" in md["face_color_cycles"], "Expected label cycles in metadata"

    # We'll check that cycle keys include our bodyparts
    label_cycles = md["face_color_cycles"]["label"]
    assert "bodypart1" in label_cycles, f"Expected bodypart1 in label cycles; got keys={list(label_cycles)[:5]}"
    assert "bodypart2" in label_cycles, f"Expected bodypart2 in label cycles; got keys={list(label_cycles)[:5]}"

    # 3) Begin editing: add bodypart1 then bodypart2
    store = controls._stores.get(placeholder)
    assert store is not None, "Expected KeypointStore to be registered for placeholder Points layer"

    # Add first point (we don't assume which keypoint is active)
    placeholder.add(np.array([0.0, 20.0, 10.0], dtype=float))
    qtbot.waitUntil(lambda: (placeholder.data is not None and len(placeholder.data) == 1), timeout=2000)

    # Wait for recolor to activate cycle mode
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5000)

    # Validate color matches metadata for the *actual label stored*
    label0 = str(placeholder.properties["label"][0])
    expected0 = np.asarray(placeholder.metadata["face_color_cycles"]["label"][label0], dtype=float)
    c0 = np.asarray(placeholder._face.colors[0], dtype=float)
    assert np.allclose(c0, expected0, atol=1e-6), f"color mismatch for {label0!r}: got={c0}, expected={expected0}"

    # Ensure the second add targets a different (unannotated) bodypart.

    # Find a different bodypart from the header ordering
    hdr = placeholder.metadata.get("header")
    assert hdr is not None, "Expected header in placeholder metadata"

    header_model = hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)
    all_bodyparts = list(header_model.bodyparts)
    assert all_bodyparts, "Header has no bodyparts; cannot drive second add deterministically."

    # Pick a different bodypart than label0
    label_alt = next((bp for bp in all_bodyparts if str(bp) != label0), None)
    assert label_alt is not None, f"Only one bodypart present; cannot add a second distinct keypoint. label0={label0!r}"

    # Clear selection so KeypointStore.current_keypoint setter applies
    placeholder.selected_data = set()
    store.current_keypoint = keypoints.Keypoint(str(label_alt), "")

    # Add second point; it should generally advance to the next keypoint
    placeholder.add(np.array([0.0, 33.0, 44.0], dtype=float))
    qtbot.waitUntil(lambda: (placeholder.data is not None and len(placeholder.data) == 2), timeout=2000)

    label1 = str(placeholder.properties["label"][1])
    expected1 = np.asarray(placeholder.metadata["face_color_cycles"]["label"][label1], dtype=float)
    c1 = np.asarray(placeholder._face.colors[1], dtype=float)
    assert np.allclose(c1, expected1, atol=1e-6), f"color mismatch for {label1!r}: got={c1}, expected={expected1}"

    # Optional sanity: if header has >=2 bodyparts, these should differ
    assert label0 != label1, f"Expected successive adds to label different keypoints, got {label0!r} then {label1!r}"
    assert not np.allclose(c0, c1, atol=1e-6), "Expected distinct colors for different labels in cycle mode"


@pytest.mark.usefixtures("qtbot")
def test_config_placeholder_multianimal_colors_by_id_after_first_keypoint_added(
    make_napari_viewer,
    qtbot,
    multianimal_config_project,
):
    """
    E2E regression: a Points layer created from a multi-animal config.yaml starts empty.
    When the user adds keypoints, the layer must switch into categorical coloring
    (cycle mode) and (in multi-animal mode) color by "id" using face_color_cycles["id"].
    """
    _, config_path = multianimal_config_project

    viewer = make_napari_viewer()
    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # 1) Open config.yaml -> empty placeholder Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # 2) Must have cycles for both label and id; id cycles must include individuals
    md = placeholder.metadata or {}
    assert "face_color_cycles" in md, "Expected face_color_cycles in metadata"
    cycles = md["face_color_cycles"]

    assert "id" in cycles, f"Expected id cycles in metadata; got keys={list(cycles.keys())}"
    id_cycles = cycles["id"]
    assert "animal1" in id_cycles, f"Expected 'animal1' in id cycles; got keys={list(id_cycles)[:10]}"
    assert "animal2" in id_cycles, f"Expected 'animal2' in id cycles; got keys={list(id_cycles)[:10]}"

    # 3) Begin editing: add a point for animal1, then animal2
    store = controls._stores.get(placeholder)
    assert store is not None, "Expected KeypointStore for placeholder Points layer"

    # Add first point: (frame, y, x)
    store.current_keypoint = keypoints.Keypoint("bodypart1", "animal1")
    placeholder.add(np.array([0.0, 12.0, 34.0], dtype=float))

    qtbot.waitUntil(lambda: (placeholder.data is not None and len(placeholder.data) == 1), timeout=2_000)

    # Wait for recolor timer to switch to cycle mode (singleShot(0))
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    # Must be coloring by id in multi-animal case
    assert placeholder._face.color_properties.name == "id"

    # Color must match the id cycle mapping for animal1
    got0 = np.asarray(placeholder._face.colors[0], dtype=float)
    exp0 = np.asarray(id_cycles["animal1"], dtype=float)
    assert np.allclose(got0, exp0, atol=1e-6), f"animal1 color mismatch: got={got0}, expected={exp0}"

    # Add second point for animal2
    store.current_keypoint = keypoints.Keypoint("bodypart2", "animal2")
    placeholder.add(np.array([0.0, 56.0, 78.0], dtype=float))

    qtbot.waitUntil(lambda: (placeholder.data is not None and len(placeholder.data) == 2), timeout=2_000)
    qtbot.wait(50)  # small buffer for color refresh

    assert placeholder.face_color_mode == "cycle"
    assert placeholder._face.color_properties.name == "id"

    got1 = np.asarray(placeholder._face.colors[1], dtype=float)
    exp1 = np.asarray(id_cycles["animal2"], dtype=float)
    assert np.allclose(got1, exp1, atol=1e-6), f"animal2 color mismatch: got={got1}, expected={exp1}"

    # Sanity check: different ids should have different colors in the cycle map
    assert not np.allclose(got0, got1, atol=1e-6), "Expected distinct colors for animal1 vs animal2"


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_panel_toggle_shows_active_then_full_config_bodyparts(
    make_napari_viewer,
    qtbot,
    tmp_path,
):
    """
    E2E:
    - open config first -> placeholder points layer
    - open dataset folder for context
    - add one visible keypoint to placeholder
    - color scheme panel (unchecked) should show only the active/current visible keypoint(s)
    - toggling config preview should show all bodyparts from config.yaml
    """
    project, config_path, labeled_folder, _h5_path = _make_minimal_dlc_project(tmp_path)

    viewer = make_napari_viewer()

    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # 1) Open config first -> placeholder Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    # 2) Open folder so image/dataset context exists
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    qtbot.wait(200)

    # Make sure the placeholder is the active target layer
    viewer.layers.selection.active = placeholder

    store = controls._stores.get(placeholder)
    assert store is not None

    # Deterministically add bodypart1
    store.current_keypoint = keypoints.Keypoint("bodypart1", "")
    placeholder.add(np.array([0.0, 20.0, 10.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    panel = controls._color_scheme_panel

    expected_active = _scheme_from_cycle(placeholder, "label", ["bodypart1"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_active, timeout=5_000)

    # Toggle full config preview
    panel._toggle.setChecked(True)

    expected_config = _scheme_from_cycle(placeholder, "label", ["bodypart1", "bodypart2"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_config, timeout=5_000)


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_panel_multianimal_toggle_shows_active_then_full_config_individuals(
    make_napari_viewer,
    qtbot,
    multianimal_config_project,
):
    """
    E2E:
    - open multi-animal config first -> placeholder points layer
    - add one keypoint for animal1
    - because KeypointControls switches multi-animal coloring to individual mode,
      the color scheme panel should:
        * unchecked: show only currently visible active individual(s)
        * checked: show all configured individuals from config.yaml
    """
    _, config_path = multianimal_config_project

    viewer = make_napari_viewer()

    from napari_deeplabcut import keypoints
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)

    placeholder = next((ly for ly in viewer.layers if isinstance(ly, Points)), None)
    assert placeholder is not None
    assert placeholder.data is None or len(placeholder.data) == 0

    store = controls._stores.get(placeholder)
    assert store is not None

    # Multi-animal config should switch controls to individual mode on first wired store
    assert controls.color_mode == "individual"

    # Add one keypoint for animal1
    store.current_keypoint = keypoints.Keypoint("bodypart1", "animal1")
    placeholder.add(np.array([0.0, 12.0, 34.0], dtype=float))

    qtbot.waitUntil(lambda: placeholder.data is not None and len(placeholder.data) == 1, timeout=2_000)
    qtbot.waitUntil(lambda: placeholder.face_color_mode == "cycle", timeout=5_000)

    panel = controls._color_scheme_panel

    expected_active = _scheme_from_cycle(placeholder, "id", ["animal1"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_active, timeout=5_000)

    # Toggle full config preview -> should show both configured individuals
    panel._toggle.setChecked(True)

    expected_config = _scheme_from_cycle(placeholder, "id", ["animal1", "animal2"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_config, timeout=5_000)
