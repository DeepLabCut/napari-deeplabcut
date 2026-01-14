import os

import numpy as np
from vispy import keys

from napari_deeplabcut import _widgets


def test_guess_continuous():
    # Hack: guess_continuous overrides napari's default logic to avoid misclassifying categorical properties
    assert _widgets.guess_continuous(np.array([0.0]))  # Floats → continuous
    assert not _widgets.guess_continuous(np.array(list("abc")))  # Strings → categorical


def test_keypoint_controls(viewer, qtbot):
    controls = _widgets.KeypointControls(viewer)
    controls.label_mode = "loop"
    assert controls._radio_group.checkedButton().text() == "Loop"
    controls.cycle_through_label_modes()
    assert controls._radio_group.checkedButton().text() == "Sequential"


def test_save_layers(viewer, points):
    controls = _widgets.KeypointControls(viewer)
    viewer.layers.selection.add(points)
    # _save_layers_dialog bypasses napari's Save dialog for Points layers (used in headless tests)
    _widgets._save_layers_dialog(controls)


def test_show_trails(viewer, store):
    controls = _widgets.KeypointControls(viewer)
    controls._stores["temp"] = store
    controls._is_saved = True
    controls._show_trails(state=2)


def test_extract_single_frame(viewer, images):
    viewer.layers.selection.add(images)
    controls = _widgets.KeypointControls(viewer)
    controls._extract_single_frame()


def test_store_crop_coordinates(viewer, images, config_path):
    viewer.layers.selection.add(images)
    _ = viewer.add_shapes(np.random.random((4, 3)), shape_type="rectangle")
    controls = _widgets.KeypointControls(viewer)
    controls._images_meta = {
        "name": "fake_video",
        "project": os.path.dirname(config_path),
    }
    # Stores crop coordinates from a rectangle shape into the project's config.yaml
    controls._store_crop_coordinates()


def test_toggle_face_color(viewer, points):
    viewer.layers.selection.add(points)
    view = viewer.window._qt_viewer
    # Shortcut 'F' toggles coloring between "id" and "label" for multi-animal datasets
    assert points._face.color_properties.name == "id"
    view.canvas.events.key_press(key=keys.Key("F"))
    assert points._face.color_properties.name == "label"
    view.canvas.events.key_press(key=keys.Key("F"))
    assert points._face.color_properties.name == "id"


def test_toggle_edge_color(viewer, points):
    viewer.layers.selection.add(points)
    view = viewer.window._qt_viewer
    # Shortcut 'E' toggles border width between 0 and 2
    np.testing.assert_array_equal(points.border_width, 0)
    view.canvas.events.key_press(key=keys.Key("E"))
    np.testing.assert_array_equal(points.border_width, 2)


def test_dropdown_menu(qtbot):
    widget = _widgets.DropdownMenu(list("abc"))
    qtbot.add_widget(widget)
    widget.update_to("c")  # Ensure update_to selects the correct item
    assert widget.currentText() == "c"
    widget.reset()  # Reset should always select the first item
    assert widget.currentText() == "a"


def test_keypoints_dropdown_menu_selection_updates_store(store, qtbot):
    widget = _widgets.KeypointsDropdownMenu(store)
    qtbot.add_widget(widget)
    id_menu = widget.menus.get("id")
    label_menu = widget.menus["label"]

    # If multi-animal, switch ID and ensure the store's id updates
    if id_menu and id_menu.count() > 1:
        id_menu.setCurrentIndex(1)
        assert store.current_id == id_menu.currentText()

    # Switch label and ensure the store's label updates
    if label_menu.count() > 1:
        label_menu.setCurrentIndex(1)
        assert store.current_label == label_menu.currentText()


def test_keypoints_dropdown_menu_single_animal_has_no_id_menu(single_animal_store, qtbot):
    widget = _widgets.KeypointsDropdownMenu(single_animal_store)
    qtbot.add_widget(widget)
    assert "id" not in widget.menus
    assert "label" in widget.menus
    assert widget.menus["label"].count() > 0


def test_keypoints_dropdown_menu(store, qtbot):
    widget = _widgets.KeypointsDropdownMenu(store)
    qtbot.add_widget(widget)
    # Menus for both "id" and "label" should exist; label menu reflects current keypoint
    # This confirms we have multi-animal data
    id_menu = widget.menus["id"]
    label_menu = widget.menus["label"]
    # Baseline: labels for the first ID
    first_id = store.ids[0]
    expected_labels_first = widget.id2label[first_id]
    assert [label_menu.itemText(i) for i in range(label_menu.count())] == expected_labels_first

    # Switch to a different valid ID (if present) and ensure labels update accordingly
    if len(store.ids) > 1:
        # Change selection in the actual menu; this triggers refresh_label_menu via signal
        id_menu.setCurrentIndex(1)
        second_id = id_menu.currentText()
        expected_labels_second = widget.id2label[second_id]
        assert [label_menu.itemText(i) for i in range(label_menu.count())] == expected_labels_second


def test_keypoints_dropdown_menu_unknown_id_yields_empty_list(store):
    # If an invalid ID is selected, the label menu should be empty
    widget = _widgets.KeypointsDropdownMenu(store)
    label_menu = widget.menus["label"]
    widget.refresh_label_menu("__NON_EXISTENT_ID__")
    assert label_menu.count() == 0  # defaultdict(list) → no labels


def test_keypoints_dropdown_menu_updates_from_store_current_properties(store, qtbot):
    widget = _widgets.KeypointsDropdownMenu(store)
    qtbot.add_widget(widget)
    id_menu = widget.menus.get("id")
    label_menu = widget.menus["label"]

    # Pick a valid keypoint (label/id pair) and set it as current
    target = store._keypoints[min(2, len(store._keypoints) - 1)]
    store.current_keypoint = target

    # Simulate event callback
    widget.update_menus(event=None)

    if id_menu:
        assert id_menu.currentText() == target.id
    assert label_menu.currentText() == target.label


def test_keypoints_dropdown_menu_smart_reset(store, qtbot):
    widget = _widgets.KeypointsDropdownMenu(store)
    qtbot.add_widget(widget)
    label_menu = widget.menus["label"]
    label_menu.update_to("kpt_2")
    widget._locked = True
    widget.smart_reset(event=None)
    # Locked state prevents reset; current selection remains unchanged
    assert label_menu.currentText() == "kpt_2"
    widget._locked = False
    # Unlocked: smart_reset picks the first unlabeled keypoint (or defaults to first)
    widget.smart_reset(event=None)
    assert label_menu.currentText() == "kpt_0"


def test_color_pair(qtbot):
    pair = _widgets.LabelPair(color="pink", name="kpt", parent=None)
    qtbot.add_widget(pair)
    # LabelPair couples a color swatch with a clickable label
    # Ensure setters update both UI and tooltip
    assert pair.part_name == "kpt"
    assert pair.color == "pink"
    pair.color = "orange"
    pair.part_name = "kpt2"
    assert pair.color_label.toolTip() == "kpt2"


def test_color_scheme_display(qtbot):
    widget = _widgets.ColorSchemeDisplay(None)
    qtbot.add_widget(widget)
    widget._build()
    # Initially empty: no color scheme entries and no layout widgets
    assert not widget.scheme_dict
    assert not widget._container.layout().count()
    widget.add_entry("keypoint", "red")
    assert widget.scheme_dict["keypoint"] == "red"
    assert widget._container.layout().count() == 1
