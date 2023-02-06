import numpy as np
import os
from napari_deeplabcut import _widgets
from vispy import keys


def test_guess_continuous():
    assert _widgets.guess_continuous(np.array([0.]))
    assert not _widgets.guess_continuous(np.array(list("abc")))


def test_keypoint_controls(viewer):
    controls = _widgets.KeypointControls(viewer)
    controls.label_mode = "loop"
    assert controls._radio_group.checkedButton().text() == "loop"
    controls.cycle_through_label_modes()
    assert controls._radio_group.checkedButton().text() == "sequential"


def test_save_layers(viewer, points):
    controls = _widgets.KeypointControls(viewer)
    viewer.layers.selection.add(points)
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
    _ = viewer.add_shapes(
        np.random.random((4, 3)),
        shape_type='rectangle',
    )
    controls = _widgets.KeypointControls(viewer)
    controls._images_meta = {
        "name": "fake_video",
        "project": os.path.dirname(config_path),
    }
    controls._store_crop_coordinates()


def test_toggle_face_color(viewer, points):
    viewer.layers.selection.add(points)
    view = viewer.window._qt_viewer
    # By default, points are colored by individual with multi-animal data
    assert points._face.color_properties.name == "id"
    view.canvas.events.key_press(key=keys.Key('F'))
    assert points._face.color_properties.name == "label"
    view.canvas.events.key_press(key=keys.Key('F'))
    assert points._face.color_properties.name == "id"


def test_toggle_edge_color(viewer, points):
    viewer.layers.selection.add(points)
    view = viewer.window._qt_viewer
    np.testing.assert_array_equal(points.edge_width, 0)
    view.canvas.events.key_press(key=keys.Key('E'))
    np.testing.assert_array_equal(points.edge_width, 2)


def test_dropdown_menu(qtbot):
    widget = _widgets.DropdownMenu(list("abc"))
    widget.update_to("c")
    assert widget.currentText() == "c"
    widget.reset()
    assert widget.currentText() == "a"
    qtbot.add_widget(widget)


def test_keypoints_dropdown_menu(store):
    widget = _widgets.KeypointsDropdownMenu(store)
    assert "id" in widget.menus
    assert "label" in widget.menus
    label_menu = widget.menus['label']
    label_menu.currentText() == "kpt_0"
    widget.update_menus(event=None)
    label_menu.currentText() == "kpt_2"
    widget.refresh_label_menu("id_0")
    assert label_menu.count() == 0


def test_keypoints_dropdown_menu_smart_reset(store):
    widget = _widgets.KeypointsDropdownMenu(store)
    label_menu = widget.menus['label']
    label_menu.update_to("kpt_2")
    widget._locked = True
    widget.smart_reset(event=None)
    assert label_menu.currentText() == "kpt_2"
    widget._locked = False
    widget.smart_reset(event=None)
    assert label_menu.currentText() == "kpt_0"


def test_color_pair():
    pair = _widgets.LabelPair(color="pink", name="kpt", parent=None)
    assert pair.part_name == "kpt"
    assert pair.color == "pink"
    pair.color = "orange"
    pair.part_name = "kpt2"
    assert pair.color_label.toolTip() == "kpt2"


def test_color_scheme_display(qtbot):
    widget = _widgets.ColorSchemeDisplay(None)
    widget._build()
    assert not widget.scheme_dict
    assert not widget._container.layout().count()
    widget.add_entry("keypoint", "red")
    assert widget.scheme_dict["keypoint"] == "red"
    assert widget._container.layout().count() == 1
    qtbot.add_widget(widget)