from napari_deeplabcut import _widgets


def test_keypoint_controls(viewer):
    controls = _widgets.KeypointControls(viewer)
    controls.label_mode = "loop"
    assert controls._radio_group.checkedButton().text() == "loop"


def test_dropdown_menu(qtbot):
    widget = _widgets.DropdownMenu(list("abc"))
    widget.update_to("c")
    assert widget.currentText() == "c"
    widget.reset()
    assert widget.currentText() == "a"
    qtbot.add_widget(widget)


def test_keypoints_dropdown_menu(store):
    widget = _widgets.KeypointsDropdownMenu(store)
    assert "id" not in widget.menus
    assert "label" in widget.menus
    label_menu = widget.menus['label']
    label_menu.currentText() == "kpt_0"
    widget.update_menus(event=None)
    label_menu.currentText() == "kpt_2"
    widget.refresh_label_menu("id_0")
    assert label_menu.count() == 0


def test_color_scheme_display(qtbot):
    widget = _widgets.ColorSchemeDisplay(None)
    widget._build()
    assert not widget.scheme_dict
    assert not widget._container.layout().count()
    widget.add_entry("keypoint", "red")
    assert widget.scheme_dict["keypoint"] == "red"
    assert widget._container.layout().count() == 1
    qtbot.add_widget(widget)