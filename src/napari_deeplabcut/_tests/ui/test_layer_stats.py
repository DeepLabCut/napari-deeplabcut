from napari_deeplabcut.ui.layer_stats import LayerStatusPanel


def test_set_invalid_points_layer_disables_slider_and_updates_text(qtbot):
    panel = LayerStatusPanel()
    qtbot.addWidget(panel)

    panel.set_invalid_points_layer()

    assert panel._progress_value.text() == "Active layer is not a DLC keypoints layer"
    assert not panel._size_slider.isEnabled()
    assert not panel._size_value.isEnabled()


def test_set_no_active_points_layer_disables_slider_and_value_label(qtbot):
    panel = LayerStatusPanel()
    qtbot.addWidget(panel)

    panel.set_point_size_enabled(True)
    assert panel._size_slider.isEnabled()
    assert panel._size_value.isEnabled()

    panel.set_no_active_points_layer()

    assert panel._progress_value.text() == "No active keypoints layer"
    assert not panel._size_slider.isEnabled()
    assert not panel._size_value.isEnabled()
