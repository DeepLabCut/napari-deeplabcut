"""Integration/smoke tests for napari API compatibility code.
Validate the overrides and monkeypatches don't crash and have the intended effect on napari layers"""
# src/napari_deeplabcut/_tests/compat/test_compat_integration.py

from __future__ import annotations

import numpy as np

from napari_deeplabcut.napari_compat import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
)


def _get_point_controls(viewer, layer):
    return viewer.window.qt_viewer.dockLayerControls.widget().widgets[layer]


def test_apply_points_layer_ui_tweaks_smoke_real_viewer(viewer, qtbot, dropdown_cls, plt_module):
    """Smoke test against a real napari viewer + real Points controls.

    This should fail if napari's private control wiring changed in a way that
    breaks our compat layer for supported versions.
    """
    layer = viewer.add_points(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        features={"label": ["a", "b"], "id": [1, 2]},
    )
    layer.metadata["colormap_name"] = "magma"

    # Ensure this layer is the active one so napari builds/shows its controls.
    viewer.layers.selection.active = layer

    qtbot.waitUntil(
        lambda: layer in viewer.window.qt_viewer.dockLayerControls.widget().widgets,
        timeout=3000,
    )

    point_controls = _get_point_controls(viewer, layer)
    before_rows = point_controls.layout().rowCount()

    selector = apply_points_layer_ui_tweaks(
        viewer,
        layer,
        dropdown_cls=dropdown_cls,
        plt_module=plt_module,
    )

    # True smoke-test expectation:
    # on supported napari versions, we expect the compat hook to really wire in.
    assert selector is not None
    assert selector.updated_to == "magma"
    assert point_controls.layout().rowCount() == before_rows + 1

    # These are the real private widgets our compat layer is supposed to hide.
    assert point_controls._face_color_control.face_color_edit.isHidden()
    assert point_controls._face_color_control.face_color_label.isHidden()
    assert point_controls._border_color_control.border_color_edit.isHidden()
    assert point_controls._border_color_control.border_color_edit_label.isHidden()
    assert point_controls._out_slice_checkbox_control.out_of_slice_checkbox.isHidden()
    assert point_controls._out_slice_checkbox_control.out_of_slice_checkbox_label.isHidden()


def test_install_add_wrapper_smoke_real_points_layer(viewer):
    """Smoke test that method rebinding works on a real napari Points layer."""
    layer = viewer.add_points(np.array([[0.0, 0.0]]))

    calls = []

    def add_impl(*args, **kwargs):
        calls.append((args, kwargs))
        return "added"

    def schedule_recolor(layer_obj):
        calls.append(layer_obj)

    install_add_wrapper(layer, add_impl=add_impl, schedule_recolor=schedule_recolor)

    # Bound method really installed on a real layer
    assert layer.add.__self__ is layer

    payload = np.array([[1.0, 2.0]])
    result = layer.add(payload, source="smoke-test")

    assert result == "added"

    add_args, add_kwargs = calls[0]
    np.testing.assert_array_equal(add_args[0], payload)
    assert add_kwargs == {"source": "smoke-test"}

    assert calls[1] is layer


def test_install_paste_patch_smoke_real_points_layer(viewer):
    """Smoke test that _paste_data can be rebound on a real napari Points layer."""
    layer = viewer.add_points(np.array([[0.0, 0.0]]))

    seen = []

    def paste_func(this):
        seen.append(this)

    install_paste_patch(layer, paste_func=paste_func)

    assert layer._paste_data.__self__ is layer

    layer._paste_data()

    assert seen == [layer]


def test_apply_points_layer_ui_tweaks_real_dropdown(qtbot):
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from qtpy.QtWidgets import QCheckBox, QFormLayout, QLabel, QLineEdit, QWidget

    from napari_deeplabcut.ui.labels_and_dropdown import DropdownMenu

    class FaceColorControl:
        def __init__(self):
            self.face_color_edit = QLineEdit()
            self.face_color_label = QLabel("face")

    class BorderColorControl:
        def __init__(self):
            self.border_color_edit = QLineEdit()
            self.border_color_edit_label = QLabel("border")

    class OutSliceControl:
        def __init__(self):
            self.out_of_slice_checkbox = QCheckBox("out")
            self.out_of_slice_checkbox_label = QLabel("out label")

    class PointControls(QWidget):
        def __init__(self):
            super().__init__()
            self._layout = QFormLayout(self)
            self.setLayout(self._layout)
            self._face_color_control = FaceColorControl()
            self._border_color_control = BorderColorControl()
            self._out_slice_checkbox_control = OutSliceControl()

        def layout(self):
            return self._layout

    layer = SimpleNamespace(metadata={"colormap_name": "magma"})
    point_controls = PointControls()
    qtbot.addWidget(point_controls)

    dock_layer_controls = SimpleNamespace(widget=lambda: SimpleNamespace(widgets={layer: point_controls}))
    SimpleNamespace(window=SimpleNamespace(qt_viewer=SimpleNamespace(dockLayerControls=dock_layer_controls)))

    selector = DropdownMenu(plt.colormaps, point_controls)
    assert selector is not None

    assert point_controls._face_color_control.face_color_edit.isHidden()
    assert point_controls._face_color_control.face_color_label.isHidden()
    assert point_controls._border_color_control.border_color_edit.isHidden()
    assert point_controls._border_color_control.border_color_edit_label.isHidden()
    assert point_controls._out_slice_checkbox_control.out_of_slice_checkbox.isHidden()
    assert point_controls._out_slice_checkbox_control.out_of_slice_checkbox_label.isHidden()
