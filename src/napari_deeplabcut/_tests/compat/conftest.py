from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)


class DummyDropdown(QWidget):
    """Simple QWidget-based dropdown stand-in for QFormLayout.addRow()."""

    def __init__(self, colormaps, parent=None):
        super().__init__(parent)
        self.colormaps = colormaps
        self.updated_to = None

    def update_to(self, value):
        self.updated_to = value


@pytest.fixture
def dropdown_cls():
    return DummyDropdown


@pytest.fixture
def plt_module():
    # minimal matplotlib-like object for compat function
    return SimpleNamespace(colormaps=["viridis", "magma", "plasma"])


class _FaceColorControl:
    def __init__(self):
        self.face_color_edit = QLineEdit()
        self.face_color_label = QLabel("face")


class _BorderColorControl:
    def __init__(self):
        self.border_color_edit = QLineEdit()
        self.border_color_edit_label = QLabel("border")


class _OutSliceCheckboxControl:
    def __init__(self):
        self.out_of_slice_checkbox = QCheckBox("out")
        self.out_of_slice_checkbox_label = QLabel("out label")


class DummyPointControls(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QFormLayout(self)
        self.setLayout(self._layout)

        self._face_color_control = _FaceColorControl()
        self._border_color_control = _BorderColorControl()
        self._out_slice_checkbox_control = _OutSliceCheckboxControl()

    def layout(self):
        return self._layout


@pytest.fixture
def ui_env(qtbot):
    """
    Minimal viewer/layer environment for apply_points_layer_ui_tweaks().
    Uses real Qt widgets and a tiny viewer-shaped object.
    """
    layer = object()

    point_controls = DummyPointControls()
    qtbot.addWidget(point_controls)

    dock_layer_controls = SimpleNamespace(widget=lambda: SimpleNamespace(widgets={layer: point_controls}))
    viewer = SimpleNamespace(window=SimpleNamespace(qt_viewer=SimpleNamespace(dockLayerControls=dock_layer_controls)))

    layer_obj = SimpleNamespace(metadata={"colormap_name": "magma"})

    plt_module = SimpleNamespace(colormaps=["viridis", "magma", "plasma"])

    return SimpleNamespace(
        viewer=viewer,
        layer_key=layer,
        layer_obj=layer_obj,
        point_controls=point_controls,
        dropdown_cls=DummyDropdown,
        plt_module=plt_module,
    )


@dataclass(frozen=True)
class Keypoint:
    label: object
    id: object


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class DummyFeatureTable:
    def __init__(self):
        self.appended = []

    def append(self, features):
        self.appended.append(features.copy())


class DummyText:
    def __init__(self):
        self.calls = []

    def _paste(self, **kwargs):
        self.calls.append(kwargs)


class DummyColorManager:
    def __init__(self):
        self.calls = []

    def _paste(self, **kwargs):
        self.calls.append(kwargs)


class DummyLayerForPaste:
    def __init__(self):
        # existing layer state
        self.data = np.array([[1, 2, 3]], dtype=float)
        self.shown = np.array([True], dtype=bool)
        self.size = np.array([5.0], dtype=float)
        self.symbol = np.array(["o"], dtype=object)
        self.edge_width = np.array([0.5], dtype=float)

        # these are the attrs mutated by make_paste_data()
        self._data = self.data.copy()
        self._shown = self.shown.copy()
        self._size = self.size.copy()
        self._symbol = self.symbol.copy()
        self._edge_width = self.edge_width.copy()

        self._view_data = np.array([[1, 2, 3]], dtype=float)

        self._slice_input = SimpleNamespace(not_displayed=[1])
        self._slice_indices = np.array([0, 7, 0])

        self._feature_table = DummyFeatureTable()
        self.text = DummyText()
        self._edge = DummyColorManager()
        self._face = DummyColorManager()

        self._selected_view = []
        self._selected_data = set()

        self.refresh_count = 0

        self._clipboard = {
            "features": pd.DataFrame(
                {
                    "label": ["nose", "tail"],
                    "id": [1, 2],
                }
            ),
            "indices": np.array([0, 5, 0]),
            "text": {
                "string": np.array(["nose-1", "tail-2"], dtype=object),
                "color": "white",
            },
            "data": np.array(
                [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                ]
            ),
            "shown": np.array([True, False], dtype=bool),
            "size": np.array([3.0, 4.0], dtype=float),
            "symbol": np.array(["x", "+"], dtype=object),
            "edge_width": np.array([1.0, 2.0], dtype=float),
            "edge_color": np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            ),
            "face_color": np.array(
                [
                    [0.5, 0.0, 0.0, 1.0],
                    [0.0, 0.5, 0.0, 1.0],
                ]
            ),
        }

    def refresh(self):
        self.refresh_count += 1


@pytest.fixture
def paste_env(monkeypatch):
    """
    Environment for make_paste_data().

    We monkeypatch the private napari helper import because compat code imports it
    inside the returned closure.
    """
    fake_layer_utils = types.ModuleType("napari.layers.utils.layer_utils")
    fake_layer_utils._features_to_properties = lambda features: {
        col: features[col].to_numpy() for col in features.columns
    }
    monkeypatch.setitem(sys.modules, "napari.layers.utils.layer_utils", fake_layer_utils)

    recolor_calls = []

    def schedule_recolor(layer):
        recolor_calls.append(layer)

    controls = SimpleNamespace(
        np=np,
        keypoints=SimpleNamespace(Keypoint=Keypoint),
        _schedule_recolor=schedule_recolor,
    )

    store_layer = object()
    store = SimpleNamespace(
        annotated_keypoints={Keypoint("nose", 1)},
        layer=store_layer,
    )

    layer = DummyLayerForPaste()

    return SimpleNamespace(
        controls=controls,
        store=store,
        layer=layer,
        recolor_calls=recolor_calls,
        Keypoint=Keypoint,
    )
