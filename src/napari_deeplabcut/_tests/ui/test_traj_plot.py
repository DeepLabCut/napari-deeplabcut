from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.ui.plots.plot_models import TrajectoryPlotState
from napari_deeplabcut.ui.plots.trajectory import TrajectoryMatplotlibCanvas


class DummyTrajectoryLayerManager:
    def __init__(self, viewer, *, plottable_layer=None, suggested_layer=None):
        self.viewer = viewer
        self.plottable_layer = plottable_layer
        self.suggested_layer = suggested_layer

    def _same_layer(self, a, b):
        if a is None or b is None:
            return False
        return getattr(a, "name", None) == getattr(b, "name", None)

    def is_plottable_traj_layer(self, layer):
        return isinstance(layer, Points) and self._same_layer(layer, self.plottable_layer)

    def active_plottable_traj_layer(self):
        active = getattr(self.viewer.layers.selection, "active", None)
        return active if self.is_plottable_traj_layer(active) else None

    def suggest_plottable_traj_layer(self):
        return self.suggested_layer

    def is_tracking_result_layer(self, layer):
        return False


def _make_canvas(viewer, qtbot, *, layer_manager=None, get_color_mode=None):
    canvas = TrajectoryMatplotlibCanvas(
        viewer,
        layer_manager=layer_manager,
        get_color_mode=get_color_mode,
    )
    qtbot.addWidget(canvas)

    # Keep tests deterministic: do not let the initial deferred refresh run.
    canvas._cancel_scheduled("initial_trajectory_refresh")

    return canvas


def _select_layer(viewer, layer):
    viewer.layers.selection.clear()
    viewer.layers.selection.add(layer)
    viewer.layers.selection.active = layer


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_get_plot_points_layer_strict_does_not_fallback_when_active_is_none(viewer, qtbot):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    viewer.layers.selection.clear()

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=points,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    assert viewer.layers.selection.active is None
    assert canvas._get_plot_points_layer(allow_fallback=False) is None


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_get_plot_points_layer_allows_fallback_for_explicit_initialization(viewer, qtbot):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    viewer.layers.selection.clear()

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=points,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    assert canvas._get_plot_points_layer(allow_fallback=True) is points


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_get_plot_points_layer_strict_returns_active_plottable_layer(viewer, qtbot):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=None,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    _select_layer(viewer, points)

    result = canvas._get_plot_points_layer(allow_fallback=False)

    assert result is not None
    assert isinstance(result, type(points))
    assert result.name == points.name


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_get_plot_points_layer_rejects_suggested_layer_not_in_viewer(viewer, qtbot):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=points,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    viewer.layers.remove(points)

    assert canvas._get_plot_points_layer(allow_fallback=True) is None


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_refresh_from_viewer_layers_strict_does_not_build_from_fallback_layer(viewer, qtbot, monkeypatch):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    viewer.layers.selection.clear()

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=points,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    def _fail_if_called(layer):
        raise AssertionError("Trajectory plot should not build from fallback layer in strict mode")

    monkeypatch.setattr(canvas, "_build_plot_state", _fail_if_called)

    canvas.refresh_from_viewer_layers(allow_fallback=False)

    assert canvas._plot_state is None
    assert canvas._plot_layer is None
    assert canvas._lines == {}


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_refresh_from_viewer_layers_allows_fallback_for_explicit_initialization(viewer, qtbot, monkeypatch):
    points = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    viewer.layers.selection.clear()

    manager = DummyTrajectoryLayerManager(
        viewer,
        plottable_layer=points,
        suggested_layer=points,
    )
    canvas = _make_canvas(viewer, qtbot, layer_manager=manager)

    calls = []

    def _fake_build_plot_state(layer):
        calls.append(layer)

        df = pd.DataFrame(index=[0, 1])
        state = TrajectoryPlotState(
            df=df,
            series=(),
            frame_min=0.0,
            frame_max=1.0,
            image_height=None,
        )
        return df, state

    monkeypatch.setattr(canvas, "_build_plot_state", _fake_build_plot_state)

    canvas.refresh_from_viewer_layers(allow_fallback=True)

    assert calls == [points]
    assert canvas._plot_state is not None
    assert canvas._plot_layer is points


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_selected_line_keys_uses_strict_layer_resolution(viewer, qtbot, monkeypatch):
    canvas = _make_canvas(viewer, qtbot)

    calls = []

    def _fake_get_plot_points_layer(*, allow_fallback=False):
        calls.append(allow_fallback)
        return None

    monkeypatch.setattr(canvas, "_get_plot_points_layer", _fake_get_plot_points_layer)

    assert canvas._selected_line_keys_from_points_layer() == set()
    assert calls == [False]
