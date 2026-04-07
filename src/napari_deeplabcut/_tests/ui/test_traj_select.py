from __future__ import annotations

import numpy as np
import pytest

from napari_deeplabcut.ui.plots.trajectory import TrajectoryMatplotlibCanvas


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_sync_visible_lines_to_points_selection_shows_all_when_no_points_selected(viewer, qtbot):
    layer = viewer.add_points(
        np.array([[0, 0], [1, 1]]),
        properties={"label": np.array(["nose", "tail"], dtype=object)},
    )

    canvas = TrajectoryMatplotlibCanvas(viewer)
    qtbot.add_widget(canvas)

    # Avoid relying on df creation for this focused visibility test
    canvas.df = object()
    (line_nose,) = canvas.ax.plot([0, 1], [0, 1])
    (line_tail,) = canvas.ax.plot([0, 1], [1, 0])
    qtbot.wait(0)  # ensure lines are fully initialized
    canvas._lines = {
        "nose": [line_nose],
        "tail": [line_tail],
    }

    layer.selected_data.clear()
    canvas.sync_visible_lines_to_points_selection()

    assert line_nose.get_visible() is True
    assert line_tail.get_visible() is True


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_sync_visible_lines_to_points_selection_filters_by_selected_labels(viewer, qtbot):
    layer = viewer.add_points(
        np.array([[0, 0], [1, 1], [2, 2]]),
        properties={"label": np.array(["nose", "tail", "nose"], dtype=object)},
    )

    canvas = TrajectoryMatplotlibCanvas(viewer)
    qtbot.add_widget(canvas)

    canvas.df = object()
    (line_nose,) = canvas.ax.plot([0, 1], [0, 1])
    (line_tail,) = canvas.ax.plot([0, 1], [1, 0])
    qtbot.wait(0)  # ensure lines are fully initialized
    canvas._lines = {
        "nose": [line_nose],
        "tail": [line_tail],
    }

    # Select a point whose label is "tail"
    layer.selected_data.select_only(1)
    canvas.sync_visible_lines_to_points_selection()

    assert line_nose.get_visible() is False
    assert line_tail.get_visible() is True


@pytest.mark.e2e
@pytest.mark.usefixtures("qtbot")
def test_sync_visible_lines_to_points_selection_shows_label_if_any_selected_point_has_that_label(viewer, qtbot):
    layer = viewer.add_points(
        np.array([[0, 0], [1, 1], [2, 2]]),
        properties={"label": np.array(["nose", "tail", "nose"], dtype=object)},
    )

    canvas = TrajectoryMatplotlibCanvas(viewer)
    qtbot.add_widget(canvas)

    canvas.df = object()
    (line_nose,) = canvas.ax.plot([0, 1], [0, 1])
    (line_tail,) = canvas.ax.plot([0, 1], [1, 0])
    qtbot.wait(0)  # ensure lines are fully initialized
    canvas._lines = {
        "nose": [line_nose],
        "tail": [line_tail],
    }

    # Select both nose points
    layer.selected_data.update({0, 2})
    canvas.sync_visible_lines_to_points_selection()

    assert line_nose.get_visible() is True
    assert line_tail.get_visible() is False
