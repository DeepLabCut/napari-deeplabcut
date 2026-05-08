from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt

from napari_deeplabcut.tracking._widgets import TrackingControls
from napari_deeplabcut.tracking.core.data import TrackingWorkerData
from napari_deeplabcut.tracking.core.models import AVAILABLE_TRACKERS

if TYPE_CHECKING:
    import napari

_DUMMY_VIDEO_N_FRAMES = 10


def _get_tracking_controls(viewer: napari.Viewer) -> TrackingControls:
    """Create and retrieve the tracking controls dock widget for a viewer."""
    viewer.window.add_dock_widget(
        TrackingControls(viewer),
        name="Tracking controls",
        area="right",
    )
    for _title, wdg in viewer.window.dock_widgets.items():
        if isinstance(wdg, TrackingControls) and wdg.property("ndlc_tracking_controls"):
            return wdg
    raise RuntimeError("Tracking controls dock widget not found")


@pytest.fixture
def patch_tracking_side_effects(monkeypatch):
    """Disable plugin hooks that assume full DLC metadata / plot state."""
    monkeypatch.setattr(
        "napari_deeplabcut._widgets.KeypointControls.on_insert",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        "napari_deeplabcut.ui.plots.trajectory.TrajectoryMatplotlibCanvas._load_dataframe",
        lambda *args, **kwargs: None,
        raising=False,
    )


@pytest.fixture
def empty_tracking_env(qtbot, viewer, patch_tracking_side_effects):
    """Fresh viewer + tracking controls, without any layers."""
    tc = _get_tracking_controls(viewer)
    qtbot.addWidget(tc)
    return viewer, tc


@pytest.fixture
def tracking_env(qtbot, viewer, patch_tracking_side_effects):
    """Fresh viewer + tracking controls + minimal deterministic video/points layers."""
    tc = _get_tracking_controls(viewer)
    qtbot.addWidget(tc)

    video_layer = viewer.add_image(
        np.zeros((_DUMMY_VIDEO_N_FRAMES, 4, 4), dtype=np.uint8),
        name="video_stack",
    )
    points_layer = viewer.add_points(
        np.array([[0, 1.0, 2.0], [0, 3.0, 4.0]]),
        features=pd.DataFrame({"id": [0, 1], "name": ["kp0", "kp1"]}),
        name="keypoints",
    )

    tc._video_layer_combo.value = video_layer
    tc._keypoint_layer_combo.value = points_layer
    return viewer, tc, video_layer, points_layer


def _put_all_points_on_frame(points_layer, frame: int) -> None:
    """Move all test points onto a specific frame while preserving row order/features."""
    data = np.asarray(points_layer.data, dtype=float).copy()
    assert data.size > 0, "Test fixture produced no points."
    data[:, 0] = float(frame)
    points_layer.data = data


def _set_current_tracking_frame(tc: TrackingControls, viewer, frame: int) -> None:
    """Mirror the real UI flow: the reference frame follows the viewer current step."""
    viewer.dims.current_step = (frame,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    assert tc._reference_spinbox.value() == frame


def _stub_worker_start(tc: TrackingControls) -> None:
    """Prevent spawning a real worker thread in request-dispatch tests."""
    tc._start_worker = lambda: setattr(tc, "worker_started", True)


def _capture_tracking_requests(tc: TrackingControls):
    captured = []
    tc.trackingRequested.connect(captured.append)
    return captured


# -----------------------------------------------------------------------------
# Light integration tests: real viewer + widget, but minimal behavior surface
# -----------------------------------------------------------------------------


def test_tracking_controls_initial_state(empty_tracking_env):
    _viewer, tc = empty_tracking_env

    items = [tc._tracking_method_combo.itemText(i) for i in range(tc._tracking_method_combo.count())]
    assert set(items) >= set(AVAILABLE_TRACKERS.keys())

    current = tc._tracking_method_combo.currentText()
    info = AVAILABLE_TRACKERS[current]["class"].info_text
    assert tc._model_info_button.toolTip() == info


def test_tracking_frame_controls_layer_selection_and_ranges(tracking_env):
    viewer, tc, _video_layer, _points_layer = tracking_env

    _set_current_tracking_frame(tc, viewer, 2)

    # Forward range
    assert tc._forward_slider.minimum() == 0
    assert tc._forward_slider.maximum() == _DUMMY_VIDEO_N_FRAMES - 1 - 2
    assert tc._forward_spinbox_absolute.minimum() == 2
    assert tc._forward_spinbox_absolute.maximum() == _DUMMY_VIDEO_N_FRAMES - 1

    # Backward range
    assert tc._backward_slider.minimum() == -2
    assert tc._backward_slider.maximum() == 0
    assert tc._backward_spinbox_absolute.minimum() == 0
    assert tc._backward_spinbox_absolute.maximum() == 2

    # Reference spinbox
    assert tc._reference_spinbox.value() == 2
    assert tc._reference_spinbox.minimum() == 0
    assert tc._reference_spinbox.maximum() == _DUMMY_VIDEO_N_FRAMES - 1

    # Swap to a larger video and re-check at a later frame.
    big_video_n_frames = 200
    new_video = np.zeros((big_video_n_frames, 4, 4), dtype=np.uint8)
    tc._video_layer_combo.value = viewer.add_image(new_video, name="big_video")
    frame = 150
    _set_current_tracking_frame(tc, viewer, frame)

    assert tc._forward_slider.minimum() == 0
    assert tc._forward_slider.maximum() == big_video_n_frames - 1 - frame
    assert tc._forward_spinbox_absolute.minimum() == frame
    assert tc._forward_spinbox_absolute.maximum() == big_video_n_frames - 1

    assert tc._backward_slider.minimum() == -frame
    assert tc._backward_slider.maximum() == 0
    assert tc._backward_spinbox_absolute.minimum() == 0
    assert tc._backward_spinbox_absolute.maximum() == frame

    assert tc._reference_spinbox.value() == frame
    assert tc._reference_spinbox.minimum() == 0
    assert tc._reference_spinbox.maximum() == big_video_n_frames - 1


# -----------------------------------------------------------------------------
# Guard / no-op behavior: direct method calls, no button clicks needed
# -----------------------------------------------------------------------------


def test_track_does_nothing_without_video_layer(empty_tracking_env, qtbot):
    _viewer, tc = empty_tracking_env

    captured = _capture_tracking_requests(tc)
    tc.track_forward()
    qtbot.wait(20)

    assert captured == []


@pytest.mark.parametrize(
    "method_name, ref_frame, target_value",
    [
        ("track_forward", 3, 3),
        ("track_backward", 3, 3),
    ],
)
def test_invalid_target_does_not_emit_request(tracking_env, qtbot, method_name, ref_frame, target_value):
    viewer, tc, _video_layer, points_layer = tracking_env

    _put_all_points_on_frame(points_layer, ref_frame)
    _set_current_tracking_frame(tc, viewer, ref_frame)

    if method_name == "track_forward":
        tc._forward_spinbox_absolute.setValue(target_value)
    else:
        tc._backward_spinbox_absolute.setValue(target_value)

    captured = _capture_tracking_requests(tc)
    getattr(tc, method_name)()
    qtbot.wait(20)

    assert captured == []


def test_track_is_ignored_while_already_tracking(tracking_env, qtbot):
    viewer, tc, _video_layer, points_layer = tracking_env

    _put_all_points_on_frame(points_layer, 0)
    _set_current_tracking_frame(tc, viewer, 0)
    tc._forward_spinbox_absolute.setValue(3)

    tc.worker_started = True
    tc.is_tracking = True

    captured = _capture_tracking_requests(tc)
    tc.track_forward()
    qtbot.wait(20)

    assert captured == []


# -----------------------------------------------------------------------------
# Request-dispatch tests: still real widget/viewer, but call slots directly
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction, ref_frame, target_frame, expected_range, expected_len, backward",
    [
        ("forward", 0, 3, (0, 4), 4, False),
        ("backward", 2, 0, (0, 3), 3, True),
    ],
)
def test_track_emits_expected_worker_data(
    tracking_env,
    direction,
    ref_frame,
    target_frame,
    expected_range,
    expected_len,
    backward,
):
    viewer, tc, _video_layer, points_layer = tracking_env
    _stub_worker_start(tc)

    _put_all_points_on_frame(points_layer, ref_frame)
    _set_current_tracking_frame(tc, viewer, ref_frame)

    if direction == "forward":
        tc._forward_spinbox_absolute.setValue(target_frame)
        action = tc.track_forward
    else:
        tc._backward_spinbox_absolute.setValue(target_frame)
        action = tc.track_backward

    captured = _capture_tracking_requests(tc)
    action()

    assert len(captured) == 1
    twd = captured[0]
    assert isinstance(twd, TrackingWorkerData)
    assert twd.tracker_name == tc._tracking_method_combo.currentText()
    assert twd.backward_tracking is backward
    assert twd.reference_frame_index == ref_frame
    assert twd.keypoint_range == expected_range
    assert twd.video.shape[0] == expected_len

    # Seed keypoints are rebased to local frame 0 inside the sliced tracking video.
    assert np.all(twd.keypoints[:, 0] == 0)

    # Original point properties should still be present on the worker input.
    assert "id" in twd.keypoint_features.columns
    assert "name" in twd.keypoint_features.columns

    # Tracking identity columns should also be present.
    assert "tracking_query_index" in twd.keypoint_features.columns
    assert "tracking_query_frame" in twd.keypoint_features.columns
    assert set(twd.keypoint_features["tracking_query_frame"]) == {ref_frame}


def test_forward_button_click_emits_request(tracking_env, qtbot):
    viewer, tc, _video_layer, points_layer = tracking_env
    _stub_worker_start(tc)

    _put_all_points_on_frame(points_layer, 0)
    _set_current_tracking_frame(tc, viewer, 0)
    tc._forward_spinbox_absolute.setValue(3)

    with qtbot.waitSignal(tc.trackingRequested, timeout=1500) as req:
        qtbot.mouseClick(tc._tracking_forward_button, Qt.LeftButton)

    twd = req.args[0]
    assert isinstance(twd, TrackingWorkerData)
    assert twd.backward_tracking is False
    assert twd.keypoint_range == (0, 4)
    assert twd.reference_frame_index == 0


def test_bothway_track_emits_forward_then_backward(tracking_env, qtbot):
    viewer, tc, _video_layer, points_layer = tracking_env
    _stub_worker_start(tc)

    _put_all_points_on_frame(points_layer, 3)
    _set_current_tracking_frame(tc, viewer, 3)
    tc._forward_spinbox_absolute.setValue(6)
    tc._backward_spinbox_absolute.setValue(0)

    # Ensure backward path doesn't fail due to missing keypoint_widget.
    tc.keypoint_widget = object()

    captured = _capture_tracking_requests(tc)

    qtbot.mouseClick(tc._tracking_bothway_button, Qt.LeftButton)
    tc.trackedKeypointsAdded.emit()
    qtbot.waitUntil(lambda: len(captured) == 2, timeout=1500)

    assert captured[0].backward_tracking is False
    assert captured[1].backward_tracking is True
    assert captured[0].reference_frame_index == 3
    assert captured[1].reference_frame_index == 3

    # When forward == reference, bothway should degenerate to backward only.
    captured.clear()
    _set_current_tracking_frame(tc, viewer, 3)
    tc._forward_spinbox_absolute.setValue(3)
    tc._backward_spinbox_absolute.setValue(0)

    tc.track_bothway()
    qtbot.waitUntil(lambda: len(captured) == 1, timeout=1000)

    assert captured[0].backward_tracking is True
    assert captured[0].reference_frame_index == 3


def test_bothway_backward_is_connected_single_shot(tracking_env, qtbot):
    viewer, tc, _video_layer, points_layer = tracking_env
    _stub_worker_start(tc)

    _put_all_points_on_frame(points_layer, 3)
    _set_current_tracking_frame(tc, viewer, 3)
    tc._forward_spinbox_absolute.setValue(6)
    tc._backward_spinbox_absolute.setValue(0)

    captured = _capture_tracking_requests(tc)
    tc.keypoint_widget = object()

    tc.track_bothway()
    assert len(captured) == 1
    assert captured[0].backward_tracking is False

    tc.trackedKeypointsAdded.emit()
    qtbot.waitUntil(lambda: len(captured) == 2, timeout=1000)
    assert captured[1].backward_tracking is True

    # Must not trigger backward a second time.
    tc.trackedKeypointsAdded.emit()
    qtbot.wait(20)
    assert len(captured) == 2


# -----------------------------------------------------------------------------
# Widget logic tests that still benefit from real controls, but avoid click paths
# -----------------------------------------------------------------------------


def test_forward_controls_stay_in_sync(tracking_env):
    viewer, tc, _video_layer, _points_layer = tracking_env

    _set_current_tracking_frame(tc, viewer, 3)

    # absolute -> relative + slider
    tc._forward_spinbox_absolute.setValue(7)
    assert tc._forward_spinbox_relative.value() == 4
    assert tc._forward_slider.value() == 4

    # slider -> absolute + relative
    tc._forward_slider.setValue(2)
    assert tc._forward_spinbox_relative.value() == 2
    assert tc._forward_spinbox_absolute.value() == 5


def test_backward_controls_stay_in_sync(tracking_env):
    viewer, tc, _video_layer, _points_layer = tracking_env

    _set_current_tracking_frame(tc, viewer, 6)

    # absolute -> relative + slider
    tc._backward_spinbox_absolute.setValue(2)
    assert tc._backward_spinbox_relative.value() == -4
    assert tc._backward_slider.value() == -4

    # slider -> absolute + relative
    tc._backward_slider.setValue(-1)
    assert tc._backward_spinbox_relative.value() == -1
    assert tc._backward_spinbox_absolute.value() == 5


def test_seed_query_points_and_features_extracts_ref_frame_only(tracking_env):
    _viewer, tc, _video_layer, points_layer = tracking_env

    points_layer.data = np.array(
        [
            [2, 1.0, 2.0],
            [2, 3.0, 4.0],
            [4, 5.0, 6.0],
        ]
    )
    points_layer.features = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "name": ["kp0", "kp1", "kp2"],
        }
    )

    keypoints, features = tc._seed_query_points_and_features(ref_frame_idx=2)

    assert keypoints.shape == (2, 3)
    assert np.all(keypoints[:, 0] == 0)
    assert list(features["id"]) == [0, 1]
    assert "tracking_query_index" in features.columns
    assert "tracking_query_frame" in features.columns
    assert set(features["tracking_query_frame"]) == {2}


def test_seed_query_points_and_features_raises_when_no_points_on_ref_frame(tracking_env):
    _viewer, tc, _video_layer, points_layer = tracking_env

    points_layer.data = np.array(
        [
            [1, 1.0, 2.0],
            [1, 3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError, match="No keypoints found on reference frame"):
        tc._seed_query_points_and_features(ref_frame_idx=3)


# -----------------------------------------------------------------------------
# Result-layer integration behavior
# -----------------------------------------------------------------------------


def test_tracking_finished_creates_and_selects_result_layer(tracking_env, qtbot):
    viewer, tc, _video_layer, points_layer = tracking_env
    tc._keypoint_layer_combo.value = points_layer

    out = type("Out", (), {})()
    out.keypoints = np.array(
        [
            [0, 1.0, 2.0],
            [1, 1.1, 2.1],
        ]
    )
    out.keypoint_features = pd.DataFrame(
        {
            "id": [0, 0],
            "name": ["kp0", "kp0"],
            "tracking_query_frame": [0, 0],
        }
    )

    before = len(viewer.layers)

    with qtbot.waitSignal(tc.trackedKeypointsAdded, timeout=1000):
        tc.tracking_finished(out)

    qtbot.waitUntil(lambda: len(viewer.layers) == before + 1, timeout=1000)
    new_layer = viewer.layers[-1]

    assert new_layer in viewer.layers.selection
    assert viewer.layers.selection.active.name == new_layer.name

    # The combo update is deferred by _select_keypoint_combo_layer(...),
    # so wait for the event loop turn that applies it.
    qtbot.waitUntil(
        lambda: tc.keypoint_layer is not None and tc.keypoint_layer.name == new_layer.name,
        timeout=1000,
    )
