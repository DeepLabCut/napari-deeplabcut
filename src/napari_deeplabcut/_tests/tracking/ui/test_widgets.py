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


def _get_tracking_controls(viewer: "napari.Viewer") -> TrackingControls:
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
def setup_tracking_widget(qtbot, viewer, monkeypatch):
    """
    Factory fixture that returns a function to set up the TrackingControls test environment.

    Usage in tests:
        tc, video_layer, points_layer = setup_tracking_env(add_data=True)
        # or for just the widget:
        tc = setup_tracking_env()
    """

    def _setup_tracking_data(*, add_data: bool = False, disable_insert_hooks: bool = True):
        tc = _get_tracking_controls(viewer)
        qtbot.addWidget(tc)

        # Optionally disable plugin insert hooks that assume DLC metadata/header.
        if add_data and disable_insert_hooks:
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

        if add_data:
            # Minimal, deterministic video and points layers
            video_layer = viewer.add_image(np.zeros((_DUMMY_VIDEO_N_FRAMES, 4, 4), dtype=np.uint8), name="video_stack")
            points_layer = viewer.add_points(
                np.array([[0, 1.0, 2.0], [0, 3.0, 4.0]]),
                features=pd.DataFrame({"id": [0, 1], "name": ["kp0", "kp1"]}),
                name="keypoints",
            )
            tc._video_layer_combo.value = video_layer
            tc._keypoint_layer_combo.value = points_layer
            return tc, video_layer, points_layer

        return tc

    return _setup_tracking_data


def _put_all_points_on_frame(points_layer, frame: int) -> None:
    """
    Move all test points onto a specific frame so tracking has a valid seed frame.
    Keeps the same number/order of points and therefore preserves features alignment.
    """
    data = np.asarray(points_layer.data, dtype=float).copy()
    assert data.size > 0, "Test fixture produced no points."
    data[:, 0] = float(frame)
    points_layer.data = data


def _set_current_tracking_frame(tc, viewer, frame: int) -> None:
    """
    Drive the tracking widget the same way the real UI does:
    the reference frame follows the viewer's current step.
    """
    viewer.dims.current_step = (frame,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    assert tc._reference_spinbox.value() == frame


@pytest.mark.usefixtures("qtbot")
def test_tracking_controls_initial_state(setup_tracking_widget):
    tc = setup_tracking_widget(add_data=False)

    items = [tc._tracking_method_combo.itemText(i) for i in range(tc._tracking_method_combo.count())]
    assert set(items) >= set(AVAILABLE_TRACKERS.keys())
    current = tc._tracking_method_combo.currentText()
    info = AVAILABLE_TRACKERS[current]["class"].info_text
    assert tc._model_info_button.toolTip() == info


@pytest.mark.usefixtures("qtbot")
def test_tracking_frame_controls_layer_selection_and_ranges(setup_tracking_widget, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    viewer.dims.current_step = (2,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    # Forward range
    assert tc._forward_slider.minimum() == 0
    assert tc._forward_slider.maximum() == _DUMMY_VIDEO_N_FRAMES - 1 - 2  # 2 steps forward possible
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

    big_video_n_frames = 200
    new_video = np.zeros((big_video_n_frames, 4, 4), dtype=np.uint8)
    tc._video_layer_combo.value = viewer.add_image(new_video, name="big_video")
    tc._video_layer_changed()
    frame = 150
    viewer.dims.current_step = (frame,) + (0,) * (viewer.dims.ndim - 1)
    # Forward range
    assert tc._forward_slider.minimum() == 0
    assert tc._forward_slider.maximum() == big_video_n_frames - 1 - frame
    assert tc._forward_spinbox_absolute.minimum() == frame
    assert tc._forward_spinbox_absolute.maximum() == big_video_n_frames - 1
    # Backward range
    assert tc._backward_slider.minimum() == -frame
    assert tc._backward_slider.maximum() == 0
    assert tc._backward_spinbox_absolute.minimum() == 0
    assert tc._backward_spinbox_absolute.maximum() == frame
    # Reference spinbox
    assert tc._reference_spinbox.value() == frame
    assert tc._reference_spinbox.minimum() == 0
    assert tc._reference_spinbox.maximum() == big_video_n_frames - 1


@pytest.mark.usefixtures("qtbot")
def test_track_does_nothing_without_video_layer(setup_tracking_widget, qtbot):
    tc = setup_tracking_widget(add_data=False)

    with qtbot.assertNotEmitted(tc.trackingRequested):
        tc.track_forward()


@pytest.mark.usefixtures("qtbot")
def test_forward_track(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    # Only change internal state
    def fake_start_worker(self):
        self.worker_started = True

    from types import MethodType

    tc._start_worker = MethodType(fake_start_worker, tc)

    # Set current frame to 0, set forward absolute to 3
    viewer.dims.current_step = (0,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    tc._reference_spinbox.setValue(0)
    tc._forward_spinbox_absolute.setValue(3)

    with qtbot.waitSignal(tc.trackingRequested, timeout=1500) as req:
        qtbot.mouseClick(tc._tracking_forward_button, Qt.LeftButton)

    twd: TrackingWorkerData = req.args[0]
    assert isinstance(twd, TrackingWorkerData)
    assert twd.tracker_name == tc._tracking_method_combo.currentText()
    assert twd.backward_tracking is False
    # video slice should have length 3 (frames 0..3 inclusive when +1 applied)
    assert twd.video.shape[0] == 4  # because track_forward uses forward_frame_idx + 1
    # keypoints should be those from ref frame with frame index reset to 0
    assert (twd.keypoints[:, 0] == 0).all()
    # features replicated per ref frame selection (only ref frame rows)
    assert len(twd.keypoint_features) == len(points_layer.features)


@pytest.mark.usefixtures("qtbot")
def test_backward_track(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    from types import MethodType

    tc._start_worker = MethodType(lambda self: setattr(self, "worker_started", True), tc)

    _put_all_points_on_frame(points_layer, 2)
    _set_current_tracking_frame(tc, viewer, 2)

    # Set ref frame to 2; backward absolute to 0 so it’s < ref
    tc._backward_spinbox_absolute.setValue(0)

    with qtbot.waitSignal(tc.trackingRequested, timeout=1500) as req:
        qtbot.mouseClick(tc._tracking_backward_button, Qt.LeftButton)
    twd = req.args[0]
    assert twd.backward_tracking is True
    assert twd.reference_frame_index == 2
    # For backward, track() reverses the video slice
    assert twd.video.shape[0] == (2 - 0 + 1)  # inclusive range when +1 is applied in TrackControls

    # Seed keypoints are re-based to local frame 0 inside the sliced tracking video.
    assert np.all(twd.keypoints[:, 0] == 0)
    # Original point properties should still be present on the worker input.
    assert "id" in twd.keypoint_features.columns
    assert "name" in twd.keypoint_features.columns

    # New tracking identity columns should also be present.
    assert "tracking_query_index" in twd.keypoint_features.columns
    assert "tracking_query_frame" in twd.keypoint_features.columns
    assert set(twd.keypoint_features["tracking_query_frame"]) == {2}


@pytest.mark.usefixtures("qtbot")
def test_bothway_track(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    from types import MethodType

    tc._start_worker = MethodType(lambda self: setattr(self, "worker_started", True), tc)

    # New invariant: current/reference frame must actually contain seed keypoints.
    _put_all_points_on_frame(points_layer, 3)
    _set_current_tracking_frame(tc, viewer, 3)

    # Forward target > ref, backward target < ref
    tc._forward_spinbox_absolute.setValue(6)
    tc._backward_spinbox_absolute.setValue(0)

    captured = []
    tc.trackingRequested.connect(lambda d: captured.append(d))

    # Ensure backward path doesn't fail due to missing keypoint_widget
    tc.keypoint_widget = object()

    with qtbot.waitSignals([tc.trackingRequested, tc.trackingRequested], timeout=2000):
        qtbot.mouseClick(tc._tracking_bothway_button, Qt.LeftButton)
        tc.trackedKeypointsAdded.emit()

    assert len(captured) == 2
    assert captured[0].backward_tracking is False
    assert captured[1].backward_tracking is True

    # Both requests should use the same seed frame, because the ref frame is still 3.
    assert captured[0].reference_frame_index == 3
    assert captured[1].reference_frame_index == 3

    # Do the same when forward == reference -> only backward tracking should run.
    captured.clear()

    # Since the widget keeps reference == current frame, make that explicit.
    _set_current_tracking_frame(tc, viewer, 3)
    tc._forward_spinbox_absolute.setValue(3)  # == ref, so forward is invalid
    tc._backward_spinbox_absolute.setValue(0)

    with qtbot.waitSignal(tc.trackingRequested, timeout=1500):
        qtbot.mouseClick(tc._tracking_bothway_button, Qt.LeftButton)

    assert len(captured) == 1
    assert captured[0].backward_tracking is True
    assert captured[0].reference_frame_index == 3


@pytest.mark.usefixtures("qtbot")
@pytest.mark.parametrize(
    "method_name, ref_frame, target_value",
    [
        ("track_forward", 3, 3),  # invalid forward
        ("track_backward", 3, 3),  # invalid backward
    ],
)
def test_invalid_target_does_not_emit_request(
    setup_tracking_widget, qtbot, viewer, method_name, ref_frame, target_value
):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    _put_all_points_on_frame(points_layer, ref_frame)
    _set_current_tracking_frame(tc, viewer, ref_frame)

    if method_name == "track_forward":
        tc._forward_spinbox_absolute.setValue(target_value)
    else:
        tc._backward_spinbox_absolute.setValue(target_value)

    captured = []
    tc.trackingRequested.connect(captured.append)

    getattr(tc, method_name)()
    qtbot.wait(50)

    assert captured == []


@pytest.mark.usefixtures("qtbot")
def test_track_is_ignored_while_already_tracking(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    _put_all_points_on_frame(points_layer, 0)
    _set_current_tracking_frame(tc, viewer, 0)
    tc._forward_spinbox_absolute.setValue(3)

    tc.worker_started = True
    tc.is_tracking = True

    captured = []
    tc.trackingRequested.connect(captured.append)

    tc.track_forward()
    qtbot.wait(50)

    assert captured == []


@pytest.mark.usefixtures("qtbot")
def test_bothway_backward_is_connected_single_shot(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    from types import MethodType

    tc._start_worker = MethodType(lambda self: setattr(self, "worker_started", True), tc)

    _put_all_points_on_frame(points_layer, 3)
    _set_current_tracking_frame(tc, viewer, 3)

    tc._forward_spinbox_absolute.setValue(6)
    tc._backward_spinbox_absolute.setValue(0)

    captured = []
    tc.trackingRequested.connect(captured.append)
    tc.keypoint_widget = object()

    tc.track_bothway()
    assert len(captured) == 1
    assert captured[0].backward_tracking is False

    tc.trackedKeypointsAdded.emit()
    qtbot.waitUntil(lambda: len(captured) == 2, timeout=1000)
    assert captured[1].backward_tracking is True

    # Should NOT trigger backward a second time
    tc.trackedKeypointsAdded.emit()
    qtbot.wait(50)
    assert len(captured) == 2


@pytest.mark.usefixtures("qtbot")
def test_tracking_finished_creates_and_selects_result_layer(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

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

    qtbot.wait(100)

    assert len(viewer.layers) == before + 1
    new_layer = viewer.layers[-1]
    assert viewer.layers.selection.active is new_layer
    assert new_layer is tc.keypoint_layer


@pytest.mark.usefixtures("qtbot")
def test_forward_controls_stay_in_sync(setup_tracking_widget, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    _set_current_tracking_frame(tc, viewer, 3)

    # absolute -> relative + slider
    tc._forward_spinbox_absolute.setValue(7)
    assert tc._forward_spinbox_relative.value() == 4
    assert tc._forward_slider.value() == 4

    # slider -> absolute + relative
    tc._forward_slider.setValue(2)
    assert tc._forward_spinbox_relative.value() == 2
    assert tc._forward_spinbox_absolute.value() == 5


@pytest.mark.usefixtures("qtbot")
def test_backward_controls_stay_in_sync(setup_tracking_widget, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    _set_current_tracking_frame(tc, viewer, 6)

    # absolute -> relative + slider
    tc._backward_spinbox_absolute.setValue(2)
    assert tc._backward_spinbox_relative.value() == -4
    assert tc._backward_slider.value() == -4

    # slider -> absolute + relative
    tc._backward_slider.setValue(-1)
    assert tc._backward_spinbox_relative.value() == -1
    assert tc._backward_spinbox_absolute.value() == 5


@pytest.mark.usefixtures("qtbot")
def test_seed_query_points_and_features_extracts_ref_frame_only(setup_tracking_widget):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

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
    assert np.all(keypoints[:, 0] == 0)  # rebased to local frame zero
    assert list(features["id"]) == [0, 1]
    assert "tracking_query_index" in features.columns
    assert "tracking_query_frame" in features.columns
    assert set(features["tracking_query_frame"]) == {2}


@pytest.mark.usefixtures("qtbot")
def test_seed_query_points_and_features_raises_when_no_points_on_ref_frame(setup_tracking_widget):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)

    points_layer.data = np.array(
        [
            [1, 1.0, 2.0],
            [1, 3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError, match="No keypoints found on reference frame"):
        tc._seed_query_points_and_features(ref_frame_idx=3)
