import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt

from napari_deeplabcut.tracking._data import TrackingWorkerData
from napari_deeplabcut.tracking._models import AVAILABLE_TRACKERS

_DUMMY_VIDEO_N_FRAMES = 10


def _get_tracking_controls(viewer):
    for title, dock in viewer.window.dock_widgets.items():
        if "Tracking controls" in title and "napari-deeplabcut" in title:
            return dock
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
                "napari_deeplabcut._widgets.KeypointMatplotlibCanvas._load_dataframe",
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


def test_tracking_controls_initial_state(setup_tracking_widget):
    tc = setup_tracking_widget(add_data=False)

    items = [tc._tracking_method_combo.itemText(i) for i in range(tc._tracking_method_combo.count())]
    assert set(items) >= set(AVAILABLE_TRACKERS.keys())
    current = tc._tracking_method_combo.currentText()
    info = AVAILABLE_TRACKERS[current]["class"].info_text
    assert tc._model_info_button.toolTip() == info


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


def test_backward_track(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    from types import MethodType

    tc._start_worker = MethodType(lambda self: setattr(self, "worker_started", True), tc)

    # Set ref frame to 2; backward absolute to 0 so it’s < ref
    viewer.dims.current_step = (2,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    tc._backward_spinbox_absolute.setValue(0)

    with qtbot.waitSignal(tc.trackingRequested, timeout=1500) as req:
        qtbot.mouseClick(tc._tracking_backward_button, Qt.LeftButton)
    twd = req.args[0]
    assert twd.backward_tracking is True
    # For backward, track() reverses the video slice
    assert twd.video.shape[0] == (2 - 0 + 1)  # inclusive range when +1 is applied in TrackControls


def test_bothway_track(setup_tracking_widget, qtbot, viewer):
    tc, video_layer, points_layer = setup_tracking_widget(add_data=True)
    from types import MethodType

    tc._start_worker = MethodType(lambda self: setattr(self, "worker_started", True), tc)

    viewer.dims.current_step = (3,) + (0,) * (viewer.dims.ndim - 1)
    tc._video_layer_changed()
    tc._reference_spinbox.setValue(0)
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

    # Do the same when forward == reference (should only do backward tracking)
    captured.clear()
    tc._forward_spinbox_absolute.setValue(0)
    tc._video_layer_changed()
    with qtbot.waitSignal(tc.trackingRequested, timeout=1500):
        qtbot.mouseClick(tc._tracking_bothway_button, Qt.LeftButton)
    assert len(captured) == 1
    assert captured[0].backward_tracking is True
