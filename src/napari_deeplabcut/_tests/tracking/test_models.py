# src/napari_deeplabcut/_tests/tracking/test_models.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.tracking.core.data import RawModelOutputs, TrackingModelInputs, TrackingWorkerData
from napari_deeplabcut.tracking.core.models import (
    AVAILABLE_TRACKERS,
    Cotracker3,
    TrackingModel,
    register_backbone,
)


def test_register_backbone_registers_class():
    name = "TemporaryUnitTestTracker"

    @register_backbone(name)
    class _TempTracker(TrackingModel):
        name = "TemporaryUnitTestTracker"
        info_text = "temp"

        def load_model(self, device: str):
            return object()

        def prepare_inputs(self, cfg, **kwargs):
            raise NotImplementedError

        def run(self, inputs, progress_callback, stop_callback, **kwargs):
            raise NotImplementedError

        def prepare_outputs(self, model_outputs, worker_inputs=None, **kwargs):
            raise NotImplementedError

        def validate_outputs(self, inputs, outputs):
            raise NotImplementedError

    assert name in AVAILABLE_TRACKERS
    assert AVAILABLE_TRACKERS[name]["class"] is _TempTracker

    # cleanup to avoid polluting global registry for later tests
    AVAILABLE_TRACKERS.pop(name, None)


def test_tracking_model_init_rejects_non_worker_cfg():
    class _MiniTracker(TrackingModel):
        name = "mini"
        info_text = "mini"

        def auto_set_device(self):
            return "cpu"

        def load_model(self, device: str):
            return "dummy-model"

        def prepare_inputs(self, cfg, **kwargs):
            raise NotImplementedError

        def run(self, inputs, progress_callback, stop_callback, **kwargs):
            raise NotImplementedError

        def prepare_outputs(self, model_outputs, worker_inputs=None, **kwargs):
            raise NotImplementedError

        def validate_outputs(self, inputs, outputs):
            raise NotImplementedError

    with pytest.raises(ValueError, match="cfg must be an instance of TrackingWorkerData"):
        _MiniTracker(cfg="not-a-worker-data")


def test_tracking_model_init_sets_device_and_model(track_worker_inputs):
    class _MiniTracker(TrackingModel):
        name = "mini"
        info_text = "mini"

        def auto_set_device(self):
            return "cpu"

        def load_model(self, device: str):
            return f"loaded-on-{device}"

        def prepare_inputs(self, cfg, **kwargs):
            raise NotImplementedError

        def run(self, inputs, progress_callback, stop_callback, **kwargs):
            raise NotImplementedError

        def prepare_outputs(self, model_outputs, worker_inputs=None, **kwargs):
            raise NotImplementedError

        def validate_outputs(self, inputs, outputs):
            raise NotImplementedError

    tracker = _MiniTracker(track_worker_inputs)

    assert tracker.device == "cpu"
    assert tracker.model == "loaded-on-cpu"
    assert tracker.cfg is track_worker_inputs


def test_cotracker3_prepare_inputs_swaps_xy_without_mutating_worker_inputs(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    original_keypoints = np.asarray(track_worker_inputs.keypoints, dtype=float).copy()

    inputs = model.prepare_inputs(track_worker_inputs)

    # video passed through as ndarray
    np.testing.assert_array_equal(inputs.video, np.asarray(track_worker_inputs.video))

    # original worker inputs must remain unchanged
    np.testing.assert_array_equal(track_worker_inputs.keypoints, original_keypoints)

    # output should be [frame, y, x]
    expected = original_keypoints.copy()
    expected[:, [1, 2]] = expected[:, [2, 1]]
    np.testing.assert_allclose(inputs.keypoints, expected)

    assert inputs.metadata["keypoint_range"] == track_worker_inputs.keypoint_range
    assert inputs.metadata["backward_tracking"] == track_worker_inputs.backward_tracking
    assert inputs.metadata["reference_frame_index"] == track_worker_inputs.reference_frame_index


def test_cotracker3_prepare_outputs_flattens_tracks_and_restores_plugin_xy(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    # T=5, K=2, model convention is (x=?, no: in this wrapper before restore it's [y, x])
    tracks = np.array(
        [
            [[20.0, 10.0], [40.0, 30.0]],
            [[21.0, 11.0], [41.0, 31.0]],
            [[22.0, 12.0], [42.0, 32.0]],
            [[23.0, 13.0], [43.0, 33.0]],
            [[24.0, 14.0], [44.0, 34.0]],
        ],
        dtype=float,
    )
    visibility = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )

    raw = RawModelOutputs(
        keypoints=tracks,
        keypoint_features={"visibility": visibility},
    )

    out = model.prepare_outputs(raw, worker_inputs=track_worker_inputs)

    # T=5, K=2 => 10 rows, plugin convention restored to [frame, x, y]
    assert out.keypoints.shape == (10, 3)
    assert out.keypoint_features.shape[0] == 10

    # first frame, first query -> [0, 10, 20]
    np.testing.assert_allclose(out.keypoints[0], np.array([0, 10.0, 20.0]))
    # first frame, second query -> [0, 30, 40]
    np.testing.assert_allclose(out.keypoints[1], np.array([0, 30.0, 40.0]))
    # second frame, first query -> [1, 11, 21]
    np.testing.assert_allclose(out.keypoints[2], np.array([1, 11.0, 21.0]))

    assert out.keypoint_features["tracking_frame"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert out.keypoint_features["tracking_visible"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert out.keypoint_features["tracking_tracker_name"].tolist() == ["Cotracker 3"] * 10


def test_cotracker3_prepare_outputs_reverses_time_axis_for_backward_tracking(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    worker_inputs = TrackingWorkerData(
        tracker_name=track_worker_inputs.tracker_name,
        video=track_worker_inputs.video,
        keypoints=track_worker_inputs.keypoints,
        keypoint_features=track_worker_inputs.keypoint_features,
        keypoint_range=track_worker_inputs.keypoint_range,
        backward_tracking=True,
        reference_frame_index=track_worker_inputs.reference_frame_index,
    )

    # tracks arrive in reversed-video time order
    tracks = np.array(
        [
            [[24.0, 14.0], [44.0, 34.0]],  # should become frame 0 after reversal
            [[23.0, 13.0], [43.0, 33.0]],
            [[22.0, 12.0], [42.0, 32.0]],
            [[21.0, 11.0], [41.0, 31.0]],
            [[20.0, 10.0], [40.0, 30.0]],
        ],
        dtype=float,
    )

    raw = RawModelOutputs(
        keypoints=tracks,
        keypoint_features={"visibility": np.ones((5, 2), dtype=bool)},
    )

    out = model.prepare_outputs(raw, worker_inputs=worker_inputs)

    # After reversal, first output row should be frame 0, x=10, y=20
    np.testing.assert_allclose(out.keypoints[0], np.array([0, 10.0, 20.0]))
    np.testing.assert_allclose(out.keypoints[1], np.array([0, 30.0, 40.0]))
    np.testing.assert_allclose(out.keypoints[-2], np.array([4, 14.0, 24.0]))
    np.testing.assert_allclose(out.keypoints[-1], np.array([4, 34.0, 44.0]))


def test_cotracker3_prepare_outputs_rejects_bad_track_shape(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    raw = RawModelOutputs(
        keypoints=np.zeros((5, 2), dtype=float),  # should be (T, K, 2)
        keypoint_features={},
    )

    with pytest.raises(ValueError, match="Expected tracks with shape"):
        model.prepare_outputs(raw, worker_inputs=track_worker_inputs)


def test_cotracker3_prepare_outputs_rejects_time_dimension_mismatch(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    raw = RawModelOutputs(
        keypoints=np.zeros((4, 2, 2), dtype=float),  # T=4, but keypoint_range implies 5 frames
        keypoint_features={},
    )

    with pytest.raises(ValueError, match="Time dimension mismatch"):
        model.prepare_outputs(raw, worker_inputs=track_worker_inputs)


def test_cotracker3_prepare_outputs_rejects_seed_feature_row_count_mismatch(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    bad_worker_inputs = TrackingWorkerData(
        tracker_name=track_worker_inputs.tracker_name,
        video=track_worker_inputs.video,
        keypoints=track_worker_inputs.keypoints,
        keypoint_features=pd.DataFrame({"id": [0]}),  # K=2 but only 1 row
        keypoint_range=track_worker_inputs.keypoint_range,
        backward_tracking=track_worker_inputs.backward_tracking,
        reference_frame_index=track_worker_inputs.reference_frame_index,
    )

    raw = RawModelOutputs(
        keypoints=np.zeros((5, 2, 2), dtype=float),
        keypoint_features={},
    )

    with pytest.raises(ValueError, match="Seed feature row count mismatch"):
        model.prepare_outputs(raw, worker_inputs=bad_worker_inputs)


def test_cotracker3_validate_outputs_accepts_valid_output(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    inputs = model.prepare_inputs(track_worker_inputs)

    raw = RawModelOutputs(
        keypoints=np.zeros((5, 2, 2), dtype=float),
        keypoint_features={},
    )
    out = model.prepare_outputs(raw, worker_inputs=track_worker_inputs)

    ok, msg = model.validate_outputs(inputs, out)

    assert ok is True
    assert msg == ""


def test_cotracker3_validate_outputs_rejects_missing_metadata(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    bad_inputs = TrackingModelInputs(
        video=np.asarray(track_worker_inputs.video),
        keypoints=np.asarray(track_worker_inputs.keypoints),
        metadata={},
    )

    out = model.prepare_outputs(
        RawModelOutputs(
            keypoints=np.zeros((5, 2, 2), dtype=float),
            keypoint_features={},
        ),
        worker_inputs=track_worker_inputs,
    )

    ok, msg = model.validate_outputs(bad_inputs, out)

    assert ok is False
    assert "keypoint_range" in msg


def test_cotracker3_validate_outputs_rejects_wrong_row_count(track_worker_inputs):
    model = Cotracker3.__new__(Cotracker3)
    model.name = "Cotracker 3"

    inputs = model.prepare_inputs(track_worker_inputs)

    # Valid shape (N, 3), but wrong N
    bad_out = type("Out", (), {})()
    bad_out.keypoints = np.zeros((3, 3), dtype=float)

    ok, msg = model.validate_outputs(inputs, bad_out)

    assert ok is False
    assert "Number of output keypoints" in msg
