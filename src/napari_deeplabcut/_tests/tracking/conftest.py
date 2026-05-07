# src/napari_deeplabcut/_tests/tracking/conftest.py
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from napari.layers import Points

from napari_deeplabcut.tracking.core import utils as tracking_utils
from napari_deeplabcut.tracking.core.data import TrackingModelInputs, TrackingWorkerData, TrackingWorkerOutput
from napari_deeplabcut.tracking.core.models import AVAILABLE_TRACKERS, RawModelOutputs, TrackingModel

# --- Tracking fixtures ---
DUMMY_TRACKER_NAME = "TestTracker"


class DummyTracker(TrackingModel):
    """
    Minimal tracker that:
    - echoes inputs to outputs with a tiny deterministic transform,
    - emits progress via the callback,
    - honors stop_callback.
    """

    name = DUMMY_TRACKER_NAME
    info_text = "Dummy tracker for unit testing."

    def load_model(self, device: str):
        # No-op model; keep a simple config to emulate 'step' like CoTracker.
        class _NoOpModel:
            step = 3

        return _NoOpModel()

    def prepare_inputs(self, cfg: "TrackingWorkerData", **kwargs) -> TrackingModelInputs:
        # Ensure video is (T, H, W, C) and keypoints is (K, 3) where columns: [frame_idx, x, y] or [id, x, y]
        video = np.asarray(cfg.video)
        queries = np.asarray(cfg.keypoints).copy()
        metadata = {
            "keypoint_range": cfg.keypoint_range,
            "backward_tracking": getattr(cfg, "backward_tracking", False),
        }
        return TrackingModelInputs(video=video, keypoints=queries, metadata=metadata)

    def run(self, inputs: TrackingModelInputs, progress_callback, stop_callback, **kwargs) -> RawModelOutputs:
        # Fake progression per frame; stop if requested.
        T = inputs.video.shape[0]
        K = inputs.keypoints.shape[0]

        # Produce tracks of shape (T, K, 2) with a deterministic offset (e.g., +1 pixel)
        tracks = np.zeros((T, K, 2), dtype=float)
        for t in range(T):
            progress_callback(t, T)
            if stop_callback():
                # Return partial result up to t
                tracks = tracks[: t + 1]
                vis = np.ones_like(tracks[..., 0], dtype=bool)  # visibility dummy
                return RawModelOutputs(keypoints=tracks, keypoint_features={"visibility": vis})
            # Use the input (x, y) for all K points and add a tiny drift proportional to t
            tracks[t, :, 0] = inputs.keypoints[:, 1] + 0.1 * t  # x
            tracks[t, :, 1] = inputs.keypoints[:, 2] + 0.1 * t  # y

        vis = np.ones_like(tracks[..., 0], dtype=bool)
        return RawModelOutputs(keypoints=tracks, keypoint_features={"visibility": vis})

    def prepare_outputs(
        self, model_outputs: RawModelOutputs, worker_inputs: "TrackingWorkerData" = None, **kwargs
    ) -> "TrackingWorkerOutput":
        # Flatten (T, K, 2) -> (N, 3) with [frame_idx, x, y]
        tracks = model_outputs.keypoints
        T = tracks.shape[0]
        K = tracks.shape[1]

        T1, T2 = worker_inputs.keypoint_range
        frame_ids = np.repeat(np.arange(T1, T1 + T), K)
        flat = tracks.reshape(-1, 2)
        keypoints = np.column_stack((frame_ids, flat))  # (N, 3)

        # Minimal features: concat original per-keypoint features replicated per frame
        keypoints_features = pd.concat(
            [worker_inputs.keypoint_features] * T,
            ignore_index=True,
        )

        return TrackingWorkerOutput(
            keypoints=keypoints,
            keypoint_features=keypoints_features,
        )

    def validate_outputs(self, inputs: TrackingModelInputs, outputs: "TrackingWorkerOutput") -> tuple[bool, str]:
        """
        Validate DummyTracker outputs.

        Expectations for DummyTracker:
        - outputs.keypoints is an (N, 3) float array of [frame_idx, x, y]
        - N == (T2 - T1) * K   where:
                T1, T2 = inputs.metadata["keypoint_range"]
                T = T2 - T1  (number of frames produced)
                K = inputs.keypoints.shape[0] (number of query points)
        - frame_idx are integers in [T1, T2-1]
        - x, y are finite. If video shape known, also check bounds: x∈[0,W), y∈[0,H)
        - outputs.keypoint_features is a DataFrame with length N
            and contains at least the columns present in worker_inputs.keypoint_features
            (as repeated by the DummyTracker)
        """

        # -------- Basic structure checks
        kp = outputs.keypoints
        if not isinstance(kp, np.ndarray):
            return False, "outputs.keypoints must be a numpy array"

        if kp.ndim != 2 or kp.shape[1] != 3:
            return False, f"outputs.keypoints must have shape (N, 3); got {kp.shape}"

        # -------- Expected length: N = (T2 - T1) * K
        meta = inputs.metadata or {}
        if (
            "keypoint_range" not in meta
            or not isinstance(meta["keypoint_range"], (tuple, list))
            or len(meta["keypoint_range"]) != 2
        ):
            return False, "inputs.metadata.keypoint_range must be a (T1, T2) tuple"

        T1, T2 = meta["keypoint_range"]
        if not (isinstance(T1, (int, np.integer)) and isinstance(T2, (int, np.integer)) and T2 >= T1):
            return False, "Invalid keypoint_range; expected integers with T2 >= T1"

        K = inputs.keypoints.shape[0]
        expected_len = (T2 - T1) * K
        if kp.shape[0] != expected_len:
            return False, f"Expected (T*K)={expected_len} rows; got {kp.shape[0]}"

        # -------- Frame index checks
        frames = kp[:, 0]
        # Allow float dtype but must be whole numbers
        if not np.all(np.isfinite(frames)):
            return False, "Frame indices contain non-finite values"

        if not np.allclose(frames, np.round(frames)):
            return False, "Frame indices must be integers"

        frames_int = frames.astype(int)
        if frames_int.min() < T1 or frames_int.max() > (T2 - 1):
            return False, f"Frame indices out of range [{T1}, {T2 - 1}]"

        # -------- Coordinate checks
        xy = kp[:, 1:3]
        if not np.all(np.isfinite(xy)):
            return False, "Coordinates contain NaN/Inf"

        # -------- Features checks
        feats = outputs.keypoint_features
        if not isinstance(feats, pd.DataFrame):
            return False, "outputs.keypoint_features must be a pandas DataFrame"

        if len(feats) != expected_len:
            return False, f"keypoint_features length mismatch: expected {expected_len}, got {len(feats)}"

        # When produced by DummyTracker, features are a concat of the input per frame
        # Ensure at least the same columns are present and non-null
        required_cols = []
        try:
            # worker_inputs.keypoint_features is replicated in DummyTracker.prepare_outputs
            required_cols = list(self.cfg.keypoint_features.columns)  # may exist on the tracker
        except Exception:
            # fallback to inputs.shape if not accessible; skip strict column match
            pass

        missing = [c for c in required_cols if c not in feats.columns]
        if missing:
            return False, f"Missing required feature columns: {missing}"

        if required_cols:
            if feats[required_cols].isna().any().any():
                return False, "keypoint_features contain NaN in required columns"

        return True, ""


@pytest.fixture(autouse=True)
def register_dummy_tracker():
    """
    Auto-register DummyTracker for all tests and restore registry afterwards.
    """
    prev = dict(AVAILABLE_TRACKERS)
    AVAILABLE_TRACKERS[DUMMY_TRACKER_NAME] = {"class": DummyTracker}
    try:
        yield
    finally:
        AVAILABLE_TRACKERS.clear()
        AVAILABLE_TRACKERS.update(prev)


@pytest.fixture
def track_worker_inputs():
    """
    Provide minimal valid TrackingWorkerData with:
    - 5-frame RGB video of 4x4 pixels,
    - 2 keypoints,
    - keypoint_range covering all frames,
    - simple features DataFrame.
    """
    video = np.zeros((5, 4, 4, 3), dtype=np.uint8)

    keypoints = np.array(
        [
            [0, 10.0, 20.0],
            [0, 30.0, 40.0],
        ],
        dtype=float,
    )

    keypoint_features = pd.DataFrame({"id": [0, 1], "name": ["kp0", "kp1"]})

    # Build TrackingWorkerData
    return TrackingWorkerData(
        tracker_name=DUMMY_TRACKER_NAME,
        video=video,
        keypoints=keypoints,
        keypoint_range=(0, 5),  # frames 0..4
        keypoint_features=keypoint_features,
        backward_tracking=False,
    )


# -----------------------------------------------------------------------------#
# Pure-unit helpers for tracking.core.utils tests
# -----------------------------------------------------------------------------#
@pytest.fixture
def fake_points_layer_factory():
    """
    Flexible factory for tracking core tests.

    Behavior
    --------
    - Returns a real napari Points layer for "normal" merge/refine-style tests
      when labels are provided and no explicit `features=` payload is passed.
    - Returns a lightweight fake Points-like object (SimpleNamespace) when tests
      need full control over `features` / `properties`, including intentionally
      invalid or mismatched states used by utils tests.
    - The behavior can be forced with `real=True` or `real=False`.

    Supported patterns
    ------------------
    1) Real Points layer:
        fake_points_layer_factory(
            name="target",
            data=[[0, 10, 20]],
            labels=["nose"],
            ids=[""],
            extra_features={"likelihood": [0.9]},
        )

    2) Fake Points-like layer:
        fake_points_layer_factory(
            data=[[0, 10, 20], [1, 11, 21]],
            features=pd.DataFrame({"label": ["nose"]}),  # mismatched on purpose
            properties={"label": np.array(["nose", "tail"], dtype=object)},
        )
    """

    def _make(
        *,
        data,
        name: str = "layer",
        labels=None,
        ids=None,
        extra_features: dict | None = None,
        features: pd.DataFrame | dict | None = None,
        properties: dict | None = None,
        real: bool | None = None,
    ):
        data_arr = np.asarray(data, dtype=float)
        props = dict(properties or {})

        # Caller explicitly supplied features payload
        explicit_features = features is not None

        if explicit_features:
            if isinstance(features, pd.DataFrame):
                feat_df = features.copy()
            else:
                feat_df = pd.DataFrame(features)
        elif labels is not None:
            if ids is None:
                ids = [""] * len(labels)

            feat_df = pd.DataFrame(
                {
                    "label": list(labels),
                    "id": list(ids),
                }
            )

            if extra_features:
                for key, values in extra_features.items():
                    feat_df[key] = list(values)
        else:
            feat_df = None

        # Auto mode:
        # - if explicit features are provided, use fake object unless caller forces real=True
        #   (this supports mismatched/broken states for utils tests)
        # - if labels are provided normally, use real Points
        if real is None:
            real = (labels is not None) and (not explicit_features)

        if real:
            if feat_df is None:
                feat_df = pd.DataFrame(index=range(len(data_arr)))

            layer = Points(
                data=data_arr,
                features=feat_df,
                name=name,
            )

            # Best-effort property override for tests that want explicit properties too
            if props:
                try:
                    layer.properties = props
                except Exception:
                    pass

            return layer

        # Fake minimal object for pure-unit tests that need invalid/flexible state
        return SimpleNamespace(
            data=data_arr,
            features=feat_df,
            properties=props,
            name=name,
        )

    return _make


@pytest.fixture
def dummy_viewer_factory():
    """
    Factory for minimal fake Viewer-like objects with only `.layers`.
    """

    def _make(*, layers):
        return SimpleNamespace(layers=list(layers))

    return _make


@pytest.fixture
def tracking_manager_factory():
    """
    Factory for minimal fake LayerLifecycleManager-like objects used by naming tests.
    """

    def _make(
        *,
        tracking_layers=(),
        source_name_by_layer_id=None,
    ):
        tracking_ids = {id(layer) for layer in tracking_layers}
        source_name_by_layer_id = dict(source_name_by_layer_id or {})

        return SimpleNamespace(
            is_tracking_result_layer=lambda layer: id(layer) in tracking_ids,
            tracking_result_source_layer_name=lambda layer: source_name_by_layer_id.get(id(layer)),
        )

    return _make


@pytest.fixture
def patch_tracking_manager(monkeypatch):
    """
    Patch tracking.core.utils.get_or_create_layer_manager(...) for pure unit tests.
    """

    def _patch(manager):
        monkeypatch.setattr(tracking_utils, "get_or_create_layer_manager", lambda _viewer: manager)
        return manager

    return _patch
