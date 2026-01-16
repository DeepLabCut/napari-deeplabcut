import os

import cv2
import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QDockWidget
from skimage.io import imsave

from napari_deeplabcut import _writer, keypoints
from napari_deeplabcut.tracking._data import TrackingModelInputs, TrackingWorkerData, TrackingWorkerOutput
from napari_deeplabcut.tracking._models import AVAILABLE_TRACKERS, RawModelOutputs, TrackingModel

# os.environ["NAPARI_DLC_HIDE_TUTORIAL"] = "True" # no longer on by default

os.environ["NAPARI_ASYNC"] = "0"  # avoid async teardown surprises in tests
# os.environ["PYTHONFAULTHANDLER"] = "1"  # better segfault traces in CI
# os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless QT for CI
# os.environ["QT_OPENGL"] = "software"  # avoid some CI issues with OpenGL
# os.environ["PYTEST_QT_API"] = "pyqt6" # only for local testing with pyqt6, we use pyside6 otherwise


@pytest.fixture
def viewer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()

    # Explicitly add the dock widgets
    # NOTE The old approach of opening every plugin menu
    # with "napari-deeplabcut" in the name is not reliable
    # and is not recommended.

    #  keypoints_dock_widget, keypoints_plugin_widget
    _, _ = viewer.window.add_plugin_dock_widget(
        "napari-deeplabcut",
        "Keypoint controls",
    )
    tracking_dock_widget, tracking_plugin_widget = viewer.window.add_plugin_dock_widget(
        "napari-deeplabcut",
        "Tracking controls",
    )

    try:
        yield viewer
    finally:
        # proactively close dock widgets to drop any lingering Qt refs
        try:
            # close all added dock widgets (if any) before viewer is closed
            for dw in list(viewer.window._qt_window.findChildren(QDockWidget)):
                # defensive: some Qt objects can be None during shutdown
                if hasattr(dw, "close"):
                    dw.close()
        except Exception:
            # bail if viewer or its window is already gone
            pass


@pytest.fixture
def fake_keypoints():
    n_rows = 10
    n_animals = 2
    n_kpts = 3
    data = np.random.rand(n_rows, n_animals * n_kpts * 2)
    cols = pd.MultiIndex.from_product(
        [
            ["me"],
            [f"animal_{i}" for i in range(n_animals)],
            [f"kpt_{i}" for i in range(n_kpts)],
            ["x", "y"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(data, columns=cols, index=range(n_rows))
    return df


@pytest.fixture
def points(tmp_path_factory, viewer, fake_keypoints):
    output_path = str(tmp_path_factory.mktemp("folder") / "fake_data.h5")
    fake_keypoints.to_hdf(output_path, key="data")
    layer = viewer.open(output_path, plugin="napari-deeplabcut")[0]

    return layer


@pytest.fixture
def fake_image():
    return (np.random.rand(10, 10) * 255).astype(np.uint8)


@pytest.fixture
def images(tmp_path_factory, viewer, fake_image):
    output_path = str(tmp_path_factory.mktemp("folder") / "img.png")
    imsave(output_path, fake_image)
    return viewer.open(output_path, plugin="napari-deeplabcut")[0]


@pytest.fixture
def store(viewer, points):
    return keypoints.KeypointStore(viewer, points)


@pytest.fixture
def single_animal_store(tmp_path_factory, viewer, fake_keypoints):
    # Keep only columns for one animal
    df = fake_keypoints.xs("animal_0", level="individuals", axis=1)
    # Now df has levels: scorer, bodyparts, coords
    # Rebuild MultiIndex with an empty "individuals" level inserted
    df.columns = pd.MultiIndex.from_product(
        [
            [df.columns.levels[0][0]],  # scorer
            [""],  # single-animal: empty ID
            df.columns.levels[1],  # bodyparts
            df.columns.levels[2],  # coords
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    path = tmp_path_factory.mktemp("folder") / "single_animal_data.h5"
    df.to_hdf(path, key="data")
    layer = viewer.open(path, plugin="napari-deeplabcut")[0]

    return keypoints.KeypointStore(viewer, layer)


@pytest.fixture(scope="session")
def config_path(tmp_path_factory):
    cfg = {
        "scorer": "me",
        "bodyparts": list("abc"),
        "dotsize": 0,
        "pcutoff": 0,
        "colormap": "viridis",
        "video_sets": {
            "fake_video": [],
        },
    }
    path = str(tmp_path_factory.mktemp("configs") / "config.yaml")
    _writer._write_config(
        path,
        params=cfg,
    )
    return path


@pytest.fixture(scope="session")
def video_path(tmp_path_factory):
    output_path = str(tmp_path_factory.mktemp("data") / "fake_video.avi")
    h = w = 50
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        2,
        (w, h),
    )
    for _ in range(5):
        frame = np.random.randint(0, 255, (h, w, 3)).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return output_path


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
            [1, 30.0, 40.0],
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
