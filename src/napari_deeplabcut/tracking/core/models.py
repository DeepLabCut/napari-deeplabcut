import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from napari_deeplabcut.tracking.core.data import (
    RawModelOutputs,
    TrackingModelInputs,
    TrackingWorkerData,
    TrackingWorkerOutput,
    coerce_features_df,
    expand_query_features_over_time,
)

if TYPE_CHECKING:
    from napari_deeplabcut.tracking.core.data import TrackingModel

# List of available tracking models.
# Automatically populated via the @register_backbone decorator.
AVAILABLE_TRACKERS: dict[str, dict[str, Any]] = {}

# TODO @C-Achard: consider splitting into base.py (TrackingModel) and putting models in core/models/

logger = logging.getLogger(__name__)


def register_backbone(model_name: str):
    def decorator(cls):
        AVAILABLE_TRACKERS[model_name] = {
            "class": cls,
        }
        return cls

    return decorator


class TrackingModel(ABC):
    """Abstract base class for tracking models.
    Use this to add new tracking models.
    """

    # These fields must be set per model
    name: str
    info_text: str

    def __init__(self, cfg: "TrackingWorkerData"):
        super().__init__()
        self.cfg: TrackingWorkerData = cfg
        if not isinstance(cfg, TrackingWorkerData):
            raise ValueError("cfg must be an instance of TrackingWorkerData")

        self.device = self.auto_set_device()
        self.model = self.load_model(self.device)

    def auto_set_device(self):
        """Automatically set the device for the model.
        Override this method if you have specific device requirements.
        """
        import torch

        # check for MPS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        return self.device

    @abstractmethod
    def load_model(self, device: str) -> Any:
        """Load the model on the specified device.
        Make sure to use proper device and set to eval mode.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, cfg: "TrackingWorkerData", **kwargs) -> TrackingModelInputs:
        """Prepare inputs for processing."""
        raise NotImplementedError

    @abstractmethod
    def run(self, inputs: TrackingModelInputs, progress_callback, stop_callback, **kwargs) -> RawModelOutputs:
        """Process the inputs and return outputs."""
        raise NotImplementedError

    @abstractmethod
    def prepare_outputs(
        self, model_outputs: RawModelOutputs, worker_inputs: "TrackingWorkerData" = None, **kwargs
    ) -> "TrackingWorkerOutput":
        """Prepare outputs after processing."""
        raise NotImplementedError

    @abstractmethod
    def validate_outputs(self, inputs: TrackingModelInputs, outputs: "TrackingWorkerOutput") -> tuple[bool, str]:
        """Validate the outputs."""
        raise NotImplementedError


# pragma: no cover
ct3 = "Cotracker 3"


@register_backbone(ct3)
class Cotracker3(TrackingModel):
    name = ct3
    info_text = (
        "Cotracker 3 model from Facebook Research.\n"
        "See https://cotracker3.github.io/ and CoTracker3: "
        "Simpler and Better Point Tracking by "
        "Pseudo-Labelling Real Videos by Karaev et al., 2024."
    )

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self, device: str):
        import torch

        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
        ).to(device)
        model.eval()
        return model

    def prepare_inputs(self, cfg: "TrackingWorkerData", **kwargs):
        self.cfg = cfg

        # video is originally (num_frames, H, W, C)
        video = np.asarray(self.cfg.video)

        # IMPORTANT: do NOT mutate worker_inputs in place
        queries = np.asarray(self.cfg.keypoints, dtype=float).copy()

        # CoTracker expects [t, y, x]
        queries[:, [1, 2]] = queries[:, [2, 1]]

        metadata = {
            "keypoint_range": self.cfg.keypoint_range,
            "backward_tracking": self.cfg.backward_tracking,
            "reference_frame_index": self.cfg.reference_frame_index,
        }

        return TrackingModelInputs(video=video, keypoints=queries, metadata=metadata)

    def run(self, inputs: TrackingModelInputs, progress_callback, stop_callback) -> RawModelOutputs:
        import torch

        if stop_callback():
            return None

        # inputs.video is (T, H, W, C)
        video = (
            torch.from_numpy(inputs.video).to(self.device).float().permute(0, 3, 1, 2)[None]  # -> (1, T, C, H, W)
        )

        # inputs.keypoints is (K, 3), already converted in prepare_inputs()
        queries = torch.from_numpy(inputs.keypoints).to(self.device).float()[None]  # -> (1, K, 3)

        total_frames = int(inputs.video.shape[0])
        progress_callback(0, total_frames)

        with torch.inference_mode():
            pred_tracks, pred_visibility = self.model(
                video,
                queries=queries,
            )

        if pred_tracks is None or pred_visibility is None:
            raise RuntimeError("CoTracker offline returned no predictions for the provided clip and queries.")

        progress_callback(total_frames, total_frames)

        if stop_callback():
            return None

        tracks = pred_tracks.detach().cpu().numpy()
        visibility = pred_visibility.detach().cpu().numpy()

        # Normalize shapes:
        # expected from model: (1, T, K, 2) and (1, T, K, 1) or similar
        if tracks.ndim == 4 and tracks.shape[0] == 1:
            tracks = tracks[0]  # -> (T, K, 2)

        if visibility.ndim == 4 and visibility.shape[0] == 1:
            visibility = visibility[0]  # -> (T, K, 1)

        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]  # -> (T, K)

        return RawModelOutputs(
            keypoints=tracks,
            keypoint_features={"visibility": visibility},
        )

    def prepare_outputs(
        self,
        model_outputs: RawModelOutputs,
        worker_inputs: TrackingWorkerData = None,
        **kwargs,
    ) -> "TrackingWorkerOutput":
        """
        Convert CoTracker outputs into canonical plugin format while preserving
        original per-query semantic identity.

        Result:
        - keypoints: (N, 3) = [frame_idx, x, y]
        - keypoint_features: one row per tracked point, aligned with keypoints
        """
        tracks = np.asarray(model_outputs.keypoints, dtype=float)  # expected (T, K, 2)

        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"Expected tracks with shape (T, K, 2), got {tracks.shape}.")

        visibility = model_outputs.keypoint_features.get("visibility")
        if visibility is not None:
            visibility = np.asarray(visibility)
            if visibility.ndim == 3 and visibility.shape[-1] == 1:
                visibility = visibility[..., 0]

        T1, T2 = map(int, worker_inputs.keypoint_range)
        frame_ids = np.arange(T1, T2, dtype=int)

        # IMPORTANT:
        # When backward_tracking=True, the model saw the time-reversed video slice.
        # So we must reverse the TIME axis before flattening, not the flattened rows.
        if worker_inputs.backward_tracking:
            tracks = tracks[::-1, :, :]
            if visibility is not None:
                visibility = visibility[::-1, :]

        T, K, _ = tracks.shape
        if T != len(frame_ids):
            raise ValueError(f"Time dimension mismatch. tracks has T={T}, frame_ids has len={len(frame_ids)}.")

        seed_features = coerce_features_df(worker_inputs.keypoint_features)
        if len(seed_features) != K:
            raise ValueError(f"Seed feature row count mismatch. Expected K={K}, got {len(seed_features)}.")

        # Flatten frame-major, preserving query order inside each frame
        xy = tracks.reshape(T * K, 2)
        keypoints = np.column_stack((np.repeat(frame_ids, K), xy))

        # Restore plugin convention from [frame, y, x] -> [frame, x, y]
        keypoints[:, [1, 2]] = keypoints[:, [2, 1]]

        keypoint_features = expand_query_features_over_time(
            seed_features,
            frame_ids=frame_ids,
            visibility=visibility,
            tracker_name=self.name,
        )

        return TrackingWorkerOutput(
            keypoints=keypoints,
            keypoint_features=keypoint_features,
        )

    def _process_step(self, window_frames, is_first_step, queries):
        """
        Internal helper for chunked processing.
        """
        import torch

        video_chunk = (
            torch.tensor(np.stack(window_frames[-self.model.step * 2 :]), device=self.device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        logger.debug(f"Video chunk shape: {video_chunk.shape}, Queries shape: {queries.shape}")
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],
            add_support_grid=True,
        )

    def validate_outputs(self, inputs: TrackingModelInputs, outputs: "TrackingWorkerOutput") -> tuple[bool, str]:
        """Validate the outputs."""
        if not isinstance(outputs.keypoints, np.ndarray):
            return False, "Outputs keypoints is not a numpy array."
        if not outputs.keypoints.ndim == 2:
            return False, "Outputs keypoints is not a 2D array."
        if not outputs.keypoints.shape[1] == 3:
            return False, "Outputs keypoints does not have 3 columns."
        # For CoTracker3, outputs contain tracked keypoints for every frame in the
        # keypoint_range. If keypoint_range = (T1, T2) and there are K input
        # keypoints, we expect (T2 - T1) * K output rows.
        metadata = getattr(inputs, "metadata", None)
        if metadata is None or "keypoint_range" not in metadata:
            return False, "Missing keypoint_range metadata required for validation."
        keypoint_range = metadata["keypoint_range"]
        if not isinstance(keypoint_range, (tuple, list)) or len(keypoint_range) != 2:
            return False, "Invalid keypoint_range metadata; expected (T1, T2)."
        T1, T2 = keypoint_range
        try:
            T1_int = int(T1)
            T2_int = int(T2)
        except (TypeError, ValueError):
            return False, "keypoint_range values must be integers."
        if T2_int <= T1_int:
            return False, "keypoint_range must satisfy T2 > T1."
        K = inputs.keypoints.shape[0]
        expected_n_keypoints = (T2_int - T1_int) * K
        if outputs.keypoints.shape[0] != expected_n_keypoints:
            return (
                False,
                f"Number of output keypoints does not match expected ((T2 - T1) * K) = {expected_n_keypoints}.",
            )
        return True, ""
