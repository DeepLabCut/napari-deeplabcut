import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from napari_deeplabcut.tracking._data import (
    RawModelOutputs,
    TrackingModelInputs,
    TrackingWorkerData,
    TrackingWorkerOutput,
)

if TYPE_CHECKING:
    from napari_deeplabcut.tracking._data import TrackingModel

# List of available tracking models.
# Automatically populated via the @register_backbone decorator.
AVAILABLE_TRACKERS: dict[str, dict[str, Any]] = {}


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

    def validate_outputs(self, inputs: TrackingModelInputs, outputs: "TrackingWorkerOutput") -> tuple[bool, str]:
        """Validate the outputs."""
        if not isinstance(outputs.keypoints, np.ndarray):
            return False, "Outputs keypoints is not a numpy array."
        if not outputs.keypoints.ndim == 2:
            return False, "Outputs keypoints is not a 2D array."
        if not outputs.keypoints.shape[1] == 3:
            return False, "Outputs keypoints does not have 3 columns."
        if not outputs.keypoints.shape[0] == inputs.keypoints.shape[0]:
            return False, "Number of output keypoints does not match number of input keypoints."
        return True, ""


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
            "cotracker3_online",
        ).to(device)
        model.eval()
        return model

    def prepare_inputs(self, cfg: "TrackingWorkerData", **kwargs):
        self.cfg = cfg
        # video is originally of shape (num_frames, height, width, channels)
        video = np.array(self.cfg.video)

        # We need to swap x, y so that it matches what cotracker expects
        self.cfg.keypoints[:, [1, 2]] = self.cfg.keypoints[:, [2, 1]]
        queries = np.asarray(self.cfg.keypoints)
        metadata = {
            "keypoint_range": self.cfg.keypoint_range,
            "backward_tracking": self.cfg.backward_tracking,
        }
        return TrackingModelInputs(
            video=video,
            keypoints=queries,
            metadata=metadata,
        )

    def run(self, inputs: TrackingModelInputs, progress_callback, stop_callback) -> RawModelOutputs:
        import torch

        video = torch.from_numpy(inputs.video).to(self.device).float()
        queries = torch.from_numpy(inputs.keypoints).to(self.device).float()

        window_frames = []
        is_first_step = True
        for i, frame in enumerate(video):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, _pred_visibility = self._process_step(window_frames, is_first_step, queries=queries)
                is_first_step = False
            window_frames.append(frame)
            progress_callback(i, len(video))
            if stop_callback():
                logger.debug("Tracking stopped.")
                return

        pred_tracks, _pred_visibility = self._process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1 :],
            is_first_step,
            queries=queries,
        )
        progress_callback(len(video), len(video))

        tracks = pred_tracks.squeeze().cpu().numpy()
        _pred_visibility = _pred_visibility.squeeze().cpu().numpy()  #

        return RawModelOutputs(
            keypoints=tracks,
            keypoint_features={"visibility": _pred_visibility},
        )

    def prepare_outputs(
        self, model_outputs: RawModelOutputs, worker_inputs: TrackingWorkerData = None, **kwargs
    ) -> "TrackingWorkerOutput":
        """
        Convert raw outputs into canonical format:
        - Flatten tracks into (N, 3): [frame_idx, x, y]
        - Restore original x/y order for worker.
        """
        tracks = model_outputs.keypoints  # shape: (T, K, 2)
        T1, T2 = worker_inputs.keypoint_range
        K = worker_inputs.keypoints.shape[0]

        # Flatten and add frame indices
        frame_ids = np.repeat(np.arange(T1, T2), K)
        tracks = tracks.reshape(-1, 2)

        # If backward tracking requested
        if worker_inputs.backward_tracking:
            tracks = tracks[::-1]
        # Combine into (N, 3)
        keypoints = np.column_stack((frame_ids, tracks))

        keypoints_features = model_outputs.keypoint_features
        keypoints_features = pd.concat(
            [worker_inputs.keypoint_features] * len(np.unique(keypoints[:, 0])),
            ignore_index=True,
        )

        # Restore original x/y order
        keypoints[:, [1, 2]] = keypoints[:, [2, 1]]

        return TrackingWorkerOutput(
            keypoints=keypoints,
            keypoint_features=keypoints_features,
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
        # logger.debug(f"Video chunk shape: {video_chunk.shape}, Queries shape: {queries.shape}")
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],
            add_support_grid=True,
        )
