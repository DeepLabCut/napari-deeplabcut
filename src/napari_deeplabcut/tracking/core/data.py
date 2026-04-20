from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackingWorkerData:
    tracker_name: str  # model name
    video: np.ndarray
    keypoints: np.ndarray  # (num_keypoint, 3)
    # [0]: frame number in `video` [1]: x, [2]: y
    keypoint_features: dict
    keypoint_range: tuple[int, int]
    backward_tracking: bool


@dataclass(frozen=True)
class TrackingModelInputs:
    """Inputs required for tracking model processing."""

    video: np.ndarray  # (num_frames, height, width, channels)
    keypoints: np.ndarray  # base on model requirements
    metadata: dict[str, Any]  # Additional metadata if needed


@dataclass
class RawModelOutputs:
    """Outputs from tracking model processing."""

    keypoints: np.ndarray  # (num_frames, num_keypoints, 2)
    keypoint_features: dict[str, Any]  # Additional features if needed


@dataclass(frozen=True)
class TrackingWorkerOutput:
    """
    Returned by models and passed on to the plugin by the worker.

    keypoints: (N, 3)
        [:, 0] = frame index (int)
        [:, 1] = x coordinate (float)
        [:, 2] = y coordinate (float)
    keypoint_features: dict
        Per-keypoint/track metadata. May contain e.g. visibility scores.
    """

    keypoints: np.ndarray
    keypoint_features: dict[str, Any]
