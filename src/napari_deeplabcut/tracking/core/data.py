from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRACKING_LAYER_METADATA_KEY = "ndlc_tracking"
TRACKING_SCHEMA_VERSION = 1


# ----- Data schemas -----
@dataclass
class TrackingWorkerData:
    tracker_name: str  # model name
    video: np.ndarray
    keypoints: np.ndarray  # (num_keypoint, 3)
    # [0]: frame number in `video` [1]: x, [2]: y

    keypoint_features: pd.DataFrame
    # one row per query keypoint. Order must be preserved.

    keypoint_range: tuple[int, int]
    backward_tracking: bool
    reference_frame_index: int | None = None


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

    keypoint_features:
        shape (N, M), one row per tracked keypoint row, aligned with `keypoints`
    """

    keypoints: np.ndarray
    keypoint_features: pd.DataFrame


# ------ Data features ------


def coerce_features_df(features) -> pd.DataFrame:
    """Return a defensive DataFrame copy with a clean RangeIndex."""
    if isinstance(features, pd.DataFrame):
        return features.reset_index(drop=True).copy()
    return pd.DataFrame(features).reset_index(drop=True).copy()


def add_query_identity_columns(
    seed_features: pd.DataFrame,
    *,
    query_frame: int,
    source_layer_name: str,
) -> pd.DataFrame:
    """
    Add stable identity columns for each seed query before tracking.
    Aims to recover semantic point identity
    after tracker inference.
    """
    df = coerce_features_df(seed_features)

    df["tracking_query_index"] = np.arange(len(df), dtype=int)
    df["tracking_query_frame"] = int(query_frame)
    df["tracking_source_layer_name"] = str(source_layer_name)

    return df


def expand_query_features_over_time(
    seed_features: pd.DataFrame,
    *,
    frame_ids: np.ndarray,
    visibility: np.ndarray | None,
    tracker_name: str,
) -> pd.DataFrame:
    """
    Repeat seed features across all tracked frames, preserving original
    semantic columns (e.g. label, id) and adding tracking-specific fields.

    Parameters
    ----------
    seed_features
        One row per seed/query point, in the same order as the model query order.
    frame_ids
        Actual frame indices corresponding to the model output time axis.
    visibility
        Optional visibility array of shape (T, K) or (T, K, 1).
    tracker_name
        Human-readable tracker name, e.g. "Cotracker 3".
    """
    seed = coerce_features_df(seed_features)

    K = len(seed)
    T = len(frame_ids)

    repeated = pd.concat([seed] * T, ignore_index=True)

    repeated["tracking_tracker_name"] = str(tracker_name)
    repeated["tracking_frame"] = np.repeat(np.asarray(frame_ids, dtype=int), K)
    repeated["tracking_is_prediction"] = True

    if visibility is not None:
        vis = np.asarray(visibility)
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]
        elif vis.ndim == 3 and vis.shape[0] == 1:
            vis = vis.squeeze(0)

        expected = (T, K)
        if vis.shape != expected:
            raise ValueError(f"Visibility shape mismatch. Expected {expected}, got {vis.shape}.")

        repeated["tracking_visible"] = vis.reshape(T * K).astype(bool)
    else:
        repeated["tracking_visible"] = True

    return repeated


def build_tracking_result_metadata(
    source_metadata: dict | None,
    *,
    tracker_name: str,
    source_layer_name: str,
    query_frame: int,
) -> dict:
    """
    Build metadata for a tracking-result Points layer while keeping the source
    metadata around as much as possible.
    """
    md = deepcopy(source_metadata or {})
    md[TRACKING_LAYER_METADATA_KEY] = {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "kind": "cotracker-result",
        "tracker_name": str(tracker_name),
        "source_layer_name": str(source_layer_name),
        "query_frame": int(query_frame),
    }
    return md


def is_tracking_result_points_layer(layer) -> bool:
    md = getattr(layer, "metadata", {}) or {}
    info = md.get(TRACKING_LAYER_METADATA_KEY)
    return isinstance(info, dict) and info.get("kind") == "cotracker-result"
