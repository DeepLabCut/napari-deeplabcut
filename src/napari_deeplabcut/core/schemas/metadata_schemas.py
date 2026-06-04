# src/napari_deeplabcut/core/schemas.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from napari_deeplabcut.config.models import DLCHeaderModel, PointsMetadata


class PointsDataModel(BaseModel):
    """Validated napari Points data for DLC keypoints writing.

    Expected napari-style layout: (N, 3) with columns [frame_index, y, x].
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any = Field(..., description="Array-like of shape (N, 3): [frame, y, x]")

    @field_validator("data")
    @classmethod
    def _validate_points_array(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Points data must have shape (N, 3) [frame, y, x]. Got {arr.shape}.")
        # Ensure numeric
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Points data must be numeric. Got dtype={arr.dtype}.")
        return arr

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def frame_inds(self) -> np.ndarray:
        # floor/cast to int is typical for frame indices
        return self.data[:, 0].astype(int)

    @property
    def xy_dlc(self) -> np.ndarray:
        # Convert napari [y, x] -> DLC [x, y]
        return self.data[:, [2, 1]]


class KeypointPropertiesModel(BaseModel):
    """Validated napari layer.properties for keypoint points.

    Napari stores per-point properties as sequences (often numpy arrays).
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    label: Sequence[str] = Field(..., description="Bodypart label per point")
    id: Sequence[str] = Field(..., description="Individual id per point ('' if single animal)")
    likelihood: Sequence[float] | None = Field(default=None, description="Likelihood per point (optional)")

    @field_validator("label", "id", mode="before")
    @classmethod
    def _coerce_str_seq(cls, v):
        # make sure we can len() it and iterate
        if v is None:
            return v
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v)

    @field_validator("likelihood", mode="before")
    @classmethod
    def _coerce_float_seq(cls, v):
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v)


class PointsWriteInputModel(BaseModel):
    """Validated bundle of what form_df needs to write DLC keypoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: PointsDataModel
    meta: PointsMetadata
    props: KeypointPropertiesModel

    @model_validator(mode="after")
    def _validate_required_header(self):
        # For writing, header must exist
        if self.meta.header is None:
            raise ValueError("PointsMetadata.header is required for writing keypoints.")
        if not isinstance(self.meta.header, DLCHeaderModel):
            raise TypeError("PointsMetadata.header must be DLCHeaderModel.")
        return self

    @model_validator(mode="after")
    def _validate_lengths(self):
        n = self.points.n
        if len(self.props.label) != n:
            raise ValueError(f"properties['label'] length {len(self.props.label)} != N points {n}")
        if len(self.props.id) != n:
            raise ValueError(f"properties['id'] length {len(self.props.id)} != N points {n}")
        if self.props.likelihood is not None and len(self.props.likelihood) != n:
            raise ValueError(f"properties['likelihood'] length {len(self.props.likelihood)} != N points {n}")
        return self

    @model_validator(mode="after")
    def _validate_paths_indexing(self):
        # If paths exist, frame indices must be in range
        if self.meta.paths:
            max_idx = len(self.meta.paths) - 1
            fi = self.points.frame_inds
            if fi.size and (fi.min() < 0 or fi.max() > max_idx):
                raise ValueError(
                    f"Frame indices out of bounds for metadata.paths: "
                    f"min={fi.min()}, max={fi.max()}, paths_len={len(self.meta.paths)}"
                )
        return self


class PointsLayerAttributesModel(BaseModel):
    """NPE2 writer attributes bundle for a Points layer."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_metadata_dict(self):
        if not isinstance(self.metadata, dict):
            raise TypeError("attributes['metadata'] must be a dict")
        if not isinstance(self.properties, dict):
            raise TypeError("attributes['properties'] must be a dict")
        return self
