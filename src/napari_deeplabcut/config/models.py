# src/napari_deeplabcut/config/models.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class PathMatchPolicy(str, Enum):
    """How image paths are matched across datasets."""

    ORDERED_DEPTHS = "ordered_depths"


class MetadataKind(str, Enum):
    """High-level metadata container type."""

    IMAGE = "image"
    POINTS = "points"


# -----------------------------------------------------------------------------
# Header model (authoritative wrapper)
# -----------------------------------------------------------------------------


class DLCHeaderModel(BaseModel):
    """
    Structured representation of a DLC header.

    Invariants
    ----------
    - `columns` must be a pandas.MultiIndex at runtime
    - This model does NOT serialize columns directly
      (napari metadata may contain non-JSON objects)

    This model allows opaque runtime storage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    columns: Any = Field(
        ...,
        description="pandas.MultiIndex defining scorer / individuals / bodyparts / coords",
    )

    @property
    def individuals(self) -> list[str]:
        inds = getattr(self.columns, "levels", None)
        if inds is None or "individuals" not in self.columns.names:
            return [""]
        return list(dict.fromkeys(self.columns.get_level_values("individuals")))

    @property
    def bodyparts(self) -> list[str]:
        return list(dict.fromkeys(self.columns.get_level_values("bodyparts")))


# -----------------------------------------------------------------------------
# Metadata models
# -----------------------------------------------------------------------------


class ImageMetadata(BaseModel):
    """
    Metadata for Image layers.

    Stored in napari layer.metadata.

    Invariants
    ----------
    - paths, if present, define frame order
    - root, if present, is a directory path
    """

    kind: MetadataKind = Field(default=MetadataKind.IMAGE)
    paths: list[str] | None = None
    root: str | None = None
    shape: tuple[int, ...] | None = None
    name: str | None = None

    def __repr__(self) -> str:
        # Only show non-None fields, truncate long lists
        fields = []
        for k in ("kind", "name", "root", "shape", "paths"):
            v = getattr(self, k)
            if v is not None:
                if k == "paths":
                    if isinstance(v, list):
                        v = f"[{len(v)} paths]"
                fields.append(f"{k}={v!r}")
        return f"ImageMetadata({', '.join(fields)})"


class PointsMetadata(BaseModel):
    """
    Metadata for Points layers.

    Invariants
    ----------
    - header defines keypoint structure
    - root + paths must align with ImageMetadata when present
    - face_color_cycles must be consistent with header
    """

    kind: MetadataKind = Field(default=MetadataKind.POINTS)

    root: str | None = None
    paths: list[str] | None = None
    shape: tuple[int, ...] | None = None
    name: str | None = None

    project: str | None = None
    header: DLCHeaderModel | None = None

    face_color_cycles: dict[str, dict[str, Any]] | None = None
    colormap_name: str | None = None

    tables: dict[str, dict[str, str]] | None = None

    # Non-serializable runtime attachments (allowed but ignored by pydantic)
    controls: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)
