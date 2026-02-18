# src/napari_deeplabcut/config/models.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

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

    model_config = ConfigDict(extra="allow")

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
    io: IOProvenance | None = None
    save_target: IOProvenance | None = None

    face_color_cycles: dict[str, dict[str, Any]] | None = None
    colormap_name: str | None = None

    tables: dict[str, dict[str, str]] | None = None

    # Non-serializable runtime attachments (allowed but ignored by pydantic)
    controls: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class AnnotationKind(str, Enum):
    """Semantic kind of keypoint annotations for deterministic IO routing.

    Notes
    -----
    This is used to enforce safe saving policies:
    - ``gt``: ground-truth labels (e.g. ``CollectedData_*.h5``)
    - ``machine``: machine predictions/refinements (e.g. ``machinelabels*.h5``)

    The napari layer display name must never be used to infer this value.
    """

    GT = "gt"
    MACHINE = "machine"


class IOProvenance(BaseModel):
    """Authoritative provenance for a Points layer.

    This model captures *identity* for IO, independent of the napari layer name.

    Design goals
    ------------
    - Prefer project-relative, OS-agnostic paths.
    - Store relative paths using POSIX separators ('/'), even on Windows.
    - Be explicit about annotation kind so saving never relies on directory ordering.

    Fields
    ------
    schema_version:
        Version marker for forward-compatible evolution.
    project_root:
        Optional project root directory. When set, ``source_relpath_posix`` is
        interpreted relative to this root.
    source_relpath_posix:
        Project-relative path encoded with POSIX separators ('/').
        Example: ``labeled-data/test/CollectedData_John.h5``.
    kind:
        Whether this layer is ground-truth or machine output.
    dataset_key:
        HDF5 key used for the keypoints table (default: ``keypoints``).
    """

    # Keep minimal but resilient to future additions
    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(default=1, description="Provenance schema version")
    project_root: str | None = Field(default=None, description="Project root directory")
    source_relpath_posix: str | None = Field(
        default=None,
        description="Project-relative POSIX path to the source .h5 (forward slashes).",
    )
    kind: AnnotationKind | None = Field(default=None, description="Annotation kind for routing")
    dataset_key: str = Field(default="keypoints", description="HDF5 key for keypoints table")

    @field_validator("source_relpath_posix")
    @classmethod
    def _normalize_relpath(cls, v: str | None) -> str | None:
        """Normalize provenance paths to POSIX separators.

        This keeps stored metadata OS-agnostic and stable across platforms.
        """
        if v is None:
            return None
        return v.replace("\\", "/")
