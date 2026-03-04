# src/napari_deeplabcut/config/models.py
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


def unsorted_unique(array: Sequence) -> np.ndarray:
    """Return the unsorted unique elements of an array."""
    _, inds = np.unique(array, return_index=True)
    return np.asarray(array)[np.sort(inds)]


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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    columns: Any = Field(...)

    # ----------------------------
    # Input normalization
    # ----------------------------
    @field_validator("columns", mode="before")
    @classmethod
    def _coerce_columns(cls, v: Any) -> Any:
        """
        Accept common header representations and normalize to a stable in-memory form.

        Supported inputs:
        - pandas.MultiIndex (stored as-is)
        - list/tuple of tuples (stored as list[tuple[str,...]])
        - dict payloads with {"columns": ...} are handled by model_validate upstream
        """
        if v is None:
            return v

        # Normalize pandas Index/MultiIndex into list of tuples for portability
        if isinstance(v, pd.MultiIndex):
            return v
        if isinstance(v, pd.Index):
            return [str(x) for x in v.to_list()]

        if isinstance(v, (list, tuple)) and v:
            # Ensure tuple-of-strings
            out = []
            for t in v:
                if isinstance(t, (list, tuple)):
                    out.append(tuple(map(str, t)))
                else:
                    out.append((str(t),))
            return out

        return v

    # ----------------------------
    # Canonical MultiIndex view
    # ----------------------------
    def as_multiindex(self) -> pd.MultiIndex:
        """
        Return a canonical 4-level DLC MultiIndex:
          (scorer, individuals, bodyparts, coords)

        Accepts:
        - already MultiIndex
        - list/tuple of tuples (3- or 4-level common cases)
        """
        cols = self.columns

        if isinstance(cols, pd.MultiIndex):
            return cols

        if isinstance(cols, (list, tuple)) and cols:
            first = cols[0]
            if not isinstance(first, (list, tuple)):
                raise TypeError(f"columns must be tuples, got {type(first)!r}")

            n = len(first)

            if n == 4:
                tuples = [tuple(map(str, t)) for t in cols]
                return pd.MultiIndex.from_tuples(
                    tuples,
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )

            if n == 3:
                # legacy: (scorer, bodyparts, coords) -> insert empty individuals
                tuples = [(str(t[0]), "", str(t[1]), str(t[2])) for t in cols]
                return pd.MultiIndex.from_tuples(
                    tuples,
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )

            tuples = [tuple(map(str, t)) for t in cols]
            return pd.MultiIndex.from_tuples(tuples, names=[None] * n)

        raise TypeError(f"Unsupported columns type: {type(cols)!r}")

    # ----------------------------
    # Self-documenting, stable API
    # ----------------------------
    @property
    def scorer(self) -> str | None:
        cols = self.as_multiindex()
        if "scorer" in (cols.names or []):
            vals = cols.get_level_values("scorer")
            return str(vals[0]) if len(vals) else None
        return str(cols.to_list()[0][0]) if len(cols) else None

    @property
    def individuals(self) -> list[str]:
        cols = self.as_multiindex()
        if "individuals" not in (cols.names or []):
            return [""]
        return list(dict.fromkeys(map(str, cols.get_level_values("individuals"))))

    @property
    def bodyparts(self) -> list[str]:
        cols = self.as_multiindex()
        if "bodyparts" in (cols.names or []):
            return list(dict.fromkeys(map(str, cols.get_level_values("bodyparts"))))
        return list(dict.fromkeys([str(t[2]) for t in cols.to_list()]))

    @property
    def coords(self) -> list[str]:
        cols = self.as_multiindex()
        if "coords" in (cols.names or []):
            return list(dict.fromkeys(map(str, cols.get_level_values("coords"))))
        # fallback: assume last level is coords
        return list(dict.fromkeys([str(t[-1]) for t in cols.to_list()]))

    def _get_unique(self, name: str) -> list[str] | None:
        if name in self.columns.names:
            return list(unsorted_unique(self.columns.get_level_values(name)))
        return None

    def form_individual_bodypart_pairs(self) -> list[tuple[str, str]]:
        """
        Return ordered list of (individual, bodypart) pairs, DLC-style.
        """
        cols = self.as_multiindex()

        names = cols.names or []
        to_drop = [n for n in names if n not in ("individuals", "bodyparts")]
        cols2 = cols.droplevel(to_drop).unique() if to_drop else cols.unique()

        # Ensure individuals level exists (single-animal legacy)
        if "individuals" not in (cols2.names or []):
            inds = self.individuals
            cols2 = pd.MultiIndex.from_product(
                [inds, cols2],
                names=["individuals"] + list(cols2.names or []),
            )

        return [(str(i), str(b)) for (i, b) in cols2.to_list()]

    # ----------------------------
    # Construction helpers
    # ----------------------------
    @classmethod
    def from_config(cls, config: dict) -> DLCHeaderModel:
        """
        Build header from DLC config.yaml content (single or multi-animal).
        """
        multi = config.get("multianimalproject", False)
        scorer = [config["scorer"]]

        if multi:
            columns = pd.MultiIndex.from_product(
                [scorer, config["individuals"], config["multianimalbodyparts"], ["x", "y"]],
                names=["scorer", "individuals", "bodyparts", "coords"],
            )
            if len(config.get("uniquebodyparts", [])):
                temp = pd.MultiIndex.from_product(
                    [scorer, ["single"], config["uniquebodyparts"], ["x", "y"]],
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )
                columns = columns.append(temp)
        else:
            columns = pd.MultiIndex.from_product(
                [scorer, config["bodyparts"], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )

        return cls(columns=columns)

    # ----------------------------
    # Serialization intended for layer.metadata
    # ----------------------------
    def to_metadata_payload(self) -> dict[str, Any]:
        """
        Return a portable payload safe to store in napari layer.metadata.
        """
        cols = self.columns
        if isinstance(cols, pd.MultiIndex):
            cols = [tuple(map(str, t)) for t in cols.to_list()]
        elif isinstance(cols, pd.Index):
            cols = [str(x) for x in cols.to_list()]
        return {"columns": cols}


# -----------------------------------------------------------------------------
# Metadata & I/O models
# -----------------------------------------------------------------------------


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
    kind: AnnotationKind | None = Field(default=None, description="Annotation kind for routing", strict=True)
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

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: AnnotationKind | str | None) -> AnnotationKind | None:
        """Validate that kind is either an AnnotationKind or a valid string."""
        if v is None:
            return None
        if isinstance(v, AnnotationKind):
            return v
        try:
            return AnnotationKind(v)
        except ValueError as e:
            raise ValueError(f"Invalid annotation kind: {v!r}") from e

    @field_validator("project_root")
    @classmethod
    def _validate_project_root(cls, v: str | None) -> str | None:
        """Store project_root without requiring the path to exist.

        Existence and type (file vs directory) checks are intentionally deferred
        to path resolution time to keep serialized provenance portable across
        machines and project locations.
        """
        if v is None:
            return None
        return str(v)


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

    kind: MetadataKind = Field(default=MetadataKind.IMAGE, strict=True)
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
