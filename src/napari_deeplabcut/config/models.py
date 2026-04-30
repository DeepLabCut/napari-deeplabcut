# src/napari_deeplabcut/config/models.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# TODO @C-Achard: move to core/


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
# Project structure models
# -----------------------------------------------------------------------------
class DLCProjectContext(BaseModel):
    """
    Best-effort DLC project/location context inferred from available hints.

    All fields are optional because users may open partial project fragments
    (e.g. only a video, only a labeled-data folder, only annotations).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    root_anchor: Path | None = Field(
        default=None,
        description="Base folder anchor used for project-relative resolution when no stronger hint exists.",
    )
    project_root: Path | None = Field(
        default=None,
        description="Folder containing config.yaml, if inferable.",
    )
    config_path: Path | None = Field(
        default=None,
        description="Resolved path to DLC config.yaml, if inferable.",
    )
    dataset_folder: Path | None = Field(
        default=None,
        description="Resolved labeled-data/<dataset> folder, if inferable.",
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> DLCProjectContext:
        def _norm(p: Path | None) -> Path | None:
            if p is None:
                return None
            try:
                return p.expanduser().resolve()
            except Exception:
                return p

        root_anchor = _norm(self.root_anchor)
        project_root = _norm(self.project_root)
        config_path = _norm(self.config_path)
        dataset_folder = _norm(self.dataset_folder)

        # If config_path is present, project_root should default to its parent
        if config_path is not None and project_root is None:
            project_root = config_path.parent

        # If project_root exists and config_path is missing, infer config path conventionally
        if project_root is not None and config_path is None:
            candidate = project_root / "config.yaml"
            if candidate.exists():
                config_path = candidate

        # If root_anchor is missing, prefer project_root, otherwise dataset_folder
        if root_anchor is None:
            root_anchor = project_root or dataset_folder

        object.__setattr__(self, "root_anchor", root_anchor)
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "config_path", config_path)
        object.__setattr__(self, "dataset_folder", dataset_folder)
        return self


# -----------------------------------------------------------------------------
# Header model (authoritative wrapper)
# -----------------------------------------------------------------------------
class DLCHeaderModel(BaseModel):
    """
    Authoritative, pandas-optional DLC header specification.

    Design goals
    ------------
    - Pandas is NOT required at runtime for this model.
    - Internal representation is always portable:
        columns: list[tuple[str, ...]]
        names: optional list[str] aligned to tuple length (may be None)
    - Accepts pandas.MultiIndex as input when pandas is installed (best-effort),
      but never stores it internally.

    Semantics
    ---------
    Canonical meaning (when present):
      scorer, individuals, bodyparts, coords

    We support both:
    - 4-level canonical tuples: (scorer, individuals, bodyparts, coords)
    - 3-level legacy tuples: (scorer, bodyparts, coords)  -> treated as individuals=""
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    columns: list[tuple[str, ...]] = Field(default_factory=list)
    names: list[str] | None = None

    # ----------------------------
    # Input normalization
    # ----------------------------
    @field_validator("columns", mode="before")
    @classmethod
    def _coerce_columns(cls, v: Any) -> list[tuple[str, ...]]:
        """
        Accept common header representations and normalize to list-of-tuples.

        Supported:
        - list/tuple of tuples
        - pandas.MultiIndex (if pandas installed)
        - pandas.Index (if pandas installed)
        """
        if v is None:
            return []

        # If payload dict accidentally arrives here, unwrap.
        if isinstance(v, dict) and "columns" in v:
            v = v["columns"]

        # pandas objects: import lazily (optional dependency)
        try:
            import pandas as pd  # type: ignore

            if isinstance(v, pd.MultiIndex):
                return [tuple(map(str, t)) for t in v.to_list()]
            if isinstance(v, pd.Index):
                return [(str(x),) for x in v.to_list()]
        except Exception:
            pass

        # list/tuple input
        if isinstance(v, (list, tuple)):
            out: list[tuple[str, ...]] = []
            for item in v:
                if isinstance(item, (list, tuple)):
                    out.append(tuple(map(str, item)))
                else:
                    out.append((str(item),))
            return out

        raise TypeError(f"Unsupported columns type: {type(v)!r}")

    @field_validator("names", mode="before")
    @classmethod
    def _coerce_names(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return None

    @field_validator("names")
    @classmethod
    def _validate_names(cls, v: list[str] | None, info) -> list[str] | None:
        # If provided, length must match tuple length (when columns non-empty).
        cols = info.data.get("columns") or []
        if v is None or not cols:
            return v
        n = len(cols[0])
        if len(v) != n:
            # Be tolerant: if mismatch, drop names rather than fail hard.
            return None
        return v

    # ----------------------------
    # Core shape helpers (pandas-free)
    # ----------------------------
    @property
    def is_single_animal(self) -> bool:
        canon = self._canonical_4()
        return bool(canon) and all(t[1] == "" for t in canon)

    def _level_index(self, name: str) -> int | None:
        if not self.names:
            return None
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def _get_level_values(self, idx: int) -> list[str]:
        if not self.columns:
            return []
        vals = []
        for t in self.columns:
            if idx < len(t):
                vals.append(str(t[idx]))
        # stable unique, preserve order
        return list(dict.fromkeys(vals))

    def _canonical_4(self) -> list[tuple[str, str, str, str]]:
        """
        Return canonical 4-tuples (scorer, individuals, bodyparts, coords).

        - If already 4-level: mapped by names if present, else by position.
        - If 3-level legacy: individuals="" inserted.
        - Otherwise: best-effort fallback to positional mapping.
        """
        out: list[tuple[str, str, str, str]] = []
        if not self.columns:
            return out

        # If names are present, use them preferentially.
        ix_scorer = self._level_index("scorer")
        ix_inds = self._level_index("individuals")
        ix_bp = self._level_index("bodyparts")
        ix_coords = self._level_index("coords")

        for t in self.columns:
            tt = tuple(map(str, t))
            if len(tt) == 4:
                if None not in (ix_scorer, ix_inds, ix_bp, ix_coords):
                    out.append((tt[ix_scorer], tt[ix_inds], tt[ix_bp], tt[ix_coords]))  # type: ignore[index]
                else:
                    out.append((tt[0], tt[1], tt[2], tt[3]))
            elif len(tt) == 3:
                # legacy (scorer, bodyparts, coords)
                if None not in (ix_scorer, ix_bp, ix_coords):
                    out.append((tt[ix_scorer], "", tt[ix_bp], tt[ix_coords]))  # type: ignore[index]
                else:
                    out.append((tt[0], "", tt[1], tt[2]))
            else:
                raise ValueError(
                    f"Unsupported DLC header shape {tt!r} (got {len(tt)} levels; "
                    f"expected 3 or 4; names={self.names!r})."
                )

        return out

    # ----------------------------
    #  Pandas interop
    # ----------------------------
    def as_multiindex(self):
        """
        OPTIONAL: Convert to pandas.MultiIndex if pandas is installed.
        This keeps pandas-specific modules working without making pandas a core invariant.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("pandas is required for as_multiindex() but is not installed") from e

        canon = self._canonical_4()
        names = ["scorer", "individuals", "bodyparts", "coords"]
        return pd.MultiIndex.from_tuples(canon, names=names)

    # ----------------------------
    # Self-documenting API (what callers should use)
    # ----------------------------
    @property
    def scorer(self) -> str | None:
        canon = self._canonical_4()
        return canon[0][0] if canon else None

    @property
    def scorers(self) -> list[str]:
        canon = self._canonical_4()
        scorers = [s for s, _, _, _ in canon]
        return list(dict.fromkeys(scorers))

    @property
    def individuals(self) -> list[str]:
        canon = self._canonical_4()
        inds = [i for _, i, _, _ in canon]
        uniq = list(dict.fromkeys(inds))
        return uniq if uniq else [""]

    @property
    def bodyparts(self) -> list[str]:
        canon = self._canonical_4()
        bps = [b for _, _, b, _ in canon]
        return list(dict.fromkeys(bps))

    @property
    def coords(self) -> list[str]:
        canon = self._canonical_4()
        cs = [c for _, _, _, c in canon]
        return list(dict.fromkeys(cs))

    def with_scorer(self, scorer: str) -> DLCHeaderModel:
        """
        Return a new header with scorer replaced (pandas-free).

        Replaces legacy `header.scorer = ...`.
        """
        new_cols = []
        for col in self.columns:
            t = tuple(map(str, col))
            if len(t) >= 1:
                new_cols.append((str(scorer), *t[1:]))
            else:
                new_cols.append((str(scorer),))
        return self.model_copy(update={"columns": new_cols, "names": self.names})

    def form_individual_bodypart_pairs(self) -> list[tuple[str, str]]:
        """
        Return ordered list of (individual, bodypart) pairs.

        This matches the previous DLCHeader behavior but is pandas-free.
        """
        canon = self._canonical_4()
        pairs = [(ind, bp) for _, ind, bp, _ in canon]
        # stable unique preserving first seen order
        return list(dict.fromkeys(pairs))

    @classmethod
    def from_config(cls, config: dict) -> DLCHeaderModel:
        """
        Build header from DLC config.yaml content (single or multi-animal),
        without requiring pandas.
        """
        multi = bool(config.get("multianimalproject", False))
        scorer = str(config["scorer"])

        cols: list[tuple[str, ...]] = []
        names: list[str]

        if multi:
            inds = [str(x) for x in config["individuals"]]
            bps = [str(x) for x in config["multianimalbodyparts"]]
            coords = ["x", "y"]
            for i in inds:
                for bp in bps:
                    for c in coords:
                        cols.append((scorer, i, bp, c))
            # unique bodyparts in "single" individual bucket
            for bp in [str(x) for x in config.get("uniquebodyparts", [])]:
                for c in coords:
                    cols.append((scorer, "single", bp, c))
            names = ["scorer", "individuals", "bodyparts", "coords"]
        else:
            bps = [str(x) for x in config["bodyparts"]]
            coords = ["x", "y"]
            for bp in bps:
                for c in coords:
                    cols.append((scorer, bp, c))
            names = ["scorer", "bodyparts", "coords"]

        return cls(columns=cols, names=names)

    def to_metadata_payload(self) -> dict[str, Any]:
        """
        Portable payload to store in napari layer.metadata.

        Never stores pandas objects.
        """
        return {"columns": self.columns, "names": self.names}


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
    dataset_key: str = Field(default="df_with_missing", description="HDF5 key for keypoints table")

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
    - config_colormap stores the configured bodypart colormap when known
    - face_color_cycles, when present, is derived display state and not authoritative
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

    config_colormap: str | None = None
    face_color_cycles: dict[str, dict[str, Any]] | None = None
    colormap_name: str | None = None

    tables: dict[str, dict[str, str]] | None = None

    # Non-serializable runtime attachments (allowed but ignored by pydantic)
    controls: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


# -----------------------------------------------------------------------------
# Save conflict models
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictEntry:
    frame_label: str
    keypoints: tuple[str, ...]


@dataclass(frozen=True)
class OverwriteConflictReport:
    """
    UI-facing overwrite-conflict contract.

    This model is intentionally decoupled from pandas so the dialog layer only
    depends on plain Python data structures.

    Semantics
    ---------
    n_overwrites:
        Number of (frame/image, keypoint) overwrite events.
    n_frames:
        Number of distinct frames/images affected.
    entries:
        Detailed per-frame/image conflict rows to display in the dialog.
    truncated_entries:
        Number of additional frame/image rows omitted from `entries`.
    layer_name:
        Optional source layer name to display in the dialog.
    destination_path:
        Optional destination path to display in the dialog.
    """

    n_overwrites: int
    n_frames: int
    entries: tuple[ConflictEntry, ...]
    truncated_entries: int = 0
    layer_name: str | None = None
    destination_path: str | None = None

    @property
    def has_conflicts(self) -> bool:
        return self.n_overwrites > 0

    @property
    def details_text(self) -> str:
        if not self.entries:
            return "No detailed conflicts."
        lines = [f"{entry.frame_label} → {', '.join(entry.keypoints)}" for entry in self.entries]
        if self.truncated_entries:
            lines.append("")
            lines.append(f"… and {self.truncated_entries} more frame/image entries.")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
#  Trails metadata and user settings models
# -----------------------------------------------------------------------------


class TrailsDisplayConfig(BaseModel):
    tail_length: int = Field(default=50, ge=0)
    head_length: int = Field(default=50, ge=0)
    tail_width: float = Field(default=6.0, gt=0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    blending: str = Field(default="translucent")
    visible: bool = Field(default=True)

    @field_validator("blending")
    @classmethod
    def _validate_blending(cls, v: str) -> str:
        # Keep this list minimal and permissive; extend if needed.
        allowed = {"translucent", "opaque", "additive", "minimum"}
        vv = str(v).strip().lower()
        return vv if vv in allowed else "translucent"


class FolderUIState(BaseModel):
    """
    Folder-scoped persisted UI state stored in .napari-deeplabcut.json.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(default=1, ge=1)
    default_scorer: str | None = None
    trails: TrailsDisplayConfig = Field(default_factory=TrailsDisplayConfig)
