# src/napari_deeplabcut/core/metadata.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from napari_deeplabcut.config.models import ImageMetadata, IOProvenance, PointsMetadata
from napari_deeplabcut.core.errors import (
    AmbiguousSaveError,
    MissingProvenanceError,
    UnresolvablePathError,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def infer_image_root(
    *,
    explicit_root: str | None = None,
    paths: Iterable[str] | None = None,
    source_path: str | None = None,
) -> str | None:
    """
    Best-effort inference of an image root directory.

    Priority (unchanged from legacy behavior):
    1. explicit_root
    2. parent of first path
    3. parent of source_path
    """
    if explicit_root:
        return explicit_root

    if paths:
        try:
            return str(Path(next(iter(paths))).expanduser().resolve().parent)
        except Exception:
            pass

    if source_path:
        try:
            return str(Path(source_path).expanduser().resolve().parent)
        except Exception:
            pass

    return None


# -----------------------------------------------------------------------------
# Safe update / merge rules
# -----------------------------------------------------------------------------


def merge_image_metadata(
    base: ImageMetadata,
    incoming: ImageMetadata,
) -> ImageMetadata:
    """
    Merge ImageMetadata without clobbering existing values.

    Non-null values in `incoming` fill missing values in `base`.
    """
    data = base.model_dump()
    for field, value in incoming.model_dump().items():
        if data.get(field) in (None, "", []) and value not in (None, "", []):
            data[field] = value
    return ImageMetadata(**data)


def merge_points_metadata(
    base: PointsMetadata,
    incoming: PointsMetadata,
) -> PointsMetadata:
    """
    Merge PointsMetadata without clobbering existing values.
    """
    data = base.model_dump()
    for field, value in incoming.model_dump().items():
        if field == "controls":
            continue
        if data.get(field) in (None, "", []) and value not in (None, "", []):
            data[field] = value
    return PointsMetadata(**data)


# -----------------------------------------------------------------------------
# Synchronization helpers
# -----------------------------------------------------------------------------


def sync_points_from_image(
    image_meta: ImageMetadata,
    points_meta: PointsMetadata,
) -> PointsMetadata:
    """
    Ensure PointsMetadata contains required image-derived fields.

    This mirrors legacy widget behavior but is centralized and explicit.
    """
    updated = points_meta.model_dump()

    for key in ("root", "paths", "shape", "name"):
        if updated.get(key) in (None, "", []):
            value = getattr(image_meta, key, None)
            if value not in (None, "", []):
                updated[key] = value

    return PointsMetadata(**updated)


def ensure_metadata_models(
    image_meta: dict | ImageMetadata | None,
    points_meta: dict | PointsMetadata | None,
) -> tuple[ImageMetadata | None, PointsMetadata | None]:
    """
    Normalize raw metadata dicts into authoritative models.

    This is the primary bridge for migration safety.
    """
    img = None
    pts = None

    if image_meta is not None:
        img = image_meta if isinstance(image_meta, ImageMetadata) else ImageMetadata(**image_meta)

    if points_meta is not None:
        pts = points_meta if isinstance(points_meta, PointsMetadata) else PointsMetadata(**points_meta)

    return img, pts


# -----------------------------------------------------------------------------
# Parsing / round-tripping
# -----------------------------------------------------------------------------


def parse_points_metadata(md: Mapping[str, Any] | PointsMetadata | None) -> PointsMetadata:
    """Parse PointsMetadata from a napari layer.metadata mapping.

    This is a *best-effort* parser intended for migration safety.
    It does not raise on unexpected keys.
    """
    if md is None:
        return PointsMetadata()
    if isinstance(md, PointsMetadata):
        return md

    try:
        # model_validate handles nested models (e.g. io) and respects extra='allow'.
        return PointsMetadata.model_validate(dict(md))
    except Exception:
        logger.debug(
            "Failed to parse PointsMetadata from dict; falling back to empty model.",
            exc_info=True,
        )
        return PointsMetadata()


def merge_model_into_metadata(
    metadata: dict[str, Any],
    model: BaseModel,
    *,
    exclude_none: bool = True,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Merge a Pydantic model into an existing metadata dict.

    This function updates only the keys owned by the model dump; it does not
    remove or overwrite unrelated keys unless they collide.

    Parameters
    ----------
    metadata:
        Existing napari metadata dict (mutated in-place and returned).
    model:
        Pydantic model to merge.
    exclude_none:
        If True, omit None values from the dump.
    exclude:
        Optional set of field names to exclude from the merge.

    Returns
    -------
    dict
        The updated metadata mapping.
    """
    if exclude is None:
        exclude = set()

    try:
        dumped = model.model_dump(exclude_none=exclude_none, exclude=exclude)
    except Exception:
        dumped = {}

    # Merge shallowly; nested dicts are replaced at the top-level key.
    # (We keep it minimal and deterministic; deeper merges can be added later.)
    for k, v in dumped.items():
        metadata[k] = v
    return metadata


# -----------------------------------------------------------------------------
# Provenance normalization and resolution
# -----------------------------------------------------------------------------


def normalize_provenance(io: IOProvenance | None) -> IOProvenance | None:
    """Normalize provenance fields for stable storage.

    - Ensures source_relpath_posix uses '/' separators.
    - Leaves unknown extra fields untouched.

    Returns the same instance type (model_copy) to avoid mutating caller state.
    """
    if io is None:
        return None

    # IOProvenance already normalizes via validator, but we may receive a partially
    # constructed instance (or older dict-derived values). Keep this minimal.
    src = io.source_relpath_posix
    if src is not None:
        src = src.replace("\\\\", "/").replace("\\", "/")

    return io.model_copy(update={"source_relpath_posix": src})


def resolve_provenance_path(
    io: IOProvenance | None,
    *,
    root_anchor: str | Path | None = None,
    allow_missing: bool = False,
) -> Path:
    """Resolve IOProvenance into a concrete filesystem Path.

    Parameters
    ----------
    io:
        Provenance model. Must contain source_relpath_posix.
    root_anchor:
        Configurable root directory used to resolve relative paths. If not
        provided, ``io.project_root`` is used.
    allow_missing:
        If True, returns the resolved path even if it does not exist.

    Raises
    ------
    MissingProvenanceError:
        If io is None or does not contain source_relpath_posix.
    UnresolvablePathError:
        If neither root_anchor nor io.project_root is available.

    Returns
    -------
    pathlib.Path
        Resolved target path.
    """
    if io is None or not io.source_relpath_posix:
        raise MissingProvenanceError("Missing IO provenance (source_relpath_posix is required).")

    io = normalize_provenance(io) or io

    anchor = root_anchor or io.project_root
    if not anchor:
        raise UnresolvablePathError(
            "Cannot resolve provenance path: no root anchor provided and io.project_root is missing."
        )

    # Resolve POSIX relpath against anchor using PurePosixPath for OS-agnostic storage.
    rel = PurePosixPath(io.source_relpath_posix)
    resolved = Path(anchor) / Path(*rel.parts)

    if not allow_missing and not resolved.exists():
        raise UnresolvablePathError(f"Resolved provenance path does not exist: {resolved}")

    return resolved


def require_unique_target(
    candidates: list[Path],
    *,
    context: str = "save target",
) -> Path:
    """Ensure a candidate list resolves to exactly one path.

    This utility is used to enforce the \"ambiguity must not be silent\" policy.
    """
    if not candidates:
        raise MissingProvenanceError(f"No candidates found for {context}.")
    if len(candidates) > 1:
        raise AmbiguousSaveError(f"Ambiguous {context}: {len(candidates)} candidates: {[c.name for c in candidates]}")
    return candidates[0]
