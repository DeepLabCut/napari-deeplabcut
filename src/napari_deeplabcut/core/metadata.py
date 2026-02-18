# src/napari_deeplabcut/core/metadata.py
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from napari_deeplabcut.config.models import ImageMetadata, IOProvenance, PointsMetadata, AnnotationKind
from napari_deeplabcut.core.paths import canonicalize_path
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
    It MUST be robust to napari runtime objects (e.g., misc.DLCHeader, Qt widgets).
    """
    if md is None:
        return PointsMetadata()
    if isinstance(md, PointsMetadata):
        return md

    try:
        raw = dict(md)

        # Drop runtime-only / non-JSON fields that can break validation.
        # We only need io/save_target/root/paths for routing decisions.
        raw.pop("controls", None)

        # `header` is often a misc.DLCHeader runtime object (not our DLCHeaderModel dict).
        # Keep it in layer.metadata, but exclude it from model_validate.
        raw.pop("header", None)

        # face_color_cycles can contain numpy objects; keep out of model parse if needed.
        # (Not strictly necessary, but safe.)
        # raw.pop("face_color_cycles", None)

        return PointsMetadata.model_validate(raw)
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


def _attach_source_and_io(metadata: dict, file_path: Path) -> None:
    """
    Attach authoritative source info + minimal IOProvenance to the layer metadata dict.

    - Keeps legacy source_h5 fields for migration.
    - Stores IOProvenance as a plain dict under metadata['metadata']['io'].
    - Uses core.paths.canonicalize_path for OS-agnostic relpaths.
    """
    meta = metadata.setdefault("metadata", {})

    # --- legacy migration fields (still useful for debugging/backfill) ---
    try:
        src_abs = str(file_path.expanduser().resolve())
    except Exception:
        src_abs = str(file_path)

    meta["source_h5"] = src_abs
    meta["source_h5_name"] = file_path.name
    meta["source_h5_stem"] = file_path.stem

    # Root anchor: default to file parent (works when only labeled-data folder is shared)
    try:
        anchor = str(file_path.expanduser().resolve().parent)
    except Exception:
        anchor = str(file_path.parent)

    # Relative path stored OS-agnostic (POSIX). With anchor=file parent, this is filename.
    relposix = canonicalize_path(file_path, n=1)
    kind = _infer_annotation_kind_from_discovery(file_path)

    io = IOProvenance(
        project_root=anchor,
        source_relpath_posix=relposix,
        kind=kind,
        dataset_key="keypoints",
    )
    # Store as plain dict so it round-trips safely in napari metadata
    meta["io"] = io.model_dump(exclude_none=True)
