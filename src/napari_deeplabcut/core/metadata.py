# src/napari_deeplabcut/core/metadata.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from napari_deeplabcut.config.models import ImageMetadata, PointsMetadata

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
