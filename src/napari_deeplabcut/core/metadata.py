# src/napari_deeplabcut/core/metadata.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel, ImageMetadata, PointsMetadata
from napari_deeplabcut.core.discovery import infer_annotation_kind_for_file
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.paths import canonicalize_path
from napari_deeplabcut.core.provenance import build_io_provenance_dict

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

    Priority:
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


def merge_image_metadata(base: ImageMetadata, incoming: ImageMetadata) -> ImageMetadata:
    """Merge ImageMetadata without clobbering existing values."""
    data = base.model_dump(mode="python")
    for field, value in incoming.model_dump(mode="python").items():
        if data.get(field) in (None, "", []) and value not in (None, "", []):
            data[field] = value
    return ImageMetadata(**data)


def merge_points_metadata(base: PointsMetadata, incoming: PointsMetadata) -> PointsMetadata:
    """Merge PointsMetadata without clobbering existing values."""
    data = base.model_dump(mode="python")
    for field, value in incoming.model_dump(mode="python").items():
        if field == "controls":
            continue
        if data.get(field) in (None, "", []) and value not in (None, "", []):
            data[field] = value
    return PointsMetadata(**data)


# -----------------------------------------------------------------------------
# Synchronization helpers
# -----------------------------------------------------------------------------


def sync_points_from_image(image_meta: ImageMetadata, points_meta: PointsMetadata) -> PointsMetadata:
    """Ensure PointsMetadata contains required image-derived fields."""
    updated = points_meta.model_dump(mode="python")
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
    """Normalize raw metadata dicts into authoritative models."""
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
def _normalize_columns(cols: Any) -> Any:
    try:
        import pandas as pd

        if isinstance(cols, pd.MultiIndex):
            return [tuple(map(str, t)) for t in cols.to_list()]
        if isinstance(cols, pd.Index):
            return [str(x) for x in cols.to_list()]
    except Exception:
        pass
    return cols


def _coerce_header_to_model(header: Any, *, strict: bool = False) -> DLCHeaderModel | None:
    """Coerce various runtime header forms into DLCHeaderModel.

    Accepts:
    - DLCHeaderModel (returned as-is)
    - dict (validated into DLCHeaderModel)
    - misc.DLCHeader-like object with `.columns` (wrapped)
    """
    if header is None:
        if strict:
            raise ValueError("Header is None; cannot write to DLCHeaderModel.")
        return None

    if isinstance(header, DLCHeaderModel):
        return header

    def _fail(msg: str, exc: Exception | None = None):
        logger.debug(msg, exc_info=exc is not None)
        if strict:
            raise ValueError(msg) from exc
        return None

    # dict-ish header
    if isinstance(header, Mapping):
        try:
            hd = dict(header)
            if "columns" in hd:
                hd["columns"] = _normalize_columns(hd["columns"])
            return DLCHeaderModel.model_validate(hd)
        except Exception:
            return _fail("Failed to parse header dict into DLCHeaderModel.", exc=Exception)

    # runtime DLCHeader-like (e.g. napari_deeplabcut.misc.DLCHeader)
    cols = getattr(header, "columns", None)
    if cols is not None:
        try:
            return DLCHeaderModel(columns=_normalize_columns(cols))
        except Exception:
            return _fail("Failed to coerce header with columns into DLCHeaderModel.", exc=Exception)

    return None


def _coerce_io_kind(d: dict, key: str = "kind") -> None:
    k = d.get(key)
    if isinstance(k, str):
        try:
            d[key] = AnnotationKind(k)  # works if enum values are "gt"/"machine"
        except Exception:
            # optionally accept upper-cased names
            try:
                d[key] = AnnotationKind[k.upper()]
            except Exception:
                pass


def parse_points_metadata(
    md: Mapping[str, Any] | PointsMetadata | None,
    *,
    drop_header: bool = False,
    drop_controls: bool = True,
    # TODO defaults may need adjusted @C-Achard
) -> PointsMetadata:
    """
    Parse PointsMetadata from a napari layer.metadata mapping.

    Robust to runtime objects (Qt widgets, misc.DLCHeader, etc.).
    - controls are dropped by default (runtime-only)
    - header is kept by default (needed for writing + conflict checking)
    """
    if md is None:
        return PointsMetadata()
    if isinstance(md, PointsMetadata):
        return md

    raw = dict(md)

    # Drop runtime-only / non-serializable fields
    if drop_controls:
        raw.pop("controls", None)

    # Coerce header unless explicitly dropped
    if drop_header:
        raw.pop("header", None)
    else:
        hdr = raw.get("header", None)
        logger.debug("Raw header type=%r", type(hdr))
        logger.debug("Raw header has columns=%s", hasattr(hdr, "columns"))
        cols = getattr(hdr, "columns", None)
        logger.debug("columns type=%r", type(cols))
        coerced = _coerce_header_to_model(hdr)
        if coerced is not None:
            raw["header"] = coerced
        else:
            # If a header was present but not usable, remove it so we can still parse.
            raw.pop("header", None)

    io_dict = raw.get("io", None)
    if isinstance(io_dict, dict):
        _coerce_io_kind(io_dict)

    st_dict = raw.get("save_target", None)
    if isinstance(st_dict, dict):
        _coerce_io_kind(st_dict)

    try:
        return PointsMetadata.model_validate(raw)
    except Exception:
        logger.debug("Failed to parse PointsMetadata; falling back to empty model.", exc_info=True)
        return PointsMetadata()


def merge_model_into_metadata(
    metadata: dict[str, Any],
    model: BaseModel,
    *,
    exclude_none: bool = True,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Merge a Pydantic model into an existing metadata dict (shallow merge)."""
    if exclude is None:
        exclude = set()

    try:
        dumped = model.model_dump(mode="python", exclude_none=exclude_none, exclude=exclude)
    except Exception:
        dumped = {}

    for k, v in dumped.items():
        metadata[k] = v
    return metadata


# -----------------------------------------------------------------------------
# Save target utilities
# -----------------------------------------------------------------------------


def require_unique_target(candidates: list[Path], *, context: str = "save target") -> Path:
    """Ensure a candidate list resolves to exactly one path."""
    if not candidates:
        raise MissingProvenanceError(f"No candidates found for {context}.")
    if len(candidates) > 1:
        raise AmbiguousSaveError(f"Ambiguous {context}: {len(candidates)} candidates: {[c.name for c in candidates]}")
    return candidates[0]


# -----------------------------------------------------------------------------
# Provenance attachment (napari metadata glue)
# -----------------------------------------------------------------------------


def attach_source_and_io(
    metadata: dict[str, Any],
    file_path: Path,
    *,
    kind: AnnotationKind | None = None,
    dataset_key: str = "keypoints",
) -> None:
    """
    Attach authoritative source info + IO provenance to napari layer metadata dict.

    - Keeps legacy fields (source_h5*) for debugging/migration.
    - Stores IOProvenance as a plain dict under metadata['metadata']['io'].
    - Stores AnnotationKind as enum object (runtime invariant).

    If kind is None, we fall back to discovery-based inference.

    # FUTURE NOTE hardcoded DLC structure:
    # kind inference relies on discovery filename patterns (CollectedData*, machinelabels*).
    """
    meta = metadata.setdefault("metadata", {})

    # Legacy migration fields
    try:
        src_abs = str(file_path.expanduser().resolve())
    except Exception:
        src_abs = str(file_path)

    meta["source_h5"] = src_abs
    meta["source_h5_name"] = file_path.name
    meta["source_h5_stem"] = file_path.stem

    # Anchor root: file parent (robust for shared labeled-data folders)
    try:
        anchor = str(file_path.expanduser().resolve().parent)
    except Exception:
        anchor = str(file_path.parent)

    # Relative path stored as POSIX (OS-agnostic)
    relposix = canonicalize_path(file_path, n=1)

    # If caller didn't provide kind, infer from discovery
    if kind is None:
        kind = infer_annotation_kind_for_file(file_path)

    meta["io"] = build_io_provenance_dict(
        project_root=anchor,
        source_relpath_posix=relposix,
        kind=kind,
        dataset_key=dataset_key,
    )
