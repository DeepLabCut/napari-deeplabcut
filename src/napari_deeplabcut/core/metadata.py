# src/napari_deeplabcut/core/metadata.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel, ImageMetadata, IOProvenance, PointsMetadata
from napari_deeplabcut.core.discovery import infer_annotation_kind_for_file
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.project_paths import canonicalize_path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def _coerce_path(p: str | None) -> Path | None:
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return Path(p)


def _is_dlc_dataset_root(p: Path) -> bool:
    """
    Heuristic: DLC dataset folder usually looks like:
      <project>/labeled-data/<dataset_name>

    True if path contains a 'labeled-data' segment AND is deeper than that folder.
    """
    parts = [s.lower() for s in p.parts]
    if "labeled-data" not in parts:
        return False
    return parts[-1] != "labeled-data"


def _paths_look_like_labeled_data(paths: list[str] | None) -> bool:
    """
    Check if any path strings contain 'labeled-data/<dataset>/'.
    Works with canonicalized paths like 'labeled-data/test/img000.png'.
    """
    if not paths:
        return False
    for s in paths:
        if isinstance(s, str) and "labeled-data" in s.replace("\\", "/").lower():
            return True
    return False


def _looks_like_project_root(points_root: str | None, project: str | None) -> bool:
    """
    Root equals project root (config parent) => this is WRONG for saving GT.
    """
    if not points_root or not project:
        return False
    try:
        return Path(points_root).expanduser().resolve() == Path(project).expanduser().resolve()
    except Exception:
        return points_root == project


def build_io_provenance_dict(
    *,
    project_root: str | Path,
    source_relpath_posix: str,
    kind: AnnotationKind | None,
    dataset_key: str,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build a provenance dict for storage in napari layer.metadata.

    Important: uses mode="python" so AnnotationKind stays an enum at runtime.
    """
    io = IOProvenance(
        project_root=str(project_root),
        source_relpath_posix=source_relpath_posix,
        kind=kind,
        dataset_key=dataset_key,
        **extra,
    )
    return io.model_dump(mode="python", exclude_none=True)


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
    """
    Ensure PointsMetadata contains required image-derived fields.

    Robust DLC policy:
    - If image root looks like a DLC dataset folder (…/labeled-data/<dataset>),
      prefer it for points_meta.root even if points_meta.root is already set
      but equals project root (config parent) or is not a dataset root.
    """
    updated = points_meta.model_dump(mode="python")

    # --- First: fill missing fields (existing behavior) ---
    for key in ("root", "paths", "shape", "name"):
        if updated.get(key) in (None, "", []):
            value = getattr(image_meta, key, None)
            if value not in (None, "", []):
                updated[key] = value

    # --- Second: if we have dataset context, correct stale root ---
    img_root_p = _coerce_path(getattr(image_meta, "root", None))
    pts_root_p = _coerce_path(updated.get("root"))
    project_p = _coerce_path(updated.get("project"))

    # Determine if the image root is a DLC dataset directory
    image_is_dataset = bool(img_root_p is not None and _is_dlc_dataset_root(img_root_p))

    # Additional hint: sometimes image_meta.root might be missing, but paths show labeled-data
    # (depends on readers / napari versions). Use that as secondary signal.
    if not image_is_dataset:
        if _paths_look_like_labeled_data(getattr(image_meta, "paths", None)):
            # If image paths look like labeled-data/... and we have a root-like string,
            # try to interpret image_meta.root anyway.
            image_is_dataset = bool(img_root_p is not None and _is_dlc_dataset_root(img_root_p))

    if image_is_dataset and img_root_p is not None:
        # Override root if:
        # - points root equals project root (typical config-first bug), OR
        # - points root exists but isn't a dataset root.
        should_override_root = False

        if _looks_like_project_root(str(pts_root_p) if pts_root_p else None, str(project_p) if project_p else None):
            should_override_root = True
        elif pts_root_p is not None and not _is_dlc_dataset_root(pts_root_p):
            should_override_root = True

        if should_override_root:
            updated["root"] = str(img_root_p)

    return PointsMetadata(**updated)


def apply_project_paths_override_to_points_meta(
    pts_meta: PointsMetadata,
    *,
    project_root: str | Path,
    rewritten_paths: list[str],
) -> PointsMetadata:
    """
    Return a copy of PointsMetadata with a save-time project/path override applied.

    This updates:
    - project
    - paths
    - io.project_root (if present)
    - save_target.project_root (if present)
    """
    project_root_str = str(project_root)

    updates = {
        "project": project_root_str,
        "paths": list(rewritten_paths),
    }

    if pts_meta.io is not None:
        updates["io"] = pts_meta.io.model_copy(update={"project_root": project_root_str})

    if pts_meta.save_target is not None:
        updates["save_target"] = pts_meta.save_target.model_copy(update={"project_root": project_root_str})

    return pts_meta.model_copy(update=updates)


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
        except Exception as e:
            return _fail("Failed to parse header dict into DLCHeaderModel.", exc=e)

    # cols = getattr(header, "columns", None)
    # if cols is not None:
    #     try:
    #         return DLCHeaderModel(columns=_normalize_columns(cols))
    #     except Exception as e:
    #         return _fail("Failed to coerce header with columns into DLCHeaderModel.", exc=e)

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

    Robust to runtime objects
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


# -------------------------------------------------------------------------
# Central metadata adapter gateway (validation + migration + controlled write)
# -------------------------------------------------------------------------
class MergePolicy(str, Enum):
    """
    How to apply an incoming model to an existing layer.metadata dict.

    - MERGE_MISSING: only fill missing/empty keys on the layer
    - MERGE: shallow update (incoming overwrites existing)
    - REPLACE: replace entire metadata mapping with incoming (rarely desired)
    """

    MERGE_MISSING = "merge_missing"
    MERGE = "merge"
    REPLACE = "replace"


# _EMPTY = (None, "", [], {})


def _is_empty_value(v: Any) -> bool:
    # Treat 0 and False as legitimate values, not "empty"
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    if isinstance(v, (list, tuple, set)) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


def _layer_metadata_dict(layer: Any) -> dict[str, Any]:
    md = getattr(layer, "metadata", None)
    if isinstance(md, dict):
        return md
    if md is None:
        return {}
    # best-effort cast if napari gives a Mapping-like object
    try:
        return dict(md)
    except Exception:
        return {}


def _infer_kind_from_source_name(p: Path) -> AnnotationKind | None:
    # best-effort legacy inference; discovery-based inference is preferred
    try:
        return infer_annotation_kind_for_file(p)
    except Exception:
        low = p.name.lower()
        if low.startswith("collecteddata"):
            return AnnotationKind.GT
        if low.startswith("machinelabels"):
            return AnnotationKind.MACHINE
    return None


def _build_io_from_source_h5(
    src: str,
    *,
    dataset_key: str = "keypoints",
) -> dict[str, Any] | None:
    """Legacy migration: build io provenance dict from source_h5 string."""
    if not isinstance(src, str) or not src:
        return None
    try:
        p = Path(src).expanduser().resolve()
    except Exception:
        p = Path(src)

    kind = _infer_kind_from_source_name(p)

    try:
        anchor = str(p.expanduser().resolve().parent)
    except Exception:
        anchor = str(p.parent)

    relposix = canonicalize_path(p, n=1)

    try:
        return build_io_provenance_dict(
            project_root=anchor,
            source_relpath_posix=relposix,
            kind=kind,
            dataset_key=dataset_key,
        )
    except Exception:
        logger.debug("Failed to build io provenance from legacy source_h5=%r", src, exc_info=True)
        return None


def _prepare_points_payload(
    md: Mapping[str, Any],
    *,
    drop_controls: bool = True,
    drop_header: bool = False,
    migrate_legacy: bool = True,
) -> dict[str, Any]:
    """
    Prepare a dict suitable for PointsMetadata.model_validate().

    - coerces header into DLCHeaderModel when possible
    - coerces io.kind / save_target.kind into AnnotationKind
    - drops runtime-only fields (controls) by default
    - optionally migrates legacy source_h5 -> io dict
    """
    raw = dict(md)

    if drop_controls:
        raw.pop("controls", None)

    # legacy migration: io from source_h5
    if migrate_legacy and not raw.get("io"):
        src = raw.get("source_h5")
        io_dict = _build_io_from_source_h5(src, dataset_key="keypoints")
        if io_dict:
            raw["io"] = io_dict

    # Coerce header into DLCHeaderModel (schema) if present
    hdr = raw.get("header", None)
    if hdr is not None:
        try:
            if isinstance(hdr, DLCHeaderModel):
                raw["header"] = hdr
            elif isinstance(hdr, dict) and "columns" in hdr:
                raw["header"] = DLCHeaderModel.model_validate(hdr)
            else:
                # support MultiIndex / list-of-tuples via validator
                raw["header"] = DLCHeaderModel(columns=hdr)
        except Exception:
            raw.pop("header", None)

    io_dict = raw.get("io", None)
    if isinstance(io_dict, dict):
        _coerce_io_kind(io_dict)

    st_dict = raw.get("save_target", None)
    if isinstance(st_dict, dict):
        _coerce_io_kind(st_dict)

    return raw


def _prepare_image_payload(
    md: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare a dict suitable for ImageMetadata.model_validate()."""
    raw = dict(md)
    return raw


# -------------------------
# Public adapter API
# -------------------------


def read_points_meta(
    layer: Any,
    *,
    migrate_legacy: bool = True,
    drop_controls: bool = True,
    drop_header: bool = False,
) -> PointsMetadata | ValidationError:
    """
    Read PointsMetadata from a layer with strict validation.

    Returns:
      - PointsMetadata on success
      - ValidationError on failure (visible to caller)
    """
    md = _layer_metadata_dict(layer)
    payload = _prepare_points_payload(
        md,
        drop_controls=drop_controls,
        drop_header=drop_header,
        migrate_legacy=migrate_legacy,
    )
    try:
        return PointsMetadata.model_validate(payload)
    except ValidationError as e:
        return e


def write_points_meta(
    layer: Any,
    model: PointsMetadata | Mapping[str, Any],
    merge_policy: MergePolicy | str = MergePolicy.MERGE_MISSING,
    *,
    fields: set[str] | None = None,
    exclude_none: bool = True,
    validate: bool = True,
    migrate_legacy: bool = True,
) -> PointsMetadata | ValidationError:
    """
    Write Points metadata through a single validated gateway.

    - Applies merge_policy to layer.metadata
    - Validates the final dict by default
    - Returns PointsMetadata on success, ValidationError on failure
    """
    if isinstance(merge_policy, str):
        merge_policy = MergePolicy(merge_policy)

    existing = _layer_metadata_dict(layer)
    # preserve a strong reference to the existing header (runtime object sometimes)
    existing_header = existing.get("header", None)

    if isinstance(model, Mapping):
        incoming = dict(model)
    else:
        incoming = model.model_dump(mode="python", exclude_none=exclude_none)

    # never write runtime-only field
    incoming.pop("controls", None)

    if fields is not None:
        incoming = {k: v for k, v in incoming.items() if k in fields}

    if merge_policy is MergePolicy.REPLACE:
        merged = dict(incoming)
    elif merge_policy is MergePolicy.MERGE:
        merged = dict(existing)
        merged.update(incoming)
    else:  # MERGE_MISSING
        merged = dict(existing)
        for k, v in incoming.items():
            if _is_empty_value(merged.get(k)):
                merged[k] = v

    # restore header if it existed but got dropped by incoming dict
    if existing_header is not None and merged.get("header") is None:
        merged["header"] = existing_header

    # legacy migration (optional): if caller writes anything, keep io stable
    if migrate_legacy and not merged.get("io") and merged.get("source_h5"):
        io_dict = _build_io_from_source_h5(str(merged.get("source_h5")), dataset_key="keypoints")
        if io_dict:
            merged["io"] = io_dict

    if validate:
        payload = _prepare_points_payload(
            merged,
            drop_controls=True,
            drop_header=False,
            migrate_legacy=migrate_legacy,
        )
        try:
            validated = PointsMetadata.model_validate(payload)
        except ValidationError as e:
            logger.warning("write_points_meta validation failed for layer=%r: %s", getattr(layer, "name", layer), e)
            return e

        # Write back validated python dict (exclude_none keeps metadata lean)
        final_dict = validated.model_dump(mode="python", exclude_none=True)
        hdr = final_dict.get("header", None)
        if isinstance(hdr, DLCHeaderModel):
            final_dict["header"] = hdr.to_metadata_payload()
        # preserve header if pydantic excluded it (or coercion removed it)
        if existing_header is not None and final_dict.get("header") is None:
            final_dict["header"] = existing_header

        # mutate in place (napari likes stable mapping refs)
        if getattr(layer, "metadata", None) is None or not isinstance(getattr(layer, "metadata", None), dict):
            layer.metadata = {}
        layer.metadata.clear()
        layer.metadata.update(final_dict)
        return validated

    # No validation mode (rare): still write merged mapping
    if getattr(layer, "metadata", None) is None or not isinstance(getattr(layer, "metadata", None), dict):
        layer.metadata = {}
    layer.metadata.clear()
    layer.metadata.update(merged)
    # best-effort return model
    try:
        return PointsMetadata.model_validate(_prepare_points_payload(merged, migrate_legacy=migrate_legacy))
    except ValidationError as e:
        return e


def read_image_meta(layer: Any) -> ImageMetadata | ValidationError:
    """
    Read ImageMetadata from a layer with strict validation.
    """
    md = _layer_metadata_dict(layer)
    payload = _prepare_image_payload(md)
    try:
        return ImageMetadata.model_validate(payload)
    except ValidationError as e:
        return e


def write_image_meta(
    layer: Any,
    model: ImageMetadata | Mapping[str, Any],
    merge_policy: MergePolicy | str = MergePolicy.MERGE_MISSING,
    *,
    fields: set[str] | None = None,
    exclude_none: bool = True,
    validate: bool = True,
) -> ImageMetadata | ValidationError:
    """
    Write Image metadata through a single validated gateway.
    """
    if isinstance(merge_policy, str):
        merge_policy = MergePolicy(merge_policy)

    existing = _layer_metadata_dict(layer)

    if isinstance(model, Mapping):
        incoming = dict(model)
    else:
        incoming = model.model_dump(mode="python", exclude_none=exclude_none)

    if fields is not None:
        incoming = {k: v for k, v in incoming.items() if k in fields}

    if merge_policy is MergePolicy.REPLACE:
        merged = dict(incoming)
    elif merge_policy is MergePolicy.MERGE:
        merged = dict(existing)
        merged.update(incoming)
    else:  # MERGE_MISSING
        merged = dict(existing)
        for k, v in incoming.items():
            if _is_empty_value(merged.get(k)):
                merged[k] = v

    if validate:
        try:
            validated = ImageMetadata.model_validate(_prepare_image_payload(merged))
        except ValidationError as e:
            logger.warning("write_image_meta validation failed for layer=%r: %s", getattr(layer, "name", layer), e)
            return e

        final_dict = validated.model_dump(mode="python", exclude_none=True)

        if getattr(layer, "metadata", None) is None or not isinstance(getattr(layer, "metadata", None), dict):
            layer.metadata = {}
        layer.metadata.clear()
        layer.metadata.update(final_dict)
        return validated

    if getattr(layer, "metadata", None) is None or not isinstance(getattr(layer, "metadata", None), dict):
        layer.metadata = {}
    layer.metadata.clear()
    layer.metadata.update(merged)
    try:
        return ImageMetadata.model_validate(_prepare_image_payload(merged))
    except ValidationError as e:
        return e


def migrate_points_layer_metadata(layer: Any) -> PointsMetadata | ValidationError:
    """
    Convenience migration entrypoint:
    - reads (with legacy migration)
    - writes back (merge_missing) through gateway
    """
    res = read_points_meta(layer, migrate_legacy=True)
    if isinstance(res, ValidationError):
        return res
    return write_points_meta(layer, res, MergePolicy.MERGE_MISSING, migrate_legacy=True)


def coerce_header_model(header: Any) -> DLCHeaderModel | None:
    """
    Convert any supported header representation to DLCHeaderModel.

    Supported:
      - DLCHeaderModel
      - dict-like payload (including {"columns": ...})
      - pandas.MultiIndex / Index via existing _coerce_header_to_model logic
    """
    if header is None:
        return None
    if isinstance(header, DLCHeaderModel):
        return header
    # fall back to existing coercion path (dict, MultiIndex, etc.)
    return _coerce_header_to_model(header)
