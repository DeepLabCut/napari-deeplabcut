# src/napari_deeplabcut/core/provenance.py
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from napari_deeplabcut.config.models import AnnotationKind, DLCProjectContext, IOProvenance, PointsMetadata
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.metadata import parse_points_metadata
from napari_deeplabcut.core.project_paths import infer_dlc_project, infer_labeled_data_folder_from_paths

logger = logging.getLogger(__name__)


def infer_dlc_project_from_points_meta(
    pts_meta: PointsMetadata,
    *,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    """
    Infer DLC dataset folder (…/labeled-data/<dataset>) from PointsMetadata.

    Uses:
      - pts_meta.project (config parent) as project root
      - pts_meta.paths (canonicalized relpaths like labeled-data/test/img000.png)
      - pts_meta.root as a fallback hint

    Returns a DLCProjectContext object representing the inferred project context.
    """
    project = getattr(pts_meta, "project", None)
    root = getattr(pts_meta, "root", None)
    paths = getattr(pts_meta, "paths", None) or []

    dataset_folder = infer_labeled_data_folder_from_paths(
        paths,
        project_root=project,
        fallback_root=root,
    )

    return infer_dlc_project(
        anchor_candidates=[project, root, dataset_folder],
        dataset_candidates=[dataset_folder],
        explicit_root=None,
        prefer_project_root=prefer_project_root,
        max_levels=max_levels,
    )


def infer_dlc_project_from_image_layer(
    layer,
    *,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    """Best-effort inference of the DLC project context from an Image/video layer using its source metadata.

    Uses:
      - layer.metadata.project as project root
      - layer.metadata.root as a fallback hint
      - layer.source.path as a fallback hint

    Returns a DLCProjectContext object representing the inferred project context.
    """
    md = getattr(layer, "metadata", {}) or {}

    candidates: list[str | Path] = []

    project = md.get("project")
    if isinstance(project, str) and project:
        candidates.append(project)

    root = md.get("root")
    if isinstance(root, str) and root:
        candidates.append(root)

    try:
        src = getattr(getattr(layer, "source", None), "path", None)
    except Exception:
        src = None

    if src:
        candidates.append(src)

    return infer_dlc_project(
        anchor_candidates=candidates,
        dataset_candidates=[],
        explicit_root=None,
        prefer_project_root=prefer_project_root,
        max_levels=max_levels,
    )


def resolve_output_path_from_metadata(metadata: dict) -> tuple[str | None, str | None, AnnotationKind | None]:
    """
    Resolve output path with promotion support.

    Returns:
      (out_path, target_scorer, source_kind)

    - Prefer PointsMetadata.save_target (promotion-to-GT).
    - For GT sources, fall back to io/source_h5.
    - For machine sources without save_target, return (None, None, "machine") to allow safe abort.
    """
    layer_meta = metadata.get("metadata")
    if not isinstance(layer_meta, dict):
        layer_meta = {}

    pts = parse_points_metadata(layer_meta)
    io = pts.io
    st = pts.save_target

    source_kind = getattr(io, "kind", None) if io is not None else None

    # Promotion target wins
    if st is not None:
        try:
            p = resolve_provenance_path(st, root_anchor=st.project_root, allow_missing=True)
            target_scorer = getattr(st, "scorer", None)
            if isinstance(target_scorer, str) and target_scorer.strip():
                return str(p), target_scorer.strip(), source_kind
            # Also accept scorer stored in dict extra
            if isinstance(layer_meta.get("save_target"), dict):
                s2 = layer_meta["save_target"].get("scorer")
                if isinstance(s2, str) and s2.strip():
                    return str(p), s2.strip(), source_kind
            return str(p), None, source_kind
        except (MissingProvenanceError, UnresolvablePathError):
            return None, None, source_kind

    # Never save back to machine sources
    if source_kind == AnnotationKind.MACHINE:
        return None, None, source_kind
    # GT source: prefer io if available
    if io is not None:
        try:
            p = resolve_provenance_path(io, root_anchor=io.project_root, allow_missing=True)
            return str(p), None, source_kind
        except (MissingProvenanceError, UnresolvablePathError):
            pass

    # Legacy fallback: source_h5 (GT only)
    src = layer_meta.get("source_h5")
    if isinstance(src, str) and src:
        return src, None, source_kind

    return None, None, source_kind


def ensure_io_provenance(obj: IOProvenance | dict | None) -> IOProvenance | None:
    """
    Validate/normalize IO provenance payload.

    Policy: runtime must carry AnnotationKind objects.
    Invalid dicts raise MissingProvenanceError for deterministic behavior.
    """
    if obj is None:
        return None
    if isinstance(obj, IOProvenance):
        return obj
    if isinstance(obj, dict):
        try:
            # This must succeed only if kinds are AnnotationKind instances
            return IOProvenance.model_validate(obj)
        except (ValidationError, TypeError) as e:
            raise MissingProvenanceError(f"Invalid IO provenance payload: {e}") from e
    raise MissingProvenanceError(f"Invalid IO provenance type: {type(obj).__name__}")


def normalize_provenance(io: IOProvenance | None) -> IOProvenance | None:
    """
    Normalize provenance fields for stable storage.

    - Ensures source_relpath_posix uses '/' separators.
    - Leaves extra fields untouched.
    """
    if io is None:
        return None

    src = io.source_relpath_posix
    if isinstance(src, str):
        src = src.replace("\\\\", "/").replace("\\", "/")

    return io.model_copy(update={"source_relpath_posix": src})


def resolve_provenance_path(
    io: IOProvenance | dict | None,
    *,
    root_anchor: str | Path | None = None,
    allow_missing: bool = False,
) -> Path:
    """
    Resolve IOProvenance into a concrete filesystem Path.

    io may be a dict stored in napari metadata; it will be validated strictly.
    """
    io2 = ensure_io_provenance(io)
    if io2 is None or not io2.source_relpath_posix:
        raise MissingProvenanceError("Missing IO provenance (source_relpath_posix is required).")

    io2 = normalize_provenance(io2) or io2

    anchor = root_anchor or io2.project_root
    if not anchor:
        raise UnresolvablePathError(
            "Cannot resolve provenance path: no root anchor provided and io.project_root is missing."
        )

    rel = PurePosixPath(io2.source_relpath_posix)
    resolved = Path(anchor) / Path(*rel.parts)

    if not allow_missing and not resolved.exists():
        raise UnresolvablePathError(f"Resolved provenance path does not exist: {resolved}")

    return resolved
