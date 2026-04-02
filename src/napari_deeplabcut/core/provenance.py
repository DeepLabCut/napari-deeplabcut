# src/napari_deeplabcut/core/provenance.py
from __future__ import annotations

import hashlib
import logging
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from napari_deeplabcut.config.models import AnnotationKind, IOProvenance, PointsMetadata
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.metadata import parse_points_metadata
from napari_deeplabcut.core.project_paths import (
    infer_dlc_project_from_points_meta,
    is_windows_absolute_path,
)

logger = logging.getLogger(__name__)


# ----------------------------------------
# Helper functions
# ----------------------------------------
def suggest_human_placeholder(anchor: str) -> str:
    """
    Deterministic fallback scorer placeholder derived from anchor path.
    """
    h = hashlib.sha1(anchor.encode("utf-8", errors="ignore")).hexdigest()[:6]
    return f"human_{h}"


def requires_gt_promotion(pts_meta: PointsMetadata) -> bool:
    """
    Return True when a machine/prediction source must be promoted to a GT save_target
    before saving.

    Rules:
    - if save_target already exists -> no promotion needed
    - if io.kind is MACHINE -> promotion required
    - otherwise -> no promotion required
    """
    if getattr(pts_meta, "save_target", None) is not None:
        return False

    io_meta = getattr(pts_meta, "io", None)
    src_kind = getattr(io_meta, "kind", None) if io_meta is not None else None
    return src_kind is AnnotationKind.MACHINE


def build_gt_save_target(
    anchor: str,
    scorer: str,
    *,
    dataset_key: str = "keypoints",
) -> IOProvenance:
    """
    Build a GT save_target pointing to CollectedData_<scorer>.h5 under a folder anchor.
    """
    scorer_clean = str(scorer).strip()
    target_name = f"CollectedData_{scorer_clean}.h5"
    return IOProvenance(
        project_root=anchor,
        source_relpath_posix=target_name,
        kind=AnnotationKind.GT,
        dataset_key=dataset_key,
        scorer=scorer_clean,
    )


def apply_gt_save_target(
    pts_meta: PointsMetadata,
    *,
    anchor: str,
    scorer: str,
    dataset_key: str = "keypoints",
) -> PointsMetadata:
    """
    Return an updated PointsMetadata with a GT promotion save_target attached.
    """
    st = build_gt_save_target(anchor, scorer, dataset_key=dataset_key)
    return pts_meta.model_copy(update={"save_target": st})


def is_projectless_folder_association_candidate(
    pts_meta: PointsMetadata,
    *,
    treat_machine_as_ineligible: bool = True,
) -> bool:
    """
    Return True for the 'associate current labeled folder with a DLC project' workflow.

    Non-candidates include:
    - machine/promotion layers (optional policy)
    - layers with an explicit save_target
    - layers that already have a resolved DLC project/config context
    - layers without a usable folder root
    - layers whose paths do not look like a simple single-folder labeling session
    """
    if treat_machine_as_ineligible and requires_gt_promotion(pts_meta):
        return False

    if getattr(pts_meta, "save_target", None) is not None:
        return False

    project_ctx = infer_dlc_project_from_points_meta(pts_meta, prefer_project_root=False)
    if project_ctx.project_root is not None and project_ctx.config_path is not None:
        return False

    root = getattr(pts_meta, "root", None)
    paths = list(getattr(pts_meta, "paths", None) or [])
    if not root or not paths:
        return False

    try:
        root_path = Path(root).expanduser().resolve(strict=False)
    except Exception:
        root_path = Path(root)

    for value in paths:
        text = str(value).replace("\\", "/")
        p = Path(value)

        parts = [part for part in text.split("/") if part]
        lowered = [part.lower() for part in parts]
        if "labeled-data" in lowered:
            idx = lowered.index("labeled-data")
            if idx + 2 < len(parts):
                continue

        is_windows_abs_misclassified = not p.is_absolute() and is_windows_absolute_path(value)
        if not p.is_absolute():
            if is_windows_abs_misclassified:
                continue

            if len(p.parts) == 1 and p.parts[0] not in (".", ".."):
                continue
            return False

        try:
            rel_to_root = p.expanduser().resolve(strict=False).relative_to(root_path)
        except Exception:
            continue

        if len(rel_to_root.parts) == 1:
            continue

        return False

    return True


# ----------------------------------------
# Core provenance logic
# ----------------------------------------


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
