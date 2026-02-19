# src/napari_deeplabcut/core/provenance.py
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from napari_deeplabcut.config.models import AnnotationKind, IOProvenance
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError

logger = logging.getLogger(__name__)


def ensure_io_provenance(obj: IOProvenance | dict | None) -> IOProvenance | None:
    """
    Validate/normalize IO provenance payload.

    Policy: runtime must carry AnnotationKind objects or valid string.
    Invalid dicts raise MissingProvenanceError for deterministic behavior.
    """
    if obj is None:
        return None
    if isinstance(obj, IOProvenance):
        return obj
    if isinstance(obj, dict):
        try:
            # This must succeed only if kinds are AnnotationKind instances or valid strings
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
