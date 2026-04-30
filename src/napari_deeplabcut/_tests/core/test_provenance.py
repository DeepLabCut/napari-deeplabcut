from __future__ import annotations

from pathlib import Path

import pytest

from napari_deeplabcut.config.models import AnnotationKind, IOProvenance
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.provenance import (
    ensure_io_provenance,
    normalize_provenance,
    resolve_provenance_path,
)

# -----------------------------------------------------------------------------
# ensure_io_provenance
# -----------------------------------------------------------------------------


def test_ensure_io_provenance_none_returns_none():
    assert ensure_io_provenance(None) is None


def test_ensure_io_provenance_accepts_model_instance(tmp_path: Path):
    io = IOProvenance(
        project_root=str(tmp_path),
        source_relpath_posix="CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="df_with_missing",
    )
    out = ensure_io_provenance(io)
    assert out is io


def test_ensure_io_provenance_accepts_dict_with_enum_kind(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,  # IMPORTANT: enum instance, not string
        "dataset_key": "df_with_missing",
    }
    out = ensure_io_provenance(payload)
    assert isinstance(out, IOProvenance)
    assert out.kind == AnnotationKind.GT
    assert out.source_relpath_posix == "CollectedData_Jane.h5"


def test_ensure_io_provenance_rejects_dict_with_string_kind(tmp_path: Path):
    """
    Contract from provenance.py docstring:
    runtime must carry AnnotationKind objects; strings invalid.
    This is enforced by IOProvenance.kind strict=True.
    """
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": "gt",  # invalid at runtime by policy
        "dataset_key": "df_with_missing",
    }
    with pytest.raises(MissingProvenanceError):
        ensure_io_provenance(payload)


def test_ensure_io_provenance_rejects_invalid_kind_value(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": "not-a-kind",
        "dataset_key": "df_with_missing",
    }
    with pytest.raises(MissingProvenanceError):
        ensure_io_provenance(payload)


def test_ensure_io_provenance_rejects_invalid_type():
    with pytest.raises(MissingProvenanceError):
        ensure_io_provenance(["not", "a", "dict"])  # type: ignore[arg-type]


def test_ensure_io_provenance_rejects_missing_required_relpath(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        # missing source_relpath_posix
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    # ensure_io_provenance validates; resolve_provenance_path is stricter about missing relpath
    out = ensure_io_provenance(payload)
    assert isinstance(out, IOProvenance)
    assert out.source_relpath_posix is None


# -----------------------------------------------------------------------------
# normalize_provenance
# -----------------------------------------------------------------------------


def test_normalize_provenance_none_returns_none():
    assert normalize_provenance(None) is None


def test_normalize_provenance_converts_backslashes(tmp_path: Path):
    io = IOProvenance(
        project_root=str(tmp_path),
        source_relpath_posix=r"labeled-data\test\CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="df_with_missing",
    )
    out = normalize_provenance(io)
    assert out is not None
    assert out.source_relpath_posix == "labeled-data/test/CollectedData_Jane.h5"


# -----------------------------------------------------------------------------
# resolve_provenance_path
# -----------------------------------------------------------------------------


def test_resolve_provenance_path_uses_root_anchor_when_provided(tmp_path: Path):
    # Two valid roots
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    other_root = tmp_path / "other_root"
    other_root.mkdir()

    # File exists ONLY under anchor
    (anchor / "CollectedData_Jane.h5").write_bytes(b"dummy")

    io = {
        "schema_version": 1,
        "project_root": str(other_root),  # valid dir, but not where file exists
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }

    resolved = resolve_provenance_path(io, root_anchor=anchor)
    assert resolved == anchor / "CollectedData_Jane.h5"


def test_resolve_provenance_path_uses_project_root_when_root_anchor_missing(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "CollectedData_Jane.h5").write_bytes(b"dummy")

    io = {
        "schema_version": 1,
        "project_root": str(root),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }

    resolved = resolve_provenance_path(io, root_anchor=None)
    assert resolved == root / "CollectedData_Jane.h5"


def test_resolve_provenance_path_requires_source_relpath_posix(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": None,
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    with pytest.raises(MissingProvenanceError):
        resolve_provenance_path(payload)


def test_resolve_provenance_path_requires_anchor_or_project_root(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": None,
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    with pytest.raises(UnresolvablePathError):
        resolve_provenance_path(payload, root_anchor=None)


def test_resolve_provenance_path_raises_if_missing_by_default(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    with pytest.raises(UnresolvablePathError):
        resolve_provenance_path(payload, allow_missing=False)


def test_resolve_provenance_path_allows_missing_when_flag_true(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    resolved = resolve_provenance_path(payload, allow_missing=True)
    assert resolved == tmp_path / "CollectedData_Jane.h5"


def test_resolve_provenance_path_normalizes_backslashes(tmp_path: Path):
    # Create expected file
    (tmp_path / "labeled-data").mkdir()
    (tmp_path / "labeled-data" / "CollectedData_Jane.h5").write_bytes(b"dummy")

    payload = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": r"labeled-data\CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "df_with_missing",
    }
    resolved = resolve_provenance_path(payload)
    assert resolved == tmp_path / "labeled-data" / "CollectedData_Jane.h5"
