# src/napari_deeplabcut/_tests/test_provenance_unit.py

from __future__ import annotations

from pathlib import Path

import pytest

from napari_deeplabcut.config.models import AnnotationKind, IOProvenance
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.provenance import (
    build_io_provenance_dict,
    ensure_io_provenance,
    normalize_provenance,
    resolve_provenance_path,
)


def test_ensure_io_provenance_none_returns_none():
    assert ensure_io_provenance(None) is None


def test_ensure_io_provenance_accepts_model_instance(tmp_path: Path):
    io = IOProvenance(
        project_root=str(tmp_path),
        source_relpath_posix="CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="keypoints",
    )
    out = ensure_io_provenance(io)
    assert out is io


def test_ensure_io_provenance_rejects_invalid_type():
    with pytest.raises(MissingProvenanceError):
        ensure_io_provenance(["not", "a", "dict"])  # type: ignore[arg-type]


def test_ensure_io_provenance_rejects_missing_required_fields():
    # Missing source_relpath_posix and dataset_key and maybe others
    with pytest.raises(MissingProvenanceError):
        ensure_io_provenance({"project_root": "X"})


def test_normalize_provenance_rewrites_backslashes(tmp_path: Path):
    io = IOProvenance(
        project_root=str(tmp_path),
        source_relpath_posix=r"labeled-data\test\CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="keypoints",
    )
    out = normalize_provenance(io)
    assert out is not None
    assert out.source_relpath_posix == "labeled-data/test/CollectedData_Jane.h5"


def test_build_io_provenance_dict_keeps_enum_kind_runtime(tmp_path: Path):
    d = build_io_provenance_dict(
        project_root=tmp_path,
        source_relpath_posix="CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="keypoints",
    )
    # Contract: `mode="python"` keeps enum objects at runtime (no stringification)
    assert isinstance(d["kind"], AnnotationKind)
    assert d["kind"] == AnnotationKind.GT
    assert d["source_relpath_posix"] == "CollectedData_Jane.h5"
    assert d["dataset_key"] == "keypoints"


def test_resolve_provenance_path_uses_root_anchor_when_provided(tmp_path: Path):
    # Create the file at anchor / relpath
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "CollectedData_Jane.h5").write_bytes(b"dummy")

    io = {
        "project_root": "SHOULD_NOT_BE_USED_IF_ROOT_ANCHOR_SET",
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "schema_version": 1,
    }

    p = resolve_provenance_path(io, root_anchor=anchor)
    assert p == anchor / "CollectedData_Jane.h5"


def test_resolve_provenance_path_requires_relpath(tmp_path: Path):
    io = {
        "project_root": str(tmp_path),
        "source_relpath_posix": "",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "schema_version": 1,
    }
    with pytest.raises(MissingProvenanceError):
        resolve_provenance_path(io)


def test_resolve_provenance_path_requires_anchor_when_project_root_missing():
    io = {
        "project_root": "",
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "schema_version": 1,
    }
    with pytest.raises(UnresolvablePathError):
        resolve_provenance_path(io, root_anchor=None)


def test_resolve_provenance_path_raises_if_missing_file_by_default(tmp_path: Path):
    io = {
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "schema_version": 1,
    }
    with pytest.raises(UnresolvablePathError):
        resolve_provenance_path(io, allow_missing=False)


def test_resolve_provenance_path_allows_missing_when_flag_true(tmp_path: Path):
    io = {
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Jane.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "schema_version": 1,
    }
    p = resolve_provenance_path(io, allow_missing=True)
    assert p == tmp_path / "CollectedData_Jane.h5"
