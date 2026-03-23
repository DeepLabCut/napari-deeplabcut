from pathlib import Path

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.metadata import build_io_provenance_dict

# -----------------------------------------------------------------------------
# build_io_provenance_dict
# -----------------------------------------------------------------------------


def test_build_io_provenance_dict_keeps_enum_kind_object(tmp_path: Path):
    d = build_io_provenance_dict(
        project_root=tmp_path,
        source_relpath_posix="CollectedData_Jane.h5",
        kind=AnnotationKind.GT,
        dataset_key="keypoints",
    )
    # mode="python" => should keep enum object at runtime
    assert isinstance(d["kind"], AnnotationKind)
    assert d["kind"] == AnnotationKind.GT
    assert d["project_root"] == str(tmp_path)
    assert d["source_relpath_posix"] == "CollectedData_Jane.h5"
    assert d["dataset_key"] == "keypoints"
    assert d["schema_version"] == 1


def test_build_io_provenance_dict_excludes_none_fields(tmp_path: Path):
    d = build_io_provenance_dict(
        project_root=tmp_path,
        source_relpath_posix="CollectedData_Jane.h5",
        kind=None,  # exclude_none=True => kind should be absent
        dataset_key="keypoints",
    )
    assert "kind" not in d
