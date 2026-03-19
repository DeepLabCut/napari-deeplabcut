from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

import napari_deeplabcut.core.metadata as metadata_mod
from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel, ImageMetadata, PointsMetadata
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.metadata import build_io_provenance_dict

# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------


def _make_validation_error() -> ValidationError:
    class TmpModel(BaseModel):
        x: int

    try:
        TmpModel.model_validate({"x": "not-an-int"})
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


class DummyLayer:
    def __init__(self, metadata=None, name="dummy-layer"):
        self.metadata = metadata
        self.name = name


# -----------------------------------------------------------------------------
# pure helper coverage
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path_str", "expected"),
    [
        ("project/labeled-data/mouse1", True),
        ("project/LABELED-DATA/mouse1", True),
        ("project/labeled-data", False),
        ("project/images/mouse1", False),
    ],
)
def test_is_dlc_dataset_root(path_str: str, expected: bool):
    assert metadata_mod._is_dlc_dataset_root(Path(path_str)) is expected


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (None, False),
        ([], False),
        (["images/img001.png"], False),
        (["labeled-data/test/img001.png"], True),
        ([r"labeled-data\test\img001.png"], True),
    ],
)
def test_paths_look_like_labeled_data(paths, expected):
    assert metadata_mod._paths_look_like_labeled_data(paths) is expected


def test_looks_like_project_root_true_when_same_path(tmp_path: Path):
    assert metadata_mod._looks_like_project_root(str(tmp_path), str(tmp_path)) is True


def test_looks_like_project_root_false_when_different(tmp_path: Path):
    other = tmp_path / "other"
    assert metadata_mod._looks_like_project_root(str(tmp_path), str(other)) is False


def test_infer_image_root_prefers_explicit_root(tmp_path: Path):
    p = tmp_path / "images" / "img001.png"
    p.parent.mkdir(parents=True)
    p.touch()

    result = metadata_mod.infer_image_root(
        explicit_root="/explicit/root",
        paths=[str(p)],
        source_path=str(p),
    )

    assert result == "/explicit/root"


def test_infer_image_root_uses_first_path_parent(tmp_path: Path):
    p = tmp_path / "images" / "img001.png"
    p.parent.mkdir(parents=True)
    p.touch()

    result = metadata_mod.infer_image_root(paths=[str(p)])
    assert result == str(p.parent.resolve())


def test_infer_image_root_falls_back_to_source_path_parent(tmp_path: Path):
    p = tmp_path / "images" / "img001.png"
    p.parent.mkdir(parents=True)
    p.touch()

    result = metadata_mod.infer_image_root(source_path=str(p))
    assert result == str(p.parent.resolve())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ([], True),
        ({}, True),
        ((), True),
        (False, False),
        (0, False),
        ("x", False),
        ([1], False),
    ],
)
def test_is_empty_value(value, expected):
    assert metadata_mod._is_empty_value(value) is expected


def test_require_unique_target_returns_single_candidate(tmp_path: Path):
    candidate = tmp_path / "CollectedData_A.h5"
    assert metadata_mod.require_unique_target([candidate]) == candidate


def test_require_unique_target_raises_when_missing():
    with pytest.raises(MissingProvenanceError, match="No candidates found"):
        metadata_mod.require_unique_target([], context="save target")


def test_require_unique_target_raises_when_ambiguous(tmp_path: Path):
    c1 = tmp_path / "CollectedData_A.h5"
    c2 = tmp_path / "CollectedData_B.h5"

    with pytest.raises(AmbiguousSaveError, match="Ambiguous save target"):
        metadata_mod.require_unique_target([c1, c2], context="save target")


# -----------------------------------------------------------------------------
# merge / sync
# -----------------------------------------------------------------------------


def test_merge_image_metadata_only_fills_missing_fields():
    base = ImageMetadata(root="rootA", name="", shape=None)
    incoming = ImageMetadata(root="rootB", name="images", shape=[10, 20])

    merged = metadata_mod.merge_image_metadata(base, incoming)

    assert merged.root == "rootA"  # preserved
    assert merged.name == "images"  # filled
    assert tuple(merged.shape) == (10, 20)  # filled


def test_merge_points_metadata_does_not_clobber_and_skips_controls():
    base = PointsMetadata(root="rootA", name="", controls={"runtime": 1})
    incoming = PointsMetadata(root="rootB", name="points", controls={"runtime": 2})

    merged = metadata_mod.merge_points_metadata(base, incoming)

    assert merged.root == "rootA"
    assert merged.name == "points"
    # controls should not be copied from incoming
    assert getattr(merged, "controls", None) in (None, {"runtime": 1})


def test_sync_points_from_image_fills_missing_fields():
    image_meta = ImageMetadata(
        root="project/labeled-data/mouse1",
        paths=["labeled-data/mouse1/img001.png"],
        shape=[100, 200],
        name="images",
    )
    points_meta = PointsMetadata()

    synced = metadata_mod.sync_points_from_image(image_meta, points_meta)

    assert synced.root == "project/labeled-data/mouse1"
    assert synced.paths == ["labeled-data/mouse1/img001.png"]
    assert tuple(synced.shape) == (100, 200)
    assert synced.name == "images"


def test_sync_points_from_image_overrides_project_root_with_dataset_root(tmp_path: Path):
    project_root = tmp_path / "project"
    dataset_root = project_root / "labeled-data" / "mouse1"
    dataset_root.mkdir(parents=True)

    image_meta = ImageMetadata(
        root=str(dataset_root),
        paths=[str(dataset_root / "img001.png")],
        name="images",
    )
    points_meta = PointsMetadata(
        root=str(project_root),  # stale / wrong
        project=str(project_root),
    )

    synced = metadata_mod.sync_points_from_image(image_meta, points_meta)

    assert synced.root == str(dataset_root)


def test_sync_points_from_image_keeps_existing_dataset_root_when_already_good(tmp_path: Path):
    project_root = tmp_path / "project"
    good_points_root = project_root / "labeled-data" / "mouse1"
    other_dataset_root = project_root / "labeled-data" / "mouse2"
    good_points_root.mkdir(parents=True)
    other_dataset_root.mkdir(parents=True)

    image_meta = ImageMetadata(root=str(other_dataset_root))
    points_meta = PointsMetadata(
        root=str(good_points_root),
        project=str(project_root),
    )

    synced = metadata_mod.sync_points_from_image(image_meta, points_meta)

    # already a valid dataset root -> do not overwrite
    assert synced.root == str(good_points_root)


def test_ensure_metadata_models_accepts_dicts_and_models():
    ImageMetadata(root="img-root")
    pts_model = PointsMetadata(root="pts-root")

    img, pts = metadata_mod.ensure_metadata_models(
        {"root": "img-root"},
        pts_model,
    )

    assert isinstance(img, ImageMetadata)
    assert img.root == "img-root"
    assert pts is pts_model


# -----------------------------------------------------------------------------
# parsing / coercion
# -----------------------------------------------------------------------------


def test_normalize_columns_handles_index_and_multiindex():
    idx = pd.Index(["a", "b"])
    mi = pd.MultiIndex.from_tuples([("scorer", "nose"), ("scorer", "tail")])

    assert metadata_mod._normalize_columns(idx) == ["a", "b"]
    assert metadata_mod._normalize_columns(mi) == [("scorer", "nose"), ("scorer", "tail")]


def test_coerce_io_kind_accepts_value_and_name():
    d1 = {"kind": "gt"}
    d2 = {"kind": "MACHINE"}

    metadata_mod._coerce_io_kind(d1)
    metadata_mod._coerce_io_kind(d2)

    assert d1["kind"] == AnnotationKind.GT
    assert d2["kind"] == AnnotationKind.MACHINE


def test_parse_points_metadata_none_returns_empty_model():
    parsed = metadata_mod.parse_points_metadata(None)
    assert isinstance(parsed, PointsMetadata)


def test_parse_points_metadata_drops_controls_and_coerces_kinds(monkeypatch):
    captured = {}

    def fake_model_validate(payload):
        captured["payload"] = payload
        return PointsMetadata()

    monkeypatch.setattr(metadata_mod.PointsMetadata, "model_validate", fake_model_validate)

    md = {
        "controls": {"runtime": object()},
        "io": {
            "kind": "gt",
            "project_root": "/tmp",
            "source_relpath_posix": "CollectedData_A.h5",
            "dataset_key": "keypoints",
        },
        "save_target": {"kind": "MACHINE"},
    }

    parsed = metadata_mod.parse_points_metadata(md)

    assert isinstance(parsed, PointsMetadata)
    assert "controls" not in captured["payload"]
    assert captured["payload"]["io"]["kind"] == AnnotationKind.GT
    assert captured["payload"]["save_target"]["kind"] == AnnotationKind.MACHINE


def test_parse_points_metadata_drop_header_removes_header(monkeypatch):
    captured = {}

    def fake_model_validate(payload):
        captured["payload"] = payload
        return PointsMetadata()

    monkeypatch.setattr(metadata_mod.PointsMetadata, "model_validate", fake_model_validate)

    parsed = metadata_mod.parse_points_metadata(
        {"header": {"columns": [("scorer", "nose")]}, "root": "x"},
        drop_header=True,
    )

    assert isinstance(parsed, PointsMetadata)
    assert "header" not in captured["payload"]


def test_parse_points_metadata_falls_back_to_empty_model_on_validation_error(monkeypatch):
    def boom(payload):
        raise RuntimeError("bad metadata")

    monkeypatch.setattr(metadata_mod.PointsMetadata, "model_validate", boom)

    parsed = metadata_mod.parse_points_metadata({"root": "x"})
    assert isinstance(parsed, PointsMetadata)


def test_coerce_header_model_none_passthrough():
    assert metadata_mod.coerce_header_model(None) is None


def test_coerce_header_model_returns_existing_model():
    # NOTE: might need to use the header fixture here
    header = DLCHeaderModel(columns=[("scorer", "nose")])
    assert metadata_mod.coerce_header_model(header) is header


# -----------------------------------------------------------------------------
# metadata dict / legacy migration helpers
# -----------------------------------------------------------------------------


def test_layer_metadata_dict_handles_none_and_mapping_like():
    assert metadata_mod._layer_metadata_dict(SimpleNamespace(metadata=None)) == {}
    assert metadata_mod._layer_metadata_dict(SimpleNamespace(metadata={"a": 1})) == {"a": 1}

    class MappingLike:
        def __iter__(self):
            return iter([("x", 1)])

    assert metadata_mod._layer_metadata_dict(SimpleNamespace(metadata=MappingLike())) == {"x": 1}


def test_build_io_from_source_h5_returns_none_for_empty_source():
    assert metadata_mod._build_io_from_source_h5("") is None
    assert metadata_mod._build_io_from_source_h5(None) is None


def test_prepare_points_payload_migrates_legacy_source_h5(monkeypatch):
    monkeypatch.setattr(
        metadata_mod,
        "_build_io_from_source_h5",
        lambda src, dataset_key="keypoints": {"kind": AnnotationKind.GT, "dataset_key": dataset_key},
    )

    payload = metadata_mod._prepare_points_payload(
        {"source_h5": "/tmp/CollectedData_A.h5"},
        migrate_legacy=True,
    )

    assert payload["io"]["kind"] == AnnotationKind.GT
    assert payload["io"]["dataset_key"] == "keypoints"


# -----------------------------------------------------------------------------
# attach_source_and_io
# -----------------------------------------------------------------------------


def test_attach_source_and_io_sets_legacy_fields_and_io(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "CollectedData_Jane.h5"
    file_path.touch()

    monkeypatch.setattr(metadata_mod, "canonicalize_path", lambda p, n=1: "CollectedData_Jane.h5")
    monkeypatch.setattr(metadata_mod, "infer_annotation_kind_for_file", lambda p: AnnotationKind.GT)

    metadata = {}
    metadata_mod.attach_source_and_io(metadata, file_path)

    inner = metadata["metadata"]
    assert inner["source_h5_name"] == "CollectedData_Jane.h5"
    assert inner["source_h5_stem"] == "CollectedData_Jane"
    assert inner["source_h5"].endswith("CollectedData_Jane.h5")
    assert inner["io"]["kind"] == AnnotationKind.GT
    assert inner["io"]["source_relpath_posix"] == "CollectedData_Jane.h5"
    assert inner["io"]["dataset_key"] == "keypoints"


# -----------------------------------------------------------------------------
# read/write adapter gateway
# -----------------------------------------------------------------------------


def test_read_points_meta_returns_validation_error(monkeypatch):
    err = _make_validation_error()

    monkeypatch.setattr(metadata_mod, "_prepare_points_payload", lambda *args, **kwargs: {"bad": "payload"})
    monkeypatch.setattr(metadata_mod.PointsMetadata, "model_validate", lambda payload: (_ for _ in ()).throw(err))

    layer = DummyLayer(metadata={"root": "x"})
    result = metadata_mod.read_points_meta(layer)

    assert isinstance(result, ValidationError)


def test_write_points_meta_merge_missing_preserves_existing_and_restores_header(monkeypatch):
    header = DLCHeaderModel(columns=[("scorer", "nose")])
    layer = DummyLayer(metadata={"root": "existing-root", "header": header})

    validated = PointsMetadata(root="existing-root", name="incoming-name", header=header)

    def fake_model_validate(payload):
        # root should remain existing because MERGE_MISSING only fills empties
        assert payload["root"] == "existing-root"
        assert payload["name"] == "incoming-name"
        return validated

    monkeypatch.setattr(metadata_mod.PointsMetadata, "model_validate", fake_model_validate)

    result = metadata_mod.write_points_meta(
        layer,
        {"root": "new-root", "name": "incoming-name"},
        metadata_mod.MergePolicy.MERGE_MISSING,
        validate=True,
    )

    assert isinstance(result, PointsMetadata)
    assert layer.metadata["root"] == "existing-root"
    assert layer.metadata["name"] == "incoming-name"
    assert "header" in layer.metadata


def test_write_points_meta_replace_without_validation_writes_raw_mapping():
    layer = DummyLayer(metadata={"root": "old", "name": "old-name"})

    result = metadata_mod.write_points_meta(
        layer,
        {"root": "new", "name": "new-name", "controls": {"runtime": 1}},
        metadata_mod.MergePolicy.REPLACE,
        validate=False,
    )

    assert layer.metadata == {"root": "new", "name": "new-name"}
    assert isinstance(result, (PointsMetadata, ValidationError))


def test_write_points_meta_returns_validation_error_and_leaves_metadata_stable(monkeypatch):
    err = _make_validation_error()
    layer = DummyLayer(metadata={"root": "old"})

    monkeypatch.setattr(
        metadata_mod.PointsMetadata,
        "model_validate",
        lambda payload: (_ for _ in ()).throw(err),
    )

    result = metadata_mod.write_points_meta(
        layer,
        {"root": "new"},
        metadata_mod.MergePolicy.MERGE,
        validate=True,
    )

    assert isinstance(result, ValidationError)
    # write should not have replaced metadata after failed validation
    assert layer.metadata == {"root": "old"}


def test_read_image_meta_returns_validation_error(monkeypatch):
    err = _make_validation_error()
    layer = DummyLayer(metadata={"root": "x"})

    monkeypatch.setattr(
        metadata_mod.ImageMetadata,
        "model_validate",
        lambda payload: (_ for _ in ()).throw(err),
    )

    result = metadata_mod.read_image_meta(layer)
    assert isinstance(result, ValidationError)


def test_write_image_meta_merge_policy_string_and_fields_filter():
    layer = DummyLayer(metadata={"root": "old", "name": ""})

    result = metadata_mod.write_image_meta(
        layer,
        {"root": "new", "name": "images", "shape": [10, 20]},
        "merge_missing",
        fields={"name"},
        validate=True,
    )

    assert isinstance(result, ImageMetadata)
    assert layer.metadata["root"] == "old"
    assert layer.metadata["name"] == "images"
    assert "shape" not in layer.metadata


def test_migrate_points_layer_metadata_round_trips_through_gateway(monkeypatch):
    layer = DummyLayer(metadata={"source_h5": "/tmp/CollectedData_A.h5"})

    read_result = PointsMetadata(root="rootA")
    write_result = PointsMetadata(root="rootA", name="points")

    monkeypatch.setattr(metadata_mod, "read_points_meta", lambda *args, **kwargs: read_result)
    monkeypatch.setattr(metadata_mod, "write_points_meta", lambda *args, **kwargs: write_result)

    result = metadata_mod.migrate_points_layer_metadata(layer)

    assert result is write_result


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
