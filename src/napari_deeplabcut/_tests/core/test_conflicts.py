from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import napari_deeplabcut.core.conflicts as conflicts_mod
import napari_deeplabcut.core.dataframes as dataframes_mod
from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.project_paths import DLCProjectContext


def _make_points_meta(
    *,
    header_scorer: str | None = "scorerA",
    save_target=None,
    io_kind=AnnotationKind.GT,
    root: str | None = None,
):
    header = None if header_scorer is None else SimpleNamespace(scorer=header_scorer)
    io = None if io_kind is None else SimpleNamespace(kind=io_kind)
    return SimpleNamespace(
        header=header,
        save_target=save_target,
        io=io,
        root=root,
    )


def _stub_validation_pipeline(
    monkeypatch,
    *,
    pts_meta,
    attrs_obj=None,
    points_obj=None,
    props_obj=None,
    ctx_obj=None,
    df_new=None,
):
    """
    Stub schema validation + parse_points_metadata + form_df_from_validated.

    This keeps tests focused on the routing/conflict logic rather than on
    pydantic/schema correctness (which is tested elsewhere).
    """
    if attrs_obj is None:
        attrs_obj = SimpleNamespace(
            metadata={"some": "meta"},
            properties={"label": ["nose"], "id": [1]},
        )
    if points_obj is None:
        points_obj = SimpleNamespace(data="POINTS")
    if props_obj is None:
        props_obj = SimpleNamespace(properties="PROPS")
    if ctx_obj is None:
        ctx_obj = SimpleNamespace(ctx="CTX")
    if df_new is None:
        df_new = pd.DataFrame({"dummy": [1]})

    monkeypatch.setattr(
        conflicts_mod.dlc_schemas.PointsLayerAttributesModel,
        "model_validate",
        lambda payload: attrs_obj,
    )
    monkeypatch.setattr(
        conflicts_mod,
        "parse_points_metadata",
        lambda metadata, drop_header=False: pts_meta,
    )
    monkeypatch.setattr(
        conflicts_mod.dlc_schemas.PointsDataModel,
        "model_validate",
        lambda payload: points_obj,
    )
    monkeypatch.setattr(
        conflicts_mod.dlc_schemas.KeypointPropertiesModel,
        "model_validate",
        lambda payload: props_obj,
    )
    monkeypatch.setattr(
        conflicts_mod.dlc_schemas.PointsWriteInputModel,
        "model_validate",
        lambda payload: ctx_obj,
    )
    monkeypatch.setattr(
        dataframes_mod,
        "form_df_from_validated",
        lambda ctx: df_new,
    )

    return SimpleNamespace(
        attrs_obj=attrs_obj,
        points_obj=points_obj,
        props_obj=props_obj,
        ctx_obj=ctx_obj,
        df_new=df_new,
    )


def test_compute_overwrite_report_raises_when_header_missing(monkeypatch):
    pts_meta = _make_points_meta(header_scorer=None)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    with pytest.raises(ValueError, match="valid DLC header"):
        conflicts_mod.compute_overwrite_report_for_points_save(
            data=[[0, 1, 2]],
            attributes={"name": "points"},
        )


def test_compute_overwrite_report_raises_for_machine_source_without_resolved_target(monkeypatch):
    pts_meta = _make_points_meta(io_kind=AnnotationKind.MACHINE)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (None, None, AnnotationKind.MACHINE),
    )

    with pytest.raises(MissingProvenanceError, match="MACHINE source"):
        conflicts_mod.compute_overwrite_report_for_points_save(
            data=[[0, 1, 2]],
            attributes={"name": "points"},
        )


def test_compute_overwrite_report_raises_when_gt_fallback_is_ambiguous(monkeypatch, tmp_path):
    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (None, None, AnnotationKind.GT),
    )
    monkeypatch.setattr(
        conflicts_mod,
        "infer_dlc_project_from_points_meta",
        lambda *args, **kwargs: DLCProjectContext(
            root_anchor=tmp_path,
            dataset_folder=tmp_path,
        ),
    )

    (tmp_path / "CollectedData_a.h5").touch()
    (tmp_path / "CollectedData_b.h5").touch()

    with pytest.raises(AmbiguousSaveError) as excinfo:
        conflicts_mod.compute_overwrite_report_for_points_save(
            data=[[0, 1, 2]],
            attributes={"name": "points"},
        )

    err = excinfo.value
    assert "Multiple CollectedData_*.h5 files found" in str(err)
    assert sorted(Path(p).name for p in err.candidates) == [
        "CollectedData_a.h5",
        "CollectedData_b.h5",
    ]


def test_compute_overwrite_report_returns_none_for_non_gt_destination(monkeypatch, tmp_path):
    out = tmp_path / "machine_output.h5"

    pts_meta = _make_points_meta(
        save_target=None,
        io_kind=AnnotationKind.MACHINE,
    )
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (str(out), None, AnnotationKind.GT),
    )

    result = conflicts_mod.compute_overwrite_report_for_points_save(
        data=[[0, 1, 2]],
        attributes={"name": "points"},
    )

    assert result is None


def test_compute_overwrite_report_returns_none_when_output_does_not_exist(monkeypatch, tmp_path):
    out = tmp_path / "CollectedData_scorerA.h5"

    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (str(out), None, AnnotationKind.GT),
    )

    result = conflicts_mod.compute_overwrite_report_for_points_save(
        data=[[0, 1, 2]],
        attributes={"name": "points"},
    )

    assert result is None


def test_compute_overwrite_report_returns_report_for_existing_gt_file_with_conflicts(monkeypatch, tmp_path):
    out = tmp_path / "CollectedData_target.h5"
    out.touch()

    old_df = pd.DataFrame({"old": [1]})
    raw_new_df = pd.DataFrame({"new": [1]})
    promoted_df = pd.DataFrame({"promoted": [1]})
    key_conflict = object()
    report = SimpleNamespace(has_conflicts=True, marker="REPORT")

    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta, df_new=raw_new_df)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (str(out), "target_scorer", AnnotationKind.GT),
    )

    seen = {}

    def fake_set_df_scorer(df, scorer):
        seen["set_df_scorer"] = (df, scorer)
        return promoted_df

    def fake_read_hdf(path, key=None):
        seen.setdefault("read_hdf_calls", []).append((Path(path), key))
        return old_df

    def fake_keypoint_conflicts(df_old, df_new):
        seen["keypoint_conflicts"] = (df_old, df_new)
        return key_conflict

    def fake_build_report(conflicts, *, layer_name, destination_path):
        seen["build_report"] = (conflicts, layer_name, destination_path)
        return report

    monkeypatch.setattr(conflicts_mod, "set_df_scorer", fake_set_df_scorer)
    monkeypatch.setattr(pd, "read_hdf", fake_read_hdf)
    monkeypatch.setattr(dataframes_mod, "keypoint_conflicts", fake_keypoint_conflicts)
    monkeypatch.setattr(dataframes_mod, "build_overwrite_conflict_report", fake_build_report)

    result = conflicts_mod.compute_overwrite_report_for_points_save(
        data=[[0, 1, 2]],
        attributes={"name": "my-points-layer"},
    )

    assert result is report
    assert seen["set_df_scorer"] == (raw_new_df, "target_scorer")
    assert seen["read_hdf_calls"] == [(out, "keypoints")]
    assert seen["keypoint_conflicts"] == (old_df, promoted_df)
    assert seen["build_report"] == (
        key_conflict,
        "my-points-layer",
        str(out),
    )


def test_compute_overwrite_report_returns_none_when_report_has_no_conflicts(monkeypatch, tmp_path):
    out = tmp_path / "CollectedData_target.h5"
    out.touch()

    old_df = pd.DataFrame({"old": [1]})
    new_df = pd.DataFrame({"new": [1]})
    report = SimpleNamespace(has_conflicts=False)

    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta, df_new=new_df)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (str(out), None, AnnotationKind.GT),
    )
    monkeypatch.setattr(pd, "read_hdf", lambda path, key=None: old_df)
    monkeypatch.setattr(dataframes_mod, "keypoint_conflicts", lambda df_old, df_new: "conflicts")
    monkeypatch.setattr(
        dataframes_mod,
        "build_overwrite_conflict_report",
        lambda conflicts, *, layer_name, destination_path: report,
    )

    result = conflicts_mod.compute_overwrite_report_for_points_save(
        data=[[0, 1, 2]],
        attributes={"name": "my-points-layer"},
    )

    assert result is None


def test_compute_overwrite_report_falls_back_when_keyed_hdf_read_fails(monkeypatch, tmp_path):
    out = tmp_path / "CollectedData_target.h5"
    out.touch()

    old_df = pd.DataFrame({"old": [1]})
    new_df = pd.DataFrame({"new": [1]})
    report = SimpleNamespace(has_conflicts=True)

    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta, df_new=new_df)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (str(out), None, AnnotationKind.GT),
    )

    calls = []

    def fake_read_hdf(path, key=None):
        calls.append((Path(path), key))
        if key == "keypoints":
            raise KeyError("missing key")
        return old_df

    monkeypatch.setattr(pd, "read_hdf", fake_read_hdf)
    monkeypatch.setattr(dataframes_mod, "keypoint_conflicts", lambda df_old, df_new: "conflicts")
    monkeypatch.setattr(
        dataframes_mod,
        "build_overwrite_conflict_report",
        lambda conflicts, *, layer_name, destination_path: report,
    )

    result = conflicts_mod.compute_overwrite_report_for_points_save(
        data=[[0, 1, 2]],
        attributes={"name": "my-points-layer"},
    )

    assert result is report
    assert calls == [
        (out, "keypoints"),
        (out, None),
    ]


def test_compute_overwrite_report_raises_when_gt_fallback_has_no_root_and_no_dataset_dir(monkeypatch):
    pts_meta = _make_points_meta(io_kind=AnnotationKind.GT, root=None)
    _stub_validation_pipeline(monkeypatch, pts_meta=pts_meta)

    monkeypatch.setattr(
        conflicts_mod,
        "resolve_output_path_from_metadata",
        lambda attributes: (None, None, AnnotationKind.GT),
    )
    monkeypatch.setattr(
        conflicts_mod,
        "infer_dlc_project_from_points_meta",
        lambda *args, **kwargs: DLCProjectContext(),
    )

    with pytest.raises(MissingProvenanceError, match="GT fallback requires root"):
        conflicts_mod.compute_overwrite_report_for_points_save(
            data=[[0, 1, 2]],
            attributes={"name": "points"},
        )
