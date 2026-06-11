from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel
from napari_deeplabcut.core.dataframes import drop_likelihood_columns
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.io import (
    _drop_likelihood_from_header,
    resolve_output_path_from_metadata,
    write_hdf,
)


def test_resolve_output_path_returns_none_for_machine_without_save_target():
    md = {
        "metadata": {
            "io": {
                "schema_version": 1,
                "project_root": str(Path.cwd()),
                "source_relpath_posix": "machinelabels-iter0.h5",
                "kind": AnnotationKind.MACHINE,
                "dataset_key": "df_with_missing",
            }
        }
    }
    out_path, scorer, kind = resolve_output_path_from_metadata(md)
    assert out_path is None
    assert scorer is None
    assert kind == AnnotationKind.MACHINE


def test_write_hdf_refuses_machine_without_promotion(tmp_path: Path):
    # minimal points + metadata for a machine source
    data = np.zeros((1, 3), dtype=float)
    attrs = {
        "metadata": {
            "root": str(tmp_path),
            "io": {
                "schema_version": 1,
                "project_root": str(tmp_path),
                "source_relpath_posix": "machinelabels-iter0.h5",
                "kind": AnnotationKind.MACHINE,
                "dataset_key": "df_with_missing",
            },
            # header is required by writer
            "header": {
                "columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")],
            },
        },
        "properties": {"label": ["bp1"], "id": [""], "likelihood": [1.0]},
    }

    with pytest.raises(MissingProvenanceError):
        write_hdf("__dlc__.h5", data, attrs)


def test_write_hdf_raises_ambiguous_when_multiple_gt_candidates_and_no_provenance(tmp_path: Path):
    # Create two GT candidates in root folder
    (tmp_path / "CollectedData_John.h5").write_bytes(b"dummy")
    (tmp_path / "CollectedData_Jane.h5").write_bytes(b"dummy")

    data = np.zeros((1, 3), dtype=float)
    attrs = {
        "metadata": {
            "root": str(tmp_path),
            "header": {
                "columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")],
            },
            # No io/source_h5 => triggers GT fallback scan => ambiguous
        },
        "properties": {"label": ["bp1"], "id": [""], "likelihood": [1.0]},
    }

    with pytest.raises(AmbiguousSaveError):
        write_hdf("__dlc__.h5", data, attrs)


def test_write_hdf_aborts_machine_without_promotion_target(tmp_path: Path):
    data = np.array([[0.0, 44.0, 55.0]], dtype=float)
    attrs = {
        "metadata": {
            "root": str(tmp_path),
            "io": {
                "schema_version": 1,
                "project_root": str(tmp_path),
                "source_relpath_posix": "machinelabels-iter0.h5",
                "kind": AnnotationKind.MACHINE,
                "dataset_key": "df_with_missing",
            },
            "header": {"columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")]},
        },
        "properties": {"label": ["bp1"], "id": [""], "likelihood": [1.0]},
    }

    with pytest.raises(MissingProvenanceError):
        write_hdf("__dlc__.h5", data, attrs)


def test_drop_likelihood_before_merge_prevents_machine_likelihood_from_leaking():
    """
    Regression test for GT merge-on-save.

    Machine labels may contain a likelihood coord, while GT labels should not.
    Because DataFrame.combine_first() keeps the union of columns, likelihood
    would survive the merge unless explicitly removed first.
    """
    gt_cols = pd.MultiIndex.from_product(
        [["John"], ["bp1"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    machine_cols = pd.MultiIndex.from_product(
        [["John"], ["bp1"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )

    # Existing GT file on disk: only x/y
    df_old = pd.DataFrame(
        [[1.0, 2.0]],
        index=["img000.png"],
        columns=gt_cols,
    )

    # New machine/promoted labels: x/y + likelihood
    df_new = pd.DataFrame(
        [[10.0, 20.0, 0.95]],
        index=["img000.png"],
        columns=machine_cols,
    )

    # Mimic writer behavior: strip likelihood on both sides before merge
    df_old = drop_likelihood_columns(df_old)
    df_new = drop_likelihood_columns(df_new)

    df_out = df_new.combine_first(df_old)

    assert "likelihood" not in df_out.columns.get_level_values("coords")
    expected = pd.DataFrame(
        [[10.0, 20.0]],
        index=["img000.png"],
        columns=gt_cols,
    )
    pd.testing.assert_frame_equal(df_out, expected)


def test_drop_likelihood_cleans_existing_gt_columns_too():
    """
    If an existing GT dataframe already contains likelihood columns from a
    previous bad write, they must still be removed before the merged result
    is saved again.
    """
    cols_with_likelihood = pd.MultiIndex.from_product(
        [["John"], ["bp1"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )

    df_old = pd.DataFrame(
        [[1.0, 2.0, 0.4]],
        index=["img000.png"],
        columns=cols_with_likelihood,
    )
    df_new = pd.DataFrame(
        [[10.0, 20.0, 0.95]],
        index=["img000.png"],
        columns=cols_with_likelihood,
    )

    df_old = drop_likelihood_columns(df_old)
    df_new = drop_likelihood_columns(df_new)
    df_out = df_new.combine_first(df_old)

    coords = df_out.columns.get_level_values("coords")
    assert list(coords) == ["x", "y"]


def test_drop_likelihood_columns_removes_likelihood_from_empty_dataframe():
    """
    Regression guard: likelihood coords must be removed even when the dataframe
    has 0 rows, e.g. an empty machine layer promoted to GT.
    """
    cols = pd.MultiIndex.from_product(
        [["John"], ["bp1"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame([], columns=cols, index=pd.Index([], name="image"))

    out = drop_likelihood_columns(df)

    assert out.empty
    assert "likelihood" not in out.columns.get_level_values("coords")
    assert list(out.columns.get_level_values("coords")) == ["x", "y"]


def test_drop_likelihood_from_header_preserves_single_animal_3level_shape():
    header = DLCHeaderModel(
        columns=[
            ("John", "bp1", "x"),
            ("John", "bp1", "y"),
            ("John", "bp1", "likelihood"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )

    out = _drop_likelihood_from_header(header)

    assert out.names == ["scorer", "bodyparts", "coords"]
    assert out.columns == [
        ("John", "bp1", "x"),
        ("John", "bp1", "y"),
    ]
    assert all(len(col) == 3 for col in out.columns)


def test_drop_likelihood_from_header_preserves_multi_animal_4level_shape():
    header = DLCHeaderModel(
        columns=[
            ("John", "mouse1", "bp1", "x"),
            ("John", "mouse1", "bp1", "y"),
            ("John", "mouse1", "bp1", "likelihood"),
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    out = _drop_likelihood_from_header(header)

    assert out.names == ["scorer", "individuals", "bodyparts", "coords"]
    assert out.columns == [
        ("John", "mouse1", "bp1", "x"),
        ("John", "mouse1", "bp1", "y"),
    ]
    assert all(len(col) == 4 for col in out.columns)
