from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.config.models import DLCHeaderModel, PointsMetadata
from napari_deeplabcut.core.dataframes import (
    align_old_new,
    build_overwrite_conflict_report,
    complete_df_for_save,
    drop_likelihood_columns,
    form_df_from_validated,
    guarantee_multiindex_rows,
    harmonize_keypoint_column_index,
    harmonize_keypoint_row_index,
    keypoint_conflicts,
    keypoint_deletions,
    merge_multiple_scorers,
    merge_save_df,
)
from napari_deeplabcut.core.schemas import PointsWriteInputModel

# -----------------------------------------------------------------------------
# Helpers: create DLC-like column MultiIndex
# -----------------------------------------------------------------------------


def cols_3level(scorer="S", bodyparts=("bp1",), coords=("x", "y")) -> pd.MultiIndex:
    """Classic single-animal DLC layout often appears as 3-level columns. [1](https://github.com/DeepLabCut/DeepLabCut/issues/3072)"""
    return pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), list(coords)],
        names=["scorer", "bodyparts", "coords"],
    )


def cols_4level(scorer="S", individuals=("",), bodyparts=("bp1",), coords=("x", "y")) -> pd.MultiIndex:
    """Multi-animal DLC layout is 4-level columns (and some pipelines may store empty individuals for single-animal). [1](https://github.com/DeepLabCut/DeepLabCut/issues/3072)"""
    return pd.MultiIndex.from_product(
        [[scorer], list(individuals), list(bodyparts), list(coords)],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )


def make_points_ctx(
    *,
    header_cols: pd.MultiIndex,
    scorer: str = "S",
    # napari points data: [frame, y, x]
    points_data: np.ndarray,
    labels: list[str],
    ids: list[str],
    likelihood: list[float] | None = None,
    paths: list[str] | None = None,
) -> PointsWriteInputModel:
    """Build a real PointsWriteInputModel from schemas.py."""
    meta = PointsMetadata(
        header=DLCHeaderModel(columns=header_cols),
        paths=paths,
    )
    ctx = PointsWriteInputModel(
        points={"data": points_data},
        meta=meta,
        props={"label": labels, "id": ids, "likelihood": likelihood},
    )
    return ctx


# -----------------------------------------------------------------------------
# 1) guarantee_multiindex_rows
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("sep_path", ["labeled-data/test/img000.png", r"labeled-data\test\img000.png"])
def test_guarantee_multiindex_rows_splits_string_index(sep_path):
    df = pd.DataFrame([[1.0]], columns=[("S", "bp1", "x")], index=[sep_path])
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scorer", "bodyparts", "coords"])

    guarantee_multiindex_rows(df)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.to_list()[0][-1] == "img000.png"


def test_guarantee_multiindex_rows_leaves_numeric_index_unchanged():
    df = pd.DataFrame([[1.0]], columns=[("S", "bp1", "x")], index=[0])
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scorer", "bodyparts", "coords"])

    guarantee_multiindex_rows(df)
    # numeric index should remain (TypeError branch)
    assert not isinstance(df.index, pd.MultiIndex)


def test_guarantee_multiindex_rows_empty_df_is_noop():
    df = pd.DataFrame(columns=["x", "y"])
    guarantee_multiindex_rows(df)
    assert len(df.index) == 0
    assert not isinstance(df.index, pd.MultiIndex)


# -----------------------------------------------------------------------------
# 2) harmonize_keypoint_column_index
# -----------------------------------------------------------------------------


def test_harmonize_keypoint_column_index_upgrades_3_to_4_with_empty_individuals():
    df = pd.DataFrame([[1.0, 2.0]], columns=cols_3level(bodyparts=("bp1",), coords=("x", "y")), index=["img000.png"])
    out = harmonize_keypoint_column_index(df)

    assert out.columns.nlevels == 4
    assert out.columns.names == ["scorer", "individuals", "bodyparts", "coords"]
    assert set(out.columns.get_level_values("individuals")) == {""}


def test_harmonize_keypoint_column_index_keeps_4level_and_sets_names():
    df = pd.DataFrame(
        [[1.0, 2.0]],
        columns=cols_4level(individuals=("animal1",), bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    out = harmonize_keypoint_column_index(df)
    assert out.columns.nlevels == 4
    assert out.columns.names == ["scorer", "individuals", "bodyparts", "coords"]


# -----------------------------------------------------------------------------
# 3) align_old_new should be stable for 3-level vs 4-level mixes
# -----------------------------------------------------------------------------


def test_align_old_new_works_when_old_is_3level_and_new_is_4level():
    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols_3level(bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    df_new = pd.DataFrame(
        [[11.0, 22.0]],
        columns=cols_4level(individuals=("",), bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )

    old2, new2 = align_old_new(df_old, df_new)
    assert old2.columns.nlevels == 4
    assert new2.columns.nlevels == 4
    assert old2.columns.equals(new2.columns)
    assert old2.index.equals(new2.index)


# Overwrite conflict detection uses harmonized index to avoid (Multi)Index mismatches
def test_align_old_new_handles_basename_index_vs_deep_path_multiindex_regression():
    """
    Regression test for overwrite-preflight crash:

    Old on-disk GT files may use basename-like row indices such as:
        Index(["img00001.png"])

    Newly formed dataframes from runtime points metadata may use deeper
    path-like row indices that become a 3-level MultiIndex after
    guarantee_multiindex_rows(), e.g.:
        MultiIndex([("labeled-data", "test", "img00001.png")])

    align_old_new() must harmonize row indices before reindex/union,
    otherwise pandas may raise:
        AssertionError: Length of new_levels (3) must be <= self.nlevels (1)
    """
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )

    # Simulate existing on-disk GT dataframe with basename-only row key
    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols,
        index=["img00001.png"],
    )
    guarantee_multiindex_rows(df_old)  # -> MultiIndex([("img00001.png",)])

    # Simulate newly formed dataframe from runtime metadata with deep relpath
    df_new = pd.DataFrame(
        [[11.0, 22.0]],
        columns=cols,
        index=["labeled-data/test/img00001.png"],
    )
    guarantee_multiindex_rows(df_new)  # -> MultiIndex([("labeled-data","test","img00001.png")])

    old2, new2 = align_old_new(df_old, df_new)

    # The key regression assertion: no crash, and indices are now aligned
    assert isinstance(old2.index, pd.MultiIndex)
    assert isinstance(new2.index, pd.MultiIndex)
    assert old2.index.equals(new2.index)

    # After harmonization, both should collapse to basename representation
    assert old2.index.nlevels == 1
    assert new2.index.nlevels == 1
    assert old2.index.to_list() == [("img00001.png",)]
    assert new2.index.to_list() == [("img00001.png",)]

    # Columns should also be aligned
    assert old2.columns.equals(new2.columns)

    # And values should still be present on the aligned row
    row = ("img00001.png",)
    assert old2.loc[row, ("S", "", "nose", "x")] == 10.0
    assert old2.loc[row, ("S", "", "nose", "y")] == 20.0
    assert new2.loc[row, ("S", "", "nose", "x")] == 11.0
    assert new2.loc[row, ("S", "", "nose", "y")] == 22.0


def test_keypoint_conflicts_handles_basename_index_vs_deep_path_multiindex_regression():
    """
    Regression test for overwrite-conflict preflight:
    keypoint_conflicts() should handle shallow-vs-deep row indices without crashing.
    """
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )

    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols,
        index=["img00001.png"],
    )
    guarantee_multiindex_rows(df_old)

    df_new = pd.DataFrame(
        [[11.0, 20.0]],  # x differs -> conflict
        columns=cols,
        index=["labeled-data/test/img00001.png"],
    )
    guarantee_multiindex_rows(df_new)

    kc = keypoint_conflicts(df_old, df_new)

    assert isinstance(kc.index, pd.MultiIndex)
    assert kc.index.nlevels == 1
    assert kc.index.to_list() == [("img00001.png",)]
    assert kc.loc[("img00001.png",)].any()


# -----------------------------------------------------------------------------
# 4) keypoint_conflicts DLC semantics
# -----------------------------------------------------------------------------


def test_keypoint_conflicts_detects_conflict_single_animal_3level():
    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols_3level(bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    df_new = pd.DataFrame(
        [[99.0, 20.0]],  # x differs
        columns=cols_3level(bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    kc = keypoint_conflicts(df_old, df_new)

    assert kc.loc[("img000.png",)].any()
    assert any("bp1" in str(c) for c in kc.columns)


def test_keypoint_conflicts_detects_conflict_multi_animal_4level():
    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols_4level(individuals=("animal1",), bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    df_new = pd.DataFrame(
        [[99.0, 20.0]],
        columns=cols_4level(individuals=("animal1",), bodyparts=("bp1",), coords=("x", "y")),
        index=["img000.png"],
    )
    kc = keypoint_conflicts(df_old, df_new)

    assert kc.loc[("img000.png",)].any()
    assert any(("animal1" in str(c) and "bp1" in str(c)) for c in kc.columns)


# -----------------------------------------------------------------------------
# 5) harmonize_keypoint_row_index: collapse deep index to basenames when overlap is high
# -----------------------------------------------------------------------------


def test_harmonize_keypoint_row_index_collapses_to_basename_when_overlap_high():
    cols = cols_3level(bodyparts=("bp1",), coords=("x", "y"))
    df_deep = pd.DataFrame(
        [[1.0, 2.0]],
        columns=cols,
        index=pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")]),
    )
    df_shallow = pd.DataFrame(
        [[1.0, 2.0]],
        columns=cols,
        index=pd.MultiIndex.from_tuples([("img000.png",)]),
    )
    a, b = harmonize_keypoint_row_index(df_deep, df_shallow)

    assert isinstance(a.index, pd.MultiIndex) and isinstance(b.index, pd.MultiIndex)
    assert a.index.nlevels == b.index.nlevels == 1
    assert a.index.to_list()[0][0] == "img000.png"


# -----------------------------------------------------------------------------
# 6) merge_multiple_scorers
# -----------------------------------------------------------------------------


def test_merge_multiple_scorers_picks_highest_likelihood_when_present():
    """
    DLC output often includes likelihood per keypoint, per frame. [1](https://github.com/DeepLabCut/DeepLabCut/issues/3072)
    If multiple scorers exist, merge should keep the max-likelihood annotation.
    """
    cols = pd.MultiIndex.from_product(
        [["A", "B"], ["bp1"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        [
            [10, 20, 0.1, 100, 200, 0.9],  # frame0 -> pick B
            [11, 21, 0.8, 101, 201, 0.2],  # frame1 -> pick A
        ],
        columns=cols,
        index=[0, 1],
    )

    out = merge_multiple_scorers(df)
    assert out.shape == (2, 3)
    assert np.allclose(out.to_numpy()[0], [100, 200, 0.9], equal_nan=True)
    assert np.allclose(out.to_numpy()[1], [11, 21, 0.8], equal_nan=True)


def test_merge_multiple_scorers_all_nan_likelihood_does_not_crash():
    cols = pd.MultiIndex.from_product(
        [["A", "B"], ["bp1"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame([[np.nan] * 6], columns=cols, index=[0])

    out = merge_multiple_scorers(df)
    assert out.shape == (1, 3)
    assert np.isnan(out.to_numpy()).all()


def test_merge_multiple_scorers_without_likelihood_picks_first_scorer():
    cols = pd.MultiIndex.from_product(
        [["A", "B"], ["bp1"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame([[10, 20, 100, 200]], columns=cols, index=[0])
    out = merge_multiple_scorers(df)

    assert set(out.columns.get_level_values("scorer")) == {"A"}
    assert out.to_numpy().tolist() == [[10, 20]]


# -----------------------------------------------------------------------------
# 7) form_df_from_validated (writer-facing) using real schemas
# -----------------------------------------------------------------------------


def test_form_df_from_validated_writes_xy_in_dlc_order_and_maps_paths_index():
    """
    PointsDataModel converts napari [frame,y,x] -> DLC [x,y].
    When meta.paths is present, df index should be replaced with those path keys.
    """
    header = cols_4level(scorer="S", individuals=("",), bodyparts=("bp1",), coords=("x", "y"))
    ctx = make_points_ctx(
        header_cols=header,
        points_data=np.array([[0.0, 44.0, 55.0]]),  # frame=0, y=44, x=55
        labels=["bp1"],
        ids=[""],
        likelihood=None,
        paths=["labeled-data/test/img000.png"],
    )

    df = form_df_from_validated(ctx)

    # Index should become path string (then guarantee_multiindex_rows may split it)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.to_list()[0][-1] == "img000.png"

    # Ensure x,y written correctly
    # Column selection: (scorer, individuals, bodyparts, coords)
    assert df.loc[df.index[0], ("S", "", "bp1", "x")] == 55.0
    assert df.loc[df.index[0], ("S", "", "bp1", "y")] == 44.0


def test_form_df_from_validated_accepts_3level_header_and_preserves_values():
    """
    DLC still uses both 3-level (single-animal) and 4-level (multi-animal) headers in the wild. [1](https://github.com/DeepLabCut/DeepLabCut/issues/3072)
    Our writer must accept a 3-level header and must not lose semantic meaning.
    The current implementation upgrades to 4-level with individuals="".
    """
    header3 = cols_3level(scorer="S", bodyparts=("bp1",), coords=("x", "y"))
    ctx = make_points_ctx(
        header_cols=header3,
        points_data=np.array([[0.0, 44.0, 55.0]]),
        labels=["bp1"],
        ids=[""],  # single-animal sentinel
        likelihood=None,
        paths=["labeled-data/test/img000.png"],
    )

    df = form_df_from_validated(ctx)
    assert isinstance(df.columns, pd.MultiIndex)

    if df.columns.nlevels == 3:
        assert df.columns.names == ["scorer", "bodyparts", "coords"]
        assert df.loc[df.index[0], ("S", "bp1", "x")] == 55.0
        assert df.loc[df.index[0], ("S", "bp1", "y")] == 44.0
    else:
        assert df.columns.nlevels == 4
        assert df.columns.names == ["scorer", "individuals", "bodyparts", "coords"]
        assert set(df.columns.get_level_values("individuals")) == {""}
        assert df.loc[df.index[0], ("S", "", "bp1", "x")] == 55.0
        assert df.loc[df.index[0], ("S", "", "bp1", "y")] == 44.0


def test_form_df_from_validated_raises_when_header_reindex_drops_all_finite_points():
    """
    The function has an invariant: if layer has finite points, df must retain finite coords.
    We trigger mismatch by giving a header that does NOT include the labeled bodypart.
    """
    header = cols_4level(scorer="S", individuals=("",), bodyparts=("DIFFERENT_BP",), coords=("x", "y"))
    ctx = make_points_ctx(
        header_cols=header,
        points_data=np.array([[0.0, 44.0, 55.0]]),
        labels=["bp1"],  # not present in header -> reindex wipes coords
        ids=[""],
        likelihood=None,
        paths=["labeled-data/test/img000.png"],
    )

    with pytest.raises(RuntimeError, match="Writer produced no finite coordinates"):
        _ = form_df_from_validated(ctx)


# -----------------------------------------------------------------------------
# 8) complete_df_for_save: expand sparse napari data to full editable save scope
# -----------------------------------------------------------------------------


def test_complete_df_for_save_expands_all_paths_and_bodyparts_with_nan_for_missing():
    header = DLCHeaderModel(
        columns=cols_4level(
            scorer="S",
            individuals=("animal1",),
            bodyparts=("nose", "tail"),
            coords=("x", "y"),
        )
    )
    pts_meta = PointsMetadata(
        header=header,
        paths=[
            "labeled-data/test/img000.png",
            "labeled-data/test/img001.png",
        ],
    )

    # Sparse df contains only img000 / nose.
    sparse = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols_4level(
            scorer="S",
            individuals=("animal1",),
            bodyparts=("nose",),
            coords=("x", "y"),
        ),
        index=["labeled-data/test/img000.png"],
    )

    out = complete_df_for_save(
        sparse,
        pts_meta=pts_meta,
        header=header,
    )

    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.to_list() == [
        ("labeled-data", "test", "img000.png"),
        ("labeled-data", "test", "img001.png"),
    ]

    assert out.columns.equals(
        cols_4level(
            scorer="S",
            individuals=("animal1",),
            bodyparts=("nose", "tail"),
            coords=("x", "y"),
        )
    )

    row0 = ("labeled-data", "test", "img000.png")
    row1 = ("labeled-data", "test", "img001.png")

    assert out.loc[row0, ("S", "animal1", "nose", "x")] == 10.0
    assert out.loc[row0, ("S", "animal1", "nose", "y")] == 20.0

    # Missing bodypart in loaded frame becomes explicit NaN.
    assert pd.isna(out.loc[row0, ("S", "animal1", "tail", "x")])
    assert pd.isna(out.loc[row0, ("S", "animal1", "tail", "y")])

    # Entire second frame is in save scope but has no points.
    assert pd.isna(out.loc[row1, ("S", "animal1", "nose", "x")])
    assert pd.isna(out.loc[row1, ("S", "animal1", "tail", "y")])


def test_complete_df_for_save_drops_likelihood_columns():
    header = DLCHeaderModel(
        columns=cols_4level(
            scorer="S",
            individuals=("animal1",),
            bodyparts=("nose",),
            coords=("x", "y", "likelihood"),
        )
    )
    pts_meta = PointsMetadata(
        header=header,
        paths=["labeled-data/test/img000.png"],
    )

    sparse = pd.DataFrame(
        [[10.0, 20.0, 0.7]],
        columns=cols_4level(
            scorer="S",
            individuals=("animal1",),
            bodyparts=("nose",),
            coords=("x", "y", "likelihood"),
        ),
        index=["labeled-data/test/img000.png"],
    )

    out = complete_df_for_save(
        sparse,
        pts_meta=pts_meta,
        header=header,
    )

    assert "likelihood" not in set(out.columns.get_level_values("coords").astype(str))
    assert out.columns.to_list() == [
        ("S", "animal1", "nose", "x"),
        ("S", "animal1", "nose", "y"),
    ]


def test_complete_df_for_save_without_paths_preserves_existing_rows_and_completes_columns():
    header = DLCHeaderModel(
        columns=cols_4level(
            scorer="S",
            individuals=("",),
            bodyparts=("nose", "tail"),
            coords=("x", "y"),
        )
    )
    pts_meta = PointsMetadata(header=header, paths=None)

    sparse = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols_4level(
            scorer="S",
            individuals=("",),
            bodyparts=("nose",),
            coords=("x", "y"),
        ),
        index=["img000.png"],
    )

    out = complete_df_for_save(
        sparse,
        pts_meta=pts_meta,
        header=header,
    )

    assert out.index.to_list() == [("img000.png",)]
    assert ("S", "", "tail", "x") in out.columns
    assert pd.isna(out.loc[("img000.png",), ("S", "", "tail", "x")])


# -----------------------------------------------------------------------------
# 9) merge_save_df: save-scope overlay preserves intentional NaN deletions
# -----------------------------------------------------------------------------


def test_merge_save_df_nan_in_new_clears_old_value():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )
    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame([[10.0, 20.0]], index=idx, columns=cols)
    df_new = pd.DataFrame([[np.nan, np.nan]], index=idx, columns=cols)

    out = merge_save_df(df_old, df_new)

    assert pd.isna(out.loc[idx[0], ("S", "", "nose", "x")])
    assert pd.isna(out.loc[idx[0], ("S", "", "nose", "y")])


def test_merge_save_df_preserves_rows_outside_new_save_scope():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )
    old_idx = pd.MultiIndex.from_tuples(
        [
            ("labeled-data", "test", "img000.png"),
            ("labeled-data", "test", "img999.png"),
        ]
    )
    new_idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        index=old_idx,
        columns=cols,
    )
    df_new = pd.DataFrame(
        [[np.nan, np.nan]],
        index=new_idx,
        columns=cols,
    )

    out = merge_save_df(df_old, df_new)

    assert pd.isna(out.loc[("labeled-data", "test", "img000.png"), ("S", "", "nose", "x")])
    assert out.loc[("labeled-data", "test", "img999.png"), ("S", "", "nose", "x")] == 30.0
    assert out.loc[("labeled-data", "test", "img999.png"), ("S", "", "nose", "y")] == 40.0


def test_merge_save_df_preserves_old_columns_outside_new_columns():
    old_cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose", "tail"),
        coords=("x", "y"),
    )
    new_cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )
    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]], index=idx, columns=old_cols)
    df_new = pd.DataFrame([[11.0, 22.0]], index=idx, columns=new_cols)

    out = merge_save_df(df_old, df_new)

    assert out.loc[idx[0], ("S", "", "nose", "x")] == 11.0
    assert out.loc[idx[0], ("S", "", "nose", "y")] == 22.0

    # Old tail columns were outside df_new's column scope, so they remain.
    assert out.loc[idx[0], ("S", "", "tail", "x")] == 30.0
    assert out.loc[idx[0], ("S", "", "tail", "y")] == 40.0


def test_merge_save_df_nan_clears_old_value_after_row_harmonization():
    """
    Deletion semantics depend on row labels matching after harmonization.

    Existing DLC files may use basename-only row keys while newly formed save
    dataframes may use deeper DLC path keys. merge_save_df() should harmonize
    those representations before assigning df_new into df_old, otherwise NaNs
    in df_new would not clear the intended old values.
    """
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )

    # Existing on-disk row uses basename representation.
    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        index=["img000.png"],
        columns=cols,
    )
    guarantee_multiindex_rows(df_old)

    # New save-scope row uses deeper DLC path representation.
    df_new = pd.DataFrame(
        [[np.nan, np.nan]],
        index=["labeled-data/test/img000.png"],
        columns=cols,
    )
    guarantee_multiindex_rows(df_new)

    out = merge_save_df(df_old, df_new)

    # harmonize_keypoint_row_index() collapses the deep row to basename so the
    # assignment hits the existing row and NaN clears the old value.
    row = ("img000.png",)
    assert row in out.index
    assert pd.isna(out.loc[row, ("S", "", "nose", "x")])
    assert pd.isna(out.loc[row, ("S", "", "nose", "y")])


def test_merge_save_df_refuses_no_row_overlap_after_harmonization():
    """
    If row labels do not overlap after harmonization, df_new cannot overwrite or
    delete existing values. merge_save_df() refuses this case rather than silently
    preserving old labels when deletion/overwrite semantics were expected.
    """
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )

    old_idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])
    new_idx = pd.MultiIndex.from_tuples([("labeled-data", "other", "img999.png")])

    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        index=old_idx,
        columns=cols,
    )
    df_new = pd.DataFrame(
        [[np.nan, np.nan]],
        index=new_idx,
        columns=cols,
    )

    with pytest.raises(ValueError, match="no row-index overlap after harmonization"):
        merge_save_df(df_old, df_new)


# -----------------------------------------------------------------------------
# 10) keypoint_deletions and keypoint_conflicts(include_deletions=True)
# -----------------------------------------------------------------------------


def test_keypoint_deletions_detects_old_value_to_new_nan():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )
    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame([[10.0, 20.0]], index=idx, columns=cols)
    df_new = pd.DataFrame([[np.nan, np.nan]], index=idx, columns=cols)

    deletions = keypoint_deletions(df_old, df_new)

    assert deletions.loc[idx[0], ("", "nose")]


def test_keypoint_deletions_does_not_flag_rows_outside_new_scope():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )
    old_idx = pd.MultiIndex.from_tuples(
        [
            ("labeled-data", "test", "img000.png"),
            ("labeled-data", "test", "img999.png"),
        ]
    )
    new_idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        index=old_idx,
        columns=cols,
    )
    df_new = pd.DataFrame(
        [[np.nan, np.nan]],
        index=new_idx,
        columns=cols,
    )

    deletions = keypoint_deletions(df_old, df_new)

    assert ("labeled-data", "test", "img000.png") in deletions.index
    assert ("labeled-data", "test", "img999.png") not in deletions.index
    assert deletions.loc[("labeled-data", "test", "img000.png"), ("", "nose")]


def test_keypoint_conflicts_include_deletions_combines_overwrites_and_deletions():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose", "tail"),
        coords=("x", "y"),
    )
    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]],
        index=idx,
        columns=cols,
    )
    df_new = pd.DataFrame(
        [[11.0, 20.0, np.nan, np.nan]],
        index=idx,
        columns=cols,
    )

    overwrites_only = keypoint_conflicts(df_old, df_new)
    combined = keypoint_conflicts(df_old, df_new, include_deletions=True)

    assert overwrites_only.loc[idx[0], ("", "nose")]
    assert not overwrites_only.loc[idx[0], ("", "tail")]

    assert combined.loc[idx[0], ("", "nose")]
    assert combined.loc[idx[0], ("", "tail")]


def test_keypoint_deletions_handles_basename_index_vs_deep_path_multiindex_regression():
    """
    Regression test for deletion detection with shallow old index and deep new save scope.

    Old on-disk dataframe may use basename rows:
        ("img00001.png",)

    New save-scope dataframe may use DLC relative paths:
        ("labeled-data", "test", "img00001.png")

    keypoint_deletions() must harmonize before restricting to the new scope.
    """
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y"),
    )

    df_old = pd.DataFrame(
        [[10.0, 20.0]],
        columns=cols,
        index=["img00001.png"],
    )
    guarantee_multiindex_rows(df_old)

    df_new = pd.DataFrame(
        [[np.nan, np.nan]],
        columns=cols,
        index=["labeled-data/test/img00001.png"],
    )
    guarantee_multiindex_rows(df_new)

    deletions = keypoint_deletions(df_old, df_new)

    assert isinstance(deletions.index, pd.MultiIndex)
    assert deletions.index.nlevels == 1
    assert deletions.index.to_list() == [("img00001.png",)]
    assert deletions.loc[("img00001.png",), ("", "nose")]


# -----------------------------------------------------------------------------
# 11) build_overwrite_conflict_report: separate modified vs deleted entries
# -----------------------------------------------------------------------------


def test_build_overwrite_conflict_report_counts_deletions_separately():
    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])
    cols = pd.MultiIndex.from_tuples(
        [
            ("", "nose"),
            ("", "tail"),
        ],
        names=["individuals", "bodyparts"],
    )

    key_conflict = pd.DataFrame(
        [[True, False]],
        index=idx,
        columns=cols,
    )
    deletion_conflict = pd.DataFrame(
        [[False, True]],
        index=idx,
        columns=cols,
    )

    report = build_overwrite_conflict_report(
        key_conflict,
        deletion_conflict=deletion_conflict,
        layer_name="CollectedData_S",
        destination_path="/tmp/CollectedData_S.h5",
    )

    assert report.has_conflicts
    assert report.n_overwrites == 1
    assert report.n_deletions == 1
    assert report.n_frames == 1
    assert len(report.entries) == 1

    entry = report.entries[0]
    assert entry.frame_label == "labeled-data/test/img000.png"
    assert entry.keypoints == ("nose",)
    assert entry.deleted_keypoints == ("tail",)


def test_build_overwrite_conflict_report_prefers_deleted_when_same_key_is_both_modified_and_deleted():
    idx = pd.MultiIndex.from_tuples([("img000.png",)])
    cols = pd.MultiIndex.from_tuples(
        [("", "nose")],
        names=["individuals", "bodyparts"],
    )

    key_conflict = pd.DataFrame([[True]], index=idx, columns=cols)
    deletion_conflict = pd.DataFrame([[True]], index=idx, columns=cols)

    report = build_overwrite_conflict_report(
        key_conflict,
        deletion_conflict=deletion_conflict,
    )

    assert report.n_overwrites == 0
    assert report.n_deletions == 1
    assert report.entries[0].keypoints == ()
    assert report.entries[0].deleted_keypoints == ("nose",)


# -----------------------------------------------------------------------------
# 12) Regression: deleting a point clears it after completing + merging save df
# -----------------------------------------------------------------------------


def test_deleted_keypoint_roundtrip_complete_then_merge_clears_old_value():
    header_cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose", "tail"),
        coords=("x", "y"),
    )
    header = DLCHeaderModel(columns=header_cols)

    pts_meta = PointsMetadata(
        header=header,
        paths=["labeled-data/test/img000.png"],
    )

    idx = pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")])

    df_old = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]],
        index=idx,
        columns=header_cols,
    )

    # Current layer contains only tail. Nose was deleted from napari Points.
    sparse_current = pd.DataFrame(
        [[31.0, 41.0]],
        index=idx,
        columns=cols_4level(
            scorer="S",
            individuals=("",),
            bodyparts=("tail",),
            coords=("x", "y"),
        ),
    )

    df_new = complete_df_for_save(
        sparse_current,
        pts_meta=pts_meta,
        header=header,
    )

    out = merge_save_df(df_old, df_new)

    # Deleted nose should be cleared.
    assert pd.isna(out.loc[idx[0], ("S", "", "nose", "x")])
    assert pd.isna(out.loc[idx[0], ("S", "", "nose", "y")])

    # Existing/current tail should be saved.
    assert out.loc[idx[0], ("S", "", "tail", "x")] == 31.0
    assert out.loc[idx[0], ("S", "", "tail", "y")] == 41.0


# -----------------------------------------------------------------------------
# 13) drop_likelihood_columns
# -----------------------------------------------------------------------------


def test_drop_likelihood_columns_removes_multiindex_likelihood_coord():
    cols = cols_4level(
        scorer="S",
        individuals=("",),
        bodyparts=("nose",),
        coords=("x", "y", "likelihood"),
    )
    df = pd.DataFrame([[1.0, 2.0, 0.9]], columns=cols)

    out = drop_likelihood_columns(df)

    assert out.columns.to_list() == [
        ("S", "", "nose", "x"),
        ("S", "", "nose", "y"),
    ]


def test_drop_likelihood_columns_removes_flat_likelihood_column():
    df = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "likelihood": [0.9],
        }
    )

    out = drop_likelihood_columns(df)

    assert list(out.columns) == ["x", "y"]
