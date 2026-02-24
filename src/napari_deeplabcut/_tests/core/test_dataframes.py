from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.config.models import DLCHeaderModel, PointsMetadata
from napari_deeplabcut.core.dataframes import (
    align_old_new,
    form_df_from_validated,
    guarantee_multiindex_rows,
    harmonize_keypoint_column_index,
    harmonize_keypoint_row_index,
    keypoint_conflicts,
    merge_multiple_scorers,
    summarize_keypoint_conflicts,
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

    assert kc.loc["img000.png"].any()
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

    assert kc.loc["img000.png"].any()
    assert any(("animal1" in str(c) and "bp1" in str(c)) for c in kc.columns)


def test_summarize_keypoint_conflicts_formats_human_readable_text():
    # Build a key_conflict frame like keypoint_conflicts returns: index=image, columns=keypoint
    key_conflict = pd.DataFrame(
        [[True, False]],
        index=pd.MultiIndex.from_tuples([("labeled-data", "test", "img000.png")]),
        columns=pd.MultiIndex.from_tuples([("animal1", "bp1"), ("animal1", "bp2")]),
    )
    txt = summarize_keypoint_conflicts(key_conflict, max_items=10)
    assert "existing keypoint" in txt.lower()
    assert "img000.png" in txt
    assert "bp1" in txt


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
