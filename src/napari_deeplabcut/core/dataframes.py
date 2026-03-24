# src/napari_deeplabcut/core/dataframes.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from napari_deeplabcut.config.models import ConflictEntry, OverwriteConflictReport
from napari_deeplabcut.core.schemas import PointsWriteInputModel

logger = logging.getLogger(__name__)


def set_df_scorer(df: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Return df with scorer level set to the given scorer (if present)."""
    scorer = (scorer or "").strip()
    if not scorer:
        return df
    if not hasattr(df.columns, "names") or "scorer" not in df.columns.names:
        return df

    try:
        cols = df.columns.to_frame(index=False)
        cols["scorer"] = scorer
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(cols)
    except Exception:
        pass
    return df


def merge_multiple_scorers(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has multiple scorers in its column MultiIndex, merge them.

    - If likelihood exists, keep the scorer with max likelihood per keypoint/frame.
    - Else, pick the first scorer deterministically.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    n_frames = df.shape[0]
    cols = df.columns
    names = list(cols.names or [])

    # Identify scorer level
    scorer_level = "scorer" if "scorer" in names else 0
    scorers = list(dict.fromkeys(cols.get_level_values(scorer_level).astype(str).tolist()))
    n_scorers = len(scorers)
    if n_scorers <= 1:
        return df

    # Identify coords level
    coords_level = "coords" if "coords" in names else (cols.nlevels - 1)
    coords_vals = cols.get_level_values(coords_level).astype(str).tolist()
    has_likelihood = "likelihood" in set(coords_vals)

    # Helper: take columns for a given scorer
    def _cols_for_scorer(scorer: str) -> pd.MultiIndex:
        mask = cols.get_level_values(scorer_level).astype(str) == str(scorer)
        return cols[mask]

    if has_likelihood:
        # Ensure each scorer block has same column ordering/shape
        cols0 = _cols_for_scorer(scorers[0])
        per_scorer = len(cols0)
        if per_scorer == 0:
            return df

        # If other scorers don't match shape, fall back to first scorer
        for s in scorers[1:]:
            if len(_cols_for_scorer(s)) != per_scorer:
                logger.debug("Scorer column blocks differ in size; falling back to first scorer.")
                return df.loc[:, cols0]

        # Stack scorer axis -> (n_frames, n_scorers, per_scorer)
        data = df.to_numpy(copy=True).reshape((n_frames, n_scorers, per_scorer))

        # We need likelihood position within per_scorer block.
        # Find likelihood columns within the first scorer block:
        coords0 = cols0.get_level_values(coords_level).astype(str).to_numpy()
        like_mask = coords0 == "likelihood"
        if not np.any(like_mask):
            # coords said likelihood exists, but not in the first scorer block - fallback
            return df.loc[:, cols0]

        # Reshape per keypoint: assume (x, y, likelihood) triplets per keypoint.
        # We infer n_keypoints from number of likelihood entries.
        n_keypoints = int(np.sum(like_mask))
        # Triplet width is per_scorer / n_keypoints if structured, but be defensive:
        triplet = per_scorer // max(1, n_keypoints)
        if triplet < 3:
            # Not in expected (x,y,likelihood) shape; fallback
            return df.loc[:, cols0]

        data3 = data.reshape((n_frames, n_scorers, n_keypoints, triplet))

        # likelihood is assumed at index 2 in each triplet (legacy DLC layout)
        try:
            idx = np.nanargmax(data3[..., 2], axis=1)
        except ValueError:  # All-NaN slice encountered
            mask = np.isnan(data3[..., 2]).all(axis=1, keepdims=True)
            mask = np.broadcast_to(mask[..., None], data3.shape)
            data3[mask] = -1
            idx = np.nanargmax(data3[..., 2], axis=1)
            data3[mask] = np.nan

        data_best = data3[np.arange(n_frames)[:, None], idx, np.arange(n_keypoints)]
        data_best = data_best.reshape((n_frames, -1))

        # Output columns: use first scorer block columns (structure preserved)
        out_cols = cols0[: data_best.shape[1]]
        return pd.DataFrame(data_best, index=df.index, columns=out_cols)

    # No likelihood: pick first scorer deterministically
    cols0 = _cols_for_scorer(scorers[0])
    return df.loc[:, cols0]


def guarantee_multiindex_rows(df: pd.DataFrame) -> None:
    """Ensure that DataFrame rows are a MultiIndex of path components.
    Legacy DLC data may use an index with pathto/video/file.png strings as Index.
    The new format uses a MultiIndex with each path component as a level.
    """
    # Make paths platform-agnostic if they are not already
    if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
        path = df.index[0]
        try:
            sep = "/" if "/" in path else "\\"
            splits = tuple(df.index.str.split(sep))
            df.index = pd.MultiIndex.from_tuples(splits)
        except TypeError:  # Ignore numerical index of frame indices
            pass


def form_df_from_validated(ctx: PointsWriteInputModel) -> pd.DataFrame:
    """Create a DLC-style DataFrame from validated napari points + metadata."""
    header = ctx.meta.header  # DLCHeaderModel (validated)
    props = ctx.props

    # DLC expects x,y columns; ctx.points.xy_dlc converts napari [y,x] -> [x,y]
    temp_df = pd.DataFrame(ctx.points.xy_dlc, columns=["x", "y"])
    temp_df["bodyparts"] = props.label
    temp_df["individuals"] = props.id
    temp_df["inds"] = ctx.points.frame_inds
    temp_df["likelihood"] = props.likelihood if props.likelihood is not None else 1.0

    temp_df["scorer"] = header.scorer or "unknown"

    # Mark rows that have actual coords
    temp_df["_has_xy"] = temp_df[["x", "y"]].notna().all(axis=1)

    # Sort so that rows WITH coords come last (so keep="last" keeps them)
    temp_df = temp_df.sort_values("_has_xy")

    # Drop duplicates on the key that defines a unique keypoint observation
    temp_df = temp_df.drop_duplicates(
        subset=["scorer", "individuals", "bodyparts", "inds"],
        keep="last",
    )
    temp_df = temp_df.drop(columns="_has_xy")

    df = temp_df.set_index(["scorer", "individuals", "bodyparts", "inds"]).stack()
    df.index.set_names("coords", level=-1, inplace=True)
    df = df.unstack(["scorer", "individuals", "bodyparts", "coords"])
    df.index.name = None

    hdr_cols = ctx.meta.header.as_multiindex()  # pandas-only helper; raises if pandas missing

    logger.debug("Before reindex: cols nlevels %s, names %s", df.columns.nlevels, df.columns.names)
    logger.debug("header cols nlevels %s, names %s", hdr_cols.nlevels, hdr_cols.names)

    # If df columns dropped individuals, drop it from header too (if present)
    # if df.columns.nlevels == 3 and isinstance(hdr_cols, pd.MultiIndex) and hdr_cols.nlevels == 4:
    #     if "individuals" in hdr_cols.names:
    #         hdr_cols = hdr_cols.droplevel("individuals")

    # If df columns kept individuals but header doesn't have it, add it (single-animal)
    if df.columns.nlevels == 4 and isinstance(hdr_cols, pd.MultiIndex) and hdr_cols.nlevels == 3:
        # Insert empty individuals level into header tuples
        frame = hdr_cols.to_frame(index=False)
        frame.insert(1, "individuals", "")
        hdr_cols = pd.MultiIndex.from_frame(frame, names=["scorer", "individuals", "bodyparts", "coords"])

    df = df.reindex(hdr_cols, axis=1)

    logger.debug("After reindex: cols nlevels %s, names %s", df.columns.nlevels, df.columns.names)
    logger.debug(
        "header cols nlevels %s, names %s",
        ctx.meta.header.as_multiindex().nlevels,
        ctx.meta.header.as_multiindex().names,
    )

    # Replace integer frame index with path keys if available
    if ctx.meta.paths:
        df.index = [ctx.meta.paths[i] for i in df.index]

    guarantee_multiindex_rows(df)

    # Writer invariant: if there are finite points in the layer, df must contain finite coords
    layer_xy = np.asarray(ctx.points.xy_dlc)  # (N, 2) in [x,y]
    n_layer = np.isfinite(layer_xy).all(axis=1).sum()

    # Count finite values in df (x/y columns only)
    n_df = np.isfinite(df.to_numpy()).sum()

    if n_layer > 0 and n_df == 0:
        raise RuntimeError(
            "Writer produced no finite coordinates although layer contains finite points. "
            "Likely a header/column MultiIndex mismatch during reindex."
        )

    return df


def harmonize_keypoint_row_index(df_new: pd.DataFrame, df_old: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize row index representation between a freshly formed points dataframe (df_new)
    and an existing on-disk dataframe (df_old) to make combine_first/align stable.

    Strategy:
    - Ensure both indices are MultiIndex via misc.guarantee_multiindex_rows
    - If nlevels differ and one is 1-level while the other is >1, attempt to collapse
      the deeper one to basename if it matches the 1-level index sufficiently.

    """
    # FUTURE NOTE @C-Achard 2026-02-18 hardcoded DLC structure:
    # DLC's CollectedData files are commonly keyed by per-folder image names (basenames),
    # even when the runtime layer may store relpaths including subfolders.
    df_new2 = df_new.copy()
    df_old2 = df_old.copy()

    # Make both MultiIndex (project convention)
    guarantee_multiindex_rows(df_new2)
    guarantee_multiindex_rows(df_old2)

    inew = df_new2.index
    iold = df_old2.index

    if not isinstance(inew, pd.MultiIndex) or not isinstance(iold, pd.MultiIndex):
        return df_new2, df_old2

    if inew.nlevels == iold.nlevels:
        return df_new2, df_old2

    # Identify which is "deep" and which is "shallow"
    if inew.nlevels > iold.nlevels:
        deep_df, shallow_df = df_new2, df_old2
    else:
        deep_df, shallow_df = df_old2, df_new2

    deep_idx = deep_df.index
    shallow_idx = shallow_df.index

    # Only try collapse when shallow is 1-level and deep is >1
    if not isinstance(shallow_idx, pd.MultiIndex) or shallow_idx.nlevels != 1:
        return df_new2, df_old2
    if not isinstance(deep_idx, pd.MultiIndex) or deep_idx.nlevels <= 1:
        return df_new2, df_old2

    # Collapse deep MultiIndex to last component (basename)
    deep_last = deep_idx.to_frame(index=False).iloc[:, -1].astype(str).tolist()
    shallow_vals = shallow_idx.to_frame(index=False).iloc[:, 0].astype(str).tolist()

    # Measure overlap (set-based)
    overlap = len(set(deep_last) & set(shallow_vals))
    denom = max(1, len(set(shallow_vals)))
    ratio = overlap / denom

    # If most shallow keys exist as basenames in deep, collapse deep to shallow representation
    if ratio >= 0.8:
        deep_df2 = deep_df.copy()
        deep_df2.index = pd.MultiIndex.from_arrays([deep_last])

        # Return in original order
        if deep_df is df_new2:
            return deep_df2, shallow_df
        else:
            return shallow_df, deep_df2

    return df_new2, df_old2


def harmonize_keypoint_column_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DLC keypoints columns are a 4-level MultiIndex with individuals inserted if missing."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    cols = df.columns

    # Already 4 levels: try to ensure correct names
    if cols.nlevels == 4:
        # set_names is safe even if already correct
        df2 = df.copy()
        df2.columns = cols.set_names(["scorer", "individuals", "bodyparts", "coords"])
        return df2

    # Legacy 3-level: (scorer, bodyparts, coords) -> insert individuals=""
    if cols.nlevels == 3:
        # We only insert individuals if it looks like the DLC pattern
        # (names might be missing/None depending on earlier ops)
        list(cols.names)
        # accept either correct names or unknown names
        # but we assume order is scorer/bodyparts/coords
        frame = cols.to_frame(index=False)

        # If names are already scorer/bodyparts/coords, this is perfect.
        # If not, we still insert individuals at position 1.
        frame.insert(1, "individuals", "")

        df2 = df.copy()
        df2.columns = pd.MultiIndex.from_frame(frame, names=["scorer", "individuals", "bodyparts", "coords"])
        return df2

    # Other nlevels not expected: leave unchanged
    return df


def align_old_new(df_old: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both dataframes to union of index and columns."""
    # First harmonize row index structure (deep path MI vs shallow basename index)
    df_new, df_old = harmonize_keypoint_row_index(df_new, df_old)

    df_old = harmonize_keypoint_column_index(df_old)
    df_new = harmonize_keypoint_column_index(df_new)

    idx = df_old.index.union(df_new.index)
    cols = df_old.columns.union(df_new.columns)
    return (
        df_old.reindex(index=idx, columns=cols),
        df_new.reindex(index=idx, columns=cols),
    )


def keypoint_conflicts(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean DataFrame indexed by image, with columns as keypoints,
    True when any coord (x/y[/likelihood]) would overwrite an existing value.

    Columns in output are MultiIndex levels subset: (individuals?, bodyparts)
    or just (bodyparts) for single animal.
    """
    old, new = align_old_new(df_old, df_new)

    old_has = old.notna()
    new_has = new.notna()

    # cell-level conflicts: both have values and differ
    cell_conflict = (old != new) & old_has & new_has

    # Identify which levels exist
    col_names = list(old.columns.names)
    has_inds = "individuals" in col_names
    has_body = "bodyparts" in col_names
    has_coords = "coords" in col_names

    if not (has_body and has_coords):
        # Unexpected format; fall back to cell-level summary
        return cell_conflict.any(axis=1).to_frame(name="conflict")

    # Drop scorer level if present (not meaningful for end-user warning)
    # We want to aggregate per (individual, bodypart) across coords.
    # Build the grouping levels that define a "keypoint".
    key_levels = []
    if has_inds:
        key_levels.append("individuals")
    key_levels.append("bodyparts")

    # Reduce across coords first -> any conflict for that coord-set
    # This yields a DataFrame with columns still multi-level including scorer and coords.
    # We then group by key_levels.
    # Step 1: ensure we can group: drop coords by grouping over it using "any".
    # We'll group over all columns that share the same (individual/bodypart), ignoring coords.
    # To do that cleanly: swap coords to last, then groupby on key_levels.
    conflict_cols = cell_conflict.copy()

    # Group columns by key_levels and reduce with any() across remaining levels (coords + scorer)
    # pandas allows groupby on axis=1 by level names:
    key_conflict = conflict_cols.T.groupby(level=key_levels).any().T

    return key_conflict


def _format_image_id(img_id) -> str:
    """Format a row index value into a user-friendly frame/image identifier."""
    if isinstance(img_id, tuple):
        # e.g. ('labeled-data', 'test', 'img000.png')
        return "/".join(map(str, img_id))
    return str(img_id)


def _format_keypoint_id(kp) -> str:
    """Format a keypoint column label into a user-friendly identifier."""
    # kp can be:
    # - scalar bodypart: "nose"
    # - tuple(individual, bodypart): ("animal1", "nose")
    # - larger tuple if upstream grouping shape changes
    if isinstance(kp, tuple):
        if len(kp) == 2:
            ind, bp = kp
            return f"{bp} (id: {ind})" if ind else str(bp)
        return " / ".join(str(x) for x in kp if x not in (None, ""))
    return str(kp)


def build_overwrite_conflict_report(
    key_conflict: pd.DataFrame,
    *,
    max_entries: int = 50,
    layer_name: str | None = None,
    destination_path: str | None = None,
) -> OverwriteConflictReport:
    """
    Convert a pandas key-conflict table into a UI-facing overwrite report.

    Parameters
    ----------
    key_conflict:
        Boolean-like DataFrame indexed by frame/image identifier, with columns
        representing keypoints. Truthy cells indicate a keypoint overwrite conflict.

    Returns
    -------
    OverwriteConflictReport
        Plain-Python UI contract describing overwrite counts and detailed entries.

    Notes
    -----
    This function is the pandas boundary for overwrite reporting. The UI should
    depend only on OverwriteConflictReport, not on DataFrame structure.
    """
    n_overwrites = int(key_conflict.to_numpy().sum())
    n_frames = int(key_conflict.any(axis=1).to_numpy().sum())

    entries: list[ConflictEntry] = []

    for img, row in key_conflict.iterrows():
        conflicted: list[str] = []
        for kp, flag in row.items():
            if bool(flag):
                conflicted.append(_format_keypoint_id(kp))

        if conflicted:
            entries.append(
                ConflictEntry(
                    frame_label=_format_image_id(img),
                    keypoints=tuple(conflicted),
                )
            )

    shown = tuple(entries[:max_entries])
    truncated = max(0, len(entries) - len(shown))

    return OverwriteConflictReport(
        n_overwrites=n_overwrites,
        n_frames=n_frames,
        entries=shown,
        truncated_entries=truncated,
        layer_name=layer_name,
        destination_path=destination_path,
    )
