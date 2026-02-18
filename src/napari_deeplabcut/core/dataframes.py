# src/napari_deeplabcut/core/dataframes.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from napari_deeplabcut import misc
from napari_deeplabcut.core.schemas import PointsWriteInputModel

logger = logging.getLogger(__name__)


def merge_multiple_scorers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    n_frames = df.shape[0]
    header = misc.DLCHeader(df.columns)
    n_scorers = len(header._get_unique("scorer"))
    if n_scorers == 1:
        return df

    if "likelihood" in header.coords:
        # Merge annotations from multiple scorers to keep
        # detections with highest confidence
        data = df.to_numpy().reshape((n_frames, n_scorers, -1, 3))
        try:
            idx = np.nanargmax(data[..., 2], axis=1)
        except ValueError:  # All-NaN slice encountered
            mask = np.isnan(data[..., 2]).all(axis=1, keepdims=True)
            mask = np.broadcast_to(mask[..., None], data.shape)
            data[mask] = -1
            idx = np.nanargmax(data[..., 2], axis=1)
            data[mask] = np.nan
        data_best = data[np.arange(n_frames)[:, None], idx, np.arange(data.shape[2])].reshape((n_frames, -1))
        df = pd.DataFrame(
            data_best,
            index=df.index,
            columns=header.columns[: data_best.shape[1]],
        )
    else:  # Arbitrarily pick data from the first scorer
        df = df.loc(axis=1)[: header.scorer]
    return df


def guarantee_multiindex_rows(df):
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

    logger.debug("Before reindex: cols nlevels %s, names %s", df.columns.nlevels, df.columns.names)
    logger.debug(
        "header cols nlevels %s, names %s",
        ctx.meta.header.as_multiindex().nlevels,
        ctx.meta.header.as_multiindex().names,
    )

    # Drop individuals if this is a single-animal layout (empty ids)
    # Here we check if all ids are '' (or falsy)
    hdr_cols = ctx.meta.header.as_multiindex()

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
    """Format DLC index row into a user-friendly image identifier."""
    if isinstance(img_id, tuple):
        # e.g. ('labeled-data', 'test', 'img000.png')
        return "/".join(map(str, img_id))
    return str(img_id)


def summarize_keypoint_conflicts(key_conflict: pd.DataFrame, max_items: int = 15) -> str:
    """
    Convert key_conflict (index=image, columns=keypoint) into readable text.
    """
    # Collect (image, keypoint) pairs where True
    pairs = []
    for img in key_conflict.index:
        row = key_conflict.loc[img]
        if hasattr(row, "items"):
            for kp, flag in row.items():
                if bool(flag):
                    pairs.append((img, kp))

    total = len(pairs)
    if total == 0:
        return "No existing keypoints will be overwritten."

    lines = []
    for img, kp in pairs[:max_items]:
        img_s = _format_image_id(img)

        # kp can be a scalar (bodypart) or a tuple (individual, bodypart)
        if isinstance(kp, tuple):
            if len(kp) == 2:
                ind, bp = kp
                kp_s = f"{bp} (id: {ind})" if ind else str(bp)
            else:
                kp_s = " / ".join(map(str, kp))
        else:
            kp_s = str(kp)

        lines.append(f"- {img_s} → {kp_s}")

    more = ""
    if total > max_items:
        more = f"\n… and {total - max_items} more."

    return f"{total} existing keypoint(s) will be overwritten.\n\nExamples:\n" + "\n".join(lines) + more
