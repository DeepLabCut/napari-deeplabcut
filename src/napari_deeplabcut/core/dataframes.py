# src/napari_deeplabcut/core/dataframes.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from napari_deeplabcut.config.models import ConflictEntry, OverwriteConflictReport, PointsMetadata
from napari_deeplabcut.core.schemas import DLCHeaderModel, PointsWriteInputModel

logger = logging.getLogger(__name__)


def drop_likelihood_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DLC likelihood columns from a dataframe if present."""
    # DLC-style wide dataframe: MultiIndex columns with a coords level
    if isinstance(df.columns, pd.MultiIndex):
        col_names = list(df.columns.names)
        if "coords" in col_names:
            mask = df.columns.get_level_values("coords").astype(str) != "likelihood"
            return df.loc[:, mask]

    # Fallback for already-stacked / flat dataframes
    if "likelihood" in df.columns:
        return df.drop(columns="likelihood")

    return df


def restore_dlc_on_disk_header_shape(
    df: pd.DataFrame, header: DLCHeaderModel, *, is_ma_project: bool | None = None
) -> pd.DataFrame:
    """
    Args:
        df: DataFrame with arbitrary column structure produced from napari Points + metadata.
        header: Authoritative DLCHeaderModel that defines the expected column structure on disk.
        is_ma_project: Optional boolean indicating if the project is multi-animal based on DLC config.

    Restore the DataFrame column structure to match the authoritative DLC header
    that should be used on disk.

    - If the logical target is single-animal, collapse an empty 'individuals' level.
    - Reindex to the authoritative header order.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    df_out = df.copy()

    # If is_ma_project is not provided, let the header report the format.
    if is_ma_project is None:
        is_ma_project = not header.is_single_animal

    # If the logical target is single-animal, collapse an empty 'individuals' level.
    if not is_ma_project:
        # If the normalized dataframe has an empty individuals level, collapse it.
        if df_out.columns.nlevels == 4 and "individuals" in (df_out.columns.names or []):
            inds = pd.Index(df_out.columns.get_level_values("individuals")).astype(str)
            non_empty_inds = {x for x in inds if x != ""}
            if non_empty_inds:
                raise ValueError(
                    "Refusing to write single-animal format because dataframe contains "
                    f"non-empty individuals: {sorted(non_empty_inds)}"
                )

            df_out = df_out.droplevel("individuals", axis=1)
            df_out.columns = df_out.columns.set_names(["scorer", "bodyparts", "coords"])

        # Reindex to the original authoritative 3-level header order
        try:
            # For an SA header, header.columns is the original 3-level tuples
            target_cols = pd.MultiIndex.from_tuples(
                header.columns,
                names=header.names or ["scorer", "bodyparts", "coords"],
            )
            df_out = df_out.reindex(target_cols, axis=1)
        except Exception:
            logger.debug("Could not reindex collapsed SA dataframe to authoritative header", exc_info=True)

        return df_out

    # Multi-animal: use canonical 4-level ordering
    try:
        target_cols = header.as_multiindex()
        df_out = df_out.reindex(target_cols, axis=1)
    except Exception:
        logger.debug("Could not reindex MA dataframe to authoritative header", exc_info=True)

    return df_out


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
    if len(df.index) == 0:
        return

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

    df = harmonize_keypoint_column_index(df)
    hdr_cols = canonical_keypoint_columns_from_header(ctx.meta.header)

    logger.debug("header cols nlevels %s, names %s", hdr_cols.nlevels, hdr_cols.names)

    df = df.reindex(hdr_cols, axis=1)

    logger.debug("After reindex: cols nlevels %s, names %s", df.columns.nlevels, df.columns.names)
    logger.debug(
        "header cols nlevels %s, names %s",
        hdr_cols.nlevels,
        hdr_cols.names,
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

    # Single animal 3-level: (scorer, bodyparts, coords) -> insert individuals=""
    # THIS IS ONLY FOR INTERNAL COMPARISONS; we do NOT want to write an SA header
    # with an empty individuals level to disk.
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


def canonical_keypoint_columns_from_header(header: DLCHeaderModel) -> pd.MultiIndex:
    """
    Return the canonical internal keypoint column index for save/merge/conflict logic.

    Internally we normalize DLC columns to 4 levels:
        scorer / individuals / bodyparts / coords
    even for single-animal projects, where individuals is represented as "".
    Final on-disk shape is restored later by restore_dlc_on_disk_header_shape().
    """
    cols = header.as_multiindex()

    if not isinstance(cols, pd.MultiIndex):
        raise TypeError("DLC header columns must be a pandas MultiIndex.")

    # Single-animal DLC header: scorer / bodyparts / coords
    if cols.nlevels == 3:
        frame = cols.to_frame(index=False)

        # Be defensive about unnamed levels. We assume DLC order:
        # scorer, bodyparts, coords.
        frame.columns = ["scorer", "bodyparts", "coords"]
        frame.insert(1, "individuals", "")

        return pd.MultiIndex.from_frame(
            frame,
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    # Multi-animal DLC header: scorer / individuals / bodyparts / coords
    if cols.nlevels == 4:
        return cols.set_names(["scorer", "individuals", "bodyparts", "coords"])

    raise ValueError(f"Unsupported DLC header column depth. Expected 3 or 4 levels, got {cols.nlevels}.")


def save_index_from_points_metadata(pts_meta: PointsMetadata) -> pd.Index | pd.MultiIndex | None:
    """
    Build the editable row index from PointsMetadata for saving.

    If pts_meta.paths is present, those paths define the set of frames/images
    currently represented by the layer. Missing keypoints within this row scope
    should be saved as NaN, which is how deleted napari points clear old labels.

    Returns None if no explicit path scope is available.
    """
    paths = list(pts_meta.paths or [])
    if not paths:
        return None

    idx = pd.Index(paths)
    dummy = pd.DataFrame(index=idx)
    guarantee_multiindex_rows(dummy)
    return dummy.index


def complete_df_for_save(
    df: pd.DataFrame,
    *,
    pts_meta: PointsMetadata,
    header: DLCHeaderModel,
) -> pd.DataFrame:
    """
    Napari Points.data contains only present/finite keypoints. If a user deletes
    a point, that point disappears from Points.data and layer.properties. For
    save semantics, absence inside the current editable scope must become NaN,
    otherwise merge-on-save will preserve the old on-disk value.

    This reindexes the dataframe to:

    - all editable rows from pts_meta.paths, when available
    - all expected keypoint columns from the DLC header

    Goal:

    - present keypoint -> finite x/y
    - deleted/missing keypoint inside save scope -> NaN
    - rows outside save scope are not included and can be preserved separately
      during merge.
    """
    df_copy = df.copy()

    guarantee_multiindex_rows(df_copy)
    df_copy = harmonize_keypoint_column_index(df_copy)
    df_copy = drop_likelihood_columns(df_copy)

    target_cols = canonical_keypoint_columns_from_header(header)

    # Always drop likelihood
    coords = target_cols.get_level_values("coords").astype(str)
    target_cols = target_cols[coords != "likelihood"]

    target_index = save_index_from_points_metadata(pts_meta)

    # If we have explicit paths, they define the editable frame/image scope.
    # Otherwise, preserve the current dataframe rows and only complete columns.
    if target_index is not None:
        df_copy = df_copy.reindex(index=target_index, columns=target_cols)
    else:
        df_copy = df_copy.reindex(columns=target_cols)

    return df_copy


def merge_save_df(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge an existing DLC dataframe with a new save dataframe.

    Semantics:
    - rows/columns outside df_new scope are preserved from df_old
    - rows/columns inside df_new scope replace df_old, including NaN
    - NaN in df_new therefore clears/deletes an old saved keypoint
    """
    df_new2, df_old2 = harmonize_keypoint_row_index(df_new, df_old)
    df_new2 = harmonize_keypoint_column_index(df_new2)
    df_old2 = harmonize_keypoint_column_index(df_old2)

    if len(df_old2.index) and len(df_new2.index):
        overlap = df_old2.index.intersection(df_new2.index)
        if overlap.empty:
            raise ValueError(
                "Cannot merge save dataframe: no row-index overlap after harmonization. "
                "Existing labels would be preserved instead of overwritten/deleted."
            )

    idx = df_old2.index.union(df_new2.index)
    cols = df_old2.columns.union(df_new2.columns)

    df_out = df_old2.reindex(index=idx, columns=cols)

    # Critical: assign df_new values directly, including NaN.
    df_out.loc[df_new2.index, df_new2.columns] = df_new2

    return df_out


def _keypoint_group_levels(columns: pd.Index | pd.MultiIndex) -> list[str] | None:
    """
    For internal DLC comparisons, columns are expected to be normalized to:
        scorer / individuals / bodyparts / coords
    Conflict reporting ignores scorer and coords, grouping by:
        individuals / bodyparts
    or just:
        bodyparts
    for single-animal data.
    """
    if not isinstance(columns, pd.MultiIndex):
        return None

    col_names = list(columns.names or [])
    has_inds = "individuals" in col_names
    has_body = "bodyparts" in col_names
    has_coords = "coords" in col_names

    if not (has_body and has_coords):
        return None

    key_levels: list[str] = []
    if has_inds:
        key_levels.append("individuals")
    key_levels.append("bodyparts")

    return key_levels


def keypoint_deletions(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Return a keypoint-level boolean table for destructive deletions.

    True means:
        old keypoint has at least one stored value,
        new save-scope keypoint has no stored value.
    """
    scoped_new = harmonize_keypoint_column_index(df_new.copy())
    guarantee_multiindex_rows(scoped_new)

    new_scope, old_scope = harmonize_keypoint_row_index(scoped_new, df_old)

    new_scope = harmonize_keypoint_column_index(new_scope)
    old_scope = harmonize_keypoint_column_index(old_scope)

    scope_index = new_scope.index
    scope_columns = new_scope.columns

    old_scoped = old_scope.reindex(index=scope_index, columns=scope_columns)
    new_scoped = new_scope.reindex(index=scope_index, columns=scope_columns)

    key_levels = _keypoint_group_levels(old_scoped.columns)

    if key_levels is None:
        return (old_scoped.notna().any(axis=1) & ~new_scoped.notna().any(axis=1)).to_frame(name="deleted")

    old_key_has = old_scoped.notna().T.groupby(level=key_levels).any().T
    new_key_has = new_scoped.notna().T.groupby(level=key_levels).any().T

    return old_key_has & ~new_key_has


def keypoint_conflicts(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    *,
    include_deletions: bool = False,
) -> pd.DataFrame:
    """
    Return a boolean DataFrame indexed by image, with columns as keypoints.

    True means saving df_new would destructively affect an existing value in df_old.

    By default this reports overwrites only:

        old finite, new finite, values differ

    If include_deletions=True, this also reports keypoint deletions:

        old keypoint has any stored value,
        new keypoint has no stored value

    Deletion detection only applies within df_new's own save scope.
    Callers should pass a df_new that has already been expanded with
    complete_df_for_save().
    """
    old, new = align_old_new(df_old, df_new)

    old_has = old.notna()
    new_has = new.notna()

    # Cell-level overwrite conflicts:
    # old and new both have values, but they differ.
    cell_conflict = (old != new) & old_has & new_has

    key_levels = _keypoint_group_levels(old.columns)

    if key_levels is None:
        overwrite_conflict = cell_conflict.any(axis=1).to_frame(name="conflict")
    else:
        overwrite_conflict = cell_conflict.T.groupby(level=key_levels).any().T

    if not include_deletions:
        return overwrite_conflict

    deletion_conflict = keypoint_deletions(df_old, df_new)

    # Align both key-level tables and combine for legacy/simple callers.
    idx = overwrite_conflict.index.union(deletion_conflict.index)
    cols = overwrite_conflict.columns.union(deletion_conflict.columns)

    overwrite_conflict = overwrite_conflict.reindex(index=idx, columns=cols, fill_value=False)
    deletion_conflict = deletion_conflict.reindex(index=idx, columns=cols, fill_value=False)

    return overwrite_conflict | deletion_conflict


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
    deletion_conflict: pd.DataFrame | None = None,
    max_entries: int = 50,
    layer_name: str | None = None,
    destination_path: str | None = None,
) -> OverwriteConflictReport:
    """
    Convert pandas key-conflict tables into a UI-facing overwrite report.

    Parameters
    ----------
    key_conflict:
        Boolean DataFrame indexed by frame/image identifier, with columns
        representing keypoints.
        Truthy cells indicate an existing keypoint value
        will be overwritten by another finite value.

    deletion_conflict:
        Optional boolean DataFrame with the same shape convention.
        Truthy cells indicate an existing keypoint value will be cleared/deleted,
        i.e. written as missing values.

    Returns
    -------
    OverwriteConflictReport
    """
    if deletion_conflict is None:
        deletion_conflict = pd.DataFrame(
            False,
            index=key_conflict.index,
            columns=key_conflict.columns,
        )

    idx = key_conflict.index.union(deletion_conflict.index)
    cols = key_conflict.columns.union(deletion_conflict.columns)

    key_conflict = key_conflict.reindex(index=idx, columns=cols, fill_value=False)
    deletion_conflict = deletion_conflict.reindex(index=idx, columns=cols, fill_value=False)

    # Prefer showing a keypoint as deleted if both flags somehow appear.
    # This avoids duplicated UI entries like "Modified" and "Deleted" for the same keypoint.
    key_conflict = key_conflict & ~deletion_conflict

    n_overwrites = int(key_conflict.to_numpy().sum())
    n_deletions = int(deletion_conflict.to_numpy().sum())

    affected = key_conflict | deletion_conflict
    n_frames = int(affected.any(axis=1).to_numpy().sum())

    entries: list[ConflictEntry] = []

    for img in idx:
        modified: list[str] = []
        deleted: list[str] = []

        for kp, flag in key_conflict.loc[img].items():
            if bool(flag):
                modified.append(_format_keypoint_id(kp))

        for kp, flag in deletion_conflict.loc[img].items():
            if bool(flag):
                deleted.append(_format_keypoint_id(kp))

        if modified or deleted:
            entries.append(
                ConflictEntry(
                    frame_label=_format_image_id(img),
                    keypoints=tuple(modified),
                    deleted_keypoints=tuple(deleted),
                )
            )

    shown = tuple(entries[:max_entries])
    truncated = max(0, len(entries) - len(shown))

    return OverwriteConflictReport(
        n_overwrites=n_overwrites,
        n_deletions=n_deletions,
        n_frames=n_frames,
        entries=shown,
        truncated_entries=truncated,
        layer_name=layer_name,
        destination_path=destination_path,
    )
