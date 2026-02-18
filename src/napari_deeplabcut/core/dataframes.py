# src/napari_deeplabcut/core/dataframes.py

from __future__ import annotations

from typing import Tuple

import pandas as pd

from napari_deeplabcut import misc


def harmonize_keypoint_row_index(df_new: pd.DataFrame, df_old: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    misc.guarantee_multiindex_rows(df_new2)
    misc.guarantee_multiindex_rows(df_old2)

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