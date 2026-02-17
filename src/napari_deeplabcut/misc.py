# src/napari_deeplabcut/misc.py
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum, EnumMeta
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from napari.utils import colormaps
from natsort import natsorted

from napari_deeplabcut.core.paths import canonicalize_path

logger = logging.getLogger(__name__)


# def canonicalize_path(p: str | Path, n: int = 3) -> str:
#     """Return canonical POSIX path built from the last n path components.

#     This is platform-agnostic: it normalizes both Windows (`\\`) and POSIX (`/`)
#     separators *before* splitting, then returns a POSIX-joined tail of length `n`.

#     Examples
#     --------
#     - "C:\\data\\frames\\test\\img001.png" -> "frames/test/img001.png" (n=3)
#     - "/home/user/frames/test/img001.png" -> "frames/test/img001.png" (n=3)
#     """
#     if n <= 0:
#         raise ValueError("n must be a positive integer")
#     try:
#         s = str(p)
#     except Exception as e:
#         logger.debug("Failed to stringify path %r (%s).", p, type(e).__name__, exc_info=True)
#         return ""

#     s = s.replace("\\", "/")
#     s = s.rstrip("/")
#     parts = [part for part in s.split("/") if part and part != "." and part != ".."]

#     if not parts:
#         return ""
#     return "/".join(parts[-n:])


# def remap_array(values, idx_map):
#     """
#     Remap integer frame indices using a mapping, safe for empty arrays.

#     Args:
#         values: Array-like of integer indices (e.g., a NumPy array) to be remapped.
#         idx_map: Mapping from original integer indices to new integer indices.

#     Returns:
#         A NumPy array of integer indices where each element of ``values`` is
#         replaced by ``idx_map[value]`` when present in the mapping; if a value
#         is not found in ``idx_map``, it is left unchanged. Empty input arrays
#         are returned unchanged.
#     """
#     values = values.astype(int, copy=False)

#     if values.size == 0:
#         return values  # important: allow empty arrays!

#     # Build array of mapped values, falling back to identity
#     mapped = np.fromiter((idx_map.get(v, v) for v in values), dtype=values.dtype, count=len(values))
#     return mapped


def find_project_config_path(labeled_data_path: str) -> str:
    return str(Path(labeled_data_path).parents[2] / "config.yaml")


def is_latest_version():
    import json
    import urllib.request

    from napari_deeplabcut import __version__

    url = "https://pypi.org/pypi/napari-deeplabcut/json"
    contents = urllib.request.urlopen(url).read()
    latest_version = json.loads(contents)["info"]["version"]
    return __version__ == latest_version, latest_version


def unsorted_unique(array: Sequence) -> np.ndarray:
    """Return the unsorted unique elements of an array."""
    _, inds = np.unique(array, return_index=True)
    return np.asarray(array)[np.sort(inds)]


def encode_categories(categories: Sequence, return_unique: bool = False, is_path: bool = True, do_sort: bool = True):
    """
    Convert a list of categories (typically filenames) into integer indices

    Args:
        categories: list of categories (strings or numbers)
        return_unique: if True, also returns a list of unique categories
        is_path: if True, canonicalize categories as paths
        do_sort: if True, sort unique categories naturally

    Returns:
        inds: array of integer indices corresponding to categories
        unique_cat: list of unique categories (if return_unique is True)
    """
    # Canonicalize all categories (important!)
    if is_path:
        categories = [canonicalize_path(c) for c in categories]

    # Determine unique values in stable order, but natural-sorted
    unique_cat = list(dict.fromkeys(categories))
    if do_sort:
        unique_cat = natsorted(unique_cat)
    map_ = {k: i for i, k in enumerate(unique_cat)}

    inds = np.array([map_[c] for c in categories], dtype=int)

    if return_unique:
        return inds, unique_cat
    return inds


def merge_multiple_scorers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    n_frames = df.shape[0]
    header = DLCHeader(df.columns)
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


def build_color_cycle(n_colors: int, colormap: str | None = "viridis") -> np.ndarray:
    cmap = colormaps.ensure_colormap(colormap)
    return cmap.map(np.linspace(0, 1, n_colors))


def build_color_cycles(header: DLCHeader, colormap: str | None = "viridis"):
    label_colors = build_color_cycle(len(header.bodyparts), colormap)
    id_colors = build_color_cycle(len(header.individuals), colormap)
    return {
        "label": dict(zip(header.bodyparts, label_colors, strict=False)),
        "id": dict(zip(header.individuals, id_colors, strict=False)),
    }


class DLCHeader:
    def __init__(self, columns: pd.MultiIndex):
        self.columns = columns

    @classmethod
    def from_config(cls, config: dict) -> DLCHeader:
        multi = config.get("multianimalproject", False)
        scorer = [config["scorer"]]
        if multi:
            columns = pd.MultiIndex.from_product(
                [
                    scorer,
                    config["individuals"],
                    config["multianimalbodyparts"],
                    ["x", "y"],
                ]
            )
            if len(config["uniquebodyparts"]):
                temp = pd.MultiIndex.from_product([scorer, ["single"], config["uniquebodyparts"], ["x", "y"]])
                columns = columns.append(temp)
            columns.set_names(["scorer", "individuals", "bodyparts", "coords"], inplace=True)
        else:
            columns = pd.MultiIndex.from_product(
                [scorer, config["bodyparts"], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
        return cls(columns)

    def form_individual_bodypart_pairs(self) -> list[tuple[str]]:
        to_drop = [name for name in self.columns.names if name not in ("individuals", "bodyparts")]
        temp = self.columns.droplevel(to_drop).unique()
        if "individuals" not in temp.names:
            temp = pd.MultiIndex.from_product([self.individuals, temp])
        return temp.to_list()

    @property
    def scorer(self) -> str:
        return self._get_unique("scorer")[0]

    @scorer.setter
    def scorer(self, scorer: str):
        self.columns = self.columns.set_levels([scorer], level="scorer")

    @property
    def individuals(self) -> list[str]:
        individuals = self._get_unique("individuals")
        if individuals is None:
            return [""]
        return individuals

    @property
    def bodyparts(self) -> list[str]:
        return self._get_unique("bodyparts")

    @property
    def coords(self) -> list[str]:
        return self._get_unique("coords")

    def _get_unique(self, name: str) -> list | None:
        if name in self.columns.names:
            return list(unsorted_unique(self.columns.get_level_values(name)))
        return None


class CycleEnumMeta(EnumMeta):
    def __new__(metacls, cls, bases, classdict, **kwargs):
        enum_ = super().__new__(metacls, cls, bases, classdict, **kwargs)
        enum_._cycle = cycle(enum_._member_map_[name] for name in enum_._member_names_)
        return enum_

    def __iter__(cls):
        return cls._cycle

    def __next__(cls):
        return next(cls.__iter__())

    def __getitem__(self, item):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)


class CycleEnum(Enum, metaclass=CycleEnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self):
        return self.value


def align_old_new(df_old: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both dataframes to union of index and columns."""
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
