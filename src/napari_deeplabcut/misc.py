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

logger = logging.getLogger(__name__)


def canonicalize_path(p: str | Path, n: int = 3) -> str:
    """Return canonical POSIX path built from the last n path components.

    This is platform-agnostic: it normalizes both Windows (`\\`) and POSIX (`/`)
    separators *before* splitting, then returns a POSIX-joined tail of length `n`.

    Examples
    --------
    - "C:\\data\\frames\\test\\img001.png" -> "frames/test/img001.png" (n=3)
    - "/home/user/frames/test/img001.png" -> "frames/test/img001.png" (n=3)
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    try:
        s = str(p)
    except Exception as e:
        logger.debug("Failed to stringify path %r (%s).", p, type(e).__name__, exc_info=True)
        return ""

    s = s.replace("\\", "/")
    s = s.rstrip("/")
    parts = [part for part in s.split("/") if part and part != "." and part != ".."]

    if not parts:
        return ""
    return "/".join(parts[-n:])


def remap_array(values, idx_map):
    """
    Remap integer frame indices using a mapping, safe for empty arrays.

    Args:
        values: Array-like of integer indices (e.g., a NumPy array) to be remapped.
        idx_map: Mapping from original integer indices to new integer indices.

    Returns:
        A NumPy array of integer indices where each element of ``values`` is
        replaced by ``idx_map[value]`` when present in the mapping; if a value
        is not found in ``idx_map``, it is left unchanged. Empty input arrays
        are returned unchanged.
    """
    values = values.astype(int, copy=False)

    if values.size == 0:
        return values  # important: allow empty arrays!

    # Build array of mapped values, falling back to identity
    mapped = np.fromiter((idx_map.get(v, v) for v in values), dtype=values.dtype, count=len(values))
    return mapped


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
