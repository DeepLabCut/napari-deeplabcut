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


# FIXME @C-Achard replace with schema in core
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
