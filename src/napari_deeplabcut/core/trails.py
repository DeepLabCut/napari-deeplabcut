from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from napari.layers import Points
from napari.utils.colormaps import Colormap

from napari_deeplabcut import keypoints


@dataclass(frozen=True)
class TrailsPayload:
    tracks_data: np.ndarray
    properties: dict[str, np.ndarray]
    color_by: str
    colormaps_dict: dict[str, Colormap]
    signature: tuple


def trails_signature(layer: Points, color_mode: str | keypoints.ColorMode) -> tuple:
    """Small signature used to decide whether trails need rebuilding."""
    props = getattr(layer, "properties", {}) or {}
    labels = props.get("label")
    ids = props.get("id")

    n_vertices = int(getattr(layer.data, "shape", [0])[0]) if layer.data is not None else 0
    n_labels = len(labels) if labels is not None else 0
    n_ids = len(ids) if ids is not None else 0

    return (
        id(layer),
        str(color_mode),
        layer.metadata.get("colormap_name"),
        n_vertices,
        n_labels,
        n_ids,
    )


def is_multianimal_points_layer(layer: Points) -> bool:
    props = getattr(layer, "properties", {}) or {}
    ids = props.get("id")
    if ids is None or len(ids) == 0:
        return False

    try:
        first = ids[0]
    except Exception:
        return False

    return isinstance(first, str) and first != ""


def active_trails_color_property(
    layer: Points,
    color_mode: str | keypoints.ColorMode,
) -> tuple[str, np.ndarray, bool]:
    """
    Return:
        color_prop: 'id' or 'label'
        categories_for_color: values to color by
        is_multi: whether the layer is multi-animal
    """
    props = getattr(layer, "properties", {}) or {}
    labels = props.get("label")
    ids = props.get("id")

    if labels is None or len(labels) == 0:
        raise ValueError("Points layer has no 'label' property; cannot build trails.")

    is_multi = is_multianimal_points_layer(layer)

    if str(color_mode) == str(keypoints.ColorMode.INDIVIDUAL) and is_multi:
        return "id", ids, True

    return "label", labels, is_multi


def _rgba_array(colors: list) -> np.ndarray:
    out = []
    for c in colors:
        c = np.asarray(c, dtype=float)
        if c.ndim == 0:
            c = np.array([0.5, 0.5, 0.5, 1.0], dtype=float)
        if c.shape[0] == 3:
            c = np.r_[c, 1.0]
        out.append(c)
    return np.asarray(out, dtype=float)


def categorical_colormap_from_points_layer(
    layer: Points,
    color_prop: str,
    categories_for_color: np.ndarray,
) -> tuple[Colormap, list[str], np.ndarray]:
    """
    Build a categorical zero-interpolation napari Colormap matching the points layer cycles.

    Returns
    -------
    cmap
    uniq_color
    codes_norm
    """
    uniq_color = list(dict.fromkeys(map(str, categories_for_color)))
    n_color = len(uniq_color)

    if n_color == 0:
        raise ValueError("No categories found for trails coloring.")

    face_cycles = (layer.metadata or {}).get("face_color_cycles", {})
    cycle_dict = face_cycles.get(color_prop, {}) or {}

    color_list = []
    for u in uniq_color:
        c = cycle_dict.get(u)
        if c is None:
            color_list = []
            break
        color_list.append(c)

    if not color_list:
        palette = plt.get_cmap("tab20").colors
        color_list = [palette[i % len(palette)] for i in range(n_color)]

    colors_rgba = _rgba_array(color_list)

    if n_color == 1:
        controls = np.array([0.0, 1.0], dtype=float)
        colors_rgba = np.vstack([colors_rgba[0], colors_rgba[0]])
    else:
        controls = np.linspace(0.0, 1.0, n_color + 1, dtype=float)

    cmap = Colormap(
        colors=colors_rgba,
        controls=controls,
        name=f"{color_prop}_categorical",
        interpolation="zero",
    )

    color_index = {u: i for i, u in enumerate(uniq_color)}
    codes = np.array([color_index[str(u)] for u in categories_for_color], dtype=int)
    codes_norm = (codes / float(max(n_color - 1, 1))).astype(float)

    return cmap, uniq_color, codes_norm


def trails_track_ids(layer: Points, *, is_multi: bool) -> np.ndarray:
    """
    Group trajectories by:
      - (id, label) for multi-animal
      - label for single-animal
    """
    props = getattr(layer, "properties", {}) or {}
    labels = props.get("label")
    ids = props.get("id")

    if labels is None or len(labels) == 0:
        raise ValueError("Points layer has no 'label' property; cannot build trails.")

    if is_multi:
        group_keys = [f"{i}|{l}" for i, l in zip(ids, labels, strict=False)]
    else:
        group_keys = [str(l) for l in labels]

    uniq_group = list(dict.fromkeys(group_keys))
    gid_map = {g: k for k, g in enumerate(uniq_group)}
    return np.array([gid_map[g] for g in group_keys], dtype=int)


def build_trails_payload(
    layer: Points,
    color_mode: str | keypoints.ColorMode,
) -> TrailsPayload:
    color_prop, categories_for_color, is_multi = active_trails_color_property(layer, color_mode)
    cmap, _, codes_norm = categorical_colormap_from_points_layer(layer, color_prop, categories_for_color)
    track_ids = trails_track_ids(layer, is_multi=is_multi)

    tracks_data = np.c_[track_ids, layer.data]
    color_key = f"{color_prop}_codes"

    return TrailsPayload(
        tracks_data=tracks_data,
        properties={color_key: codes_norm},
        color_by=color_key,
        colormaps_dict={color_key: cmap},
        signature=trails_signature(layer, color_mode),
    )
