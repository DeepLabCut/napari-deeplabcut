"""Helper functions for building trails layers from points layers."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from napari.layers import Points, Tracks
from napari.utils.colormaps import Colormap

from napari_deeplabcut.config.models import TrailsDisplayConfig
from napari_deeplabcut.core import keypoints


@dataclass(frozen=True)
class TrailsPayload:
    tracks_data: np.ndarray
    properties: dict[str, np.ndarray]
    color_by: str
    colormaps_dict: dict[str, Colormap]
    signature: tuple
    geometry_signature: tuple


def trails_signature(layer: Points, color_mode: str | keypoints.ColorMode) -> tuple:
    """
    Full signature used to decide whether trails style needs refreshing.

    Includes:
    - source layer identity
    - active color mode
    - configured colormap name
    - number of vertices
    - label/id contents
    """
    props = getattr(layer, "properties", {}) or {}
    labels = tuple(map(str, props.get("label", [])))
    ids = tuple(map(str, props.get("id", [])))

    n_vertices = int(getattr(layer.data, "shape", [0])[0]) if layer.data is not None else 0

    return (
        id(layer),
        str(color_mode),
        (layer.metadata or {}).get("colormap_name"),
        n_vertices,
        labels,
        ids,
    )


def trails_geometry_signature(layer: Points) -> tuple:
    """
    Signature used to decide whether trails geometry must be rebuilt.

    Includes:
    - source layer identity
    - raw data shape
    - label/id contents (because they define grouping)
    """
    props = getattr(layer, "properties", {}) or {}
    labels = tuple(map(str, props.get("label", [])))
    ids = tuple(map(str, props.get("id", [])))
    data_shape = tuple(np.asarray(layer.data).shape) if layer.data is not None else (0,)

    return (
        id(layer),
        data_shape,
        labels,
        ids,
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


def _trails_rgba_array(colors: list) -> np.ndarray:
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
    *,
    cycle_override: dict | None = None,
) -> tuple[Colormap, list[str], np.ndarray]:
    """
    Build a categorical zero-interpolation napari Colormap matching the points layer cycles.

    Parameters
    ----------
    layer
        Source points layer.
    color_prop
        'id' or 'label'.
    categories_for_color
        Property values used for color coding.
    cycle_override
        Optional resolved category->color mapping to use instead of raw metadata.
    """
    uniq_color = list(dict.fromkeys(map(str, categories_for_color)))
    n_color = len(uniq_color)

    if n_color == 0:
        raise ValueError("No categories found for trails coloring.")

    if cycle_override is not None:
        cycle_dict = {str(k): np.asarray(v, dtype=float) for k, v in cycle_override.items()}
    else:
        face_cycles = (layer.metadata or {}).get("face_color_cycles", {})
        cycle_dict = {str(k): np.asarray(v, dtype=float) for k, v in (face_cycles.get(color_prop, {}) or {}).items()}

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

    colors_rgba = _trails_rgba_array(color_list)

    color_index = {u: i for i, u in enumerate(uniq_color)}
    codes = np.array([color_index[str(u)] for u in categories_for_color], dtype=int)

    if n_color == 1:
        # zero-interpolation colormap still expects len(controls) == len(colors) + 1
        colors_rgba = np.vstack([colors_rgba[0], colors_rgba[0]])
        controls = np.array([0.0, 0.5, 1.0], dtype=float)
        codes_norm = np.full(len(categories_for_color), 0.5, dtype=float)
    else:
        controls = np.linspace(0.0, 1.0, n_color + 1, dtype=float)
        codes_norm = ((codes + 0.5) / n_color).astype(float)

    cmap = Colormap(
        colors=colors_rgba,
        controls=controls,
        name=f"{color_prop}_categorical",
        interpolation="zero",
    )

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
    *,
    cycle_override: dict | None = None,
) -> TrailsPayload:
    color_prop, categories_for_color, is_multi = active_trails_color_property(layer, color_mode)
    cmap, _, codes_norm = categorical_colormap_from_points_layer(
        layer,
        color_prop,
        categories_for_color,
        cycle_override=cycle_override,
    )
    track_ids = trails_track_ids(layer, is_multi=is_multi)

    tracks_data = np.c_[track_ids, layer.data]
    color_key = f"{color_prop}_codes"

    return TrailsPayload(
        tracks_data=tracks_data,
        properties={color_key: codes_norm},
        color_by=color_key,
        colormaps_dict={color_key: cmap},
        signature=trails_signature(layer, color_mode),
        geometry_signature=trails_geometry_signature(layer),
    )


def tracks_kwargs_from_display_config(cfg: TrailsDisplayConfig) -> dict:
    """
    Convert persistent trails display config into kwargs suitable for viewer.add_tracks().

    Notes
    -----
    We intentionally do NOT include `visible` here, because visibility is UI state:
    when the user explicitly checks "Show trails", the created layer should be shown.
    """
    return {
        "tail_length": int(cfg.tail_length),
        "head_length": int(cfg.head_length),
        "tail_width": float(cfg.tail_width),
        "opacity": float(cfg.opacity),
        "blending": str(cfg.blending),
    }


def display_config_from_tracks_layer(layer: Tracks, *, visible: bool | None = None) -> TrailsDisplayConfig:
    """
    Extract persistent trails display config from an existing Tracks layer.

    Parameters
    ----------
    layer
        Tracks layer to snapshot.
    visible
        Optional visibility override. Useful when persisting a deliberate hide/show
        action without mutating the layer first.
    """
    return TrailsDisplayConfig(
        tail_length=int(getattr(layer, "tail_length", 50)),
        head_length=int(getattr(layer, "head_length", 50)),
        tail_width=float(getattr(layer, "tail_width", 6.0)),
        opacity=float(getattr(layer, "opacity", 1.0)),
        blending=str(getattr(layer, "blending", "translucent")),
        visible=bool(layer.visible if visible is None else visible),
    )
