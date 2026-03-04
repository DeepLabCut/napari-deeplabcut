from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import numpy as np

try:
    # napari is an optional dependency at import time in some test setups
    from napari.layers import Image, Layer, Points, Shapes, Tracks
except Exception:  # pragma: no cover
    Image = Points = Shapes = Tracks = Layer = object  # type: ignore

from napari_deeplabcut import misc
from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel

T = TypeVar("T")

# TODO move to a layers/ folder?
logger = logging.getLogger(__name__)


# Helper to populate keypoint layer metadata
def populate_keypoint_layer_metadata(
    header: DLCHeaderModel,
    *,
    labels: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    likelihood: Sequence[float] | None = None,
    paths: list[str] | None = None,
    size: int | None = 8,
    pcutoff: float | None = 0.6,
    colormap: str | None = "viridis",
) -> dict:
    """
    Populate metadata and display properties for a keypoint Points layer.

    Notes
    -----
    - Single-animal DLC: "individuals" level is effectively absent; we represent
      that as ids[0] == "" (falsy) => color/text by label.
    - Multi-animal DLC: ids[0] is a non-empty individual identifier => color/text by id.
    - Must accept empty labels/ids/likelihood and must not assume ≥ 1 entry.
    """

    if labels is None:
        labels = header.bodyparts
    if ids is None:
        ids = header.individuals
    if likelihood is None:
        likelihood_arr = np.ones(len(labels), dtype=float)
    else:
        likelihood_arr = np.asarray(likelihood, dtype=float)

    # 1) Normalize inputs to plain lists (Series-safe)
    #    This prevents pandas Series truthiness errors.
    labels_list = list(labels) if labels is not None else []
    ids_list = list(ids) if ids is not None else []

    # 2) Likelihood: always numeric ndarray for vector ops
    if likelihood is None:
        likelihood_arr = np.ones(len(labels_list), dtype=float)
    else:
        likelihood_arr = np.asarray(list(likelihood), dtype=float)

    # 3) Determine single vs multi animal:
    #    - empty ids => treat as single-animal (label-based)
    #    - ids[0] == "" => also single-animal (label-based)
    first_id = ids_list[0] if len(ids_list) > 0 else ""
    use_id = bool(first_id)

    face_color_cycle_maps = misc.build_color_cycles(header, colormap)
    face_color_prop = "id" if use_id else "label"

    return {
        "name": "keypoints",
        "text": "{id}–{label}" if use_id else "label",
        "properties": {
            "label": list(labels),
            "id": list(ids),
            "likelihood": likelihood_arr,
            "valid": likelihood_arr > pcutoff,
        },
        "face_color_cycle": face_color_cycle_maps[face_color_prop],
        "face_color": face_color_prop,
        "face_colormap": colormap,
        "border_color": "valid",
        "border_color_cycle": ["black", "red"],
        "border_width": 0,
        "border_width_is_relative": False,
        "size": size,
        "metadata": {
            "header": header,
            "face_color_cycles": face_color_cycle_maps,
            "colormap_name": colormap,
            "paths": paths or [],
        },
    }


def is_machine_layer(layer) -> bool:
    md = getattr(layer, "metadata", {}) or {}
    io = md.get("io") or {}
    k = io.get("kind")
    # allow enum or string
    if k is AnnotationKind.MACHINE:
        return True
    is_machine = str(k).lower() == "machine"
    if is_machine:
        logger.info(
            "A literal 'machine' str was used for io.kind; please use AnnotationKind.MACHINE for better validation."
        )
    return is_machine


# -----------------------------------------------
#  Layer-finding utilities
# -----------------------------------------------
def iter_layers(viewer_or_layers: Any) -> Iterable[Any]:
    """Yield layers from a napari Viewer or an iterable of layers."""
    layers = getattr(viewer_or_layers, "layers", viewer_or_layers)
    return layers


def find_first_layer(
    viewer_or_layers: Any,
    layer_type: type[T],
    predicate: Callable[[T], bool] | None = None,
) -> T | None:
    """Return the first layer of type ``layer_type`` that matches ``predicate``.

    Parameters
    ----------
    viewer_or_layers:
        A napari Viewer, LayerList, or any iterable of layers.
    layer_type:
        The desired layer type (e.g., napari.layers.Points).
    predicate:
        Optional function to further filter matching layers.

    Notes
    -----
    This intentionally mirrors the common pattern used throughout the plugin:
    "iterate viewer.layers in order and pick the first match".
    """
    pred = predicate or (lambda _ly: True)
    for ly in iter_layers(viewer_or_layers):
        if isinstance(ly, layer_type) and pred(ly):
            return ly
    return None


def find_last_layer(
    viewer_or_layers: Any,
    layer_type: type[T],
    predicate: Callable[[T], bool] | None = None,
) -> T | None:
    """Return the last layer of type ``layer_type`` that matches ``predicate``."""
    pred = predicate or (lambda _ly: True)
    last: T | None = None
    for ly in iter_layers(viewer_or_layers):
        if isinstance(ly, layer_type) and pred(ly):
            last = ly
    return last


# ---- Convenience wrappers used by deeplabcut widgets ----


def get_first_points_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Points)


def get_first_image_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Image)


def get_first_video_image_layer(viewer_or_layers: Any) -> Any | None:
    """First Image layer that looks like a video (>=3D data)."""

    def _is_video(img: Any) -> bool:
        try:
            return hasattr(img, "data") and getattr(img.data, "ndim", 0) >= 3
        except Exception:
            return False

    return find_first_layer(viewer_or_layers, Image, _is_video)


def get_points_layer_with_tables(viewer_or_layers: Any) -> Any | None:
    """First Points layer whose metadata has a non-empty 'tables' entry."""

    def _has_tables(pts: Any) -> bool:
        try:
            md = getattr(pts, "metadata", None) or {}
            return bool(md.get("tables"))
        except Exception:
            return False

    return find_first_layer(viewer_or_layers, Points, _has_tables)


def get_first_shapes_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Shapes)


def get_first_tracks_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Tracks)
