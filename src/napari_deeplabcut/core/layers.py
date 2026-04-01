from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from napari.layers import Image, Points, Shapes, Tracks

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel
from napari_deeplabcut.core.keypoints import build_color_cycles

T = TypeVar("T")

# TODO move to a layers/ folder?
logger = logging.getLogger(__name__)


# Helper to populate keypoint layer properties
def populate_keypoint_layer_properties(
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

    face_color_cycle_maps = build_color_cycles(header, colormap)
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


# --------------------
# Convenience wrappers
# --------------------


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


@dataclass(frozen=True)
class LabelProgress:
    labeled_points: int
    total_points: int
    labeled_percent: float
    remaining_percent: float
    frame_count: int
    bodypart_count: int
    individual_count: int


def _get_header_model_from_metadata(md: dict) -> DLCHeaderModel | None:
    if not isinstance(md, dict):
        return None

    hdr = md.get("header")
    if hdr is None:
        return None

    if isinstance(hdr, DLCHeaderModel):
        return hdr

    if isinstance(hdr, dict):
        try:
            return DLCHeaderModel.model_validate(hdr)
        except Exception:
            return None

    try:
        return DLCHeaderModel(columns=hdr)
    except Exception:
        return None


def get_uniform_point_size(layer: Points, *, default: int = 6) -> int:
    size = getattr(layer, "size", default)
    try:
        arr = np.asarray(size, dtype=float).ravel()
        if arr.size == 0:
            return default
        return int(round(float(np.nanmean(arr))))
    except Exception:
        try:
            return int(round(float(size)))
        except Exception:
            return default


def set_uniform_point_size(layer: Points, size: int) -> None:
    # Scalar assignment keeps it lightweight and applies uniformly.
    layer.size = float(size)


def infer_frame_count(layer: Points, *, fallback_paths: list[str] | None = None) -> int:
    md = getattr(layer, "metadata", {}) or {}

    paths = md.get("paths") or fallback_paths or []
    if paths:
        return len(paths)

    data = np.asarray(getattr(layer, "data", []))
    if data.size == 0:
        return 0

    try:
        # Points layers use frame/time in first column
        return int(np.nanmax(data[:, 0])) + 1
    except Exception:
        return 0


def infer_bodypart_count(layer: Points) -> int:
    hdr = _get_header_model_from_metadata(getattr(layer, "metadata", {}) or {})
    if hdr is None:
        return 0

    try:
        return len([bp for bp in hdr.bodyparts if str(bp) != ""])
    except Exception:
        return 0


def infer_individual_count(layer: Points) -> int:
    """
    Returns the number of valid DLC individuals.

    Single-animal convention:
    - if no individuals are defined
    - or individuals are empty / blank
    => returns 1
    """
    hdr = _get_header_model_from_metadata(getattr(layer, "metadata", {}) or {})
    if hdr is None:
        return 1

    try:
        inds = [str(ind) for ind in hdr.individuals if str(ind) != ""]
        return max(1, len(inds))
    except Exception:
        return 1


def compute_label_progress(layer: Points, *, fallback_paths: list[str] | None = None) -> LabelProgress:
    frame_count = infer_frame_count(layer, fallback_paths=fallback_paths)
    bodypart_count = infer_bodypart_count(layer)
    individual_count = infer_individual_count(layer)

    total_points = frame_count * bodypart_count * individual_count

    data = np.asarray(getattr(layer, "data", []))
    labeled_points = int(data.shape[0]) if data.ndim >= 2 else 0

    if total_points > 0:
        labeled_points = min(labeled_points, total_points)
        labeled_percent = 100.0 * labeled_points / total_points
    else:
        labeled_percent = 0.0

    remaining_percent = max(0.0, 100.0 - labeled_percent)

    return LabelProgress(
        labeled_points=labeled_points,
        total_points=total_points,
        labeled_percent=labeled_percent,
        remaining_percent=remaining_percent,
        frame_count=frame_count,
        bodypart_count=bodypart_count,
        individual_count=individual_count,
    )


def infer_folder_display_name(
    active_layer,
    *,
    fallback_root: str | None = None,
) -> str:
    """
    Best-effort label for the current image/video folder context.
    """
    if active_layer is None:
        return "—"

    md = getattr(active_layer, "metadata", {}) or {}

    paths = md.get("paths") or []
    if paths:
        try:
            return Path(paths[0]).expanduser().parent.name or "—"
        except Exception:
            pass

    root = md.get("root") or fallback_root
    if root:
        try:
            return Path(root).expanduser().name or "—"
        except Exception:
            pass

    try:
        src = getattr(getattr(active_layer, "source", None), "path", None)
        if src:
            p = Path(str(src))
            if p.is_file():
                # video source: show parent folder name
                return p.parent.name or p.stem or "—"
            return p.name or "—"
    except Exception:
        pass

    return "—"


def find_relevant_image_layer(viewer) -> Image | None:
    active = viewer.layers.selection.active
    if isinstance(active, Image):
        return active

    for layer in viewer.layers:
        if isinstance(layer, Image):
            return layer

    return None
