import logging
from collections.abc import Sequence

import numpy as np

from napari_deeplabcut import misc
from napari_deeplabcut.config.models import AnnotationKind

# TODO move to a layers/ folder?
logger = logging.getLogger(__name__)


# Helper to populate keypoint layer metadata
def populate_keypoint_layer_metadata(
    header: misc.DLCHeader,
    *,
    labels: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    likelihood: Sequence[float] | None = None,
    paths: list[str] | None = None,
    size: int | None = 8,
    pcutoff: float | None = 0.6,
    colormap: str | None = "viridis",
) -> dict:
    if labels is None:
        labels = header.bodyparts
    if ids is None:
        ids = header.individuals
    if likelihood is None:
        likelihood = np.ones(len(labels))
    face_color_cycle_maps = misc.build_color_cycles(header, colormap)
    face_color_prop = "id" if ids[0] else "label"
    return {
        "name": "keypoints",
        "text": "{id}–{label}" if ids[0] else "label",
        "properties": {
            "label": list(labels),
            "id": list(ids),
            "likelihood": likelihood,
            "valid": likelihood > pcutoff,
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
