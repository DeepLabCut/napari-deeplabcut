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
