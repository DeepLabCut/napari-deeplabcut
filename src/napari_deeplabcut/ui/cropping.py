# src/napari_deeplabcut/ui/cropping.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from napari.layers import Image, Shapes
from qtpy.QtWidgets import QGroupBox, QPushButton, QVBoxLayout

import napari_deeplabcut.core.io as io
from napari_deeplabcut.core.project_paths import infer_dlc_project, infer_dlc_project_from_image_layer


def build_video_action_menu(*, on_extract_frame, on_store_crop) -> QGroupBox:
    """Create the small 'Video' action panel used by KeypointControls."""
    group_box = QGroupBox("Video")
    layout = QVBoxLayout()

    extract_button = QPushButton("Extract frame")
    extract_button.clicked.connect(on_extract_frame)
    layout.addWidget(extract_button)

    crop_button = QPushButton("Store crop coordinates")
    crop_button.clicked.connect(on_store_crop)
    layout.addWidget(crop_button)

    group_box.setLayout(layout)
    return group_box


def _resolve_video_key(image_layer: Image, project_path: str, fallback_video_name: str | None = None) -> str | None:
    """
    Determine the key used under cfg['video_sets'].

    Prefer the actual loaded video source path if available, otherwise fall back
    to <project>/videos/<video_name>.
    """
    try:
        src = getattr(getattr(image_layer, "source", None), "path", None)
    except Exception:
        src = None

    if src:
        try:
            return str(Path(src).expanduser().resolve())
        except Exception:
            return str(Path(src))

    video_name = fallback_video_name or getattr(image_layer, "name", None)
    if not video_name:
        return None

    return os.path.join(project_path, "videos", video_name)


def _find_latest_rectangle_crop(viewer) -> tuple[int, int, int, int] | None:
    """
    Return (x1, x2, y1, y2) from the latest rectangle in a Shapes layer.

    This preserves the current coordinate convention used by _widgets.py.
    """
    shape_layers = [layer for layer in viewer.layers if isinstance(layer, Shapes)]
    if not shape_layers:
        return None

    for layer in reversed(shape_layers):
        try:
            rect_indices = [i for i, shape in enumerate(layer.shape_type) if shape == "rectangle"]
        except Exception:
            continue

        if not rect_indices:
            continue

        ind = rect_indices[-1]
        try:
            bbox = np.asarray(layer.data[ind])[:, 1:].copy()
        except Exception:
            continue

        if bbox.ndim != 2 or bbox.shape[1] != 2:
            continue

        try:
            # Preserve the existing convention from _widgets.py
            h = viewer.dims.range[2][1]
        except Exception:
            return None

        bbox[:, 0] = h - bbox[:, 0]
        bbox = np.clip(bbox, 0, a_max=None).astype(int)

        y1, x1 = bbox.min(axis=0)
        y2, x2 = bbox.max(axis=0)
        return int(x1), int(x2), int(y1), int(y2)

    return None


def store_crop_coordinates(
    viewer,
    *,
    image_layer: Image,
    explicit_project_path: str | None = None,
    fallback_video_name: str | None = None,
) -> bool:
    """
    Persist crop coordinates into the project's config.yaml.

    Returns
    -------
    bool
        True if the crop was written, False otherwise.
    """
    if image_layer is None:
        return False

    if explicit_project_path:
        ctx = infer_dlc_project(
            explicit_root=explicit_project_path,
            anchor_candidates=[getattr(getattr(image_layer, "source", None), "path", None)],
            prefer_project_root=True,
        )
    else:
        ctx = infer_dlc_project_from_image_layer(image_layer, prefer_project_root=True)

    config_path = ctx.config_path
    project_root = ctx.project_root or ctx.root_anchor

    if config_path is None or project_root is None:
        return False

    crop = _find_latest_rectangle_crop(viewer)
    if crop is None:
        return False

    video_key = _resolve_video_key(image_layer, str(project_root), fallback_video_name=fallback_video_name)
    if not video_key:
        return False

    cfg = io.load_config(str(config_path))
    video_sets = cfg.setdefault("video_sets", {})
    existing = dict(video_sets.get(video_key, {}))
    existing["crop"] = ", ".join(map(str, crop))
    video_sets[video_key] = existing
    io.write_config(str(config_path), cfg)
    return True
