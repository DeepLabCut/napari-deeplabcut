# src/napari_deeplabcut/_reader.py
"""Readers for DeepLabCut data formats."""

from __future__ import annotations

import logging
from pathlib import Path

from napari_deeplabcut.core.discovery import discover_annotations
from napari_deeplabcut.core.io import (
    SUPPORTED_IMAGES,
    SUPPORTED_VIDEOS,
    read_config,
    read_hdf,
    read_hdf_single,
    read_images,
    read_video,
)
from napari_deeplabcut.core.paths import looks_like_dlc_labeled_folder

logger = logging.getLogger(__name__)


def is_video(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEOS)


def get_hdf_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not str(path).endswith(".h5"):
        return None
    return read_hdf


def get_image_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not any(str(path).lower().endswith(ext) for ext in SUPPORTED_IMAGES):
        return None
    return read_images


def get_video_reader(path):
    if isinstance(path, str) and any(path.lower().endswith(ext) for ext in SUPPORTED_VIDEOS):
        return read_video
    return None


def get_config_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not str(path).endswith(".yaml"):
        return None
    return read_config


def _filter_extensions(image_paths, valid_extensions=SUPPORTED_IMAGES) -> list[Path]:
    return [Path(p) for p in image_paths if Path(p).suffix.lower() in valid_extensions]


def get_folder_parser(path):
    if not path or not Path(path).is_dir():
        return None
    if not looks_like_dlc_labeled_folder(path):
        return None

    layers = []
    images = _filter_extensions(Path(path).iterdir(), valid_extensions=SUPPORTED_IMAGES)

    if not images:
        has_video = any(Path(path).glob(f"*{ext}") for ext in SUPPORTED_VIDEOS)
        if has_video:
            logger.info(
                "No supported images found in '%s' (extensions: %s). "
                "A supported video appears to be present; open the video directly to view frames.",
                path,
                SUPPORTED_IMAGES,
            )
        else:
            logger.warning(
                "No supported images found in '%s' (extensions: %s), and no supported videos found (extensions: %s).",
                path,
                SUPPORTED_IMAGES,
                SUPPORTED_VIDEOS,
            )
        return None

    layers.extend(read_images(images))

    # Deterministic discovery: load ALL H5 artifacts
    artifacts = discover_annotations(path)
    h5_artifacts = [(Path(a.h5_path), a.kind) for a in artifacts if a.h5_path is not None]

    if not h5_artifacts:
        return lambda _: layers

    errors = []
    for h5_path, kind in h5_artifacts:
        try:
            layers.extend(read_hdf_single(h5_path, kind=kind))
        except Exception as e:
            logger.debug("Could not read annotation data from %s", h5_path, exc_info=True)
            errors.append((RuntimeError, f"Could not read annotation data from {h5_path}", e))

    n_points_layers = sum(1 for _, _, layer_type in layers if layer_type == "points")
    if n_points_layers == 0 and errors:
        exc_type, msg, cause = errors[0]
        raise exc_type(msg) from cause

    return lambda _: layers
