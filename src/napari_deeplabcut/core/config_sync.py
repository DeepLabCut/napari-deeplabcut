from __future__ import annotations

import logging
from pathlib import Path

import napari_deeplabcut.core.io as io
from napari_deeplabcut.core.metadata import read_points_meta
from napari_deeplabcut.core.project_paths import (
    find_nearest_config,
    infer_dlc_project,
    infer_dlc_project_from_image_layer,
    infer_dlc_project_from_points_meta,
)

logger = logging.getLogger("napari-deeplabcut.core.config_sync")

_POINT_SIZE_KEY = "dotsize"


def _coerce_point_size(value, *, default: int = 6, minimum: int = 1, maximum: int = 100) -> int:
    try:
        size = int(round(float(value)))
    except Exception:
        size = default
    return max(minimum, min(maximum, size))


def _layer_source_path(layer) -> str | None:
    try:
        src = getattr(layer, "source", None)
        p = getattr(src, "path", None) if src is not None else None
        return str(p) if p else None
    except Exception:
        return None


def resolve_config_path_from_layer(
    layer,
    *,
    fallback_project: str | Path | None = None,
    fallback_root: str | Path | None = None,
    image_layer=None,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> Path | None:
    """
    Best-effort, lightweight config resolution using centralized DLC project inference.

    Resolution order
    ----------------
    1. Infer from Points metadata via infer_dlc_project_from_points_meta(...)
    2. Infer from current image/video layer via infer_dlc_project_from_image_layer(...)
    3. Infer from generic path-like hints via infer_dlc_project(...)
    4. Last-resort upward search with find_nearest_config(...)

    This intentionally:
    - does not do recursive filesystem crawling
    - only searches upward with bounded max_levels
    - reuses the plugin's root-anchor / project-context semantics
    """
    # ------------------------------------------------------------------
    # 1) Points-layer-centric inference (authoritative when available)
    # ------------------------------------------------------------------
    try:
        pts_meta = read_points_meta(
            layer,
            migrate_legacy=True,
            drop_controls=True,
            drop_header=False,
        )
    except Exception:
        pts_meta = None

    if pts_meta is not None and not hasattr(pts_meta, "errors"):
        try:
            ctx = infer_dlc_project_from_points_meta(
                pts_meta,
                prefer_project_root=prefer_project_root,
                max_levels=max_levels,
            )
            if ctx.config_path is not None and ctx.config_path.is_file():
                return ctx.config_path
        except Exception:
            logger.debug("Failed to infer config from points metadata", exc_info=True)

    # ------------------------------------------------------------------
    # 2) Image/video-layer-centric inference
    # ------------------------------------------------------------------
    if image_layer is not None:
        try:
            ctx = infer_dlc_project_from_image_layer(
                image_layer,
                prefer_project_root=prefer_project_root,
                max_levels=max_levels,
            )
            if ctx.config_path is not None and ctx.config_path.is_file():
                return ctx.config_path
        except Exception:
            logger.debug("Failed to infer config from image layer", exc_info=True)

    # ------------------------------------------------------------------
    # 3) Generic fallback inference from path-like hints
    # ------------------------------------------------------------------
    md = getattr(layer, "metadata", {}) or {}
    paths = md.get("paths") or []

    anchor_candidates: list[str | Path] = []
    dataset_candidates: list[str | Path] = []

    for value in (
        md.get("project"),
        md.get("root"),
        _layer_source_path(layer),
        fallback_project,
        fallback_root,
    ):
        if value:
            anchor_candidates.append(value)

    # Paths can help infer the labeled-data dataset folder/root anchor
    if paths:
        # dataset candidate: first opened path / row-key hint
        dataset_candidates.append(paths[0])

        # Add a few paths as anchors (bounded/lightweight)
        for value in paths[:3]:
            anchor_candidates.append(value)

    try:
        ctx = infer_dlc_project(
            anchor_candidates=anchor_candidates,
            dataset_candidates=dataset_candidates,
            explicit_root=None,
            prefer_project_root=prefer_project_root,
            max_levels=max_levels,
        )
        if ctx.config_path is not None and ctx.config_path.is_file():
            return ctx.config_path
    except Exception:
        logger.debug("Failed to infer config from generic path hints", exc_info=True)

    # ------------------------------------------------------------------
    # 4) Fallback upward search on a bounded set of candidates
    # ------------------------------------------------------------------
    for candidate in anchor_candidates:
        try:
            cfg = find_nearest_config(candidate, max_levels=max_levels)
            if cfg is not None and cfg.is_file():
                return cfg
        except Exception:
            logger.debug("find_nearest_config failed for %r", candidate, exc_info=True)

    return None


def load_point_size_from_config(config_path: str | Path | None) -> int | None:
    if not config_path:
        return None

    try:
        cfg = io.load_config(str(config_path))
    except Exception:
        logger.debug("Could not read config file %r", config_path, exc_info=True)
        return None

    if _POINT_SIZE_KEY in cfg:
        return _coerce_point_size(cfg.get(_POINT_SIZE_KEY))
    return None


def save_point_size_to_config(config_path: str | Path | None, size: int) -> bool:
    """
    Persist point size in config.yaml if possible.

    Returns
    -------
    bool
        True if the config was changed and written, False otherwise.
    """
    if not config_path:
        logger.debug("Skipping point-size config sync: no config path resolved.")
        return False

    size = _coerce_point_size(size)

    try:
        cfg = io.load_config(str(config_path))
    except Exception:
        logger.debug("Could not read config file %r", config_path, exc_info=True)
        return False

    old_value = cfg.get(_POINT_SIZE_KEY, None)

    try:
        if old_value is not None and _coerce_point_size(old_value) == size:
            logger.debug("Skipping point-size config sync: dotsize already %s", size)
            return False
    except Exception:
        pass

    cfg[_POINT_SIZE_KEY] = size

    try:
        io.write_config(str(config_path), cfg)
        logger.debug("Updated dotsize=%s in %s", size, config_path)
        return True
    except Exception:
        logger.debug("Could not write config file %r", config_path, exc_info=True)
        return False
