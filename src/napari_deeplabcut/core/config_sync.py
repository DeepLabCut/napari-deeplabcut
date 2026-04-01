from __future__ import annotations

import logging
from pathlib import Path

import napari_deeplabcut.core.io as io

logger = logging.getLogger("napari-deeplabcut.core.config_sync")

_POINT_SIZE_KEY = "dotsize"


def _coerce_point_size(value, *, default: int = 6, minimum: int = 1, maximum: int = 100) -> int:
    try:
        size = int(round(float(value)))
    except Exception:
        size = default
    return max(minimum, min(maximum, size))


# FIXME likely duplicated logic
def resolve_config_path_from_layer(layer) -> Path | None:
    """
    Lightweight, safe config resolution:
    - uses layer.metadata["project"] if present
    - does not recursively search the filesystem
    """
    md = getattr(layer, "metadata", {}) or {}

    project = md.get("project")
    if not project:
        return None

    try:
        project_path = Path(project).expanduser()
    except Exception:
        return None

    if project_path.is_file() and project_path.name == "config.yaml":
        return project_path

    cfg = project_path / "config.yaml"
    if cfg.is_file():
        return cfg

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
        return False

    size = _coerce_point_size(size)

    try:
        cfg = io.load_config(str(config_path))
    except Exception:
        logger.debug("Could not read config file %r", config_path, exc_info=True)
        return False

    existing_key = _POINT_SIZE_KEY
    old_value = cfg.get(existing_key, None)

    try:
        if old_value is not None and _coerce_point_size(old_value) == size:
            return False
    except Exception:
        pass

    cfg[existing_key] = size

    try:
        io.write_config(str(config_path), cfg)
        return True
    except Exception:
        logger.debug("Could not write config file %r", config_path, exc_info=True)
        return False
