"""
Wrapper around anything that touches private napari internals, to isolate potential breakages to a single file.
Last updated with version of napari: 0.9.1 by @C-Achard - 2026-09-08
"""

from __future__ import annotations

from .points_layer import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
    register_points_action,
)
from .proxy import unwrap
from .window import qt_dock_widgets

__all__ = [
    "apply_points_layer_ui_tweaks",
    "install_add_wrapper",
    "install_paste_patch",
    "qt_dock_widgets",
    "register_points_action",
    "unwrap",
]
