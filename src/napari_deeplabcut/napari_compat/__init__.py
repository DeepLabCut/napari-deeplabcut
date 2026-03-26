"""
Wrapper around anything that touches private napari internals, to isolate potential breakages to a single file.
Last updated with version of napari: 0.6.6 by @C-Achard - 2026-03-04
"""

from __future__ import annotations

from .color import patch_color_manager_guess_continuous
from .points_layer import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
    register_points_action,
)

__all__ = [
    "patch_color_manager_guess_continuous",
    "apply_points_layer_ui_tweaks",
    "install_add_wrapper",
    "install_paste_patch",
    "register_points_action",
]
