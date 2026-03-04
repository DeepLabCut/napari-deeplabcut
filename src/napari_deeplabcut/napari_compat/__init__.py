from __future__ import annotations

from .color import patch_color_manager_guess_continuous
from .points_layer import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
)

__all__ = [
    "patch_color_manager_guess_continuous",
    "apply_points_layer_ui_tweaks",
    "install_add_wrapper",
    "install_paste_patch",
]
