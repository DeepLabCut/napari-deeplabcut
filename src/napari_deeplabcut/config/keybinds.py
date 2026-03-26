"""Future file for centralizing keybinds."""

# src/napari_deeplabcut/config/keybinds.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from napari.layers import Points

_global_points_bindings_installed = False


@dataclass(frozen=True)
class ShortcutSpec:
    key: str
    description: str
    scope: str  # e.g. "points-layer", "global-points"
    overwrite: bool = False


# ---- Single source of truth for displayed shortcuts ----
CYCLE_LABEL_MODE = ShortcutSpec(
    key="M",
    description="Change labeling mode",
    scope="points-layer",
)

CYCLE_COLOR_MODE = ShortcutSpec(
    key="F",
    description="Change color mode",
    scope="points-layer",
)

NEXT_KEYPOINT = ShortcutSpec(
    key="Down",
    description="Select next keypoint",
    scope="points-layer",
    overwrite=True,
)

PREV_KEYPOINT = ShortcutSpec(
    key="Up",
    description="Select previous keypoint",
    scope="points-layer",
    overwrite=True,
)

NEXT_UNLABELED_RIGHT = ShortcutSpec(
    key="Shift-Right",
    description="Jump to first unlabeled frame",
    scope="points-layer",
)

NEXT_UNLABELED_LEFT = ShortcutSpec(
    key="Shift-Left",
    description="Jump to first unlabeled frame",
    scope="points-layer",
)

TOGGLE_EDGE_COLOR = ShortcutSpec(
    key="E",
    description="Toggle point edge color",
    scope="global-points",
)


def install_points_layer_keybindings(layer: Points, controls, store) -> None:
    """Bind per-layer keybindings for a DLC-managed Points layer."""
    layer.bind_key(CYCLE_LABEL_MODE.key, controls.cycle_through_label_modes)
    layer.bind_key(CYCLE_COLOR_MODE.key, controls.cycle_through_color_modes)

    layer.bind_key(NEXT_UNLABELED_RIGHT.key, store._find_first_unlabeled_frame)
    layer.bind_key(NEXT_UNLABELED_LEFT.key, store._find_first_unlabeled_frame)

    layer.bind_key(NEXT_KEYPOINT.key, store.next_keypoint, overwrite=NEXT_KEYPOINT.overwrite)
    layer.bind_key(PREV_KEYPOINT.key, store.prev_keypoint, overwrite=PREV_KEYPOINT.overwrite)


def toggle_edge_color(layer):
    """Toggle point border width between 0 and 2."""
    layer.border_width = np.bitwise_xor(layer.border_width, 2)


def install_global_points_keybindings() -> None:
    """Install Points-class-wide keybindings exactly once."""
    global _global_points_bindings_installed
    if _global_points_bindings_installed:
        return

    Points.bind_key(TOGGLE_EDGE_COLOR.key)(toggle_edge_color)
    _global_points_bindings_installed = True


def iter_shortcuts():
    """Return all known shortcuts for help / docs UI."""
    return [
        CYCLE_LABEL_MODE,
        CYCLE_COLOR_MODE,
        NEXT_KEYPOINT,
        PREV_KEYPOINT,
        NEXT_UNLABELED_RIGHT,
        NEXT_UNLABELED_LEFT,
        TOGGLE_EDGE_COLOR,
    ]
