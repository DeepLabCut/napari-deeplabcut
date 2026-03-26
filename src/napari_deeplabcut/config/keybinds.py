"""Future file for centralizing keybinds."""

# src/napari_deeplabcut/config/keybinds.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from napari.layers import Points

_global_points_bindings_installed = False


@dataclass(frozen=True)
class ShortcutSpec:
    keys: tuple[str, ...]
    description: str
    group: str
    scope: str
    overwrite: bool = False
    when: str | None = None  # optional UI note, e.g. "Multi-animal layers only"


# ---- Single source of truth for displayed shortcuts ----

SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec(
        keys=("M",),
        description="Change labeling mode",
        group="Annotation",
        scope="points-layer",
    ),
    ShortcutSpec(
        keys=("F",),
        description="Change color mode",
        group="Display",
        scope="points-layer",
        when="Only cycles beyond bodypart mode for multi-animal layers",
    ),
    ShortcutSpec(
        keys=("Down",),
        description="Select next keypoint",
        group="Navigation",
        scope="points-layer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("Up",),
        description="Select previous keypoint",
        group="Navigation",
        scope="points-layer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("Shift-Right", "Shift-Left"),
        description="Jump to first unlabeled frame",
        group="Navigation",
        scope="points-layer",
    ),
    ShortcutSpec(
        keys=("E",),
        description="Toggle point edge color",
        group="Display",
        scope="global-points",
    ),
)


def iter_shortcuts() -> Iterable[ShortcutSpec]:
    return SHORTCUTS


def _bind_each_key(layer: Points, keys: tuple[str, ...], callback, *, overwrite: bool = False) -> None:
    for key in keys:
        layer.bind_key(key, callback, overwrite=overwrite)


def install_points_layer_keybindings(layer: Points, controls, store) -> None:
    _bind_each_key(layer, ("M",), controls.cycle_through_label_modes)
    _bind_each_key(layer, ("F",), controls.cycle_through_color_modes)
    _bind_each_key(layer, ("Shift-Right", "Shift-Left"), store._find_first_unlabeled_frame)
    _bind_each_key(layer, ("Down",), store.next_keypoint, overwrite=True)
    _bind_each_key(layer, ("Up",), store.prev_keypoint, overwrite=True)


def toggle_edge_color(layer):
    layer.border_width = np.bitwise_xor(layer.border_width, 2)


def install_global_points_keybindings() -> None:
    global _global_points_bindings_installed
    if _global_points_bindings_installed:
        return

    for key in ("E",):
        Points.bind_key(key)(toggle_edge_color)

    _global_points_bindings_installed = True
