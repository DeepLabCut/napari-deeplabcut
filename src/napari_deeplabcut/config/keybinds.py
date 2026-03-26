"""Future file for centralizing keybinds."""

# src/napari_deeplabcut/config/keybinds.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from napari.layers import Points

_global_points_bindings_installed = False


@dataclass(frozen=True)
class BindingContext:
    controls: object
    store: object


@dataclass(frozen=True)
class ShortcutSpec:
    keys: tuple[str, ...]
    action: ShortcutAction | None = None  # optional enum for programmatic reference
    get_callback: Callable[[BindingContext], Callable] | None = None
    description: str
    group: str
    scope: str
    overwrite: bool = False
    when: str | None = None  # optional UI note, e.g. "Multi-animal layers only"


class ShortcutAction(Enum):
    CYCLE_LABEL_MODE = auto()
    CYCLE_COLOR_MODE = auto()
    NEXT_KEYPOINT = auto()
    PREV_KEYPOINT = auto()
    JUMP_UNLABELED_FRAME = auto()
    TOGGLE_EDGE_COLOR = auto()


# ----------------------------------------
#  Functions with associated keybind callbacks
# ----------------------------------------
def _cycle_label_mode(ctx: BindingContext):
    return ctx.controls.cycle_through_label_modes


def _cycle_color_mode(ctx: BindingContext):
    return ctx.controls.cycle_through_color_modes


def _next_keypoint(ctx: BindingContext):
    return ctx.store.next_keypoint


def _prev_keypoint(ctx: BindingContext):
    return ctx.store.prev_keypoint


def _jump_unlabeled_frame(ctx: BindingContext):
    return ctx.store._find_first_unlabeled_frame


# ---- Single source of truth for displayed shortcuts ----

SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec(
        keys=("M",),
        action=ShortcutAction.CYCLE_LABEL_MODE,
        get_callback=_cycle_label_mode,
        description="Change labeling mode",
        group="Annotation",
        scope="points-layer",
    ),
    ShortcutSpec(
        keys=("F",),
        action=ShortcutAction.CYCLE_COLOR_MODE,
        get_callback=_cycle_color_mode,
        description="Change color mode",
        group="Display",
        scope="points-layer",
        when="Only cycles beyond bodypart mode for multi-animal layers",
    ),
    ShortcutSpec(
        keys=("Down",),
        action=ShortcutAction.NEXT_KEYPOINT,
        get_callback=_next_keypoint,
        description="Select next keypoint",
        group="Navigation",
        scope="points-layer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("Up",),
        action=ShortcutAction.PREV_KEYPOINT,
        get_callback=_prev_keypoint,
        description="Select previous keypoint",
        group="Navigation",
        scope="points-layer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("Shift-Right", "Shift-Left"),
        action=ShortcutAction.JUMP_UNLABELED_FRAME,
        get_callback=_jump_unlabeled_frame,
        description="Jump to first unlabeled frame",
        group="Navigation",
        scope="points-layer",
    ),
    ShortcutSpec(
        keys=("E",),
        action=ShortcutAction.TOGGLE_EDGE_COLOR,
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
    ctx = BindingContext(controls=controls, store=store)

    for spec in SHORTCUTS:
        if spec.scope != "points-layer" or spec.get_callback is None:
            continue

        callback = spec.get_callback(ctx)
        _bind_each_key(layer, spec.keys, callback, overwrite=spec.overwrite)


# ------- Global keybinds that apply to all points layers, e.g. toggling edge color -------


def toggle_edge_color(layer):
    layer.border_width = np.bitwise_xor(layer.border_width, 2)


def install_global_points_keybindings() -> None:
    global _global_points_bindings_installed
    if _global_points_bindings_installed:
        return

    for key in ("E",):
        Points.bind_key(key)(toggle_edge_color)

    _global_points_bindings_installed = True
