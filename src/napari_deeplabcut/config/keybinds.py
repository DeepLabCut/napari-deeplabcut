"""Central registry and installers for napari-deeplabcut keybindings (source of truth)."""

# src/napari_deeplabcut/config/keybinds.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from napari import Viewer
from napari.components._viewer_key_bindings import (
    increment_dims_left,
    increment_dims_right,
)
from napari.layers import Points
from qtpy.QtCore import QTimer

from .settings import TRACKING_SHORTCUTS_ENABLED

_global_points_bindings_installed = False


@dataclass(frozen=True)
class BindingContext:
    controls: object
    store: object
    viewer: object | None = (
        None
        # only needed for viewer-scoped keybinds,
        # can be left out for points-layer-scoped keybinds
    )


@dataclass(frozen=True)
class ShortcutSpec:
    keys: tuple[str, ...]
    description: str
    group: str
    scope: str
    action: ShortcutAction | None = None  # optional enum for programmatic reference
    get_callback: Callable[[BindingContext], Callable] | None = None
    overwrite: bool = False
    when: str | None = None  # optional UI note, e.g. "Multi-animal layers only"


class ShortcutAction(Enum):
    CYCLE_LABEL_MODE = auto()
    CYCLE_COLOR_MODE = auto()
    NEXT_KEYPOINT = auto()
    PREV_KEYPOINT = auto()
    JUMP_UNLABELED_FRAME = auto()
    TOGGLE_EDGE_COLOR = auto()
    NEXT_FRAME = auto()
    PREV_FRAME = auto()


# ----------------------------------------
#  Functions with associated keybind callbacks
# ----------------------------------------

_FRAME_REPEAT_INTERVAL_MS = 60
_frame_repeat_timers: dict[tuple[int, str], QTimer] = {}


def _viewer_from_callback_arg(ctx: BindingContext, obj):
    """
    Get the viewer either from context or from the callback argument itself.
    Requires a napari action that uses viewer as the callback argument.
    """
    if ctx.viewer is not None:
        return ctx.viewer

    if isinstance(obj, Viewer):
        return obj

    return None


def _make_repeating_viewer_callback(ctx: BindingContext, action, repeat_id: str):
    """
    Call a napari viewer action once, then continue calling it while the key is held.

    This reuses napari's own increment_dims_* functions, but adds hold-to-repeat
    for A/D, which napari otherwise filters as non-navigation autorepeat keys.
    """

    def callback(obj):
        viewer = _viewer_from_callback_arg(ctx, obj)
        if viewer is None:
            return

        timer_key = (id(viewer), repeat_id)

        # Avoid duplicate timers if repeat key-press events sneak through.
        if timer_key in _frame_repeat_timers:
            return

        # Move once immediately.
        action(viewer)

        timer = QTimer()
        timer.setInterval(_FRAME_REPEAT_INTERVAL_MS)
        timer.timeout.connect(lambda: action(viewer))

        _frame_repeat_timers[timer_key] = timer
        timer.start()

        try:
            yield
        finally:
            timer.stop()
            timer.deleteLater()
            _frame_repeat_timers.pop(timer_key, None)

    return callback


def _prev_frame(ctx: BindingContext):
    return _make_repeating_viewer_callback(
        ctx,
        increment_dims_left,
        "prev_frame",
    )


def _next_frame(ctx: BindingContext):
    return _make_repeating_viewer_callback(
        ctx,
        increment_dims_right,
        "next_frame",
    )


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
        keys=("Down", "S"),
        action=ShortcutAction.NEXT_KEYPOINT,
        get_callback=_next_keypoint,
        description="Select next keypoint",
        group="Navigation",
        scope="points-layer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("Up", "W"),
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
    ShortcutSpec(
        keys=("A",),
        action=ShortcutAction.PREV_FRAME,
        get_callback=_prev_frame,
        description="Previous frame",
        group="Navigation",
        scope="viewer",
        overwrite=True,
    ),
    ShortcutSpec(
        keys=("D",),
        action=ShortcutAction.NEXT_FRAME,
        get_callback=_next_frame,
        description="Next frame",
        group="Navigation",
        scope="viewer",
        overwrite=True,
    ),
)

# --------------------------------
# Tracking shortcuts
# --------------------------------


@dataclass(frozen=True)
class TrackingKeybindConfig:
    key: str
    tooltip: str

    def get_display(self) -> str:
        txt = self.tooltip
        if TRACKING_SHORTCUTS_ENABLED:
            txt += f" ({self.key})"
        return txt


TRACK_FORWARD = TrackingKeybindConfig(key="L", tooltip="Track forward")
TRACK_FORWARD_END = TrackingKeybindConfig(key="K", tooltip="Track forward to end")
TRACK_BACKWARD = TrackingKeybindConfig(key="H", tooltip="Track backward")
TRACK_BACKWARD_END = TrackingKeybindConfig(key="J", tooltip="Track backward to start")
MOVE_FORWARD_FRAME = TrackingKeybindConfig(key="I", tooltip="Move forward one frame")
MOVE_BACKWARD_FRAME = TrackingKeybindConfig(key="U", tooltip="Move backward one frame")


TRACKING_SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec(
        keys=(TRACK_FORWARD.key,),
        description=TRACK_FORWARD.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
    ShortcutSpec(
        keys=(TRACK_FORWARD_END.key,),
        description=TRACK_FORWARD_END.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
    ShortcutSpec(
        keys=(TRACK_BACKWARD.key,),
        description=TRACK_BACKWARD.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
    ShortcutSpec(
        keys=(TRACK_BACKWARD_END.key,),
        description=TRACK_BACKWARD_END.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
    ShortcutSpec(
        keys=(MOVE_FORWARD_FRAME.key,),
        description=MOVE_FORWARD_FRAME.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
    ShortcutSpec(
        keys=(MOVE_BACKWARD_FRAME.key,),
        description=MOVE_BACKWARD_FRAME.tooltip,
        group="Tracking",
        scope="tracking-points-layer",
        when="Tracking widget is open",
    ),
)

# ----- Keybind registry functions ------


def iter_shortcuts() -> Iterable[ShortcutSpec]:
    yield from SHORTCUTS
    if TRACKING_SHORTCUTS_ENABLED:
        yield from TRACKING_SHORTCUTS


def _bind_each_key(layer: Points, keys: tuple[str, ...], callback, *, overwrite: bool = False) -> None:
    for key in keys:
        layer.bind_key(key, callback, overwrite=overwrite)


def install_points_layer_keybindings(layer: Points, controls, store, viewer=None) -> None:
    ctx = BindingContext(controls=controls, store=store, viewer=viewer)

    for spec in SHORTCUTS:
        if spec.get_callback is None:
            continue

        if spec.scope == "points-layer":
            callback = spec.get_callback(ctx)
            _bind_each_key(layer, spec.keys, callback, overwrite=spec.overwrite)

        elif spec.scope == "viewer" and viewer is not None:
            callback = spec.get_callback(ctx)
            _bind_each_key(layer, spec.keys, callback, overwrite=spec.overwrite)


def install_viewer_keybindings(viewer, controls=None, store=None) -> None:
    # Still needed so A/D work without a Points layer
    ctx = BindingContext(controls=controls, store=store, viewer=viewer)

    for spec in SHORTCUTS:
        if spec.scope != "viewer" or spec.get_callback is None:
            continue

        callback = spec.get_callback(ctx)

        for key in spec.keys:
            viewer.bind_key(key, callback, overwrite=spec.overwrite)


# ------- Global keybinds that apply to all points layers, e.g. toggling edge color -------


def toggle_edge_color(layer):
    layer.border_width = np.bitwise_xor(layer.border_width, 2)


def install_global_points_keybindings() -> None:
    global _global_points_bindings_installed
    if _global_points_bindings_installed:
        return

    for spec in SHORTCUTS:
        if spec.scope == "global-points" and spec.action == ShortcutAction.TOGGLE_EDGE_COLOR:
            for key in spec.keys:
                Points.bind_key(key)(toggle_edge_color)

    _global_points_bindings_installed = True
