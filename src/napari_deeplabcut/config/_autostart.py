from __future__ import annotations

import logging
from weakref import WeakSet

import napari
from napari.layers import Points
from napari.utils.events import Event
from qtpy.QtCore import QTimer

from napari_deeplabcut.config.settings import get_auto_open_keypoint_controls
from napari_deeplabcut.core.metadata import read_points_meta
from napari_deeplabcut.widget_factory import get_existing_keypoint_controls

logger = logging.getLogger(__name__)

# Track viewers where the observer has already been installed.
_INSTALLED_VIEWERS: WeakSet = WeakSet()


def _is_dlc_points_layer(layer) -> bool:
    """Return True if layer looks like a valid DLC Points layer."""
    if not isinstance(layer, Points):
        return False

    res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
    if hasattr(res, "errors"):
        return False
    return res.header is not None


def _ensure_keypoint_controls_open(viewer) -> None:
    """Open Keypoint controls dock widget if enabled in settings."""
    if viewer is None or not get_auto_open_keypoint_controls():
        return
    if get_existing_keypoint_controls(viewer) is not None:
        return
    try:
        # Public API: returns the existing widget if already docked.
        viewer.window.add_plugin_dock_widget(
            "napari-deeplabcut",
            "Keypoint controls",
        )
    except Exception:
        logger.debug("Failed to open Keypoint controls dock widget.", exc_info=True)
        napari.utils.notifications.show_info(
            "Failed to open Keypoint controls. Please open manually from the Plugins menu.",
        )


def _maybe_open_for_inserted_layer(viewer, layer) -> None:
    """Open controls when a qualifying DLC points layer is present."""
    if viewer is None or layer is None:
        return
    if not _is_dlc_points_layer(layer):
        return

    # Defer slightly to avoid re-entrancy during layer insertion.
    QTimer.singleShot(0, lambda: _ensure_keypoint_controls_open(viewer))


def maybe_install_keypoint_controls_autostart(viewer=None) -> None:
    """
    Install a per-viewer observer that auto-opens Keypoint controls when a valid
    DLC Points layer is inserted.

    Safe to call repeatedly; installation happens once per viewer.
    """
    if viewer is None:
        viewer = napari.current_viewer()
    if viewer is None:
        return

    if viewer in _INSTALLED_VIEWERS:
        return

    _INSTALLED_VIEWERS.add(viewer)

    def _on_insert(event: Event) -> None:
        try:
            layer = event.value if hasattr(event, "value") else event.source[-1]
        except Exception:
            layer = None
        _maybe_open_for_inserted_layer(viewer, layer)

    viewer.layers.events.inserted.connect(_on_insert)

    # Also scan already-present layers in case installation happens late.
    for layer in list(viewer.layers):
        if _is_dlc_points_layer(layer):
            QTimer.singleShot(0, lambda v=viewer: _ensure_keypoint_controls_open(v))
            break
