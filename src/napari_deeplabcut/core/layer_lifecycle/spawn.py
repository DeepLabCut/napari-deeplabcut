# src/napari_deeplabcut/core/layer_lifecycle/spawn.py
from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject

from ...napari_compat.proxy import unwrap
from .manager import LayerLifecycleManager

if TYPE_CHECKING:
    import napari

_MANAGER_REGISTRY: weakref.WeakKeyDictionary[object, LayerLifecycleManager] = weakref.WeakKeyDictionary()
_MANAGER_LOCK = threading.RLock()


def _viewer_qparent(viewer: napari.Viewer) -> QObject | None:
    """Qt parent for the manager. Expects an already-unwrapped viewer.

    Reached through a PublicOnlyProxy, ``_qt_window`` comes back wrapped, and wrapt
    spoofs ``__class__`` so the isinstance check below still passes -- but
    ``QObject(parent=<proxy>)`` then raises on PySide6.
    """
    try:
        window = getattr(viewer, "window", None)
        qt_window = getattr(window, "_qt_window", None)
        return qt_window if isinstance(qt_window, QObject) else None
    except Exception:
        return None


def get_layer_manager(viewer: napari.Viewer) -> LayerLifecycleManager | None:
    viewer = unwrap(viewer)
    with _MANAGER_LOCK:
        return _MANAGER_REGISTRY.get(viewer)


def get_or_create_layer_manager(viewer: napari.Viewer) -> LayerLifecycleManager:
    viewer = unwrap(viewer)
    with _MANAGER_LOCK:
        mgr = get_layer_manager(viewer)
        if mgr is not None:
            return mgr

        mgr = LayerLifecycleManager(
            viewer=viewer,
            parent=_viewer_qparent(viewer),
        )
        mgr.attach()

        _MANAGER_REGISTRY[viewer] = mgr
        return mgr
