# src/napari_deeplabcut/core/layer_lifecycle/spawn.py
from __future__ import annotations

import threading
import weakref

from qtpy.QtCore import QObject

from .manager import LayerLifecycleManager

_MANAGER_REGISTRY: weakref.WeakKeyDictionary[object, LayerLifecycleManager] = weakref.WeakKeyDictionary()
_MANAGER_LOCK = threading.RLock()
_VIEWER_ATTR = "_ndlc_layer_manager"


def _viewer_qparent(viewer) -> QObject | None:
    try:
        window = getattr(viewer, "window", None)
        qt_window = getattr(window, "_qt_window", None)
        return qt_window if isinstance(qt_window, QObject) else None
    except Exception:
        return None


def get_layer_manager(viewer) -> LayerLifecycleManager | None:
    with _MANAGER_LOCK:
        mgr = _MANAGER_REGISTRY.get(viewer)
        if mgr is not None:
            return mgr

        try:
            mgr = getattr(viewer, _VIEWER_ATTR, None)
        except Exception:
            mgr = None

        if mgr is not None and getattr(mgr, "viewer", None) is viewer:
            _MANAGER_REGISTRY[viewer] = mgr
            return mgr

        return None


def get_or_create_layer_manager(viewer) -> LayerLifecycleManager:
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
        try:
            setattr(viewer, _VIEWER_ATTR, mgr)
        except Exception:
            pass

        return mgr
