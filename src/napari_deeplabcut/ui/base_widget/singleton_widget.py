from __future__ import annotations

import weakref
from typing import ClassVar

from qtpy.QtWidgets import QWidget


class ViewerSingletonWidget(QWidget):
    """Base QWidget enforcing at most one live instance per viewer per subclass."""

    _instances_by_cls: ClassVar[
        dict[type, weakref.WeakKeyDictionary[object, weakref.ReferenceType[ViewerSingletonWidget]]]
    ] = {}

    # ------------------------------------------------------------------ #
    # Viewer extraction / normalization                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_viewer_from_call(args, kwargs):
        if args:
            return args[0]
        if "napari_viewer" in kwargs:
            return kwargs["napari_viewer"]
        if "viewer" in kwargs:
            return kwargs["viewer"]
        return None

    @staticmethod
    def canonical_viewer(viewer):
        current = viewer
        seen = set()

        while True:
            wrapped = getattr(current, "__wrapped__", None)
            if wrapped is None:
                wrapped = getattr(current, "_obj", None)

            if wrapped is None or wrapped is current or id(wrapped) in seen:
                return current

            seen.add(id(current))
            current = wrapped

    # ------------------------------------------------------------------ #
    # Registry helpers                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _instance_registry(cls):
        reg = ViewerSingletonWidget._instances_by_cls.get(cls)
        if reg is None:
            reg = weakref.WeakKeyDictionary()
            ViewerSingletonWidget._instances_by_cls[cls] = reg
        return reg

    @staticmethod
    def _is_qt_alive(widget) -> bool:
        try:
            widget.objectName()  # any QObject call is enough
        except RuntimeError:
            return False
        except Exception:
            return True
        return True

    @classmethod
    def get_existing(cls, viewer):
        canonical = cls.canonical_viewer(viewer)
        ref = cls._instance_registry().get(canonical)
        widget = ref() if ref is not None else None
        if widget is None:
            return None
        if not cls._is_qt_alive(widget):
            cls._instance_registry().pop(canonical, None)
            return None
        return widget

    @classmethod
    def get_or_create(cls, viewer, *args, **kwargs):
        existing = cls.get_existing(viewer)
        if existing is not None:
            return existing
        return cls(viewer, *args, **kwargs)

    @classmethod
    def is_docked(cls, viewer, widget) -> bool:
        try:
            for obj in viewer.window.dock_widgets.values():
                if obj is widget:
                    return True
                try:
                    if obj.widget() is widget:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------ #
    # Singleton construction                                             #
    # ------------------------------------------------------------------ #

    def __new__(cls, *args, **kwargs):
        viewer = cls._extract_viewer_from_call(args, kwargs)
        if viewer is None:
            raise TypeError(
                f"{cls.__name__} requires a viewer argument (positional, napari_viewer=..., or viewer=...)."
            )

        canonical = cls.canonical_viewer(viewer)
        existing = cls.get_existing(canonical)
        if existing is not None:
            return existing

        obj = super().__new__(cls)
        cls._instance_registry()[canonical] = weakref.ref(obj)
        return obj

    def _singleton_prepare_init(self, *args, **kwargs) -> bool:
        """Pre-Qt-init guard. Safe to call before QWidget.__init__()."""
        if getattr(self, "_viewer_singleton_initialized", False):
            return False

        viewer = self._extract_viewer_from_call(args, kwargs)
        if viewer is None:
            raise TypeError(f"{self.__class__.__name__} requires a viewer argument during initialization.")

        self._viewer_singleton_initialized = True
        self._viewer_singleton_key = self.canonical_viewer(viewer)
        return True

    def _singleton_finalize_init(self) -> None:
        """Post-Qt-init finalization. Call after QWidget.__init__()."""
        # only connect once, and only after QObject exists
        if getattr(self, "_viewer_singleton_finalize_done", False):
            return
        self._viewer_singleton_finalize_done = True
        self.destroyed.connect(self._on_singleton_destroyed)

    def _on_singleton_destroyed(self, *args) -> None:
        key = getattr(self, "_viewer_singleton_key", None)
        if key is None:
            return

        reg = self.__class__._instance_registry()
        ref = reg.get(key)
        if ref is not None and ref() is self:
            reg.pop(key, None)
