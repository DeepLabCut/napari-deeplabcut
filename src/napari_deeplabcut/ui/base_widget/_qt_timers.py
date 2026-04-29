# src/napari_deeplabcut/ui/base_widget/_qt_timers.py
from __future__ import annotations

from qtpy.QtCore import QTimer


class OwnedTimersMixin:
    """Reusable QObject-owned timer helpers.

    Requirements
    ------------
    - `self` must already be a live QObject when `_init_owned_timers()` is called.
    - The mixin does not subclass QObject itself; it only assumes QObject behavior.
    """

    def _init_owned_timers(self) -> None:
        if getattr(self, "_owned_timers_initialized", False):
            return

        self._owned_timers_initialized = True
        self._timers: dict[str, QTimer] = {}
        self._temp_timers: set[QTimer] = set()

        try:
            self.destroyed.connect(self._cleanup_owned_timers)
        except Exception:
            # Defensive: if destroyed is unavailable during odd proxy/test states,
            # explicit cleanup can still be called manually.
            pass

    def _cleanup_owned_timers(self, *_args) -> None:
        """Stop/delete all timers owned by this QObject."""
        for timer in list(getattr(self, "_timers", {}).values()):
            try:
                timer.stop()
                timer.deleteLater()
            except RuntimeError:
                pass
        self._timers = {}

        for timer in list(getattr(self, "_temp_timers", set())):
            try:
                timer.stop()
                timer.deleteLater()
            except RuntimeError:
                pass
        if hasattr(self, "_temp_timers"):
            self._temp_timers.clear()

    def _schedule_once(self, name: str, msec: int, callback) -> None:
        """Schedule/coalesce a named single-shot callback owned by this QObject."""
        timer = self._timers.get(name)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(callback)
            self._timers[name] = timer

        try:
            timer.start(msec)
        except RuntimeError:
            # QObject/timer already tearing down
            pass

    def _single_shot_owned(self, msec: int, callback) -> None:
        """Schedule a one-off callback via a child QTimer tracked by this QObject."""
        timer = QTimer(self)
        timer.setSingleShot(True)
        self._temp_timers.add(timer)

        def _fire():
            try:
                callback()
            finally:
                self._temp_timers.discard(timer)
                try:
                    timer.deleteLater()
                except RuntimeError:
                    pass

        timer.timeout.connect(_fire)

        try:
            timer.start(msec)
        except RuntimeError:
            self._temp_timers.discard(timer)

    def _cancel_scheduled(self, name: str) -> None:
        timer = getattr(self, "_timers", {}).get(name)
        if timer is None:
            return
        try:
            timer.stop()
        except RuntimeError:
            pass
