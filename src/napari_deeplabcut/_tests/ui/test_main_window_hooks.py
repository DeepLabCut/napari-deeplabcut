"""Coverage for the hooks KeypointControls installs into napari's main window."""

from __future__ import annotations

from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QMainWindow


def _hooked_window(keypoint_controls, qtbot) -> QMainWindow:
    """Wait for the deferred hook to run and return the main window it attached to."""
    qtbot.waitUntil(lambda: keypoint_controls._main_window_hooked, timeout=5_000)

    window = keypoint_controls._main_window()
    assert window is not None, "widget is docked, so window() should be napari's main window"
    return window


def test_main_window_is_reached_through_the_qt_parent_chain(keypoint_controls, qtbot):
    window = _hooked_window(keypoint_controls, qtbot)

    assert isinstance(window, QMainWindow)
    assert window is keypoint_controls.window()


def test_last_saved_label_is_added_to_the_status_bar(keypoint_controls, qtbot):
    window = _hooked_window(keypoint_controls, qtbot)

    label = keypoint_controls.last_saved_label
    assert window.statusBar().isAncestorOf(label), "label should be parented into the status bar"
    # Unparented it would show as its own top-level window instead of in the status bar.
    assert not label.isWindow()


def test_hooks_are_installed_only_once(keypoint_controls, qtbot):
    window = _hooked_window(keypoint_controls, qtbot)
    close_event_handler = window.closeEvent

    keypoint_controls._ensure_main_window_hooks()

    assert window.closeEvent is close_event_handler, "second call should not re-wrap the handler"


def test_closing_the_window_routes_through_on_close(keypoint_controls, qtbot, monkeypatch):
    window = _hooked_window(keypoint_controls, qtbot)

    seen = []

    def fake_on_close(event):
        seen.append(event)
        # Ignore it, or the wrapper falls through to napari's own handler and really
        # closes the viewer mid-test.
        event.ignore()

    monkeypatch.setattr(keypoint_controls, "on_close", fake_on_close)

    window.closeEvent(QCloseEvent())

    assert len(seen) == 1
