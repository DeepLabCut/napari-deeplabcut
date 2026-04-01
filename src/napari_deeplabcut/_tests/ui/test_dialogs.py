from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from napari.layers import Image, Points
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton, QScrollArea

from napari_deeplabcut.config.keybinds import iter_shortcuts
from napari_deeplabcut.ui.dialogs import (
    OverwriteConflictsDialog,
    ShortcutRow,
    Shortcuts,
    Tutorial,
    maybe_confirm_overwrite,
)

# -----------------------------------------------------------------------------
# Shortcuts
# -----------------------------------------------------------------------------


class _DummyEmitter:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def disconnect(self, callback):
        self._callbacks.remove(callback)

    def emit(self, event=None):
        for callback in list(self._callbacks):
            callback(event)


class _DummyViewer:
    def __init__(self, *, active=None):
        self.layers = SimpleNamespace(
            selection=SimpleNamespace(
                active=active,
                events=SimpleNamespace(active=_DummyEmitter()),
            )
        )


def _points_layer(name: str = "points"):
    return Points(np.empty((0, 2)), name=name)


def _image_layer(name: str = "image"):
    return Image(np.zeros((8, 8)), name=name)


def test_shortcuts_dialog_without_viewer_renders_registry(dialog_parent, qtbot):
    dlg = Shortcuts(dialog_parent)
    qtbot.addWidget(dlg)

    assert dlg.parent() is dialog_parent
    assert dlg.windowTitle() == "Keyboard shortcuts"
    assert dlg.testAttribute(Qt.WA_DeleteOnClose)

    scroll_areas = dlg.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    rows = dlg.findChildren(ShortcutRow)
    assert len(rows) == len(tuple(iter_shortcuts()))

    assert (
        dlg.context_banner.text()
        == "Showing all known shortcuts. Availability cannot be determined without a viewer context."
    )


def test_shortcuts_dialog_marks_points_shortcuts_unavailable_for_non_points_layer(dialog_parent, qtbot):
    viewer = _DummyViewer(active=_image_layer("raw image"))
    dlg = Shortcuts(dialog_parent, viewer=viewer)
    qtbot.addWidget(dlg)

    assert "No active <b>Points</b> layer" in dlg.context_banner.text()

    rows = dlg.findChildren(ShortcutRow)
    points_rows = [row for row in rows if row.spec.scope == "points-layer"]
    global_rows = [row for row in rows if row.spec.scope == "global-points"]

    assert points_rows, "expected at least one points-layer shortcut row"
    assert global_rows, "expected at least one global-points shortcut row"

    for row in points_rows:
        assert row.graphicsEffect().opacity() == 0.45
        assert "No active Points layer." in row.toolTip()

    for row in global_rows:
        assert row.graphicsEffect().opacity() == 1.0
        assert "No active Points layer." not in row.toolTip()


def test_shortcuts_dialog_updates_when_active_layer_changes_and_escapes_layer_name(dialog_parent, qtbot):
    viewer = _DummyViewer(active=_image_layer("raw image"))
    dlg = Shortcuts(dialog_parent, viewer=viewer)
    qtbot.addWidget(dlg)

    # Start with a non-Points layer: points shortcuts are dimmed.
    row = next(row for row in dlg.findChildren(ShortcutRow) if row.spec.scope == "points-layer")
    assert row.graphicsEffect().opacity() == 0.45
    assert "No active Points layer." in row.toolTip()

    # Switch to a Points layer with characters that must be escaped in rich text.
    viewer.layers.selection.active = _points_layer("a <bizarre> & layer")
    viewer.layers.selection.events.active.emit()

    banner = dlg.context_banner.text()
    assert "Active Points layer:" in banner
    assert "&lt;bizarre&gt;" in banner
    assert "&amp; layer" in banner
    assert "a <bizarre> & layer" not in banner  # raw rich-text-breaking text should not appear

    assert row.graphicsEffect().opacity() == 1.0
    assert "No active Points layer." not in row.toolTip()


def test_shortcuts_dialog_disconnects_from_viewer_on_close(dialog_parent, qtbot):
    viewer = _DummyViewer(active=_image_layer("raw image"))
    emitter = viewer.layers.selection.events.active

    dlg = Shortcuts(dialog_parent, viewer=viewer)
    qtbot.addWidget(dlg)

    assert len(emitter._callbacks) == 1

    dlg.show()
    dlg.close()
    qtbot.wait(0)

    assert emitter._callbacks == []


# -----------------------------------------------------------------------------
# Tutorial
# -----------------------------------------------------------------------------


def test_tutorial_initial_state(dialog_parent, qtbot):
    dlg = Tutorial(dialog_parent)
    qtbot.addWidget(dlg)

    assert dlg.parent() is dialog_parent
    assert dlg.isModal()
    assert dlg._current_tip == -1
    assert dlg.count.text() == ""

    # initial nav state with "intro" screen before first tip
    assert not dlg.prev_button.isEnabled()
    assert dlg.next_button.isEnabled()

    assert dlg.message.openExternalLinks()
    assert dlg.message.textInteractionFlags() & Qt.LinksAccessibleByMouse


def test_tutorial_next_advances_to_first_tip_and_updates_position(dialog_parent, qtbot):
    dlg = Tutorial(dialog_parent)
    qtbot.addWidget(dlg)

    qtbot.mouseClick(dlg.next_button, Qt.LeftButton)

    assert dlg._current_tip == 0
    assert dlg.count.text() == f"Tip 1|{len(dlg._tips)}"

    # first real tip still has prev disabled, next enabled
    assert not dlg.prev_button.isEnabled()
    assert dlg.next_button.isEnabled()

    xrel, yrel = dlg._tips[0].pos
    geom = dialog_parent.geometry()
    expected = QPoint(
        int(geom.left() + geom.width() * xrel),
        int(geom.top() + geom.height() * yrel),
    )
    assert dlg.pos() == expected


def test_tutorial_navigation_enables_and_disables_buttons(dialog_parent, qtbot):
    dlg = Tutorial(dialog_parent)
    qtbot.addWidget(dlg)

    qtbot.mouseClick(dlg.next_button, Qt.LeftButton)  # tip 1
    qtbot.mouseClick(dlg.next_button, Qt.LeftButton)  # tip 2

    assert dlg._current_tip == 1
    assert dlg.prev_button.isEnabled()
    assert dlg.next_button.isEnabled()
    assert dlg.count.text() == f"Tip 2|{len(dlg._tips)}"

    qtbot.mouseClick(dlg.prev_button, Qt.LeftButton)
    assert dlg._current_tip == 0
    assert not dlg.prev_button.isEnabled()
    assert dlg.next_button.isEnabled()


def test_tutorial_last_tip_has_no_emoji_prefix_and_disables_next(dialog_parent, qtbot):
    dlg = Tutorial(dialog_parent)
    qtbot.addWidget(dlg)

    for _ in range(len(dlg._tips)):
        qtbot.mouseClick(dlg.next_button, Qt.LeftButton)

    assert dlg._current_tip == len(dlg._tips) - 1
    assert dlg.prev_button.isEnabled()
    assert not dlg.next_button.isEnabled()

    # last tip should not be prefixed with the emoji
    assert "napari-deeplabcut" in dlg.message.text()


# -----------------------------------------------------------------------------
# OverwriteConflictsDialog
# -----------------------------------------------------------------------------


def test_overwrite_conflicts_dialog_smoke(dialog_parent, qtbot):
    dlg = OverwriteConflictsDialog(
        dialog_parent,
        title="Overwrite warning",
        summary="Saving will overwrite existing keypoints in the destination file.",
        layer_text="points",
        dest_text="/tmp/labels.h5",
        affected_text="3 keypoint overwrite(s) across 2 frame(s)/image(s).",
        details="img001.png -> nose, tail\nimg002.png -> paw",
    )
    qtbot.addWidget(dlg)

    assert dlg.windowTitle() == "Overwrite warning"
    assert dlg.isModal()
    assert dlg.text is not None
    assert isinstance(dlg.text, QPlainTextEdit)
    assert dlg.text.isReadOnly()
    assert dlg.text.toPlainText() == "img001.png -> nose, tail\nimg002.png -> paw"

    assert isinstance(dlg.cancel_btn, QPushButton)
    assert isinstance(dlg.overwrite_btn, QPushButton)
    assert dlg.overwrite_btn.isDefault()
    assert dlg.overwrite_btn.autoDefault()

    labels = [w.text() for w in dlg.findChildren(QLabel)]
    assert any("Saving will overwrite existing keypoints" in text for text in labels)
    assert any("<b>Layer:</b> points" == text for text in labels)
    assert any("<b>Destination:</b> /tmp/labels.h5" == text for text in labels)
    assert any("<b>Affected:</b> 3 keypoint overwrite(s) across 2 frame(s)/image(s)." == text for text in labels)
    assert any("Conflicts (frame/image" in text for text in labels)


def test_overwrite_conflicts_dialog_cancel_button_rejects(dialog_parent, qtbot):
    dlg = OverwriteConflictsDialog(
        dialog_parent,
        title="Overwrite warning",
        summary="summary",
        layer_text="layer",
        dest_text="dest",
        affected_text="affected",
        details="details",
    )
    qtbot.addWidget(dlg)

    dlg.show()
    qtbot.mouseClick(dlg.cancel_btn, Qt.LeftButton)

    assert dlg.result() == QDialog.Rejected


def test_overwrite_conflicts_dialog_overwrite_button_accepts(dialog_parent, qtbot):
    dlg = OverwriteConflictsDialog(
        dialog_parent,
        title="Overwrite warning",
        summary="summary",
        layer_text="layer",
        dest_text="dest",
        affected_text="affected",
        details="details",
    )
    qtbot.addWidget(dlg)

    dlg.show()
    qtbot.mouseClick(dlg.overwrite_btn, Qt.LeftButton)

    assert dlg.result() == QDialog.Accepted


#  Confirm
def test_overwrite_conflicts_dialog_confirm_returns_true_on_accept(monkeypatch, dialog_parent):
    def fake_exec(self):
        return QDialog.Accepted

    monkeypatch.setattr(OverwriteConflictsDialog, "exec_", fake_exec)

    result = OverwriteConflictsDialog.confirm(
        dialog_parent,
        summary="summary",
        layer_text="layer",
        dest_text="dest",
        affected_text="affected",
        details="details",
        title="Overwrite warning",
    )

    assert result is True


def test_overwrite_conflicts_dialog_confirm_returns_false_on_reject(monkeypatch, dialog_parent):
    def fake_exec(self):
        return QDialog.Rejected

    monkeypatch.setattr(OverwriteConflictsDialog, "exec_", fake_exec)

    result = OverwriteConflictsDialog.confirm(
        dialog_parent,
        summary="summary",
        layer_text="layer",
        dest_text="dest",
        affected_text="affected",
        details="details",
        title="Overwrite warning",
    )

    assert result is False


# -----------------------------------------------------------------------------
# maybe_confirm_overwrite
# -----------------------------------------------------------------------------


def test_maybe_confirm_overwrite_returns_true_when_no_conflicts(monkeypatch, dialog_parent):
    report = SimpleNamespace(
        has_conflicts=False,
        layer_name="layer",
        destination_path="/tmp/file.h5",
        n_overwrites=0,
        n_frames=0,
        details_text="",
    )

    called = []

    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.get_overwrite_confirmation_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.OverwriteConflictsDialog.confirm",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    result = maybe_confirm_overwrite(dialog_parent, report)

    assert result is True
    assert called == []


def test_maybe_confirm_overwrite_returns_true_when_confirmation_disabled(monkeypatch, dialog_parent):
    report = SimpleNamespace(
        has_conflicts=True,
        layer_name="layer",
        destination_path="/tmp/file.h5",
        n_overwrites=3,
        n_frames=2,
        details_text="details",
    )

    called = []

    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.get_overwrite_confirmation_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.OverwriteConflictsDialog.confirm",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    result = maybe_confirm_overwrite(dialog_parent, report)

    assert result is True
    assert called == []


def test_maybe_confirm_overwrite_delegates_to_confirm(monkeypatch, dialog_parent):
    report = SimpleNamespace(
        has_conflicts=True,
        layer_name="pose-layer",
        destination_path="/tmp/labels.h5",
        n_overwrites=3,
        n_frames=2,
        details_text="img001.png -> nose, tail",
    )

    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.get_overwrite_confirmation_enabled",
        lambda: True,
    )

    captured = {}

    def fake_confirm(parent, **kwargs):
        captured["parent"] = parent
        captured["kwargs"] = kwargs
        return False

    monkeypatch.setattr(
        "napari_deeplabcut.ui.dialogs.OverwriteConflictsDialog.confirm",
        fake_confirm,
    )

    result = maybe_confirm_overwrite(dialog_parent, report)

    assert result is False
    assert captured["parent"] is dialog_parent
    assert captured["kwargs"] == {
        "summary": "Saving will overwrite existing keypoints in the destination file.",
        "layer_text": "pose-layer",
        "dest_text": "/tmp/labels.h5",
        "affected_text": "3 keypoint overwrite(s) across 2 frame(s)/image(s).",
        "details": "img001.png -> nose, tail",
    }
