from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from napari_deeplabcut.ui.debug_window import (
    DebugTextWindow,
    make_issue_report_provider,
    make_log_text_provider,
)


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def render_text(self, *, limit: int = 0) -> str:
        self.calls.append(limit)
        return f"rendered:{limit}"


def test_make_log_text_provider_uses_recorder_limit():
    recorder = FakeRecorder()
    provider = make_log_text_provider(recorder=recorder, limit=123)

    text = provider()

    assert text == "rendered:123"
    assert recorder.calls == [123]


def test_make_log_text_provider_handles_missing_recorder():
    provider = make_log_text_provider(recorder=None, limit=5)
    assert provider() == "<debug recorder unavailable>"


def test_make_issue_report_provider_calls_build_debug_report(monkeypatch):
    calls = []

    def fake_build_debug_report(*, viewer, recorder, log_limit):
        calls.append((viewer, recorder, log_limit))
        return "full-report"

    monkeypatch.setattr(
        "napari_deeplabcut.ui.debug_window.build_debug_report",
        fake_build_debug_report,
    )

    viewer = object()
    recorder = object()
    provider = make_issue_report_provider(viewer=viewer, recorder=recorder, log_limit=77)

    text = provider()

    assert text == "full-report"
    assert calls == [(viewer, recorder, 77)]


@pytest.mark.usefixtures("qtbot")
def test_debug_window_shows_initial_text(qtbot):
    provider_calls = []

    def provider():
        provider_calls.append("called")
        return "hello debug world"

    window = DebugTextWindow(
        title="Debug",
        text_provider=provider,
        initial_hint="hint",
    )
    qtbot.addWidget(window)
    window.show()

    assert provider_calls == ["called"]
    assert window.windowTitle() == "Debug"
    assert "hello debug world" in window._text_edit.toPlainText()
    assert window._text_edit.isReadOnly()


@pytest.mark.usefixtures("qtbot")
def test_debug_window_refresh_reloads_text(qtbot):
    state = {"n": 0}

    def provider():
        state["n"] += 1
        return f"value:{state['n']}"

    window = DebugTextWindow(
        title="Debug",
        text_provider=provider,
    )
    qtbot.addWidget(window)
    window.show()

    assert window._text_edit.toPlainText() == "value:1"

    qtbot.mouseClick(window._refresh_btn, Qt.LeftButton)
    assert window._text_edit.toPlainText() == "value:2"


@pytest.mark.usefixtures("qtbot")
def test_debug_window_copy_to_clipboard(qtbot):
    def provider():
        return "copy me"

    window = DebugTextWindow(
        title="Debug",
        text_provider=provider,
    )
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window._copy_btn, Qt.LeftButton)

    assert QApplication.clipboard().text() == "copy me"
    assert "Copied to clipboard" in window._status_label.text()


@pytest.mark.usefixtures("qtbot")
def test_debug_window_handles_provider_failure(qtbot):
    def provider():
        raise RuntimeError("provider failed")

    window = DebugTextWindow(
        title="Debug",
        text_provider=provider,
    )
    qtbot.addWidget(window)
    window.show()

    text = window._text_edit.toPlainText()
    assert "[debug-window] failed to build debug text" in text
    assert "RuntimeError" in text


@pytest.mark.usefixtures("qtbot")
def test_debug_window_copy_shortcut_works(qtbot):
    QApplication.clipboard().clear()

    def provider():
        return "shortcut copy"

    window = DebugTextWindow(
        title="Debug",
        text_provider=provider,
    )
    qtbot.addWidget(window)
    window.show()
    window.raise_()
    window.activateWindow()
    window.setFocus()
    qtbot.wait(50)

    qtbot.keyClick(window, Qt.Key_C, modifier=Qt.ControlModifier)

    assert QApplication.clipboard().text() == "shortcut copy"
