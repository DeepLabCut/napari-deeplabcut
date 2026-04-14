# src/napari_deeplabcut/ui/debug_window.py
from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QFontDatabase, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut.utils.debug import InMemoryDebugRecorder, build_debug_report


def make_log_text_provider(
    *,
    recorder: InMemoryDebugRecorder | None,
    limit: int = 300,
) -> Callable[[], str]:
    def _provider() -> str:
        if recorder is None:
            return "<debug recorder unavailable>"
        return recorder.render_text(limit=limit)

    return _provider


def make_issue_report_provider(
    *,
    viewer,
    recorder: InMemoryDebugRecorder | None,
    log_limit: int = 300,
) -> Callable[[], str]:
    def _provider() -> str:
        return build_debug_report(viewer=viewer, recorder=recorder, log_limit=log_limit)

    return _provider


class DebugTextWindow(QDialog):
    """
    Minimal, shape-agnostic debug text viewer.

    This widget only knows how to:
    - fetch text from a callable
    - display it read-only
    - copy it to clipboard
    - refresh it on demand

    It intentionally knows nothing about:
    - log recorder internals
    - napari viewer internals
    - environment/report formatting
    """

    def __init__(
        self,
        *,
        title: str,
        text_provider: Callable[[], str],
        parent: QWidget | None = None,
        initial_hint: str = "Read-only diagnostic output",
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(900, 650)

        self._text_provider = text_provider

        self._build_ui(initial_hint=initial_hint)
        self.refresh_text()

    def _build_ui(self, *, initial_hint: str) -> None:
        layout = QVBoxLayout(self)

        self._hint_label = QLabel(initial_hint, self)
        self._hint_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._hint_label)

        self._text_edit = QPlainTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Use a fixed-width system font for logs / reports
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._text_edit.setFont(font)

        layout.addWidget(self._text_edit, stretch=1)

        button_row = QHBoxLayout()

        self._status_label = QLabel("", self)
        self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        button_row.addWidget(self._status_label, stretch=1)

        self._refresh_btn = QPushButton("Refresh", self)
        self._refresh_btn.clicked.connect(self.refresh_text)
        button_row.addWidget(self._refresh_btn)

        self._copy_btn = QPushButton("Copy to clipboard", self)
        self._copy_btn.clicked.connect(self.copy_to_clipboard)
        button_row.addWidget(self._copy_btn)

        self._close_btn = QPushButton("Close", self)
        self._close_btn.clicked.connect(self.close)
        button_row.addWidget(self._close_btn)

        layout.addLayout(button_row)

        # Optional keyboard shortcut
        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_to_clipboard)
        self.addAction(copy_action)

    def refresh_text(self) -> None:
        try:
            text = self._text_provider()
        except Exception as exc:
            text = f"[debug-window] failed to build debug text\n\n{exc!r}"

        self._text_edit.setPlainText(text or "<no debug text available>")
        self._text_edit.moveCursor(self._text_edit.textCursor().Start)
        self._status_label.setText("")

    def copy_to_clipboard(self) -> None:
        try:
            text = self._text_edit.toPlainText()
            QApplication.clipboard().setText(text)
            self._status_label.setText("Copied to clipboard")
        except Exception:
            self._status_label.setText("Could not copy to clipboard")
