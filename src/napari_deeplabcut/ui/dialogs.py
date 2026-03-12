# src/napari_deeplabcut/ui/dialogs.py
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pandas as pd
from qtpy.QtCore import QPoint, Qt
from qtpy.QtSvgWidgets import QSvgWidget
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout

from napari_deeplabcut.core.dataframes import summarize_keypoint_conflicts

Tip = namedtuple("Tip", ["msg", "pos"])


class Shortcuts(QDialog):
    """Opens a window displaying available napari-deeplabcut shortcuts."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.setWindowTitle("Shortcuts")

        image_path = str(Path(__file__).resolve().parents[1] / "assets" / "napari_shortcuts.svg")

        vlayout = QVBoxLayout()
        svg_widget = QSvgWidget(image_path)
        svg_widget.setStyleSheet("background-color: white;")
        vlayout.addWidget(svg_widget)
        self.setLayout(vlayout)


class Tutorial(QDialog):
    """Walkthrough window with a small set of tips."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.setWindowTitle("Tutorial")
        self.setModal(True)
        self.setStyleSheet("background:#361AE5")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowOpacity(0.95)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        self._current_tip = -1
        self._tips = [
            Tip(
                "Load a folder of annotated data\n"
                "(and optionally a config file if labeling from scratch)\n"
                "from the menu File > Open File or Open Folder.\n"
                "Alternatively, files and folders of images can be dragged\n"
                "and dropped onto the main window.",
                (0.35, 0.15),
            ),
            Tip(
                "Data layers will be listed at the bottom left;\n"
                "their visibility can be toggled by clicking on the small eye icon.",
                (0.1, 0.65),
            ),
            Tip(
                "Corresponding layer controls can be found at the top left.\n"
                "Switch between labeling and selection mode using the numeric keys 2 and 3,\n"
                "or clicking on the + or -> icons.",
                (0.1, 0.2),
            ),
            Tip(
                "There are three keypoint labeling modes:\nthe key M can be used to cycle between them.",
                (0.65, 0.05),
            ),
            Tip(
                "When done labeling, save your data by selecting the Points layer\n"
                "and hitting Ctrl+S (or File > Save Selected Layer(s)...).",
                (0.1, 0.65),
            ),
            Tip(
                "Read more at <a href='https://github.com/DeepLabCut/napari-deeplabcut#usage'>napari-deeplabcut</a>",
                (0.4, 0.4),
            ),
        ]

        vlayout = QVBoxLayout()
        self.message = QLabel("💡\n\nLet's get started with a quick walkthrough!")
        self.message.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.message.setOpenExternalLinks(True)
        vlayout.addWidget(self.message)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self.prev_tip)
        nav_layout.addWidget(self.prev_button)
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.next_tip)
        nav_layout.addWidget(self.next_button)

        self.update_nav_buttons()

        hlayout = QHBoxLayout()
        self.count = QLabel("")
        hlayout.addWidget(self.count)
        hlayout.addLayout(nav_layout)
        vlayout.addLayout(hlayout)
        self.setLayout(vlayout)

    def prev_tip(self):
        self._current_tip = (self._current_tip - 1) % len(self._tips)
        self.update_tip()
        self.update_nav_buttons()

    def next_tip(self):
        self._current_tip = (self._current_tip + 1) % len(self._tips)
        self.update_tip()
        self.update_nav_buttons()

    def update_tip(self):
        tip = self._tips[self._current_tip]
        msg = tip.msg
        # No emoji in the last tip otherwise the hyperlink breaks
        if self._current_tip < len(self._tips) - 1:
            msg = "💡\n\n" + msg
        self.message.setText(msg)
        self.count.setText(f"Tip {self._current_tip + 1}|{len(self._tips)}")
        self.adjustSize()

        xrel, yrel = tip.pos
        geom = self.parent().geometry()
        p = QPoint(
            int(geom.left() + geom.width() * xrel),
            int(geom.top() + geom.height() * yrel),
        )
        self.move(p)

    def update_nav_buttons(self):
        self.prev_button.setEnabled(self._current_tip > 0)
        self.next_button.setEnabled(self._current_tip < len(self._tips) - 1)


# --------------------------------------------------------------------------------------
# Conflict resolution dialog for overwriting existing keypoints when saving annotations.
# --------------------------------------------------------------------------------------
def _conflict_stats(key_conflict: pd.DataFrame) -> tuple[int, int]:
    """
    Returns:
      n_pairs  = number of (image, keypoint) pairs that will be overwritten
      n_images = number of distinct images affected
    """
    n_pairs = int(key_conflict.to_numpy().sum())
    n_images = int(key_conflict.any(axis=1).to_numpy().sum())
    return n_pairs, n_images


def _build_overwrite_warning_text(key_conflict: pd.DataFrame, max_items: int = 15) -> tuple[str, str]:
    """
    Returns (summary, details).
    Summary: short sentence for dialog top.
    Details: scrollable text body listing image → keypoint examples.
    """
    n_pairs, n_images = _conflict_stats(key_conflict)
    summary = f"{n_pairs} existing keypoint(s) will be overwritten across {n_images} image(s)."
    details = summarize_keypoint_conflicts(key_conflict, max_items=max_items)
    return summary, details


def maybe_confirm_overwrite(metadata: dict, key_conflict: pd.DataFrame, allow_missing_controls=False) -> bool:
    """
    Returns True if save should proceed, False if user cancels.
    If no GUI controls are present, returns True (non-interactive).
    """
    if not key_conflict.to_numpy().any():
        return True

    controls = metadata.get("controls")
    if controls is None:
        if allow_missing_controls:
            return True  # headless/scripted save: no dialog
        else:
            raise RuntimeError("Keypoint conflicts detected but no GUI controls found.")

    summary, details = _build_overwrite_warning_text(key_conflict)
    return OverwriteConflictsDialog.confirm(controls, summary=summary, details=details)


class OverwriteConflictsDialog(QDialog):
    """
    Scrollable warning dialog listing keypoints that will be overwritten.
    Returns True if user chooses to proceed, False otherwise.
    """

    def __init__(self, parent, *, title: str, summary: str, details: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 450)

        layout = QVBoxLayout(self)

        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(details)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.overwrite_btn = QPushButton("Overwrite")
        self.overwrite_btn.setDefault(True)
        self.overwrite_btn.setAutoDefault(True)
        self.overwrite_btn.clicked.connect(self.accept)

        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.overwrite_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def confirm(parent, *, summary: str, details: str, title="Overwrite warning") -> bool:
        dlg = OverwriteConflictsDialog(parent, title=title, summary=summary, details=details)
        return dlg.exec_() == QDialog.Accepted
