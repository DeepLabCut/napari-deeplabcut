# src/napari_deeplabcut/ui/dialogs.py
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from qtpy.QtCore import QPoint, Qt
from qtpy.QtSvgWidgets import QSvgWidget
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout

from napari_deeplabcut.config.settings import get_overwrite_confirmation_enabled
from napari_deeplabcut.core.conflicts import OverwriteConflictReport

# from napari_deeplabcut.core.dataframes import summarize_keypoint_conflicts

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
class OverwriteConflictsDialog(QDialog):
    """
    Warning dialog listing keypoints that will be overwritten.

    Design goals:
    - compact initial size
    - no horizontal scrolling
    - conflict list clearly separated from summary/context
    - details show which frame/image and which keypoints conflict
    """

    def __init__(
        self,
        parent,
        *,
        title: str,
        summary: str,
        layer_text: str,
        dest_text: str,
        affected_text: str,
        details: str,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Summary
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(summary_label)

        # Context block
        layer_label = QLabel(f"<b>Layer:</b> {layer_text}")
        layer_label.setWordWrap(True)
        layer_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layer_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(layer_label)

        dest_label = QLabel(f"<b>Destination:</b> {dest_text}")
        dest_label.setWordWrap(True)
        dest_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        dest_label.setToolTip(dest_text)  # helpful for long paths
        dest_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(dest_label)

        affected_label = QLabel(f"<b>Affected:</b> {affected_text}")
        affected_label.setWordWrap(True)
        affected_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        affected_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(affected_label)

        # Detail section label
        details_label = QLabel("Conflicts (frame/image → keypoints):")
        details_label.setWordWrap(True)
        details_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(details_label)

        # Scrollable conflict list
        text = QPlainTextEdit(self)
        text.setReadOnly(True)
        text.setPlainText(details)
        text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        text.setCenterOnScroll(False)
        self.text = text

        # Keep default height compact, but allow the user to resize larger
        fm = text.fontMetrics()
        line_h = fm.lineSpacing()
        text.setMinimumHeight(line_h * 5 + 16)
        text.setMaximumHeight(line_h * 14 + 24)

        layout.addWidget(text)

        # Buttons
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

        # Let content drive the initial size instead of hardcoding a large minimum
        self.adjustSize()
        self.sizeHint()
        # self.resize(
        #     min(max(hint.width(), 480), 820),
        #     min(max(hint.height(), 240), 520),
        # )

    @staticmethod
    def confirm(
        parent,
        *,
        summary: str,
        layer_text: str,
        dest_text: str,
        affected_text: str,
        details: str,
        title: str = "Overwrite warning",
    ) -> bool:
        dlg = OverwriteConflictsDialog(
            parent,
            title=title,
            summary=summary,
            layer_text=layer_text,
            dest_text=dest_text,
            affected_text=affected_text,
            details=details,
        )
        return dlg.exec_() == QDialog.Accepted


def maybe_confirm_overwrite(
    parent,
    report: OverwriteConflictReport,
) -> bool:
    if not report.has_conflicts:
        return True

    if not get_overwrite_confirmation_enabled():
        return True

    return OverwriteConflictsDialog.confirm(
        parent,
        summary="Saving will overwrite existing keypoints in the destination file.",
        layer_text=report.layer_name or "Unknown layer",
        dest_text=report.destination_path or "Unknown destination",
        affected_text=f"{report.n_overwrites} keypoint overwrite(s) across {report.n_frames} frame(s)/image(s).",
        details=report.details_text,
    )
