# src/napari_deeplabcut/ui/dialogs.py
from __future__ import annotations

import html
from collections import defaultdict, namedtuple

from napari.layers import Points
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut.config.keybinds import iter_shortcuts
from napari_deeplabcut.config.settings import get_overwrite_confirmation_enabled
from napari_deeplabcut.core.conflicts import OverwriteConflictReport

# from napari_deeplabcut.core.dataframes import summarize_keypoint_conflicts

Tip = namedtuple("Tip", ["msg", "pos"])

# --------------------------------------------------------------------------------
# Keyboard shortcuts dialog, generated from the keybinding registry
# --------------------------------------------------------------------------------

_KEY_LABELS = {
    "Shift": "Shift",
    "Right": "→",
    "Left": "←",
    "Up": "↑",
    "Down": "↓",
    "Space": "Space",
    "Enter": "Enter",
    "Return": "Enter",
    "Backspace": "⌫",
    "Delete": "Del",
    "Escape": "Esc",
    "Ctrl": "Ctrl",
    "Alt": "Alt",
    "Meta": "Meta",
}


def _split_key_sequence(seq: str) -> list[str]:
    """Convert a napari key sequence like 'Shift-Right' into display parts."""
    return [_KEY_LABELS.get(part, part) for part in seq.split("-")]


def _scope_label(scope: str) -> str:
    if scope == "points-layer":
        return "Points layer"
    if scope == "global-points":
        return "All Points layers"
    return scope


class KeycapLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px solid palette(mid);
                border-radius: 4px;
                padding: 1px 5px;
                background: palette(base);
                font-weight: 600;
                min-width: 16px;
            }
            """
        )


class ShortcutKeysWidget(QWidget):
    def __init__(self, keys: tuple[str, ...], parent=None):
        super().__init__(parent=parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        for i, seq in enumerate(keys):
            if i > 0:
                sep = QLabel("/")
                layout.addWidget(sep)

            parts = _split_key_sequence(seq)
            for j, part in enumerate(parts):
                if j > 0:
                    plus = QLabel("+")
                    layout.addWidget(plus)

                layout.addWidget(KeycapLabel(part))

        layout.addStretch(1)


class ShortcutRow(QFrame):
    """One shortcut row that can be enabled/dimmed based on availability."""

    KEY_COL_WIDTH = 200

    def __init__(self, spec, parent=None):
        super().__init__(parent=parent)
        self.spec = spec

        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(10)

        self.keys_widget = ShortcutKeysWidget(spec.keys)
        self.keys_widget.setFixedWidth(self.KEY_COL_WIDTH)
        self.keys_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.title_label = QLabel(spec.description)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.title_label.setWordWrap(False)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.keys_widget, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.title_label, 1, Qt.AlignLeft | Qt.AlignVCenter)

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

    def set_available(self, available: bool, reason: str | None = None) -> None:
        """Dim row when unavailable and expose context via tooltip only."""
        self._opacity_effect.setOpacity(1.0 if available else 0.45)

        tooltip_parts = [self.spec.description, _scope_label(self.spec.scope)]

        if getattr(self.spec, "when", None):
            tooltip_parts.append(self.spec.when)

        if not available and reason:
            tooltip_parts.append(reason)

        tooltip = "\n".join(tooltip_parts)

        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.keys_widget.setToolTip(tooltip)


class Shortcuts(QDialog):
    def __init__(self, parent=None, *, viewer=None):
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.viewer = viewer
        self._rows: list[ShortcutRow] = []

        self.setWindowTitle("Keyboard shortcuts")
        self.resize(720, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        intro = QLabel("These shortcuts are generated from the napari-deeplabcut keybinding registry.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        root.addWidget(intro)

        self.context_banner = QLabel("")
        self.context_banner.setTextFormat(Qt.RichText)
        self.context_banner.setWordWrap(True)
        self.context_banner.setStyleSheet(
            """
            QLabel {
                border: 1px solid palette(mid);
                border-radius: 8px;
                padding: 8px;
                background: palette(alternate-base);
            }
            """
        )
        root.addWidget(self.context_banner)

        # Scroll area for future scalability
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(scroll, 1)

        self._scroll_container = QWidget()
        scroll.setWidget(self._scroll_container)

        self._content_layout = QVBoxLayout(self._scroll_container)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(12)

        self._build_rows()
        self._refresh_availability()

        # Live updates as the active layer changes
        if self.viewer is not None:
            self._active_event_emitter = self.viewer.layers.selection.events.active
            self._active_event_emitter.connect(self._refresh_availability)

    def closeEvent(self, event):
        if self._active_event_emitter is not None:
            try:
                self._active_event_emitter.disconnect(self._refresh_availability)
            except (TypeError, RuntimeError):
                pass
            self._active_event_emitter = None
        super().closeEvent(event)

    def _build_rows(self) -> None:
        grouped = defaultdict(list)
        for spec in iter_shortcuts():
            grouped[spec.group].append(spec)

        for group_name in sorted(grouped):
            box = QGroupBox(group_name)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(6, 14, 6, 4)
            box_layout.setSpacing(1)

            for spec in grouped[group_name]:
                row = ShortcutRow(spec)
                self._rows.append(row)
                box_layout.addWidget(row)

            self._content_layout.addWidget(box)

        self._content_layout.addStretch(1)

    def _active_layer(self):
        if self.viewer is None:
            return None
        try:
            return self.viewer.layers.selection.active
        except Exception:
            return None

    def _availability_for_spec(self, spec) -> tuple[bool, str | None]:
        """
        Return (available, reason) for a shortcut in the current viewer context.
        """
        active = self._active_layer()
        active_is_points = isinstance(active, Points)

        if spec.scope == "points-layer" and not active_is_points:
            return False, "No active Points layer."

        # Optional: support extra conditions later, e.g. multi-animal-only
        # if getattr(spec, "requires_multianimal", False):
        #     if not active_is_points or not self._is_multianimal(active):
        #         return False, "Only available for multi-animal Points layers."

        return True, None

    def _refresh_availability(self, event=None) -> None:
        active = self._active_layer()
        active_is_points = isinstance(active, Points)

        if self.viewer is None:
            self.context_banner.setText(
                "Showing all known shortcuts. Availability cannot be determined without a viewer context."
            )
        elif active_is_points:
            layer_name = html.escape(getattr(active, "name", "active layer"))
            self.context_banner.setText(
                f"Active Points layer: <b>{layer_name}</b>. "
                "Shortcuts specific to Points layers are currently available."
            )
        else:
            self.context_banner.setText("No active <b>Points</b> layer — some shortcuts are currently unavailable.")

        for row in self._rows:
            available, reason = self._availability_for_spec(row.spec)
            row.set_available(available, reason)


# --------------------------------------------------------------------------------------
# Tutorial dialog with a small set of tips for new users
# --------------------------------------------------------------------------------------


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
