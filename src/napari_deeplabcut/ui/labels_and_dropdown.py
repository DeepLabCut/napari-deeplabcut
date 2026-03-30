"""Dropdown and color-scheme UI components for napari-deeplabcut.

Notes
-----
KeypointsDropdownMenu intentionally depends on ``keypoints.KeypointStore``.
"""
# src/napari_deeplabcut/ui/colors_and_dropdown.py

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut.core import keypoints

logger = logging.getLogger(__name__)


class DropdownMenu(QComboBox):
    def __init__(self, labels: Sequence[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.update_items(labels)

    def update_to(self, text: str):
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def reset(self):
        self.setCurrentIndex(0)

    def update_items(self, items: Sequence[str]):
        self.clear()
        self.addItems(list(items))


def create_dropdown_menu(store: keypoints.KeypointStore, items: Sequence[str], attr: str) -> DropdownMenu:
    menu = DropdownMenu(items)

    def item_changed(ind: int):
        current_item = menu.itemText(ind)
        if current_item is not None:
            setattr(store, f"current_{attr}", current_item)

    menu.currentIndexChanged.connect(item_changed)
    return menu


class KeypointsDropdownMenu(QWidget):
    """Keypoint selection UI bound to a KeypointStore."""

    def __init__(
        self,
        store: keypoints.KeypointStore,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.store = store
        self.store.layer.events.current_properties.connect(self.update_menus)
        self._locked = False

        self.id2label: dict[str, list[str]] = defaultdict(list)
        self.menus: dict[str, DropdownMenu] = {}
        self._map_individuals_to_bodyparts()
        self._populate_menus()

        layout1 = QVBoxLayout()
        layout1.addStretch(1)
        group_box = QGroupBox("Keypoint selection")
        layout2 = QVBoxLayout()
        for menu in self.menus.values():
            layout2.addWidget(menu)
        group_box.setLayout(layout2)
        layout1.addWidget(group_box)
        self.setLayout(layout1)

    def _get_ids(self) -> list[str]:
        ids = self.store.ids
        return ids if ids else [""]  # single-animal case: fake one id for consistent logic

    def _map_individuals_to_bodyparts(self) -> None:
        self.id2label.clear()  # preserve config order
        for keypoint in self.store._keypoints:
            label = keypoint.label
            id_ = keypoint.id
            if label not in self.id2label[id_]:
                self.id2label[id_].append(label)

    def _populate_menus(self) -> None:
        ids = self._get_ids()
        if not ids:
            return
        id_ = ids[0]
        if id_:
            menu = create_dropdown_menu(self.store, list(self.id2label), "id")
            menu.currentTextChanged.connect(self.refresh_label_menu)
            self.menus["id"] = menu
        self.menus["label"] = create_dropdown_menu(self.store, self.id2label[id_], "label")

    def _update_items(self) -> None:
        ids = self._get_ids()
        if not ids:
            return
        id_ = ids[0]
        if id_:
            self.menus["id"].update_items(list(self.id2label))
        self.menus["label"].update_items(self.id2label[id_])

    def update_menus(self, event=None) -> None:
        keypoint = self.store.current_keypoint
        for attr, menu in self.menus.items():
            val = getattr(keypoint, attr)
            if menu.currentText() != val:
                menu.update_to(val)

    def refresh_label_menu(self, text: str) -> None:
        menu = self.menus["label"]
        menu.blockSignals(True)
        menu.clear()
        menu.blockSignals(False)
        menu.addItems(self.id2label[text])

    def smart_reset(self, event=None) -> None:
        """Set current keypoint to the first unlabeled one."""
        if self._locked:
            return
        try:
            if not self.isVisible():
                return
        except RuntimeError:
            return

        keypoints_ = getattr(self.store, "_keypoints", None) or []
        if not keypoints_:
            return

        unannotated = ""
        already_annotated = self.store.annotated_keypoints
        for keypoint in keypoints_:
            if keypoint not in already_annotated:
                unannotated = keypoint
                break

        target = unannotated if unannotated else keypoints_[0]

        # Avoid redundant setter/event churn on every frame step.
        if self.store.current_keypoint != target:
            self.store.current_keypoint = target


class ClickableLabel(QLabel):
    clicked = Signal(str)

    def __init__(self, text: str = "", color: str = "turquoise", parent=None):
        super().__init__(text, parent)
        self._default_style = self.styleSheet()
        self.color = color

    def mousePressEvent(self, event):  # type: ignore[override]
        self.clicked.emit(self.text())

    def enterEvent(self, event):  # type: ignore[override]
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet(f"color: {self.color}")

    def leaveEvent(self, event):  # type: ignore[override]
        self.unsetCursor()
        self.setStyleSheet(self._default_style)


class LabelPair(QWidget):
    def __init__(self, color: str, name: str, parent: QWidget):
        super().__init__(parent)
        self._color = color
        self._part_name = name

        self.color_label = QLabel("", parent=self)
        self.part_label = ClickableLabel(name, color=color, parent=self)

        self.color_label.setToolTip(name)
        self.part_label.setToolTip(name)

        self._format_label(self.color_label, 10, 10)
        self._format_label(self.part_label)

        self.color_label.setStyleSheet(f"background-color: {color};")
        self._build()

    @staticmethod
    def _format_label(label: QLabel, height: int | None = None, width: int | None = None) -> None:
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if height is not None:
            label.setMaximumHeight(height)
        if width is not None:
            label.setMaximumWidth(width)

    def _build(self) -> None:
        layout = QHBoxLayout()
        layout.addWidget(self.color_label, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.part_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color
        self.color_label.setStyleSheet(f"background-color: {color};")

    @property
    def part_name(self) -> str:
        return self._part_name

    @part_name.setter
    def part_name(self, part_name: str) -> None:
        self._part_name = part_name
        self.part_label.setText(part_name)
        self.part_label.setToolTip(part_name)
        self.color_label.setToolTip(part_name)
