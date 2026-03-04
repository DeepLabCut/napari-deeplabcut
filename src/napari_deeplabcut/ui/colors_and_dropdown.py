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
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut import keypoints

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

    def _map_individuals_to_bodyparts(self) -> None:
        self.id2label.clear()  # preserve config order
        for keypoint in self.store._keypoints:
            label = keypoint.label
            id_ = keypoint.id
            if label not in self.id2label[id_]:
                self.id2label[id_].append(label)

    def _populate_menus(self) -> None:
        id_ = self.store.ids[0]
        if id_:
            menu = create_dropdown_menu(self.store, list(self.id2label), "id")
            menu.currentTextChanged.connect(self.refresh_label_menu)
            self.menus["id"] = menu
        self.menus["label"] = create_dropdown_menu(self.store, self.id2label[id_], "label")

    def _update_items(self) -> None:
        id_ = self.store.ids[0]
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
        unannotated = ""
        already_annotated = self.store.annotated_keypoints
        for keypoint in self.store._keypoints:
            if keypoint not in already_annotated:
                unannotated = keypoint
                break
        self.store.current_keypoint = unannotated if unannotated else self.store._keypoints[0]


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


class ColorSchemeDisplay(QScrollArea):
    """Scrollable list of keypoint labels and their associated colors."""

    added = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.scheme_dict: dict[str, str] = {}
        self._layout = QVBoxLayout()
        self._layout.setSpacing(0)
        # container required by QScrollArea.setWidget
        self._container = QWidget(parent=self)
        self._build()

    @property
    def labels(self):
        labels = []
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if w := item.widget():
                labels.append(w)
        return labels

    def _build(self) -> None:
        self._container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self._container.setLayout(self._layout)
        self._container.adjustSize()

        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setBaseSize(100, 200)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def add_entry(self, name: str, color: str) -> None:
        self.scheme_dict.update({name: color})
        widget = LabelPair(color, name, self)
        self._layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignLeft)
        self.added.emit(widget)

    def update_color_scheme(self, new_color_scheme: dict[str, str]) -> None:
        logger.debug("Updating color scheme: %s widgets", self._layout.count())
        self.scheme_dict = {name: color for name, color in new_color_scheme.items()}
        names = list(new_color_scheme.keys())
        existing_widgets = self._layout.count()
        required_widgets = len(self.scheme_dict)

        # update existing widgets
        for idx in range(min(existing_widgets, required_widgets)):
            w = self._layout.itemAt(idx).widget()
            if w is None:
                continue
            w.setVisible(True)
            w.part_name = names[idx]
            w.color = self.scheme_dict[names[idx]]

        # hide extra widgets
        for i in range(max(existing_widgets - required_widgets, 0)):
            if w := self._layout.itemAt(required_widgets + i).widget():
                w.setVisible(False)

        # add missing widgets
        for i in range(max(required_widgets - existing_widgets, 0)):
            name = names[existing_widgets + i]
            self.add_entry(name, self.scheme_dict[name])

    def reset(self) -> None:
        self.scheme_dict = {}
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w is not None:
                w.setVisible(False)
