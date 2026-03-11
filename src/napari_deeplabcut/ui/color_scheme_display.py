# src/napari_deeplabcut/ui/color_scheme_display.py
from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

import numpy as np
from napari.layers import Points
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut import keypoints, misc
from napari_deeplabcut.ui.labels_and_dropdown import LabelPair

logger = logging.getLogger(__name__)


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


class SchemeSource(StrEnum):
    ACTIVE = "active"
    CONFIG = "config"


def _to_hex(rgba) -> str:
    arr = np.asarray(rgba, dtype=float).ravel()
    if arr.size < 3:
        return "#000000"

    if arr.size == 3:
        arr = np.r_[arr, 1.0]

    arr = np.clip(arr[:4], 0, 1)
    r, g, b, _a = (arr * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class ColorSchemeResolver:
    def __init__(
        self,
        viewer,
        get_color_mode: Callable[[], str],
        get_header_model: Callable[[dict], object | None],
    ):
        self.viewer = viewer
        self._get_color_mode = get_color_mode
        self._get_header_model = get_header_model

    def get_target_layer(self) -> Points | None:
        active = self.viewer.layers.selection.active
        if isinstance(active, Points) and getattr(active, "visible", True):
            return active

        for layer in reversed(list(self.viewer.layers)):
            if isinstance(layer, Points) and getattr(layer, "visible", True):
                return layer

        return None

    def get_color_property(self, layer: Points) -> str | None:
        md = layer.metadata or {}
        cycles = md.get("face_color_cycles") or {}
        if not cycles:
            return None

        header = self._get_header_model(md)

        is_multi = False
        try:
            inds = getattr(header, "individuals", None)
            is_multi = bool(inds and len(inds) > 0 and str(inds[0]) != "")
        except Exception:
            pass

        prop = "id" if (is_multi and "id" in cycles) else "label"

        color_mode = self._get_color_mode()
        if color_mode == str(keypoints.ColorMode.INDIVIDUAL) and "id" in cycles:
            prop = "id"
        elif color_mode == str(keypoints.ColorMode.BODYPART) and "label" in cycles:
            prop = "label"

        if prop not in cycles:
            if "label" in cycles:
                return "label"
            if "id" in cycles:
                return "id"
            return None

        return prop

    def get_visible_categories(self, layer: Points, prop: str) -> list[str]:
        props = getattr(layer, "properties", {}) or {}
        values = props.get(prop, None)
        if values is None or len(values) == 0:
            return []

        values = np.asarray(values, dtype=object).ravel()
        mask = np.ones(len(values), dtype=bool)

        try:
            data = np.asarray(layer.data)
            if len(data) == len(values) and data.ndim == 2 and data.shape[1] > 0:
                current_step = self.viewer.dims.current_step
                if len(current_step) > 0:
                    frame = current_step[0]
                    mask &= data[:, 0] == frame
        except Exception:
            pass

        try:
            shown = getattr(layer, "shown", None)
            if shown is not None and len(shown) == len(values):
                mask &= np.asarray(shown, dtype=bool)
        except Exception:
            pass

        out = []
        seen = set()
        for v in values[mask]:
            if v in ("", None) or misc._is_nan_value(v):
                continue
            s = str(v)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def get_config_categories(self, layer: Points, prop: str) -> list[str]:
        md = layer.metadata or {}
        cycles = md.get("face_color_cycles") or {}
        mapping = cycles.get(prop) or {}
        return list(mapping.keys())

    def resolve(self, *, show_config_keypoints: bool) -> dict[str, str]:
        layer = self.get_target_layer()
        if layer is None:
            return {}

        md = layer.metadata or {}
        cycles = md.get("face_color_cycles") or {}
        if not cycles:
            return {}

        prop = self.get_color_property(layer)
        if prop is None:
            return {}

        mapping = cycles.get(prop) or {}
        if not mapping:
            return {}

        if show_config_keypoints:
            names = self.get_config_categories(layer, prop)
        else:
            names = self.get_visible_categories(layer, prop)

        scheme = {}
        for name in names:
            color = mapping.get(name)
            if color is None:
                continue
            scheme[str(name)] = _to_hex(color)
        return scheme


class ColorSchemePanel(QWidget):
    def __init__(
        self,
        viewer,
        get_color_mode,
        get_header_model,
        parent=None,
    ):
        super().__init__(parent)
        self.viewer = viewer
        self._update_pending = False

        self._resolver = ColorSchemeResolver(
            viewer=viewer,
            get_color_mode=get_color_mode,
            get_header_model=get_header_model,
        )

        self._toggle = QCheckBox("Show config keypoints instead of active", self)
        self._toggle.setToolTip(
            "If checked, show all keypoints from the config/header.\n"
            "If unchecked, show only the currently visible keypoints in the active layer/frame."
        )
        self._toggle.toggled.connect(self.schedule_update)

        self.display = ColorSchemeDisplay(parent=self)

        layout = QVBoxLayout(self)
        layout.addWidget(self._toggle)
        layout.addWidget(self.display)

        self._connect_viewer_events()

    @property
    def show_config_keypoints(self) -> bool:
        return self._toggle.isChecked()

    def _connect_viewer_events(self) -> None:
        self.viewer.layers.selection.events.active.connect(self.schedule_update)
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self.schedule_update)
        self.viewer.dims.events.current_step.connect(self.schedule_update)

        for layer in list(self.viewer.layers):
            self._maybe_wire_layer(layer)

    def _on_layers_changed(self, event=None) -> None:
        try:
            layer = event.value if event is not None else None
        except Exception:
            layer = None

        if layer is None:
            try:
                layer = event.source[-1]
            except Exception:
                layer = None

        self._maybe_wire_layer(layer)
        self.schedule_update()

    def _maybe_wire_layer(self, layer) -> None:
        if not isinstance(layer, Points):
            return

        for event_name in ("visible", "data", "properties", "shown", "current_properties"):
            try:
                getattr(layer.events, event_name).connect(self.schedule_update)
            except Exception:
                pass

    def schedule_update(self, event=None) -> None:
        if self._update_pending:
            return

        self._update_pending = True

        def _do():
            try:
                self.update_scheme()
            finally:
                self._update_pending = False

        QTimer.singleShot(0, _do)

    def update_scheme(self) -> None:
        scheme = self._resolver.resolve(show_config_keypoints=self.show_config_keypoints)
        self.display.reset()
        if scheme:
            self.display.update_color_scheme(scheme)
