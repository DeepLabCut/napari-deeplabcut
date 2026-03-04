"""Matplotlib trajectory plot widget for napari-deeplabcut.

This module intentionally does not make decisions about what constitutes
valid DLC points metadata; it only reads what it needs (face_color_cycles).
"""

# src/napari_deeplabcut/ui/plots/trajectory.py
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.style as mplstyle
import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from napari.utils.events import Event
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

import napari_deeplabcut.core.io as io
from napari_deeplabcut.core.layers import get_first_points_layer, get_first_video_image_layer

logger = logging.getLogger(__name__)


class NapariNavigationToolbar(NavigationToolbar2QT):
    """Custom Toolbar style for Napari."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(28, 28))

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()
        icon_dir = self.parentWidget()._get_path_to_icon()

        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(QIcon(os.path.join(icon_dir, "Pan_checked.png")))
            else:
                self._actions["pan"].setIcon(QIcon(os.path.join(icon_dir, "Pan.png")))

        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(QIcon(os.path.join(icon_dir, "Zoom_checked.png")))
            else:
                self._actions["zoom"].setIcon(QIcon(os.path.join(icon_dir, "Zoom.png")))


class KeypointMatplotlibCanvas(QWidget):
    """Trajectory plot using matplotlib for keypoints (t-y plot)."""

    # FIXME: y axis should be reversed due to napari using top-left as origin

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        with mplstyle.context(self.mpl_style_sheet_path):
            self.canvas = FigureCanvas()
            self.canvas.figure.set_size_inches(4, 2, forward=True)
            self.canvas.figure.set_layout_engine("constrained")
            self.ax = self.canvas.figure.subplots()

        self.toolbar = NapariNavigationToolbar(self.canvas, parent=self)
        self._replace_toolbar_icons()
        self.canvas.mpl_connect("button_press_event", self.on_doubleclick)

        self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Y position")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(10000)
        self.slider.setValue(50)
        self.slider.setToolTip("Adjust the range of frames to display around the current frame")
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider_value = QLabel(str(self.slider.value()))
        self._window = self.slider.value()
        self.slider.valueChanged.connect(self.set_window)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.slider)
        layout2.addWidget(self.slider_value)
        layout.addLayout(layout2)

        self.setLayout(layout)

        self.frames = []
        self.keypoints = []
        self.df = None
        self.setMinimumHeight(300)

        self.viewer.dims.events.current_step.connect(self.update_plot_range)
        self._n = 0
        self.update_plot_range(Event(type_name="", value=[self.viewer.dims.current_step[0]]))

        self.viewer.layers.events.inserted.connect(self._load_dataframe)
        self.viewer.dims.events.range.connect(self._update_slider_max)
        self._lines: dict[str, list] = {}

    def on_doubleclick(self, event):
        if getattr(event, "dblclick", False):
            if not self._lines:
                return
            show = list(self._lines.values())[0][0].get_visible()
            for lines in self._lines.values():
                for l in lines:
                    l.set_visible(not show)
            self._refresh_canvas(value=self._n)

    def _napari_theme_has_light_bg(self) -> bool:
        theme = napari.utils.theme.get_theme(self.viewer.theme)
        _, _, bg_lightness = theme.background.as_hsl_tuple()
        return bg_lightness > 0.5

    @property
    def mpl_style_sheet_path(self) -> Path:
        if self._napari_theme_has_light_bg():
            return Path(__file__).resolve().parents[1] / "styles" / "light.mplstyle"
        else:
            return Path(__file__).resolve().parents[1] / "styles" / "dark.mplstyle"

    def _get_path_to_icon(self) -> Path:
        icon_root = Path(__file__).resolve().parents[1] / "assets"
        if self._napari_theme_has_light_bg():
            return icon_root / "black"
        else:
            return icon_root / "white"

    def _replace_toolbar_icons(self) -> None:
        icon_dir = self._get_path_to_icon()
        for action in self.toolbar.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; Click once to activate; Click again to deactivate"
                )
            if text == "Zoom":
                action.setToolTip("Zoom to rectangle; Click once to activate; Click again to deactivate")
            if len(text) > 0:
                icon_path = os.path.join(icon_dir, text + ".png")
                action.setIcon(QIcon(icon_path))

    def _load_dataframe(self, event=None) -> None:
        points_layer = get_first_points_layer(self.viewer)
        if points_layer is None:
            return
        # Preserve existing semantics (numpy bool inversion) from original code
        try:
            if ~np.any(points_layer.data):
                return
        except Exception:
            return

        # Silly hack so the window does not hang the first time it is shown
        self.show()
        self.hide()

        try:
            self.df = io.form_df(
                points_layer.data,
                layer_metadata=points_layer.metadata,
                layer_properties=points_layer.properties,
            )
        except Exception as e:
            logger.error("Failed to form DataFrame from points layer: %r", e, exc_info=True)
            return

        self._lines.clear()
        self.ax.clear()
        self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Y position")

        for keypoint in self.df.columns.get_level_values("bodyparts").unique():
            y = self.df.xs((keypoint, "y"), axis=1, level=["bodyparts", "coords"])
            x = np.arange(len(y))
            try:
                color = points_layer.metadata["face_color_cycles"]["label"][keypoint]
            except Exception:
                color = "C0"
            lines = self.ax.plot(x, y, color=color, label=str(keypoint))
            self._lines[str(keypoint)] = lines

        self._refresh_canvas(value=self._n)

    def _toggle_line_visibility(self, keypoint: str) -> None:
        if keypoint not in self._lines:
            return
        for artist in self._lines[keypoint]:
            artist.set_visible(not artist.get_visible())
        self._refresh_canvas(value=self._n)

    def _refresh_canvas(self, value: int) -> None:
        if self.df is None:
            return
        start = max(0, value - self._window // 2)
        end = min(value + self._window // 2, len(self.df))
        self.ax.set_xlim(start, end)
        self.vline.set_xdata([value])
        self.canvas.draw()

    def set_window(self, value: int) -> None:
        self._window = value
        self.slider_value.setText(str(value))
        self.update_plot_range(Event(type_name="", value=[self._n]))

    def update_plot_range(self, event) -> None:
        value = event.value[0]
        self._n = value
        if self.df is None:
            return
        self._refresh_canvas(value)

    def _update_slider_max(self, event=None) -> None:
        img = get_first_video_image_layer(self.viewer)
        if img is None:
            return
        try:
            n_frames = img.data.shape[0]
        except Exception:
            return

        if n_frames < self.slider.minimum():
            self.slider.setMaximum(self.slider.minimum())
        else:
            self.slider.setMaximum(n_frames - 1)
