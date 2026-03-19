"""Matplotlib trajectory plot widget for napari-deeplabcut.

This module intentionally does not make decisions about what constitutes
valid DLC points metadata; it only reads what it needs (face_color_cycles).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import matplotlib.style as mplstyle
import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from napari.utils.events import Event
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget

import napari_deeplabcut.core.io as io
from napari_deeplabcut.core.layers import (
    get_first_image_layer,
    get_first_points_layer,
    get_first_video_image_layer,
)
from napari_deeplabcut.utils.deprecations import deprecated

logger = logging.getLogger(__name__)

_PACKAGE = "napari_deeplabcut"


@lru_cache(maxsize=1)
def _pkg_root():
    return files(_PACKAGE)


def _styles_traversable():
    return _pkg_root() / "styles"


def _assets_traversable():
    return _pkg_root() / "assets"


# TODO: @C-Achard 2026-03-04 - Check if this is still needed after refactor
class NapariNavigationToolbar(NavigationToolbar2QT):
    """Custom Toolbar style for Napari."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(28, 28))

    @staticmethod
    def _qicon(pathlike) -> QIcon:
        """Build a QIcon safely from any path-like object."""
        return QIcon(str(pathlike))

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()
        icon_dir = self.parentWidget()._get_path_to_icon()

        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(self._qicon(Path(icon_dir) / "Pan_checked.png"))
            else:
                self._actions["pan"].setIcon(self._qicon(Path(icon_dir) / "Pan.png"))

        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(self._qicon(Path(icon_dir) / "Zoom_checked.png"))
            else:
                self._actions["zoom"].setIcon(self._qicon(Path(icon_dir) / "Zoom.png"))


class KeypointMatplotlibCanvas(QWidget):
    """Trajectory plot using matplotlib for keypoints (t-y plot)."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer

        with mplstyle.context(self.mpl_style_sheet_path):
            self.canvas = FigureCanvas()
            # self.canvas.figure.set_size_inches(4, 2, forward=True)
            self.canvas.figure.set_layout_engine("constrained")
            self.ax = self.canvas.figure.subplots()
            self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("Y position")

        # self.toolbar = NapariNavigationToolbar(self.canvas, parent=self)
        self.toolbar = NavigationToolbar2QT(self.canvas, parent=self)
        self.toolbar.setIconSize(QSize(28, 28))
        self._set_toolbar_tooltips()
        self.canvas.mpl_connect("button_press_event", self.on_doubleclick)

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

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider_value.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Give almost all vertical stretch to the plot canvas
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar, stretch=0)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.slider_value, stretch=0)
        layout.addLayout(slider_row, stretch=0)

        self.setLayout(layout)

        self.frames = []
        self.keypoints = []
        self.df = None
        self.setMinimumSize(280, 350)

        self.viewer.dims.events.current_step.connect(self.update_plot_range)
        self._n = 0
        self.update_plot_range(
            Event(type_name="", value=[self.viewer.dims.current_step[0]]),
            force=True,
        )
        self._apply_axis_theme()

        self.viewer.layers.events.inserted.connect(self._load_dataframe)
        self.viewer.dims.events.range.connect(self._update_slider_max)
        self._lines: dict[str, list] = {}

        self._apply_napari_theme()
        self._connect_theme_events()
        # If layers already existed before this widget was created
        # (e.g. drag-and-drop load before opening the plugin), populate
        # the plot from the current viewer state on the next event-loop turn.
        QTimer.singleShot(0, self.refresh_from_viewer_layers)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.canvas.draw_idle()

    def sizeHint(self) -> QSize:
        """
        Preferred initial size for the trajectory plot dock widget.

        Wide enough for the toolbar + slider, and tall enough that the plot
        is useful by default without preventing later resizing.
        """
        return QSize(480, 340)

    def minimumSizeHint(self) -> QSize:
        """
        Smallest comfortable size before the widget becomes cramped.
        """
        return QSize(280, 440)

    def on_doubleclick(self, event):
        if getattr(event, "dblclick", False):
            if not self._lines:
                return
            show = list(self._lines.values())[0][0].get_visible()
            for lines in self._lines.values():
                for line in lines:
                    line.set_visible(not show)
            self._refresh_canvas(value=self._n)

    def refresh_from_viewer_layers(self) -> None:
        """
        Refresh the trajectory plot from the current viewer state.

        This is safe to call:
        - after drag/drop loads that happened before the widget was opened
        - after layer adoption / remap
        - after later explicit layer changes

        It intentionally:
        1) reloads the dataframe from the current Points layer
        2) updates the slider max from the current Image/Video layer
        3) re-syncs visible lines to current point selection
        """
        try:
            self._load_dataframe()
        except Exception:
            logger.debug("Trajectory plot: failed to load dataframe from viewer layers", exc_info=True)

        try:
            self._update_slider_max()
        except Exception:
            logger.debug("Trajectory plot: failed to update slider max", exc_info=True)

        try:
            self.sync_visible_lines_to_points_selection()
        except Exception:
            logger.debug("Trajectory plot: failed to sync visible lines to point selection", exc_info=True)

    def _apply_axis_theme(self) -> None:
        """Force axis/text colors to match napari theme."""
        is_light = self._napari_theme_has_light_bg()
        fg = "black" if is_light else "white"

        self.ax.tick_params(axis="both", colors=fg, which="both")
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        self.vline.set_color(fg)

    def _napari_theme_has_light_bg(self) -> bool:
        theme = napari.utils.theme.get_theme(self.viewer.theme)
        _, _, bg_lightness = theme.background.as_hsl_tuple()
        return bg_lightness > 0.5

    @property
    def mpl_style_sheet_path(self) -> Path:
        if self._napari_theme_has_light_bg():
            return _styles_traversable() / "light.mplstyle"
        return _styles_traversable() / "dark.mplstyle"

    def _get_path_to_icon(self) -> Path:
        icon_root = _assets_traversable() / "icons"
        if self._napari_theme_has_light_bg():
            return icon_root / "black"
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
            if text:
                icon_path = Path(icon_dir) / f"{text}.png"
                try:
                    action.setIcon(self.toolbar._qicon(icon_path))
                except AttributeError:
                    logger.debug(f"Failed to set toolbar icon from {icon_path}", exc_info=True)

    def _load_dataframe(self, event=None) -> None:
        with mplstyle.context(self.mpl_style_sheet_path):
            points_layer = get_first_points_layer(self.viewer)
            data = getattr(points_layer, "data", None)

            if points_layer is None or data is None or len(data) == 0:
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

            image_layer = get_first_video_image_layer(self.viewer)
            if image_layer is None:
                image_layer = get_first_image_layer(self.viewer)

            self._lines = {}
            self.ax.clear()
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("Y position")
            self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
            self._apply_napari_theme()

            height = None
            if image_layer is not None:
                try:
                    img_data = image_layer.data
                    if getattr(image_layer, "rgb", False):
                        height = img_data.shape[-3]
                    else:
                        height = img_data.shape[-2]
                except Exception:
                    height = None

            for keypoint in self.df.columns.get_level_values("bodyparts").unique():
                y = (
                    self.df.xs(
                        (keypoint, "y"),
                        axis=1,
                        level=["bodyparts", "coords"],
                    )
                    .to_numpy()
                    .squeeze()
                )
                x = np.arange(len(y))
                try:
                    color = points_layer.metadata["face_color_cycles"]["label"][keypoint]
                except Exception:
                    color = "C0"
                lines = self.ax.plot(x, y, color=color, label=str(keypoint))
                self._lines[str(keypoint)] = lines

            # Match napari image coordinates: y increases downward
            if height is not None:
                self.ax.set_ylim(height, 0)
            else:
                self.ax.invert_yaxis()

            self._refresh_canvas(value=self._n)

    @deprecated(
        details="No longer used, instead visibility is based on napari Points selection.",
        replacement="sync_visible_lines_to_points_selection",
    )
    def _toggle_line_visibility(self, keypoint: str) -> None:
        if keypoint not in self._lines:
            return
        for artist in self._lines[keypoint]:
            artist.set_visible(not artist.get_visible())
        self._refresh_canvas(value=self._n)

    def show_only_keypoint(self, keypoint: str) -> None:
        """Show only one keypoint trajectory; if unknown, show all."""
        if keypoint not in self._lines:
            self._show_all_keypoints()
            return
        self._set_visible_keypoints({keypoint})

    def _refresh_canvas(self, value: int) -> None:
        if self.df is None:
            return

        with mplstyle.context(self.mpl_style_sheet_path):
            start = max(0, value - self._window // 2)
            end = min(value + self._window // 2, len(self.df))
            self.ax.set_xlim(start, end)
            self.vline.set_xdata([value])
            self.canvas.draw()

    def set_window(self, value: int) -> None:
        self._window = value
        self.slider_value.setText(str(value))
        self.update_plot_range(Event(type_name="", value=[self._n]))

    def update_plot_range(self, event, force: bool = False) -> None:
        if not self.isVisible() and not force:
            return

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

    def _set_visible_keypoints(self, visible_keypoints: set[str]) -> None:
        """Show only the given keypoint trajectories."""
        if not self._lines:
            return

        for keypoint, artists in self._lines.items():
            show = keypoint in visible_keypoints
            for artist in artists:
                artist.set_visible(show)

        self._refresh_canvas(value=self._n)

    def _show_all_keypoints(self) -> None:
        """Show all trajectories."""
        if not self._lines:
            return

        for artists in self._lines.values():
            for artist in artists:
                artist.set_visible(True)

        self._refresh_canvas(value=self._n)

    def _selected_bodyparts_from_points_layer(self) -> set[str]:
        """
        Return the set of selected bodypart labels from the first Points layer.

        Notes
        -----
        - We intentionally key visibility by `label` because this plot currently
        groups trajectories by bodypart name.
        - If nothing is selected, returns an empty set.
        """
        points_layer = get_first_points_layer(self.viewer)
        if points_layer is None:
            return set()

        selected = getattr(points_layer, "selected_data", None)
        if not selected:
            return set()

        props = getattr(points_layer, "properties", {}) or {}
        labels = props.get("label", None)
        if labels is None:
            return set()

        try:
            labels_arr = np.asarray(labels, dtype=object).ravel()
        except Exception:
            return set()

        visible: set[str] = set()
        for idx in selected:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < len(labels_arr):
                val = labels_arr[i]
                if val is None:
                    continue
                text = str(val)
                if text:
                    visible.add(text)

        return visible

    def sync_visible_lines_to_points_selection(self) -> None:
        """
        Sync trajectory visibility to the current napari Points selection.

        Behavior:
        - no selected points -> show all trajectories
        - selected points    -> show only trajectories whose bodypart label is selected
        """
        if not self._lines:
            return

        visible = self._selected_bodyparts_from_points_layer()
        if not visible:
            self._show_all_keypoints()
            return

        self._set_visible_keypoints(visible)

    def _set_toolbar_tooltips(self) -> None:
        """Set clearer tooltips for the stock matplotlib toolbar."""
        for action in self.toolbar.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; Click once to activate; Click again to deactivate"
                )
            elif text == "Zoom":
                action.setToolTip("Zoom to rectangle; Click once to activate; Click again to deactivate")

    # def _apply_toolbar_icons(self) -> None:
    #     """
    #     Apply static black/white toolbar icons based on the current napari theme.

    #     This does not override toolbar behavior; it only replaces the displayed icons.
    #     """
    #     icon_dir = self._get_path_to_icon()
    #     for action in self.toolbar.actions():
    #         text = action.text()
    #         if not text:
    #             continue

    #         icon_path = Path(icon_dir) / f"{text}.png"
    #         if icon_path.exists():
    #             action.setIcon(QIcon(str(icon_path)))

    def _apply_toolbar_stylesheet(self) -> None:
        """
        Apply a minimal Qt stylesheet so the toolbar background has enough contrast.

        In light mode:
        - toolbar background becomes light gray
        - buttons remain mostly transparent until hover/checked

        In dark mode:
        - keep a subtle dark/transparent look
        """
        is_light = self._napari_theme_has_light_bg()

        if is_light:
            fg = "#202020"
            toolbar_bg = "#ececec"  # light gray background for the whole toolbar
            hover = "rgba(0, 0, 0, 0.06)"
            pressed = "rgba(0, 0, 0, 0.12)"
            border = "rgba(0, 0, 0, 0.10)"
        else:
            fg = "#f2f2f2"
            toolbar_bg = "transparent"
            hover = "rgba(255, 255, 255, 0.10)"
            pressed = "rgba(255, 255, 255, 0.18)"
            border = "rgba(255, 255, 255, 0.12)"

        self.toolbar.setStyleSheet(
            f"""
            QToolBar {{
                background: {toolbar_bg};
                border: none;
                spacing: 2px;
                padding: 2px;
            }}
            QToolButton {{
                color: {fg};
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px;
                margin: 1px;
            }}
            QToolButton:hover {{
                background: {hover};
                border: 1px solid {border};
            }}
            QToolButton:pressed {{
                background: {pressed};
            }}
            QToolButton:checked {{
                background: {pressed};
                border: 1px solid {border};
            }}
            """
        )

    def _apply_label_styles(self) -> None:
        """Apply light/dark text color to simple Qt labels in this widget."""
        fg = "black" if self._napari_theme_has_light_bg() else "white"
        self.slider_value.setStyleSheet(f"color: {fg};")

    def _apply_napari_theme(self) -> None:
        """
        Re-apply all napari-dependent styling.

        Safe to call repeatedly.
        """
        self._apply_axis_theme()
        # self._apply_toolbar_icons()
        self._apply_toolbar_stylesheet()
        self._apply_label_styles()
        self.canvas.draw_idle()

    def _connect_theme_events(self) -> None:
        """
        Re-apply theme when the viewer theme changes, if the event exists.

        Safe across versions: if the event is absent, this becomes a no-op.
        """
        viewer_events = getattr(self.viewer, "events", None)
        theme_emitter = getattr(viewer_events, "theme", None)
        if theme_emitter is not None:
            theme_emitter.connect(lambda event=None: self._apply_napari_theme())
