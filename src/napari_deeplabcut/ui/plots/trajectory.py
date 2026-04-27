"""Matplotlib trajectory plot widget for napari-deeplabcut.

This module intentionally does not make decisions about what constitutes
valid DLC points metadata; it only reads what it needs (face_color_cycles).
"""

# src/napari_deeplabcut/ui/plots/trajectory.py
from __future__ import annotations

import logging
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import matplotlib.style as mplstyle
import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from napari.layers import Points
from napari.utils.events import Event
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget

import napari_deeplabcut.core.io as io
from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.config.settings import (
    DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP,
    DEFAULT_SINGLE_ANIMAL_CMAP,
)
from napari_deeplabcut.core.keypoints import build_color_cycles
from napari_deeplabcut.core.layers import (
    get_first_image_layer,
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


class TrajectoryMatplotlibCanvas(QWidget):
    """Trajectory plot using matplotlib for keypoints (t-y plot)."""

    def __init__(self, napari_viewer, parent=None, get_color_mode: callable = None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self._get_color_mode = get_color_mode or (lambda: "bodypart")

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

        # self.setLayout(layout)

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
        self._lines: dict[tuple[str, str], list] = {}

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
        return QSize(280, 340)

    def _get_header_model_from_metadata(self, md: dict) -> DLCHeaderModel | None:
        """Return DLCHeaderModel from metadata['header'] when possible.
        TODO: Check codebase for duplicate logic and centralize if needed.
        """
        if not isinstance(md, dict):
            return None

        hdr = md.get("header", None)
        if hdr is None:
            return None

        if isinstance(hdr, DLCHeaderModel):
            return hdr

        if isinstance(hdr, dict):
            try:
                return DLCHeaderModel.model_validate(hdr)
            except Exception:
                return None

        try:
            return DLCHeaderModel(columns=hdr)
        except Exception:
            return None

    def _is_multianimal_layer(self, layer: Points) -> bool:
        """TODO: Check codebase for duplicate logic and centralize if needed."""
        md = getattr(layer, "metadata", None) or {}
        header = self._get_header_model_from_metadata(md)
        if header is None:
            return False

        try:
            inds = getattr(header, "individuals", None)
            return bool(inds and len(inds) > 0 and str(inds[0]) != "")
        except Exception:
            return False

    def _get_config_colormap(self, layer: Points) -> str:
        """Return the colormap for the given layer based on its metadata.
        TODO: Check codebase for duplicate logic and centralize if needed.
        """
        md = getattr(layer, "metadata", None) or {}
        cmap = md.get("config_colormap")
        if isinstance(cmap, str) and cmap:
            return cmap
        return DEFAULT_SINGLE_ANIMAL_CMAP

    def _plot_mode(self) -> str:
        try:
            mode = str(self._get_color_mode()).lower()
        except Exception:
            mode = "bodypart"

        if "individual" in mode:
            return "individual"
        return "bodypart"

    @staticmethod
    def _normalized_cycle(mapping) -> dict[str, object]:
        """Return a cycle mapping with string keys for robust lookups."""
        try:
            return {str(k): v for k, v in (mapping or {}).items()}
        except Exception:
            return {}

    @staticmethod
    def _normalized_individual_name(value) -> str:
        """Best-effort normalization for individual/id values used as cycle keys."""
        if value is None:
            return ""
        text = str(value)
        if not text or text.lower() == "nan":
            return ""
        return text

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

    def _get_plot_points_layer(self):
        """
        Return the active plottable Points layer for DLC trajectories when possible.

        A generic napari Points layer may not have the DLC header required by io.form_df.
        If the active layer is not a suitable DLC Points layer, fall back to the first
        plottable Points layer in the viewer.
        """

        def _is_plottable_points_layer(layer) -> bool:
            if not isinstance(layer, Points):
                return False

            md = getattr(layer, "metadata", None) or {}
            data = getattr(layer, "data", None)

            if md.get("header") is None:
                return False
            if data is None or len(data) == 0:
                return False

            return True

        active_layer = getattr(getattr(self.viewer.layers, "selection", None), "active", None)
        if _is_plottable_points_layer(active_layer):
            return active_layer

        for layer in self.viewer.layers:
            if _is_plottable_points_layer(layer):
                return layer

        return None

    def _clear_plot(self) -> None:
        """Clear plotted trajectories and reset axes."""
        self.df = None
        self._lines = {}

        self.ax.clear()
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Y position")
        self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
        self._apply_napari_theme()

    def _has_refreshable_df(self) -> bool:
        """
        Return True if self.df looks like something we can safely use with len(...).

        This keeps selection-only tests and partial UI states from crashing.
        """
        if self.df is None:
            return False
        try:
            len(self.df)
        except Exception:
            return False
        return True

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
                if icon_path.is_file():
                    action.setIcon(QIcon(str(icon_path)))
                else:
                    logger.debug(f"Failed to set toolbar icon from {icon_path}: file does not exist")

    def _load_dataframe(self, event=None) -> None:
        with mplstyle.context(self.mpl_style_sheet_path):
            points_layer = self._get_plot_points_layer()
            if points_layer is None:
                # No plottable DLC points layer present -> clear the plot quietly.
                self._clear_plot()
                return

            # Silly hack so the window does not hang the first time it is shown
            was_visible = self.isVisible()
            self.show()
            if not was_visible:
                self.hide()

            try:
                self.df = io.form_df(
                    points_layer.data,
                    layer_metadata=points_layer.metadata,
                    layer_properties=points_layer.properties,
                )
            except KeyError as e:
                # Generic / incomplete points layer: not an error for the UI, just skip plotting.
                logger.debug("Trajectory plot skipped for non-DLC/incomplete points layer: %r", e)
                self._clear_plot()
                return
            except Exception as e:
                logger.error("Failed to form DataFrame from points layer: %r", e, exc_info=True)
                self._clear_plot()
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

            self._lines = {}

            for individual, bodypart, y in self._iter_series():
                x = np.arange(y.shape[0])
                color = self._line_color_for(points_layer, individual, bodypart)
                label = self._legend_text_for(individual, bodypart)
                artists = self.ax.plot(x, y, color=color, label=label)
                self._lines[(individual, bodypart)] = artists

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
        """Show all trajectories matching one bodypart; if unknown, show all."""
        matches = {k for k in self._lines if k[1] == keypoint}
        if not matches:
            self._show_all_keypoints()
            return
        self._set_visible_keypoints(matches)

    def _refresh_canvas(self, value: int) -> None:
        if not self._has_refreshable_df():
            self.canvas.draw_idle()
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

    def _set_visible_keypoints(self, visible_keys: set) -> None:
        if not self._lines:
            return

        mode = self._plot_mode()

        for (individual, bodypart), artists in self._lines.items():
            if mode == "individual":
                show = (individual, bodypart) in visible_keys
            else:
                show = bodypart in visible_keys

            for artist in artists:
                artist.set_visible(show)

        if self.isVisible():
            self._refresh_canvas(value=self._n)

    def _show_all_keypoints(self) -> None:
        """Show all trajectories."""
        if not self._lines:
            return

        for artists in self._lines.values():
            for artist in artists:
                artist.set_visible(True)

        if self.isVisible():
            self._refresh_canvas(value=self._n)

    def _selected_line_keys_from_points_layer(self) -> set:
        points_layer = self._get_plot_points_layer()
        if points_layer is None:
            return set()

        selected = getattr(points_layer, "selected_data", None)
        if not selected:
            return set()

        props = getattr(points_layer, "properties", {}) or {}
        labels = props.get("label", None)
        ids = props.get("id", None)

        if labels is None:
            return set()

        try:
            labels_arr = np.asarray(labels, dtype=object).ravel()
        except Exception:
            return set()

        try:
            ids_arr = np.asarray(ids, dtype=object).ravel() if ids is not None else None
        except Exception:
            ids_arr = None

        mode = self._plot_mode()
        visible = set()

        for idx in selected:
            try:
                i = int(idx)
            except Exception:
                continue
            if not (0 <= i < len(labels_arr)):
                continue

            label = str(labels_arr[i])
            if not label:
                continue

            if mode == "individual":
                individual = ""
                if ids_arr is not None and i < len(ids_arr):
                    val = ids_arr[i]
                    if val is not None:
                        text = str(val)
                        if text and text.lower() != "nan":
                            individual = text
                visible.add((individual, label))
            else:
                # bodypart mode -> show all series with this bodypart
                visible.add(label)

        return visible

    def sync_visible_lines_to_points_selection(self) -> None:
        """
        Sync trajectory visibility to the current napari Points selection.

        Behavior:
        - no selected points -> show all trajectories
        - selected points    -> show only trajectories for the selected (id, label) pairs
        """
        if not self._lines:
            return

        visible = self._selected_line_keys_from_points_layer()
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

    def _df_has_individuals(self) -> bool:
        if self.df is None:
            return False
        try:
            cols = self.df.columns
            if "individuals" not in cols.names:
                return False
            vals = [str(v) for v in cols.get_level_values("individuals").unique()]
            return any(v != "" for v in vals)
        except Exception:
            return False

    def _resolved_face_color_cycles(self, layer: Points) -> dict[str, dict]:
        """
        Resolve the same label/id color cycles as ColorSchemeResolver.

        - label cycle uses the current config colormap
        - id cycle uses DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP for multi-animal
        - single-animal falls back to the bodypart cycle for both
        """
        md = getattr(layer, "metadata", None) or {}
        header = self._get_header_model_from_metadata(md)
        if header is None:
            return {}

        config_cmap = self._get_config_colormap(layer)

        try:
            bodypart_cycles = build_color_cycles(header, config_cmap) or {}
        except Exception:
            logger.debug("Trajectory plot: failed to build bodypart color cycles", exc_info=True)
            bodypart_cycles = {}

        if self._is_multianimal_layer(layer):
            try:
                individual_cycles = build_color_cycles(header, DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP) or {}
            except Exception:
                logger.debug("Trajectory plot: failed to build individual color cycles", exc_info=True)
                individual_cycles = {}
        else:
            individual_cycles = bodypart_cycles

        return {
            "label": self._normalized_cycle(bodypart_cycles.get("label", {})),
            "id": self._normalized_cycle(individual_cycles.get("id", {})),
        }

    def _line_color_for(self, points_layer: Points, individual: str, bodypart: str):
        """
        Resolve line color from the same logic as ColorSchemeResolver.

        - individual mode -> prefer id cycle
        - bodypart mode   -> prefer label cycle
        - fallback        -> whichever exists
        - final fallback  -> Matplotlib default
        """
        cycles = self._resolved_face_color_cycles(points_layer)
        mode = self._plot_mode()

        id_cycle = cycles.get("id", {}) or {}
        label_cycle = cycles.get("label", {}) or {}

        individual_key = self._normalized_individual_name(individual)
        bodypart_key = str(bodypart)

        if mode == "individual" and self._is_multianimal_layer(points_layer):
            if individual_key and individual_key in id_cycle:
                return id_cycle[individual_key]
            if bodypart_key in label_cycle:
                return label_cycle[bodypart_key]
            return "C0"

        if bodypart_key in label_cycle:
            return label_cycle[bodypart_key]
        if individual_key and individual_key in id_cycle:
            return id_cycle[individual_key]
        return "C0"

    def _legend_text_for(self, individual: str, bodypart: str) -> str:
        if self._plot_mode() == "individual" and individual:
            return f"{individual} • {bodypart}"
        return bodypart

    def _iter_series(self):
        if self.df is None:
            return

        cols = self.df.columns
        names = list(cols.names)

        has_inds = "individuals" in names
        if has_inds:
            individuals = [self._normalized_individual_name(v) for v in cols.get_level_values("individuals").unique()]
            # preserve order while removing duplicates
            seen = set()
            individuals = [v for v in individuals if not (v in seen or seen.add(v))]
        else:
            individuals = [""]

        for individual in individuals:
            for bodypart in cols.get_level_values("bodyparts").unique():
                mask = cols.get_level_values("bodyparts") == bodypart
                mask &= cols.get_level_values("coords") == "y"

                if has_inds:
                    mask &= cols.get_level_values("individuals") == individual

                y_df = self.df.loc[:, mask]
                if y_df.shape[1] == 0:
                    continue

                y = np.asarray(y_df.iloc[:, 0].to_numpy(), dtype=float).ravel()
                if y.size == 0:
                    continue

                yield self._normalized_individual_name(individual), str(bodypart), y

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
