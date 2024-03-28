import logging
import os
from collections import defaultdict, namedtuple
from copy import deepcopy
from datetime import datetime
from functools import partial, cached_property
from math import ceil, log10
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import napari
import pandas as pd
from pathlib import Path
from types import MethodType
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT

import numpy as np
from napari._qt.widgets.qt_welcome import QtWelcomeLabel
from napari.layers import Image, Points, Shapes, Tracks
from napari.layers.points._points_key_bindings import register_points_action
from napari.layers.utils import color_manager
from napari.layers.utils.layer_utils import _features_to_properties
from napari.utils.events import Event
from napari.utils.history import get_save_history, update_save_history
from qtpy.QtCore import Qt, QTimer, Signal, QPoint, QSettings, QSize
from qtpy.QtGui import QPainter, QAction, QCursor, QIcon
from qtpy.QtSvgWidgets import QSvgWidget
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut import keypoints
from napari_deeplabcut._reader import _load_config
from napari_deeplabcut._writer import _write_config, _write_image, _form_df
from napari_deeplabcut.misc import (
    encode_categories,
    to_os_dir_sep,
    guarantee_multiindex_rows,
    build_color_cycles,
)

Tip = namedtuple("Tip", ["msg", "pos"])


class Shortcuts(QDialog):
    """Opens a window displaying available napari-deeplabcut shortcuts"""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.setWindowTitle("Shortcuts")

        image_path = str(Path(__file__).parent / "assets" / "napari_shortcuts.svg")

        vlayout = QVBoxLayout()
        svg_widget = QSvgWidget(image_path)
        svg_widget.setStyleSheet("background-color: white;")
        vlayout.addWidget(svg_widget)
        self.setLayout(vlayout)


class Tutorial(QDialog):
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
                "Load a folder of annotated data\n(and optionally a config file if labeling from scratch)\nfrom the menu File > Open File or Open Folder.\nAlternatively, files and folders of images can be dragged\nand dropped onto the main window.",
                (0.35, 0.15),
            ),
            Tip(
                "Data layers will be listed at the bottom left;\ntheir visibility can be toggled by clicking on the small eye icon.",
                (0.1, 0.65),
            ),
            Tip(
                "Corresponding layer controls can be found at the top left.\nSwitch between labeling and selection mode using the numeric keys 2 and 3,\nor clicking on the + or -> icons.",
                (0.1, 0.2),
            ),
            Tip(
                "There are three keypoint labeling modes:\nthe key M can be used to cycle between them.",
                (0.65, 0.05),
            ),
            Tip(
                "When done labeling, save your data by selecting the Points layer\nand hitting Ctrl+S (or File > Save Selected Layer(s)...).",
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
        if (
            self._current_tip < len(self._tips) - 1
        ):  # No emoji in the last tip otherwise the hyperlink breaks
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


def _get_and_try_preferred_reader(
    self,
    dialog,
    *args,
):
    try:
        self.viewer.open(
            dialog._current_file,
            plugin="napari-deeplabcut",
        )
    except ValueError:
        self.viewer.open(
            dialog._current_file,
            plugin="builtins",
        )


# Hack to avoid napari's silly variable type guess,
# where property is understood as continuous if
# there are more than 16 unique categories...
def guess_continuous(property):
    if issubclass(property.dtype.type, np.floating):
        return True
    else:
        return False


color_manager.guess_continuous = guess_continuous


def _paste_data(self, store):
    """Paste only currently unannotated data."""
    features = self._clipboard.pop("features", None)
    if features is None:
        return

    unannotated = [
        keypoints.Keypoint(label, id_) not in store.annotated_keypoints
        for label, id_ in zip(features["label"], features["id"])
    ]
    if not any(unannotated):
        return

    new_features = features.iloc[unannotated]
    indices_ = self._clipboard.pop("indices")
    text_ = self._clipboard.pop("text")
    self._clipboard = {k: v[unannotated] for k, v in self._clipboard.items()}
    self._clipboard["features"] = new_features
    self._clipboard["indices"] = indices_
    if text_ is not None:
        new_text = {
            "string": text_["string"][unannotated],
            "color": text_["color"],
        }
        self._clipboard["text"] = new_text

    npoints = len(self._view_data)
    totpoints = len(self.data)

    if len(self._clipboard.keys()) > 0:
        not_disp = self._slice_input.not_displayed
        data = deepcopy(self._clipboard["data"])
        offset = [
            self._slice_indices[i] - self._clipboard["indices"][i] for i in not_disp
        ]
        data[:, not_disp] = data[:, not_disp] + np.array(offset)
        self._data = np.append(self.data, data, axis=0)
        self._shown = np.append(self.shown, deepcopy(self._clipboard["shown"]), axis=0)
        self._size = np.append(self.size, deepcopy(self._clipboard["size"]), axis=0)
        self._symbol = np.append(
            self.symbol, deepcopy(self._clipboard["symbol"]), axis=0
        )

        self._feature_table.append(self._clipboard["features"])

        self.text._paste(**self._clipboard["text"])

        self._edge_width = np.append(
            self.edge_width,
            deepcopy(self._clipboard["edge_width"]),
            axis=0,
        )
        self._edge._paste(
            colors=self._clipboard["edge_color"],
            properties=_features_to_properties(self._clipboard["features"]),
        )
        self._face._paste(
            colors=self._clipboard["face_color"],
            properties=_features_to_properties(self._clipboard["features"]),
        )

        self._selected_view = list(
            range(npoints, npoints + len(self._clipboard["data"]))
        )
        self._selected_data = set(
            range(totpoints, totpoints + len(self._clipboard["data"]))
        )
        self.refresh()


# Hack to save a KeyPoints layer without showing the Save dialog
def _save_layers_dialog(self, selected=False):
    """Save layers (all or selected) to disk, using ``LayerList.save()``.
    Parameters
    ----------
    selected : bool
        If True, only layers that are selected in the viewer will be saved.
        By default, all layers are saved.
    """
    selected_layers = list(self.viewer.layers.selection)
    msg = ""
    if not len(self.viewer.layers):
        msg = "There are no layers in the viewer to save."
    elif selected and not len(selected_layers):
        msg = "Please select a Points layer to save."
    if msg:
        QMessageBox.warning(self, "Nothing to save", msg, QMessageBox.Ok)
        return
    if len(selected_layers) == 1 and isinstance(selected_layers[0], Points):
        self.viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
        self.viewer.status = "Data successfully saved"
    else:
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)
        filename, _ = dlg.getSaveFileName(
            caption=f'Save {"selected" if selected else "all"} layers',
            dir=hist[0],  # home dir by default
        )
        if filename:
            self.viewer.layers.save(filename, selected=selected)
        else:
            return
    self._is_saved = True
    self.last_saved_label.setText(
        f'Last saved at {str(datetime.now().time()).split(".")[0]}'
    )
    self.last_saved_label.show()


def on_close(self, event, widget):
    if widget._stores and not widget._is_saved:
        choice = QMessageBox.warning(
            widget,
            "Warning",
            "Data were not saved. Are you certain you want to leave?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    else:
        event.accept()


# Class taken from https://github.com/matplotlib/napari-matplotlib/blob/53aa5ec95c1f3901e21dedce8347d3f95efe1f79/src/napari_matplotlib/base.py#L309
class NapariNavigationToolbar(NavigationToolbar2QT):
    """Custom Toolbar style for Napari."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(28, 28))

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()
        icon_dir = self.parentWidget()._get_path_to_icon()

        # changes pan/zoom icons depending on state (checked or not)
        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(icon_dir, "Pan_checked.png"))
                )
            else:
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(icon_dir, "Pan.png"))
                )
        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(icon_dir, "Zoom_checked.png"))
                )
            else:
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(icon_dir, "Zoom.png"))
                )


class KeypointMatplotlibCanvas(QWidget):
    """
    Class about matplotlib canvas in which I will draw the keypoints over a range of frames
    It will be at the bottom of the screen and will use the keypoints from the range of frames to plot them on a x-y time series.
    """

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        with mplstyle.context(self.mpl_style_sheet_path):
            self.canvas = FigureCanvas()
            self.canvas.figure.set_layout_engine("constrained")
            self.ax = self.canvas.figure.subplots()
        self.toolbar = NapariNavigationToolbar(self.canvas, parent=self)
        self._replace_toolbar_icons()
        self.canvas.mpl_connect("button_press_event", self.on_doubleclick)
        self.vline = self.ax.axvline(0, 0, 1, color="k", linestyle="--")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Y position")
        # Add a slot to specify the range of frames to plot
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(10000)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider_value = QLabel(str(self.slider.value()))
        self._window = self.slider.value()
        # Connect slider to window setter
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
        # Make widget larger
        self.setMinimumHeight(300)
        # connect sliders to update plot
        self.viewer.dims.events.current_step.connect(self.update_plot_range)

        # Run update plot range once to initialize the plot
        self._n = 0
        self.update_plot_range(
            Event(type_name="", value=[self.viewer.dims.current_step[0]])
        )

        self.viewer.layers.events.inserted.connect(self._load_dataframe)
        self._lines = {}

    def on_doubleclick(self, event):
        if event.dblclick:
            show = list(self._lines.values())[0][0].get_visible()
            for lines in self._lines.values():
                for l in lines:
                    l.set_visible(not show)
            self._refresh_canvas(value=self._n)

    def _napari_theme_has_light_bg(self) -> bool:
        """
        Does this theme have a light background?

        Returns
        -------
        bool
            True if theme's background colour has hsl lighter than 50%, False if darker.
        """
        theme = napari.utils.theme.get_theme(self.viewer.theme, as_dict=False)
        _, _, bg_lightness = theme.background.as_hsl_tuple()
        return bg_lightness > 0.5

    @property
    def mpl_style_sheet_path(self) -> Path:
        """
        Path to the set Matplotlib style sheet.
        """
        if self._napari_theme_has_light_bg():
            return Path(__file__).parent / "styles" / "light.mplstyle"
        else:
            return Path(__file__).parent / "styles" / "dark.mplstyle"

    def _get_path_to_icon(self) -> Path:
        """
        Get the icons directory (which is theme-dependent).

        Icons modified from
        https://github.com/matplotlib/matplotlib/tree/main/lib/matplotlib/mpl-data/images
        """
        icon_root = Path(__file__).parent / "assets"
        if self._napari_theme_has_light_bg():
            return icon_root / "black"
        else:
            return icon_root / "white"

    def _replace_toolbar_icons(self) -> None:
        """
        Modifies toolbar icons to match the napari theme, and add some tooltips.
        """
        icon_dir = self._get_path_to_icon()
        for action in self.toolbar.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; "
                    "Click once to activate; Click again to deactivate"
                )
            if text == "Zoom":
                action.setToolTip(
                    "Zoom to rectangle; Click once to activate; "
                    "Click again to deactivate"
                )
            if len(text) > 0:  # i.e. not a separator item
                icon_path = os.path.join(icon_dir, text + ".png")
                action.setIcon(QIcon(icon_path))

    def _load_dataframe(self):
        points_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                points_layer = layer
                break

        if points_layer is None or ~np.any(points_layer.data):
            return

        self.viewer.window.add_dock_widget(self, name="Trajectory plot", area="right")
        self.hide()

        self.df = _form_df(
            points_layer.data,
            {
                "metadata": points_layer.metadata,
                "properties": points_layer.properties,
            },
        )
        for keypoint in self.df.columns.get_level_values("bodyparts").unique():
            y = self.df.xs((keypoint, "y"), axis=1, level=["bodyparts", "coords"])
            x = np.arange(len(y))
            color = points_layer.metadata["face_color_cycles"]["label"][keypoint]
            lines = self.ax.plot(x, y, color=color, label=keypoint)
            self._lines[keypoint] = lines

        self._refresh_canvas(value=self._n)

    def _toggle_line_visibility(self, keypoint):
        for artist in self._lines[keypoint]:
            artist.set_visible(not artist.get_visible())
        self._refresh_canvas(value=self._n)

    def _refresh_canvas(self, value):
        start = max(0, value - self._window // 2)
        end = min(value + self._window // 2, len(self.df))

        self.ax.set_xlim(start, end)
        self.vline.set_xdata(value)
        self.canvas.draw()

    def set_window(self, value):
        self._window = value
        self.slider_value.setText(str(value))
        self.update_plot_range(Event(type_name="", value=[self._n]))

    def update_plot_range(self, event):
        value = event.value[0]
        self._n = value

        if self.df is None:
            return

        self._refresh_canvas(value)


class KeypointControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._is_saved = False

        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.on_insert)
        self.viewer.layers.events.removed.connect(self.on_remove)

        self.viewer.window.qt_viewer._get_and_try_preferred_reader = MethodType(
            _get_and_try_preferred_reader,
            self.viewer.window.qt_viewer,
        )

        status_bar = self.viewer.window._qt_window.statusBar()
        self.last_saved_label = QLabel("")
        self.last_saved_label.hide()
        status_bar.addPermanentWidget(self.last_saved_label)

        # Hack napari's Welcome overlay to show more relevant instructions
        overlay = self.viewer.window._qt_viewer._welcome_widget
        welcome_widget = overlay.layout().itemAt(1).widget()
        welcome_widget.deleteLater()
        w = QtWelcomeWidget(None)
        overlay._overlay = w
        overlay.addWidget(w)
        overlay._overlay.sig_dropped.connect(overlay.sig_dropped)

        self._color_mode = keypoints.ColorMode.default()
        self._label_mode = keypoints.LabelMode.default()

        # Hold references to the KeypointStores
        self._stores = {}
        # Intercept close event if data were not saved
        self.viewer.window._qt_window.closeEvent = partial(
            on_close,
            self.viewer.window._qt_window,
            widget=self,
        )

        # Storage for extra image metadata that are relevant to other layers.
        # These are updated anytime images are added to the Viewer
        # and passed on to the other layers upon creation.
        self._images_meta = dict()

        # Add some more controls
        self._layout = QVBoxLayout(self)
        self._menus = []
        self._layer_to_menu = {}
        self.viewer.layers.selection.events.active.connect(self.on_active_layer_change)

        self._video_group = self._form_video_action_menu()
        self.video_widget = self.viewer.window.add_dock_widget(
            self._video_group, name="video", area="right"
        )
        self.video_widget.setVisible(False)

        # form helper display
        help_buttons = self._form_help_buttons()
        self._layout.addLayout(help_buttons)

        hlayout = QHBoxLayout()
        trail_label = QLabel("Show trails")
        self._trail_cb = QCheckBox()
        self._trail_cb.setToolTip("toggle trails visibility")
        self._trail_cb.setChecked(False)
        self._trail_cb.setEnabled(False)
        self._trail_cb.stateChanged.connect(self._show_trails)
        self._trails = None

        matplotlib_label = QLabel("Show matplotlib canvas")
        self._matplotlib_canvas = KeypointMatplotlibCanvas(self.viewer)
        self._matplotlib_cb = QCheckBox()
        self._matplotlib_cb.setToolTip("toggle matplotlib canvas visibility")
        self._matplotlib_cb.stateChanged.connect(self._show_matplotlib_canvas)
        self._matplotlib_cb.setChecked(False)
        self._matplotlib_cb.setEnabled(False)
        self._view_scheme_cb = QCheckBox("Show color scheme", parent=self)

        hlayout.addWidget(self._matplotlib_cb)
        hlayout.addWidget(matplotlib_label)
        hlayout.addWidget(self._trail_cb)
        hlayout.addWidget(trail_label)
        hlayout.addWidget(self._view_scheme_cb)

        self._layout.addLayout(hlayout)

        # form buttons for selection of annotation mode
        self._radio_group = self._form_mode_radio_buttons()

        # form color scheme display + color mode selector
        self._color_grp, self._color_mode_selector = self._form_color_mode_selector()
        self._display = ColorSchemeDisplay(parent=self)
        self._color_scheme_display = self._form_color_scheme_display(self.viewer)
        self._view_scheme_cb.toggled.connect(self._show_color_scheme)
        self._view_scheme_cb.toggle()
        self._display.added.connect(
            lambda w: w.part_label.clicked.connect(
                self._matplotlib_canvas._toggle_line_visibility
            ),
        )

        # Substitute default menu action with custom one
        for action in self.viewer.window.file_menu.actions()[::-1]:
            action_name = action.text().lower()
            if "save selected layer" in action_name:
                action.triggered.disconnect()
                action.triggered.connect(
                    lambda: _save_layers_dialog(
                        self,
                        selected=True,
                    )
                )
            elif "save all layers" in action_name:
                self.viewer.window.file_menu.removeAction(action)

        # Add action to show the walkthrough again
        launch_tutorial = QAction("&Launch Tutorial", self)
        launch_tutorial.triggered.connect(self.start_tutorial)
        self.viewer.window.view_menu.addAction(launch_tutorial)

        # Add action to view keyboard shortcuts
        display_shortcuts_action = QAction("&Shortcuts", self)
        display_shortcuts_action.triggered.connect(self.display_shortcuts)
        self.viewer.window.help_menu.addAction(display_shortcuts_action)

        # Hide some unused viewer buttons
        self.viewer.window._qt_viewer.viewerButtons.gridViewButton.hide()
        self.viewer.window._qt_viewer.viewerButtons.rollDimsButton.hide()
        self.viewer.window._qt_viewer.viewerButtons.transposeDimsButton.hide()
        self.viewer.window._qt_viewer.layerButtons.newPointsButton.setDisabled(True)
        self.viewer.window._qt_viewer.layerButtons.newLabelsButton.setDisabled(True)

        if self.settings.value("first_launch", True) and not os.environ.get(
            "hide_tutorial", False
        ):
            QTimer.singleShot(10, self.start_tutorial)
            self.settings.setValue("first_launch", False)

    @cached_property
    def settings(self):
        return QSettings()

    def start_tutorial(self):
        Tutorial(self.viewer.window._qt_window.current()).show()

    def display_shortcuts(self):
        Shortcuts(self.viewer.window._qt_window.current()).show()

    def _move_image_layer_to_bottom(self, index):
        if (ind := index) != 0:
            self.viewer.layers.move_selected(ind, 0)
            self.viewer.layers.select_next()  # Auto-select the Points layer

    def _show_color_scheme(self):
        show = self._view_scheme_cb.isChecked()
        self._color_scheme_display.setVisible(show)

    def _show_trails(self, state):
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            if self._trails is None:
                store = list(self._stores.values())[0]
                categories = store.layer.properties["id"]
                if not categories[0]:  # Single animal data
                    categories = store.layer.properties["label"]
                inds = encode_categories(categories)
                temp = np.c_[inds, store.layer.data]
                cmap = "viridis"
                for layer in self.viewer.layers:
                    if isinstance(layer, Points) and layer.metadata:
                        cmap = layer.metadata["colormap_name"]
                self._trails = self.viewer.add_tracks(
                    temp,
                    tail_length=50,
                    head_length=50,
                    tail_width=6,
                    name="trails",
                    colormap=cmap,
                )
            self._trails.visible = True
        elif self._trails is not None:
            self._trails.visible = False

    def _show_matplotlib_canvas(self, state):
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            self._matplotlib_canvas.show()
        else:
            self._matplotlib_canvas.hide()

    def _form_video_action_menu(self):
        group_box = QGroupBox("Video")
        layout = QVBoxLayout()
        extract_button = QPushButton("Extract frame")
        extract_button.clicked.connect(self._extract_single_frame)
        layout.addWidget(extract_button)
        crop_button = QPushButton("Store crop coordinates")
        crop_button.clicked.connect(self._store_crop_coordinates)
        layout.addWidget(crop_button)
        group_box.setLayout(layout)
        return group_box

    def _form_help_buttons(self):
        layout = QHBoxLayout()
        show_shortcuts = QPushButton("View shortcuts")
        show_shortcuts.clicked.connect(self.display_shortcuts)
        layout.addWidget(show_shortcuts)
        tutorial = QPushButton("Start tutorial")
        tutorial.clicked.connect(self.start_tutorial)
        layout.addWidget(tutorial)
        return layout

    def _extract_single_frame(self, *args):
        image_layer = None
        points_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                image_layer = layer
            elif isinstance(layer, Points):
                points_layer = layer
        if image_layer is not None:
            ind = self.viewer.dims.current_step[0]
            frame = image_layer.data[ind]
            n_frames = image_layer.data.shape[0]
            name = f"img{str(ind).zfill(int(ceil(log10(n_frames))))}.png"
            output_path = os.path.join(image_layer.metadata["root"], name)
            _write_image(frame, str(output_path))

            # If annotations were loaded, they should be written to a machinefile.h5 file
            if points_layer is not None:
                df = _form_df(
                    points_layer.data,
                    {
                        "metadata": points_layer.metadata,
                        "properties": points_layer.properties,
                    },
                )
                df = df.iloc[ind: ind + 1]
                df.index = pd.MultiIndex.from_tuples([Path(output_path).parts[-3:]])
                filepath = os.path.join(
                    image_layer.metadata["root"], "machinelabels-iter0.h5"
                )
                if Path(filepath).is_file():
                    df_prev = pd.read_hdf(filepath)
                    guarantee_multiindex_rows(df_prev)
                    df = pd.concat([df_prev, df])
                    df = df[~df.index.duplicated(keep="first")]
                df.to_hdf(filepath, key="machinelabels")

    def _store_crop_coordinates(self, *args):
        if not (project_path := self._images_meta.get("project")):
            return
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                try:
                    ind = layer.shape_type.index("rectangle")
                except ValueError:
                    return
                bbox = layer.data[ind][:, 1:]
                h = self.viewer.dims.range[2][1]
                bbox[:, 0] = h - bbox[:, 0]
                bbox = np.clip(bbox, 0, a_max=None).astype(int)
                y1, x1 = bbox.min(axis=0)
                y2, x2 = bbox.max(axis=0)
                temp = {"crop": ", ".join(map(str, [x1, x2, y1, y2]))}
                config_path = os.path.join(project_path, "config.yaml")
                cfg = _load_config(config_path)
                cfg["video_sets"][
                    os.path.join(project_path, "videos", self._images_meta["name"])
                ] = temp
                _write_config(config_path, cfg)
                break

    def _form_dropdown_menus(self, store):
        menu = KeypointsDropdownMenu(store)
        self.viewer.dims.events.current_step.connect(
            menu.smart_reset,
            position="last",
        )
        menu.smart_reset(event=None)
        self._menus.append(menu)
        self._layer_to_menu[store.layer] = len(self._menus) - 1
        layout = QVBoxLayout()
        layout.addWidget(menu)
        self._layout.addLayout(layout)

    def _form_mode_radio_buttons(self):
        group_box = QGroupBox("Labeling mode")
        layout = QHBoxLayout()
        group = QButtonGroup(self)
        for i, mode in enumerate(keypoints.LabelMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            btn.setToolTip(keypoints.TOOLTIPS[mode])
            group.addButton(btn, i)
            layout.addWidget(btn)
        group.button(1).setChecked(True)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)

        def _func():
            self.label_mode = group.checkedButton().text()

        group.buttonClicked.connect(_func)
        return group

    def _form_color_mode_selector(self):
        group_box = QGroupBox("Keypoint coloring mode")
        layout = QHBoxLayout()
        group = QButtonGroup(self)
        for i, mode in enumerate(keypoints.ColorMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            group.addButton(btn, i)
            layout.addWidget(btn)
        group.button(1).setChecked(True)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)

        def _func():
            self.color_mode = group.checkedButton().text()

        group.buttonClicked.connect(_func)
        return group_box, group

    def _form_color_scheme_display(self, viewer):
        self.viewer.layers.events.inserted.connect(self._update_color_scheme)
        return viewer.window.add_dock_widget(
            self._display, name="Color scheme reference", area="left"
        )

    def _update_color_scheme(self):
        def to_hex(nparray):
            a = np.array(nparray * 255, dtype=int)
            rgb2hex = lambda r, g, b, _: f"#{r:02x}{g:02x}{b:02x}"
            res = rgb2hex(*a)
            return res

        self._display.reset()
        mode = "label"
        if self.color_mode == str(keypoints.ColorMode.INDIVIDUAL):
            mode = "id"

        for layer in self.viewer.layers:
            if isinstance(layer, Points) and layer.metadata:
                self._display.update_color_scheme(
                    {
                        name: to_hex(color)
                        for name, color in layer.metadata["face_color_cycles"][mode].items()
                    }
                )

    def _remap_frame_indices(self, layer):
        if not self._images_meta.get("paths"):
            return

        new_paths = [to_os_dir_sep(p) for p in self._images_meta["paths"]]
        paths = layer.metadata.get("paths")
        if paths is not None and np.any(layer.data):
            paths_map = dict(zip(range(len(paths)), map(to_os_dir_sep, paths)))
            # Discard data if there are missing frames
            missing = [i for i, path in paths_map.items() if path not in new_paths]
            if missing:
                if isinstance(layer.data, list):
                    inds_to_remove = [
                        i
                        for i, verts in enumerate(layer.data)
                        if verts[0, 0] in missing
                    ]
                else:
                    inds_to_remove = np.flatnonzero(np.isin(layer.data[:, 0], missing))
                layer.selected_data = inds_to_remove
                layer.remove_selected()
                for i in missing:
                    paths_map.pop(i)

            # Check now whether there are new frames
            temp = {k: new_paths.index(v) for k, v in paths_map.items()}
            data = layer.data
            if isinstance(data, list):
                for verts in data:
                    verts[:, 0] = np.vectorize(temp.get)(verts[:, 0])
            else:
                data[:, 0] = np.vectorize(temp.get)(data[:, 0])
            layer.data = data
        layer.metadata.update(self._images_meta)

    def on_insert(self, event):
        layer = event.source[-1]
        logging.debug(f"Inserting Layer {layer}")
        if isinstance(layer, Image):
            paths = layer.metadata.get("paths")
            if paths is None:  # Then it's a video file
                self.video_widget.setVisible(True)
            # Store the metadata and pass them on to the other layers
            self._images_meta.update(
                {
                    "paths": paths,
                    "shape": layer.level_shapes[0],
                    "root": layer.metadata["root"],
                    "name": layer.name,
                }
            )
            # Delay layer sorting
            QTimer.singleShot(
                10, partial(self._move_image_layer_to_bottom, event.index)
            )
        elif isinstance(layer, Points):
            # If the current Points layer comes from a config file, some have already
            # been added and the body part names are different from the existing ones,
            # then we update store's metadata and menus.
            if layer.metadata.get("project", "") and self._stores:
                new_metadata = layer.metadata.copy()

                keypoints_menu = self._menus[0].menus["label"]
                current_keypoint_set = set(
                    keypoints_menu.itemText(i) for i in range(keypoints_menu.count())
                )
                new_keypoint_set = set(new_metadata["header"].bodyparts)
                diff = new_keypoint_set.difference(current_keypoint_set)
                if diff:
                    answer = QMessageBox.question(
                        self, "", "Do you want to display the new keypoints only?"
                    )
                    if answer == QMessageBox.Yes:
                        self.viewer.layers[-2].shown = False

                    self.viewer.status = f"New keypoint{'s' if len(diff) > 1 else ''} {', '.join(diff)} found."
                    for _layer, store in self._stores.items():
                        _layer.metadata["header"] = new_metadata["header"]
                        store.layer = _layer

                    for menu in self._menus:
                        menu._map_individuals_to_bodyparts()
                        menu._update_items()

                # Remove the unnecessary layer newly added
                QTimer.singleShot(10, self.viewer.layers.pop)

                # Always update the colormap to reflect the one in the config.yaml file
                for _layer, store in self._stores.items():
                    _layer.metadata["face_color_cycles"] = new_metadata[
                        "face_color_cycles"
                    ]
                    _layer.face_color = "label"
                    _layer.face_color_cycle = new_metadata["face_color_cycles"][
                        "label"
                    ]
                    _layer.events.face_color()
                    store.layer = _layer
                self._update_color_scheme()

                return

            store = keypoints.KeypointStore(self.viewer, layer)
            self._stores[layer] = store
            # TODO Set default dir of the save file dialog
            if root := layer.metadata.get("root"):
                update_save_history(root)
            layer.metadata["controls"] = self
            layer.text.visible = False
            layer.bind_key("M", self.cycle_through_label_modes)
            layer.bind_key("F", self.cycle_through_color_modes)
            func = partial(_paste_data, store=store)
            layer._paste_data = MethodType(func, layer)
            layer.add = MethodType(keypoints._add, store)
            layer.events.add(query_next_frame=Event)
            layer.events.query_next_frame.connect(store._advance_step)
            layer.bind_key("Shift-Right", store._find_first_unlabeled_frame)
            layer.bind_key("Shift-Left", store._find_first_unlabeled_frame)

            layer.bind_key("Down", store.next_keypoint, overwrite=True)
            layer.bind_key("Up", store.prev_keypoint, overwrite=True)
            layer.face_color_mode = "cycle"
            self._form_dropdown_menus(store)

            self._images_meta.update(
                {
                    "project": layer.metadata.get("project"),
                }
            )
            self._trail_cb.setEnabled(True)
            self._matplotlib_cb.setEnabled(True)

            # Hide the color pickers, as colormaps are strictly defined by users
            controls = self.viewer.window.qt_viewer.dockLayerControls
            point_controls = controls.widget().widgets[layer]
            point_controls.faceColorEdit.hide()
            point_controls.edgeColorEdit.hide()
            point_controls.layout().itemAt(9).widget().hide()
            point_controls.layout().itemAt(11).widget().hide()
            # Hide out of slice checkbox
            point_controls.outOfSliceCheckBox.hide()
            point_controls.layout().itemAt(15).widget().hide()
            # Add dropdown menu for colormap picking
            colormap_selector = DropdownMenu(plt.colormaps, self)
            colormap_selector.update_to(layer.metadata["colormap_name"])
            colormap_selector.currentTextChanged.connect(self._update_colormap)
            point_controls.layout().addRow("colormap", colormap_selector)

        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self._remap_frame_indices(layer_)

    def on_remove(self, event):
        layer = event.value
        n_points_layer = sum(isinstance(l, Points) for l in self.viewer.layers)
        if isinstance(layer, Points) and n_points_layer == 0:
            if self._color_scheme_display is not None:
                self._display.reset()
            self._stores.pop(layer, None)
            while self._menus:
                menu = self._menus.pop()
                self._layout.removeWidget(menu)
                menu.deleteLater()
                menu.destroy()
            self._layer_to_menu = {}
            self._trail_cb.setEnabled(False)
            self._matplotlib_cb.setEnabled(False)
            self.last_saved_label.hide()
        elif isinstance(layer, Image):
            self._images_meta = dict()
            paths = layer.metadata.get("paths")
            if paths is None:
                self.video_widget.setVisible(False)
        elif isinstance(layer, Tracks):
            self._trail_cb.setChecked(False)
            self._matplotlib_cb.setChecked(False)
            self._trails = None

    def on_active_layer_change(self, event) -> None:
        """Updates the GUI when the active layer changes
            * Hides all KeypointsDropdownMenu that aren't for the selected layer
            * Sets the visibility of the "Color mode" box to True if the selected layer
                is a multi-animal one, or False otherwise
        """
        self._color_grp.setVisible(self._is_multianimal(event.value))
        menu_idx = -1
        if event.value is not None and isinstance(event.value, Points):
            menu_idx = self._layer_to_menu.get(event.value, -1)

        for idx, menu in enumerate(self._menus):
            if idx == menu_idx:
                menu.setHidden(False)
            else:
                menu.setHidden(True)

    def _update_colormap(self, colormap_name):
        for layer in self.viewer.layers.selection:
            if isinstance(layer, Points) and layer.metadata:
                face_color_cycle_maps = build_color_cycles(
                    layer.metadata["header"], colormap_name,
                )
                layer.metadata["face_color_cycles"] = face_color_cycle_maps
                face_color_prop = "label"
                if self.color_mode == str(keypoints.ColorMode.INDIVIDUAL):
                    face_color_prop = "id"

                layer.face_color = face_color_prop
                layer.face_color_cycle = face_color_cycle_maps[face_color_prop]
                layer.events.face_color()
                self._update_color_scheme()

    @register_points_action("Change labeling mode")
    def cycle_through_label_modes(self, *args):
        self.label_mode = next(keypoints.LabelMode)

    @register_points_action("Change color mode")
    def cycle_through_color_modes(self, *args):
        if (
            self._active_layer_is_multianimal()
            or self.color_mode != str(keypoints.ColorMode.BODYPART)
        ):
            self.color_mode = next(keypoints.ColorMode)

    @property
    def label_mode(self):
        return str(self._label_mode)

    @label_mode.setter
    def label_mode(self, mode: Union[str, keypoints.LabelMode]):
        self._label_mode = keypoints.LabelMode(mode)
        self.viewer.status = self.label_mode
        mode_ = str(mode)
        if mode_ == "loop":
            for menu in self._menus:
                menu._locked = True
        else:
            for menu in self._menus:
                menu._locked = False
        for btn in self._radio_group.buttons():
            if btn.text() == mode_:
                btn.setChecked(True)
                break

    @property
    def color_mode(self):
        return str(self._color_mode)

    @color_mode.setter
    def color_mode(self, mode: Union[str, keypoints.ColorMode]):
        self._color_mode = keypoints.ColorMode(mode)
        if self._color_mode == keypoints.ColorMode.BODYPART:
            face_color_mode = "label"
        else:
            face_color_mode = "id"

        for layer in self.viewer.layers:
            if isinstance(layer, Points) and layer.metadata:
                layer.face_color = face_color_mode
                layer.face_color_cycle = layer.metadata["face_color_cycles"][
                    face_color_mode]
                layer.events.face_color()

        for btn in self._color_mode_selector.buttons():
            if btn.text() == str(mode):
                btn.setChecked(True)
                break

        self._update_color_scheme()

    def _is_multianimal(self, layer) -> bool:
        is_multi = False
        if layer is not None and isinstance(layer, Points):
            try:
                header = layer.metadata.get("header")
                if header is not None:
                    ids = header.individuals
                    is_multi = len(ids) > 0 and ids[0] != ""
            except AttributeError:
                pass

        return is_multi

    def _active_layer_is_multianimal(self) -> bool:
        """Returns: whether the active layer is a multi-animal points layer"""
        for layer in self.viewer.layers.selection:
            if self._is_multianimal(layer):
                return True

        return False


@Points.bind_key("E")
def toggle_edge_color(layer):
    # Trick to toggle between 0 and 2
    layer.edge_width = np.bitwise_xor(layer.edge_width, 2)


class DropdownMenu(QComboBox):
    def __init__(self, labels: Sequence[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.update_items(labels)

    def update_to(self, text: str):
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def reset(self):
        self.setCurrentIndex(0)

    def update_items(self, items):
        self.clear()
        self.addItems(items)


class KeypointsDropdownMenu(QWidget):
    def __init__(
        self,
        store: keypoints.KeypointStore,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.store = store
        self.store.layer.events.current_properties.connect(self.update_menus)
        self._locked = False

        self.id2label = defaultdict(list)
        self.menus = dict()
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

    def _map_individuals_to_bodyparts(self):
        self.id2label.clear()  # Empty dict so entries are ordered as in the config
        for keypoint in self.store._keypoints:
            label = keypoint.label
            id_ = keypoint.id
            if label not in self.id2label[id_]:
                self.id2label[id_].append(label)

    def _populate_menus(self):
        id_ = self.store.ids[0]
        if id_:
            menu = create_dropdown_menu(self.store, list(self.id2label), "id")
            menu.currentTextChanged.connect(self.refresh_label_menu)
            self.menus["id"] = menu
        self.menus["label"] = create_dropdown_menu(
            self.store,
            self.id2label[id_],
            "label",
        )

    def _update_items(self):
        id_ = self.store.ids[0]
        if id_:
            self.menus["id"].update_items(list(self.id2label))
        self.menus["label"].update_items(self.id2label[id_])

    def update_menus(self, event):
        keypoint = self.store.current_keypoint
        for attr, menu in self.menus.items():
            val = getattr(keypoint, attr)
            if menu.currentText() != val:
                menu.update_to(val)

    def refresh_label_menu(self, text: str):
        menu = self.menus["label"]
        menu.blockSignals(True)
        menu.clear()
        menu.blockSignals(False)
        menu.addItems(self.id2label[text])

    def smart_reset(self, event):
        """Set current keypoint to the first unlabeled one."""
        if self._locked:  # The currently selected point is not updated
            return
        unannotated = ""
        already_annotated = self.store.annotated_keypoints
        for keypoint in self.store._keypoints:
            if keypoint not in already_annotated:
                unannotated = keypoint
                break
        self.store.current_keypoint = (
            unannotated if unannotated else self.store._keypoints[0]
        )


def create_dropdown_menu(store, items, attr):
    menu = DropdownMenu(items)

    def item_changed(ind):
        current_item = menu.itemText(ind)
        if current_item is not None:
            setattr(store, f"current_{attr}", current_item)

    menu.currentIndexChanged.connect(item_changed)
    return menu


# WelcomeWidget modified from:
# https://github.com/napari/napari/blob/a72d512972a274380645dae16b9aa93de38c3ba2/napari/_qt/widgets/qt_welcome.py#L28
class QtWelcomeWidget(QWidget):
    """Welcome widget to display initial information and shortcuts to user."""

    sig_dropped = Signal("QEvent")

    def __init__(self, parent):
        super().__init__(parent)

        # Create colored icon using theme
        self._image = QLabel()
        self._image.setObjectName("logo_silhouette")
        self._image.setMinimumSize(300, 300)
        self._label = QtWelcomeLabel(
            """
            Drop a folder from within a DeepLabCut's labeled-data directory,
            and,  if labeling from scratch,
            the corresponding project's config.yaml file.
            """
        )

        # Widget setup
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self._image.setAlignment(Qt.AlignCenter)
        self._label.setAlignment(Qt.AlignCenter)

        # Layout
        text_layout = QVBoxLayout()
        text_layout.addWidget(self._label)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.setSpacing(30)
        layout.addWidget(self._image)
        layout.addLayout(text_layout)
        layout.addStretch()

        self.setLayout(layout)

    def paintEvent(self, event):
        """Override Qt method.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        option = QStyleOption()
        option.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, p, self)

    def _update_property(self, prop, value):
        """Update properties of widget to update style.

        Parameters
        ----------
        prop : str
            Property name to update.
        value : bool
            Property value to update.
        """
        self.setProperty(prop, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", True)
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Override Qt method.

        Provide style updates on event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", False)

    def dropEvent(self, event):
        """Override Qt method.

        Provide style updates on event and emit the drop event.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._update_property("drag", False)
        self.sig_dropped.emit(event)


class ClickableLabel(QLabel):
    clicked = Signal(str)

    def __init__(self, text="", color="turquoise", parent=None):
        super().__init__(text, parent)
        self._default_style = self.styleSheet()
        self.color = color

    def mousePressEvent(self, event):
        self.clicked.emit(self.text())

    def enterEvent(self, event):
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet(f"color: {self.color}")

    def leaveEvent(self, event):
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
    def _format_label(label: QLabel, height: int = None, width: int = None):
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if height is not None:
            label.setMaximumHeight(height)
        if width is not None:
            label.setMaximumWidth(width)

    def _build(self):
        layout = QHBoxLayout()
        layout.addWidget(self.color_label, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.part_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = color
        self.color_label.setStyleSheet(f"background-color: {color};")

    @property
    def part_name(self):
        return self._part_name

    @part_name.setter
    def part_name(self, part_name: str):
        self._part_name = part_name
        self.part_label.setText(part_name)
        self.part_label.setToolTip(part_name)
        self.color_label.setToolTip(part_name)


class ColorSchemeDisplay(QScrollArea):
    added = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)

        self.scheme_dict = {}  # {name: color} mapping
        self._layout = QVBoxLayout()
        self._layout.setSpacing(0)
        self._container = QWidget(
            parent=self
        )  # workaround to use setWidget, let me know if there's a better option

        self._build()

    @property
    def labels(self):
        labels = []
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if w := item.widget():
                labels.append(w)
        return labels

    def _build(self):
        self._container.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Maximum
        )  # feel free to change those
        self._container.setLayout(self._layout)
        self._container.adjustSize()

        self.setWidget(self._container)

        self.setWidgetResizable(True)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )  # feel free to change those
        # self.setMaximumHeight(150)
        self.setBaseSize(100, 200)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def add_entry(self, name, color):
        self.scheme_dict.update({name: color})

        widget = LabelPair(color, name, self)
        self._layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignLeft)
        self.added.emit(widget)

    def update_color_scheme(self, new_color_scheme) -> None:
        logging.debug(f"Updating color scheme: {self._layout.count()} widgets")
        self.scheme_dict = {name: color for name, color in new_color_scheme.items()}
        names = list(new_color_scheme.keys())
        existing_widgets = self._layout.count()
        required_widgets = len(self.scheme_dict)

        # update existing widgets
        for idx in range(min(existing_widgets, required_widgets)):
            logging.debug(f"  updating {idx}")
            w = self._layout.itemAt(idx).widget()
            w.setVisible(True)
            w.part_name = names[idx]
            w.color = self.scheme_dict[names[idx]]

        # remove extra widgets
        for i in range(max(existing_widgets - required_widgets, 0)):
            logging.debug(f"  hiding {required_widgets + i}")
            if w := self._layout.itemAt(required_widgets + i).widget():
                logging.debug(f"  done!")
                w.setVisible(False)

        # add missing widgets
        for i in range(max(required_widgets - existing_widgets, 0)):
            logging.debug(f"  adding {existing_widgets + i}")
            name = names[existing_widgets + i]
            self.add_entry(name, self.scheme_dict[name])
        logging.debug(f"  done!")

    def reset(self):
        self.scheme_dict = {}
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            logging.debug(f"making {w} invisible")
            w.setVisible(False)
