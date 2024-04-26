### ------------- Custom widgets for tracking module -------------- ###
import logging
import threading
from datetime import datetime
from functools import partial
from typing import Optional
from datetime import datetime


import napari
from qtpy import QtCore
from qtpy.QtCore import QObject
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


### -------------- UI utilities -------------- ###
class ContainerWidget(QWidget):
    """Class for a container widget that can contain other widgets."""

    def __init__(
        self, l=0, t=0, r=1, b=11, vertical=True, parent=None, fixed=True
    ):
        """Creates a container widget that can contain other widgets.

        Args:
            l: left margin in pixels
            t: top margin in pixels
            r: right margin in pixels
            b: bottom margin in pixels
            vertical: if True, renders vertically. Horizontal otherwise
            parent: parent QWidget
            fixed: uses QLayout.SetFixedSize if True
        """
        super().__init__(parent)
        self.layout = None

        if vertical:
            self.layout = QVBoxLayout(self)
        else:
            self.layout = QHBoxLayout(self)

        self.layout.setContentsMargins(l, t, r, b)
        if fixed:
            self.layout.setSizeConstraint(QLayout.SetFixedSize)


def get_time():
    """Get time in the following format : hour:minute:second. NOT COMPATIBLE with file paths (saving with ":" is invalid)."""
    return f"{datetime.now():%H:%M:%S}"


def add_widgets(layout, widgets):
    """Adds all widgets in the list to layout, with the specified alignment.

    If alignment is None, no alignment is set.

    Args:
        layout: layout to add widgets in
        widgets: list of QWidgets to add to layout
    """
    for w in widgets:
        layout.addWidget(w)


def make_label(name, parent=None):  # TODO update to child class
    """Creates a QLabel.

    Args:
        name: string with name
        parent: parent widget

    Returns: created label

    """
    label = QLabel(name, parent) if parent is not None else QLabel(name)
    return label


class QWidgetSingleton(type(QObject)):
    """To be used as a metaclass when making a singleton QWidget, meaning only one instance exists at a time.

    Avoids unnecessary memory overhead and keeps user parameters even when a widget is closed.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Ensure only one instance of a QWidget with QWidgetSingleton as a metaclass exists at a time."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


### -------------- Tracking widgets -------------- ###


class DropdownMenu(QComboBox):
    """Creates a dropdown menu with a title and adds specified entries to it."""

    def __init__(
        self,
        entries: Optional[list] = None,
        parent: Optional[QWidget] = None,
        text_label: Optional[str] = None,
        fixed: Optional[bool] = True,
    ):
        """Creates a dropdown menu with a title and adds specified entries to it.

        Args:
            entries (array(str)): Entries to add to the dropdown menu. Defaults to None, no entries if None
            parent (QWidget): parent QWidget to add dropdown menu to. Defaults to None, no parent is set if None
            text_label (str) : if not None, creates a QLabel with the contents of 'label', and returns the label as well
            fixed (bool): if True, will set the size policy of the dropdown menu to Fixed in h and w. Defaults to True.
        """
        super().__init__(parent)
        self.label = None
        if entries is not None:
            self.addItems(entries)
        if text_label is not None:
            self.label = QLabel(text_label)
        if fixed:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def get_items(self):
        """Returns the items in the dropdown menu."""
        return [self.itemText(i) for i in range(self.count())]


class LayerSelecter(ContainerWidget):
    """Class that creates a dropdown menu to select a layer from a napari viewer."""

    def __init__(
        self, viewer, name="Layer", layer_type=napari.layers.Layer, parent=None
    ):
        """Creates an instance of LayerSelecter."""
        super().__init__(parent=parent, fixed=False)
        self._viewer = viewer
        self.layer_type = layer_type

        self.layer_list = DropdownMenu(
            parent=self, text_label=name, fixed=False
        )
        # if it's a keypoint layer, show the number of keypoints
        if layer_type == napari.layers.Points:
            self.layer_description = make_label(
                "Number of keypoints:", parent=self
            )
        else:
            self.layer_description = make_label("Video :", parent=self)
        self.layer_description.setVisible(False)
        # self.layer_list.setSizeAdjustPolicy(QComboBox.AdjustToContents) # use tooltip instead ?

        # connect to LayerList events
        self._viewer.layers.events.inserted.connect(partial(self._add_layer))
        self._viewer.layers.events.removed.connect(partial(self._remove_layer))
        self._viewer.layers.events.changed.connect(self._check_for_layers)

        # update self.layer_list when layers are added or removed
        self.layer_list.currentIndexChanged.connect(self._update_tooltip)
        self.layer_list.currentTextChanged.connect(self._update_description)

        add_widgets(
            self.layout,
            [self.layer_list.label, self.layer_list, self.layer_description],
        )
        self._check_for_layers()

    def _get_all_layers(self):
        return [
            self.layer_list.itemText(i) for i in range(self.layer_list.count())
        ]

    def _check_for_layers(self):
        """Check for layers of the correct type and update the dropdown menu.

        Also removes layers that have been removed from the viewer.
        """
        for layer in self._viewer.layers:
            layer.events.name.connect(self._rename_layer)

            if (
                isinstance(layer, self.layer_type)
                and layer.name not in self._get_all_layers()
            ):
                logger.debug(
                    f"Layer {layer.name} - List : {self._get_all_layers()}"
                )
                # add new layers of correct type
                self.layer_list.addItem(layer.name)
                logger.debug(f"Layer {layer.name} has been added to the menu")
                # break
                # once added, check again for previously renamed layers
                self._check_for_removed_layer(layer)

            if layer.name in self._get_all_layers() and not isinstance(
                layer, self.layer_type
            ):
                # remove layers of incorrect type
                index = self.layer_list.findText(layer.name)
                self.layer_list.removeItem(index)
                logger.debug(
                    f"Layer {layer.name} has been removed from the menu"
                )

        self._check_for_removed_layers()
        self._update_tooltip()
        self._update_description()

    def _check_for_removed_layer(self, layer):
        """Check if a specific layer has been removed from the viewer and must be removed from the menu."""
        if isinstance(layer, str):
            name = layer
        elif isinstance(layer, self.layer_type):
            name = layer.name
        else:
            logger.warning("Layer is not a string or a valid napari layer")
            return

        if name in self._get_all_layers() and name not in [
            l.name for l in self._viewer.layers
        ]:
            index = self.layer_list.findText(name)
            self.layer_list.removeItem(index)
            logger.debug(f"Layer {name} has been removed from the menu")

    def _check_for_removed_layers(self):
        """Check for layers that have been removed from the viewer and must be removed from the menu."""
        for layer in self._get_all_layers():
            self._check_for_removed_layer(layer)

    def _update_tooltip(self):
        self.layer_list.setToolTip(self.layer_list.currentText())

    def _update_description(self):
        try:
            if self.layer_list.currentText() != "":
                try:
                    if self.layer_type == napari.layers.Points:
                        shape_desc = f"{len(self.layer_data())} keypoints"
                    else:
                        shape_desc = f"{self.layer_data().shape} frames"
                    self.layer_description.setText(shape_desc)
                    self.layer_description.setVisible(True)
                except AttributeError:
                    self.layer_description.setVisible(False)
            else:
                self.layer_description.setVisible(False)
        except KeyError:
            self.layer_description.setVisible(False)

    def _add_layer(self, event):
        inserted_layer = event.value

        if isinstance(inserted_layer, self.layer_type):
            self.layer_list.addItem(inserted_layer.name)

        # check for renaming
        inserted_layer.events.name.connect(self._rename_layer)

    def _rename_layer(self, _):
        # on layer rename, check for removed/new layers
        self._check_for_layers()

    def _remove_layer(self, event):
        removed_layer = event.value

        if isinstance(
            removed_layer, self.layer_type
        ) and removed_layer.name in [
            self.layer_list.itemText(i) for i in range(self.layer_list.count())
        ]:
            index = self.layer_list.findText(removed_layer.name)
            self.layer_list.removeItem(index)

    def layer(self):
        """Returns the layer selected in the dropdown menu."""
        try:
            return self._viewer.layers[self.layer_name()]
        except ValueError:
            return None

    def layer_name(self):
        """Returns the name of the layer selected in the dropdown menu."""
        try:
            return self.layer_list.currentText()
        except (KeyError, ValueError):
            logger.warning("Layer list is empty")
            return None

    def layer_data(self):
        """Returns the data of the layer selected in the dropdown menu."""
        if self.layer_list.count() < 1:
            logger.debug("Layer list is empty")
            return None
        try:
            if self.layer_type == napari.layers.Points:
                return self.layer().features
            else:
                return self.layer().data
        except (KeyError, ValueError):
            msg = f"Layer {self.layer_name()} has no data. Layer might have been renamed or removed."
            logger.warning(msg)
            return None


class Log(QTextEdit):
    """Class to implement a log for important user info. Should be thread-safe."""

    def __init__(self, parent=None):
        """Creates a log with a lock for multithreading.

        Args:
            parent (QWidget): parent widget to add Log instance to.
        """
        super().__init__(parent)

        # from qtpy.QtCore import QMetaType
        # parent.qRegisterMetaType<QTextCursor>("QTextCursor")

        self.lock = threading.Lock()

    def flush(self):
        """Flush the log."""

    def write(self, message):
        """Write message to log in a thread-safe manner.

        Args:
            message: string to be printed
        """
        self.lock.acquire()
        try:
            if not hasattr(self, "flag"):
                self.flag = False
            message = message.replace("\r", "").rstrip()
            if message:
                method = "replace_last_line" if self.flag else "append"
                QtCore.QMetaObject.invokeMethod(
                    self,
                    method,
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, message),
                )
                self.flag = True
            else:
                self.flag = False

        finally:
            self.lock.release()

    @QtCore.Slot(str)
    def replace_last_line(self, text):
        """Replace last line. For use in progress bar.

        Args:
            text: string to be printed
        """
        self.lock.acquire()
        try:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.insertBlock()
            self.setTextCursor(cursor)
            self.insertPlainText(text)
        finally:
            self.lock.release()

    def print_and_log(self, text, printing=True):
        """Utility used to both print to terminal and log text to a QTextEdit item in a thread-safe manner. Use only for important user info.

        Args:
            text (str): Text to be printed and logged
            printing (bool): Whether to print the message as well or not using logger.info(). Defaults to True.

        """
        self.lock.acquire()
        try:
            if printing:
                logger.info(text)
            # causes issue if you clik on terminal (tied to CMD QuickEdit mode on Windows)
            self.moveCursor(QTextCursor.End)
            self.insertPlainText(f"\n{text}")
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )
        finally:
            self.lock.release()

    def warn(self, warning):
        """Show logger.warning from another thread.

        Args:
            warning: warning to be printed
        """
        self.lock.acquire()
        try:
            logger.warning(warning)
        finally:
            self.lock.release()

    def error(self, error, msg=None):
        """Show exception and message from another thread.

        Args:
            error: error to be printed
            msg: message to be printed
        """
        self.lock.acquire()
        try:
            logger.error(error, exc_info=True)
            if msg is not None:
                self.print_and_log(f"{msg} : {error}", printing=False)
            else:
                self.print_and_log(
                    f"Exception caught in another thread : {error}",
                    printing=False,
                )
            raise error
        finally:
            self.lock.release()
