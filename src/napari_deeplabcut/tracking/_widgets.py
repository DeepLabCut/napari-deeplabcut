import logging
import os
from dataclasses import dataclass
from functools import partial

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, create_widget
from napari.layers import Image, Points
from napari.viewer import Viewer
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut.keypoints import KeypointStore
from napari_deeplabcut.tracking._data import TrackingWorkerData, TrackingWorkerOutput
from napari_deeplabcut.tracking._models import AVAILABLE_TRACKERS
from napari_deeplabcut.tracking._worker import TrackingWorker

# Keybinds
TRACKING_SHORTCUTS_ENABLED = os.environ.get("NAPARI_DLC_TRACKING_SHORTCUTS_ENABLED", "1") == "1"


@dataclass(frozen=True)
class TrackingKeybindConfig:
    key: str
    tooltip: str

    def get_display(self) -> str:
        txt = self.tooltip
        if TRACKING_SHORTCUTS_ENABLED:
            txt += f" ({self.key})"
        return txt


TRACK_FORWARD = TrackingKeybindConfig(key="l", tooltip="Track forward")
TRACK_FORWARD_END = TrackingKeybindConfig(key="k", tooltip="Track forward to end")
TRACK_BACKWARD = TrackingKeybindConfig(key="h", tooltip="Track backward")
TRACK_BACKWARD_END = TrackingKeybindConfig(key="j", tooltip="Track backward to start")
MOVE_FORWARD_FRAME = TrackingKeybindConfig(key="i", tooltip="Move forward one frame")
MOVE_BACKWARD_FRAME = TrackingKeybindConfig(key="u", tooltip="Move backward one frame")

logger = logging.getLogger(__name__)


class TrackingControls(QWidget):
    trackingRequested = Signal(TrackingWorkerData)
    trackedKeypointsAdded = Signal()

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        from napari_deeplabcut._widgets import KeypointControls

        self._viewer: Viewer = viewer

        # Layout
        ## Data and model selection
        self._tracking_method_combo = QComboBox()
        self._model_info_button = QToolButton()
        self._model_info_button.setIcon(QIcon.fromTheme("help-about"))
        self._model_info_button.setIconSize(self._model_info_button.iconSize() * 1.2)
        self._keypoint_layer_combo: ComboBox = create_widget(annotation=Points)
        self._video_layer_combo: ComboBox = create_widget(annotation=Image)
        self._video_layer_combo.changed.connect(self._video_layer_changed)
        ## Frame selection controls
        self._set_ref_button = QPushButton()
        self._reference_spinbox = QSpinBox()
        self._reference_spinbox.setReadOnly(True)
        self._reference_spinbox.setButtonSymbols(QSpinBox.NoButtons)
        # self._reference_spinbox.setEnabled(False)
        self._updating_controls = False
        ### Backward
        self._backward_slider = QSlider(Qt.Horizontal)
        self._backward_spinbox_absolute = QSpinBox()
        self._backward_spinbox_relative = QSpinBox()
        ### Forward
        self._forward_slider = QSlider(Qt.Horizontal)
        self._forward_spinbox_absolute = QSpinBox()
        self._forward_spinbox_relative = QSpinBox()
        ## Tracking controls
        self._tracking_stop_button = QPushButton()
        self._tracking_forward_button = QPushButton()
        self._tracking_forward_button.clicked.connect(self.track_forward)
        self._tracking_forward_end_button = QPushButton()
        self._tracking_forward_end_button.clicked.connect(self.track_forward_end)
        self._tracking_backward_button = QPushButton()
        self._tracking_backward_end_button = QPushButton()
        self._tracking_backward_end_button.clicked.connect(self.track_backward_end)
        self._tracking_backward_button.clicked.connect(self.track_backward)
        self._tracking_bothway_button = QPushButton()
        self._tracking_bothway_button.clicked.connect(self.track_bothway)
        self._tracking_progress_bar = QProgressBar()

        # Controls
        ## Forward controls
        self._forward_slider.valueChanged.connect(partial(self._forward_update, from_absolute=False, from_slider=True))
        self._forward_spinbox_relative.valueChanged.connect(
            partial(self._forward_update, from_absolute=False, from_slider=False)
        )
        self._forward_spinbox_absolute.valueChanged.connect(
            partial(self._forward_update, from_absolute=True, from_slider=False)
        )
        ## Backward controls
        self._backward_slider.valueChanged.connect(
            partial(self._backward_update, from_absolute=False, from_slider=True)
        )
        self._backward_spinbox_relative.valueChanged.connect(
            partial(self._backward_update, from_absolute=False, from_slider=False)
        )
        self._backward_spinbox_absolute.valueChanged.connect(
            partial(self._backward_update, from_absolute=True, from_slider=False)
        )

        # when the range of viewer dims changes (e.g. on opening a new video), update the reference spinbox max
        self._viewer.dims.events.current_step.connect(lambda e: self._reference_spinbox.setValue(int(e.value[0])))
        self._viewer.dims.events.current_step.connect(self._set_frame_controls_range)
        self._reference_spinbox.valueChanged.connect(self._set_frame_controls_range)

        # Worker
        self.is_tracking = False
        self.worker_started = False
        self.worker: TrackingWorker | None = None

        # Reference to the keypoint control widget.
        # this gets assigned after the user requests tracking for the first time.
        self.keypoint_widget: KeypointControls | None = None

        self._setup_keybindings(viewer=viewer)

        self._build_layout()

    def _set_model_info_tooltip(self, current_model_name: str = None):
        """Retrieves the display info for the selected model and sets it as tooltip for the model info button."""
        tracker_info = AVAILABLE_TRACKERS.get(current_model_name, None)
        tracker_info = tracker_info["class"].info_text if tracker_info is not None else None
        if tracker_info is not None:
            self._model_info_button.setToolTip(tracker_info)
        else:
            self._model_info_button.setToolTip("")

    def _set_tooltips(self):
        self._tracking_forward_button.setToolTip(TRACK_FORWARD.get_display())
        self._tracking_forward_end_button.setToolTip(TRACK_FORWARD_END.get_display())
        self._tracking_backward_button.setToolTip(TRACK_BACKWARD.get_display())
        self._tracking_backward_end_button.setToolTip(TRACK_BACKWARD_END.get_display())
        self._tracking_bothway_button.setToolTip("Track both ways")
        self._tracking_stop_button.setToolTip("Stop tracking")
        self._set_ref_button.setToolTip("Set reference frame")

    def _setup_keybindings(self, viewer: "napari.viewer.Viewer"):
        @Points.bind_key(TRACK_FORWARD.key, overwrite=True)
        def track_forward(event):
            self.track_forward()

        @Points.bind_key(TRACK_FORWARD_END.key, overwrite=True)
        def track_forward_end(event):
            self.track_forward_end()

        @Points.bind_key(TRACK_BACKWARD.key, overwrite=True)
        def track_backward(event):
            self.track_backward()

        @Points.bind_key(TRACK_BACKWARD_END.key, overwrite=True)
        def track_backward_end(event):
            self.track_backward_end()

        @Points.bind_key(MOVE_FORWARD_FRAME.key, overwrite=True)
        def move_forward_frame(event):
            viewer.dims.current_step = (
                viewer.dims.current_step[0] + 1,
                *viewer.dims.current_step[1:],
            )

        @Points.bind_key(MOVE_BACKWARD_FRAME.key, overwrite=True)
        def move_backward_frame(event):
            viewer.dims.current_step = (
                viewer.dims.current_step[0] - 1,
                *viewer.dims.current_step[1:],
            )

        self._set_tooltips()

    def _update_frame_controls(
        self,
        slider,
        relative_spinbox,
        absolute_spinbox,
        reference_spinbox,
        value,
        direction,
        from_absolute=False,
        from_slider=False,
    ):
        """
        Generic function to update slider, relative spinbox, and absolute spinbox.

        Parameters:
        - slider: The slider widget.
        - relative_spinbox: The relative spinbox widget.
        - absolute_spinbox: The absolute spinbox widget.
        - reference_spinbox: The reference spinbox widget.
        - value: The new value to set.
        - direction: "forward" or "backward".
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            if from_absolute:
                # Update relative and slider from absolute spinbox
                relative_value = value - reference_spinbox.value()
                if direction == "forward":
                    relative_value = max(0, relative_value)
                else:  # backward
                    relative_value = min(0, relative_value)
                relative_spinbox.setValue(relative_value)
                slider.setValue(relative_value)
            elif from_slider:
                # Update relative and absolute spinboxes from slider
                relative_spinbox.setValue(value)
                absolute_spinbox.setValue(reference_spinbox.value() + value)
            else:
                # Update slider and absolute spinbox from relative spinbox
                slider.setValue(value)
                absolute_spinbox.setValue(reference_spinbox.value() + value)
        finally:
            self._updating_controls = False

    def _forward_update(self, value: int, from_absolute: bool, from_slider: bool):
        """Helper to update forward controls.

        Parameters:
        - value: The new value to set.
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        self._update_frame_controls(
            slider=self._forward_slider,
            relative_spinbox=self._forward_spinbox_relative,
            absolute_spinbox=self._forward_spinbox_absolute,
            reference_spinbox=self._reference_spinbox,
            value=value,
            direction="forward",
            from_absolute=from_absolute,
            from_slider=from_slider,
        )

    def _backward_update(self, value: int, from_absolute: bool, from_slider: bool):
        """Helper to update backward controls.

        Parameters:
        - value: The new value to set.
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        self._update_frame_controls(
            slider=self._backward_slider,
            relative_spinbox=self._backward_spinbox_relative,
            absolute_spinbox=self._backward_spinbox_absolute,
            reference_spinbox=self._reference_spinbox,
            value=value,
            direction="backward",
            from_absolute=from_absolute,
            from_slider=from_slider,
        )

    @Slot()
    def _set_frame_controls_range(self):
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            if self.video_layer is None:
                return

            max_frames = self.video_layer.data.shape[0] - 1
            logger.debug(f"Updating tracking controls for video with {max_frames + 1} frames.")
            current_frame = max(0, min(self._viewer.dims.current_step[0], max_frames))
            logger.debug(f"Current frame: {current_frame}")
            self._reference_spinbox.setRange(0, max_frames)
            self._reference_spinbox.setValue(current_frame)

            forward_delta = self._forward_spinbox_relative.value()
            self._forward_slider.setRange(0, max_frames - current_frame)
            self._forward_slider.setValue(forward_delta)
            self._forward_spinbox_relative.setRange(0, max_frames - current_frame)
            self._forward_spinbox_relative.setValue(forward_delta)
            self._forward_spinbox_absolute.setRange(current_frame, max_frames)
            self._forward_spinbox_absolute.setValue(current_frame + forward_delta)  # see _forward_update

            backward_delta = self._backward_spinbox_relative.value()
            self._backward_slider.setRange(-current_frame, 0)
            self._backward_slider.setValue(backward_delta)
            self._backward_spinbox_relative.setRange(-current_frame, 0)
            self._backward_spinbox_relative.setValue(backward_delta)
            self._backward_spinbox_absolute.setRange(0, current_frame)
            self._backward_spinbox_absolute.setValue(
                current_frame + self._backward_spinbox_relative.value()
            )  # see _backward_update

            # self._viewer.dims.current_step = (current_frame, *self._viewer.dims.current_step[1:])
        finally:
            self._updating_controls = False

    def _start_worker(self):
        self.is_tracking = False
        self.worker_started = False
        self.worker = TrackingWorker()
        self.worker.trackingStarted.connect(self.tracking_started)
        self.worker.started.connect(partial(setattr, self, "worker_started", True))
        self.worker.finished.connect(partial(setattr, self, "worker_started", False))
        self.worker.progress.connect(
            lambda x: (
                (
                    self._tracking_progress_bar.setMaximum(x[1]),
                    self._tracking_progress_bar.setValue(x[0]),
                )
                if self._tracking_progress_bar.maximum() != x[1]
                else self._tracking_progress_bar.setValue(x[0])
            )
        )
        self.worker.trackingFinished.connect(self.tracking_finished)
        self.worker.trackingStopped.connect(self.tracking_stopped)
        self.trackingRequested.connect(self.worker.track)
        self._tracking_stop_button.clicked.connect(self.worker.stop_tracking)

        self.worker.start()

    @property
    def keypoint_layer(self) -> Points | None:
        return self._keypoint_layer_combo.value

    @property
    def keypoint_store(self) -> KeypointStore | None:
        return self.keypoint_widget._stores[self.keypoint_layer] if self.keypoint_widget else None

    @property
    def video_layer(self) -> Image | None:
        return self._video_layer_combo.value

    @Slot()
    def _video_layer_changed(self):
        if self._viewer.dims.ndim != 3:
            return
        self._set_frame_controls_range()

    @Slot()
    def tracking_started(self):
        self.is_tracking = True
        self._tracking_progress_bar.setValue(0)

    @Slot(TrackingWorkerOutput)
    def tracking_finished(self, out: TrackingWorkerOutput):
        self.is_tracking = False
        try:
            try:
                new_features_df = (
                    out.keypoint_features
                    if isinstance(out.keypoint_features, pd.DataFrame)
                    else pd.DataFrame(out.keypoint_features)
                )
            except ValueError as e:
                logger.error(f"Failed to convert keypoint features to DataFrame: {e}")
                new_features_df = pd.DataFrame()
            self.add_keypoints_to_layer(out.keypoints, new_features_df)
        except Exception as e:
            print(e)
        self._tracking_progress_bar.setValue(self._tracking_progress_bar.maximum())
        self._tracking_progress_bar.setFormat("%p% Done")
        self.trackedKeypointsAdded.emit()

    @Slot()
    def tracking_stopped(self):
        self.is_tracking = False
        self._tracking_progress_bar.setValue(self._tracking_progress_bar.maximum())
        self._tracking_progress_bar.setFormat("%p% Stopped")

    def add_keypoints_to_layer(self, new_keypoints: np.ndarray, new_features: pd.DataFrame):
        current_keypoints = self.keypoint_layer.data
        current_features: pd.DataFrame = self.keypoint_layer.features

        # Extract unique frame indices
        unique_frames = np.sort(np.unique(np.concatenate((current_keypoints[:, 0], new_keypoints[:, 0]))))

        merged_keypoints = []
        merged_features = []

        for frame in unique_frames:
            # Select keypoints and features for the current frame
            frame_old_keypoints = current_keypoints[current_keypoints[:, 0] == frame]
            frame_old_features = current_features[current_keypoints[:, 0] == frame]

            frame_new_keypoints = new_keypoints[new_keypoints[:, 0] == frame]
            frame_new_features = new_features[new_keypoints[:, 0] == frame]

            # Here we can add custom logic when merging. Right now we overwrite any previous keypoints.
            if len(frame_new_keypoints) > 0:
                # If there are keypoints in new, take those
                merged_keypoints.append(frame_new_keypoints)
                merged_features.append(frame_new_features)
            else:
                merged_keypoints.append(frame_old_keypoints)
                merged_features.append(frame_old_features)

        merged_keypoints = (
            np.vstack(merged_keypoints) if merged_keypoints else np.empty((0, current_keypoints.shape[1]))
        )
        merged_feature_df = pd.concat(merged_features, ignore_index=True)

        self.keypoint_layer.data = merged_keypoints
        self.keypoint_layer.features = merged_feature_df

    @Slot()
    def track_forward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        forward_frame_idx: int = self._forward_spinbox_absolute.value()
        if forward_frame_idx <= ref_frame_idx:
            return
        self.track(
            (ref_frame_idx, forward_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=False,
        )

    @Slot()
    def track_forward_end(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        forward_frame_idx: int = self.video_layer.data.shape[0] - 1
        if forward_frame_idx <= ref_frame_idx:
            return
        self.track(
            (ref_frame_idx, forward_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=False,
        )

    @Slot()
    def track_backward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        backward_frame_idx: int = self._backward_spinbox_absolute.value()
        if backward_frame_idx >= ref_frame_idx:
            return
        print(backward_frame_idx, ref_frame_idx)
        self.track(
            (backward_frame_idx, ref_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=True,
        )

    @Slot()
    def track_backward_end(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        backward_frame_idx: int = 0
        if backward_frame_idx >= ref_frame_idx:
            return
        self.track(
            (backward_frame_idx, ref_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=True,
        )

    @Slot()
    def track_bothway(self):
        # if forward target is invalid, go directly backward
        ref = self._reference_spinbox.value()
        fwd = self._forward_spinbox_absolute.value()
        if fwd <= ref:
            self.track_backward()
            return

        self.track_forward()
        self.trackedKeypointsAdded.connect(self.track_backward, type=Qt.ConnectionType.SingleShotConnection)

    def track(self, keypoint_range: tuple[int, int], ref_frame_idx, backward_tracking=False):
        if not self.worker_started:
            self._start_worker()
        if not self.keypoint_widget:
            for k, v in self._viewer.window._dock_widgets.items():
                if "Keypoint controls" in k and "napari-deeplabcut" in k:
                    self.keypoint_widget = v.widget()
                    break
        if self.is_tracking:
            return
        self._tracking_progress_bar.setFormat("%p%")

        if backward_tracking:
            video_slice = self.video_layer.data[keypoint_range[0] : keypoint_range[1]][::-1]
        else:
            video_slice = self.video_layer.data[keypoint_range[0] : keypoint_range[1]]
        keypoints = self.keypoint_layer.data[self.keypoint_layer.data[:, 0] == ref_frame_idx]
        keypoints[:, 0] = 0
        keypoint_features = self.keypoint_layer.features[self.keypoint_layer.data[:, 0] == ref_frame_idx]
        tracking_data = TrackingWorkerData(
            tracker_name=self._tracking_method_combo.currentText(),  # do not instantiate yet
            video=video_slice,
            keypoints=keypoints,
            keypoint_features=keypoint_features,
            keypoint_range=keypoint_range,
            backward_tracking=backward_tracking,
        )
        self.trackingRequested.emit(tracking_data)

    def _build_layout(self):
        # Layout
        self.setLayout(QVBoxLayout())
        # self._tracking_method_combo.addItems(["Cotracker", "PIP"])
        ## Model selection
        self._tracking_method_combo.addItems(AVAILABLE_TRACKERS.keys())
        # self._tracking_method_combo.setCurrentText("Cotracker")
        self._tracking_method_combo.setCurrentIndex(0)

        _model_info_layout = QHBoxLayout()
        _model_info_layout.addWidget(QLabel("Tracker"))
        _model_info_layout.addWidget(self._model_info_button)
        # _model_info_layout.addStretch(1)

        _tracking_method_layout = QHBoxLayout()
        _tracking_method_layout.addLayout(_model_info_layout)
        _tracking_method_layout.addWidget(self._tracking_method_combo)
        self._tracking_method_combo.currentTextChanged.connect(self._set_model_info_tooltip)
        self._set_model_info_tooltip(self._tracking_method_combo.currentText())
        ## Layer selection
        ### Keypoint layer
        self._viewer.layers.events.inserted.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._keypoint_layer_combo.reset_choices)
        _keypoint_layer_method_layout = QHBoxLayout()
        _keypoint_layer_method_layout.addWidget(QLabel("Keypoints"))
        _keypoint_layer_method_layout.addWidget(self._keypoint_layer_combo.native)
        ### Video layer
        self._viewer.layers.events.inserted.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._video_layer_combo.reset_choices)
        _video_layer_method_layout = QHBoxLayout()
        _video_layer_method_layout.addWidget(QLabel("Video"))
        _video_layer_method_layout.addWidget(self._video_layer_combo.native)

        # Stack previous layouts
        self.layout().addLayout(_tracking_method_layout)
        self.layout().addLayout(_keypoint_layer_method_layout)
        self.layout().addLayout(_video_layer_method_layout)

        ## Frame range controls
        range_controls_layout = QGridLayout()  # 3 by 5
        self._backward_slider.setRange(-100, 0)
        # self._backward_slider.setInvertedAppearance(True)
        range_controls_layout.addWidget(self._backward_slider, 0, 0, 1, 2)
        self._backward_spinbox_absolute.setRange(0, 100)
        self._backward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_absolute.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(self._backward_spinbox_absolute, 1, 1)
        range_controls_layout.addWidget(QLabel("<< Abs"), 1, 0)
        self._backward_spinbox_relative.setRange(-100, 0)
        self._backward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_relative.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("<< Rel"), 2, 0)
        range_controls_layout.addWidget(self._backward_spinbox_relative, 2, 1)
        _ref_label = QLabel("Current")
        self._reference_spinbox.setRange(0, 100)
        self._reference_spinbox.setAlignment(Qt.AlignCenter)
        self._reference_spinbox.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(self._reference_spinbox, 1, 2)
        _ref_label.setAlignment(Qt.AlignCenter)
        range_controls_layout.addWidget(_ref_label, 0, 2)
        # self._set_ref_button.setText("Set")
        # range_controls_layout.addWidget(self._set_ref_button, 2, 2)
        self._forward_slider.setRange(0, 100)
        range_controls_layout.addWidget(self._forward_slider, 0, 3, 1, 2)
        self._forward_spinbox_absolute.setRange(0, 100)
        self._forward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_absolute.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("Abs >>"), 1, 4)
        range_controls_layout.addWidget(self._forward_spinbox_absolute, 1, 3)
        self._forward_spinbox_relative.setRange(0, 100)
        self._forward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_relative.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("Rel >>"), 2, 4)
        range_controls_layout.addWidget(self._forward_spinbox_relative, 2, 3)

        self.layout().addLayout(range_controls_layout)

        ## Start/stop tracking controls

        def themed_icon(name: str, fallback: QStyle.StandardPixmap) -> QIcon:
            """Use napari's QApplication instance to get style and fallback icons."""
            # More consistent than using Unicode characters that may vary in appearance across platforms
            # Esp. on high-res displays they are very small (4K laptop screens etc)
            style = QApplication.instance().style()  # reuse existing app
            return QIcon.fromTheme(name, style.standardIcon(fallback))

        tracking_controls_layout = QGridLayout()  # 2 by 3
        # self._tracking_backward_button.setText("⇤")
        self._tracking_backward_button.setIcon(themed_icon("go-previous", QStyle.SP_ArrowLeft))
        tracking_controls_layout.addWidget(self._tracking_backward_button, 0, 0)
        # self._tracking_backward_end_button.setText("⇤⇤")
        self._tracking_backward_end_button.setIcon(themed_icon("media-seek-backward", QStyle.SP_MediaSeekBackward))
        tracking_controls_layout.addWidget(self._tracking_backward_end_button, 1, 0)

        # self._tracking_stop_button.setText("□")
        self._tracking_stop_button.setIcon(themed_icon("media-playback-stop", QStyle.SP_MediaStop))
        tracking_controls_layout.addWidget(self._tracking_stop_button, 0, 1)

        self._tracking_forward_button.setIcon(themed_icon("go-next", QStyle.SP_ArrowRight))
        # self._tracking_forward_button.setText("⇥")
        tracking_controls_layout.addWidget(self._tracking_forward_button, 0, 2)
        # self._tracking_forward_end_button.setText("⇥⇥")
        self._tracking_forward_end_button.setIcon(themed_icon("media-seek-forward", QStyle.SP_MediaSeekForward))
        tracking_controls_layout.addWidget(self._tracking_forward_end_button, 1, 2)

        # self._tracking_bothway_button.setText("↹")
        # TODO : better icon ? Not really any standard icon for "both way"
        self._tracking_bothway_button.setIcon(themed_icon("view-refresh", QStyle.SP_BrowserReload))
        tracking_controls_layout.addWidget(self._tracking_bothway_button, 1, 1)

        self._tracking_progress_bar.setRange(0, 100)
        self.layout().addLayout(tracking_controls_layout)
        self.layout().addWidget(self._tracking_progress_bar)
