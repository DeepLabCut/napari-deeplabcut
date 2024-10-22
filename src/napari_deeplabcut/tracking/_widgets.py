from magicgui import magic_factory
import pandas as pd
import numpy as np
import debugpy
from functools import partial
from magicgui.widgets import CheckBox, Container, create_widget, ComboBox, Slider, SpinBox
from napari.types import PointsData, ImageData
from napari.layers import Points, Image
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QProgressBar
from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QSlider, QLabel, QComboBox, QSizePolicy, QGridLayout
from qtpy.QtCore import Qt, Slot, Signal
from skimage.util import img_as_float
import napari
from napari.viewer import Viewer
from napari.utils.events.event import Event

from napari_deeplabcut.tracking._worker import TrackingWorker, TrackingWorkerData
from napari_deeplabcut.keypoints import KeypointStore


class TrackingControls(QWidget):
    trackingRequested = Signal(TrackingWorkerData)
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        from napari_deeplabcut._widgets import KeypointControls
        self._viewer: Viewer = viewer

        # Layout
        self._tracking_method_combo = QComboBox()
        self._keypoint_layer_combo: ComboBox = create_widget(annotation=Points)
        self._video_layer_combo: ComboBox = create_widget(annotation=Image)
        self._video_layer_combo.changed.connect(self._video_layer_changed)
        self._backward_slider = QSlider(Qt.Horizontal)
        self._backward_spinbox_absolute = QSpinBox()
        self._backward_spinbox_relative = QSpinBox()
        self._reference_spinbox = QSpinBox()
        self._set_ref_button = QPushButton()
        self._forward_slider = QSlider(Qt.Horizontal)
        self._forward_spinbox_absolute = QSpinBox()
        self._forward_spinbox_relative = QSpinBox()
        self._tracking_stop_button = QPushButton()
        self._tracking_forward_button = QPushButton()
        self._tracking_forward_button.clicked.connect(self.track_forward)
        self._tracking_backward_button = QPushButton()
        self._tracking_backward_button.clicked.connect(self.track_backward)
        self._tracking_bothway_button = QPushButton()
        self._tracking_bothway_button.clicked.connect(self.track_bothway)
        self._tracking_progress_bar = QProgressBar()

        # Controls
        self._forward_slider.valueChanged.connect(self._forward_spinbox_relative.setValue)
        self._forward_spinbox_relative.valueChanged.connect(lambda x : (self._forward_slider.setValue(x), self._forward_spinbox_absolute.setValue(self._reference_spinbox.value() + x)))
        self._forward_spinbox_absolute.valueChanged.connect(lambda x : self._forward_spinbox_relative.setValue(x - self._reference_spinbox.value() if x - self._reference_spinbox.value() > 0 else 0))

        self._backward_slider.valueChanged.connect(self._backward_spinbox_relative.setValue)
        self._backward_spinbox_relative.valueChanged.connect(lambda x : (self._backward_slider.setValue(x), self._backward_spinbox_absolute.setValue(self._reference_spinbox.value() + x)))
        self._backward_spinbox_absolute.valueChanged.connect(lambda x : self._backward_spinbox_relative.setValue(x - self._reference_spinbox.value() if x - self._reference_spinbox.value() < 0 else 0))

        self._viewer.dims.events.current_step.connect(lambda e: self._reference_spinbox.setValue(e.value[0]))
        self._reference_spinbox.valueChanged.connect(self._update_controls)

        # Worker
        self.is_tracking = False
        self.worker_started = False
        self.worker: TrackingWorker | None = None

        # Reference to the keypoint control widget.
        # this gets assigned after the user requests tracking for the first time.
        self.keypoint_widget: KeypointControls | None = None

        self._build_layout()

    @Slot(int)
    def _update_controls(self, new_val: int):
        if self.video_layer is None:
            return
        self._forward_slider.setRange(0, self.video_layer.data.shape[0] - 1 - new_val)
        self._forward_spinbox_relative.setRange(0, self.video_layer.data.shape[0] - 1 - new_val)
        self._forward_spinbox_absolute.setRange(new_val, self.video_layer.data.shape[0] - 1)
        self._forward_spinbox_absolute.setValue(new_val+self._forward_spinbox_relative.value())

        self._backward_slider.setRange(-new_val, 0)
        self._backward_spinbox_relative.setRange(-new_val, 0)
        self._backward_spinbox_absolute.setRange(0, new_val)
        self._backward_spinbox_absolute.setValue(new_val+self._backward_spinbox_relative.value())

        self._viewer.dims.current_step = (new_val, *self._viewer.dims.current_step[1:])

    def _start_worker(self):
        self.is_tracking = False
        self.worker_started = False
        self.worker = TrackingWorker()
        self.worker.trackingStarted.connect(self.tracking_started)
        self.worker.started.connect(partial(setattr, self, 'worker_started', True))
        self.worker.finished.connect(partial(setattr, self, 'worker_started', False))
        self.worker.progress.connect(lambda x: (self._tracking_progress_bar.setMaximum(x[1]), self._tracking_progress_bar.setValue(x[0])) if self._tracking_progress_bar.maximum() != x[1] else self._tracking_progress_bar.setValue(x[0]))
        self.worker.trackingFinished.connect(self.tracking_finished)
        self.trackingRequested.connect(self.worker.track)

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
        self._update_controls(0)

    @Slot()
    def tracking_started(self):
        self.is_tracking = True
        self._tracking_progress_bar.setValue(0)

    @Slot(TrackingWorkerData)
    def tracking_finished(self, trackingdata: TrackingWorkerData):
        self.is_tracking = False
        try:
            self.add_keypoints_to_layer(trackingdata.keypoints, trackingdata.keypoint_features)
        except Exception as e:
            print(e)
        self._tracking_progress_bar.setValue(self._tracking_progress_bar.maximum())

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

        merged_keypoints = np.vstack(merged_keypoints) if merged_keypoints else np.empty((0, current_keypoints.shape[1]))
        merged_feature_df = pd.concat(merged_features, ignore_index=True)

        self.keypoint_layer.data = merged_keypoints
        self.keypoint_layer.features = merged_feature_df


    @Slot()
    def track_forward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        forward_frame_idx: int = self._forward_spinbox_absolute.value()
        self.track((ref_frame_idx, forward_frame_idx+1), ref_frame_idx, backward_tracking=False)

    @Slot()
    def track_backward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        backward_frame_idx: int = self._backward_spinbox_absolute.value()
        self.track((backward_frame_idx, ref_frame_idx+1), ref_frame_idx, backward_tracking=True)

    @Slot()
    def track_bothway(self):
        pass


    def track(self, keypoint_range: tuple[int, int], ref_frame_idx, backward_tracking=False):
        if not self.worker_started:
            self._start_worker()
        if not self.keypoint_widget:
            for k,v in self._viewer.window._dock_widgets.items():
                if 'Keypoint controls' in k and 'napari-deeplabcut' in k:
                    self.keypoint_widget = v.widget()
                    break
        if self.is_tracking:
            return

        if backward_tracking:
            video_slice = self.video_layer.data[keypoint_range[0]:keypoint_range[1]][::-1]
        else:
            video_slice = self.video_layer.data[keypoint_range[0]:keypoint_range[1]]
        keypoints = self.keypoint_layer.data[self.keypoint_layer.data[:, 0] == ref_frame_idx]
        keypoints[:, 0] = 0
        keypoint_features = self.keypoint_layer.features[self.keypoint_layer.data[:, 0] == ref_frame_idx]
        tracking_data = TrackingWorkerData(
            tracker=self._tracking_method_combo.currentText(),
            video=video_slice,
            keypoints=keypoints,
            keypoint_features=keypoint_features,
            keypoint_range=keypoint_range,
            backward_tracking=backward_tracking
        )
        self.trackingRequested.emit(tracking_data)
        

    def _build_layout(self):
        self.setLayout(QVBoxLayout())
        self._tracking_method_combo.addItems(["Cotracker", "PIP"])
        self._tracking_method_combo.setCurrentText("Cotracker")
        _tracking_method_layout = QHBoxLayout()
        _tracking_method_layout.addWidget(QLabel("Tracker"))
        _tracking_method_layout.addWidget(self._tracking_method_combo)
        self._viewer.layers.events.inserted.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._keypoint_layer_combo.reset_choices)
        _keypoint_layer_method_layout = QHBoxLayout()
        _keypoint_layer_method_layout.addWidget(QLabel("Keypoints"))
        _keypoint_layer_method_layout.addWidget(self._keypoint_layer_combo.native)
        self._viewer.layers.events.inserted.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._video_layer_combo.reset_choices)
        _video_layer_method_layout = QHBoxLayout()
        _video_layer_method_layout.addWidget(QLabel("Video"))
        _video_layer_method_layout.addWidget(self._video_layer_combo.native)

        self.layout().addLayout(_tracking_method_layout)
        self.layout().addLayout(_keypoint_layer_method_layout)
        self.layout().addLayout(_video_layer_method_layout)
        range_controls_layout = QGridLayout() # 3 by 5
        self._backward_slider.setRange(-100, 0)
        # self._backward_slider.setInvertedAppearance(True)
        range_controls_layout.addWidget(self._backward_slider, 0, 0, 1, 2)
        self._backward_spinbox_absolute.setRange(0, 100)
        self._backward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_absolute.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(self._backward_spinbox_absolute, 1, 1)
        range_controls_layout.addWidget(QLabel("<< Abs"), 1, 0)
        self._backward_spinbox_relative.setRange(-100, 0)
        self._backward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_relative.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("<< Rel"), 2, 0)
        range_controls_layout.addWidget(self._backward_spinbox_relative, 2, 1)
        _ref_label = QLabel("Ref")
        self._reference_spinbox.setRange(0, 100)
        self._reference_spinbox.setAlignment(Qt.AlignCenter)
        self._reference_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(self._reference_spinbox, 1, 2)
        _ref_label.setAlignment(Qt.AlignCenter)
        range_controls_layout.addWidget(_ref_label, 0, 2)
        self._set_ref_button.setText("Set")
        range_controls_layout.addWidget(self._set_ref_button, 2, 2)
        self._forward_slider.setRange(0, 100)
        range_controls_layout.addWidget(self._forward_slider, 0, 3, 1, 2)
        self._forward_spinbox_absolute.setRange(0, 100)
        self._forward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_absolute.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("Abs >>"), 1, 4)
        range_controls_layout.addWidget(self._forward_spinbox_absolute, 1, 3)
        self._forward_spinbox_relative.setRange(0, 100)
        self._forward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_relative.setStyleSheet("""
            QSpinBox {
                padding: 0;
            }
        """)
        range_controls_layout.addWidget(QLabel("Rel >>"), 2, 4)
        range_controls_layout.addWidget(self._forward_spinbox_relative, 2, 3)

        self.layout().addLayout(range_controls_layout)
        tracking_controls_layout = QGridLayout() # 2 by 5
        self._tracking_stop_button.setText("□")
        tracking_controls_layout.addWidget(self._tracking_stop_button, 0, 2)
        self._tracking_forward_button.setText("⇥")
        tracking_controls_layout.addWidget(self._tracking_forward_button, 0, 3)
        self._tracking_bothway_button.setText("↹")
        self._tracking_backward_button.setText("⇤")
        tracking_controls_layout.addWidget(self._tracking_backward_button, 0, 1)
        tracking_controls_layout.addWidget(self._tracking_bothway_button, 1, 2)

        self._tracking_progress_bar.setRange(0, 100)
        self.layout().addLayout(tracking_controls_layout)
        self.layout().addWidget(self._tracking_progress_bar)
