# TODO :
# - Implement the tracking worker with multithreading
# - Implement a Log widget to display the tracking progress + a progress bar
# - Prepare I/O with the actual tracking backend
from functools import partial
from pathlib import Path

import napari
import numpy as np
from napari._qt.qthreading import GeneratorWorker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.utils._qthreading import GeneratorWorkerSignals, WorkerBaseSignals

from napari_deeplabcut._tracking_utils import (
    ContainerWidget,
    LayerSelecter,
    Log,
    QWidgetSingleton,
    add_widgets,
    get_time,
)


class TrackingModule(QWidget, metaclass=QWidgetSingleton):
    """Plugin for tracking."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        """Creates a widget with links to documentation and about page."""
        super().__init__()
        self._viewer = napari_viewer
        self._worker = None
        self._keypoint_layer = None
        ### Widgets ###
        self.video_layer_dropdown = LayerSelecter(
            self._viewer,
            name="Video layer",
            layer_type=napari.layers.Image,
            parent=self,
        )
        self.keypoint_layer_dropdown = LayerSelecter(
            self._viewer,
            name="Keypoint layer",
            layer_type=napari.layers.Points,
            parent=self,
        )
        self.start_button = QPushButton("Start tracking")
        self.start_button.clicked.connect(self._start)
        #############################
        # status report docked widget
        self.container_docked = False  # check if already docked

        self.report_container = ContainerWidget(l=10, t=5, r=5, b=5)

        self.report_container.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Minimum
        )
        self.progress = QProgressBar(self.report_container)
        self.progress.setVisible(False)
        """Widget for the progress bar"""

        self.log = Log(self.report_container)
        self.log.setVisible(False)
        """Read-only display for process-related info. Use only for info destined to user."""
        self._build()

    # Use @property to get/set the keypoint layer
    @property
    def keypoint_layer(self):
        """Get the keypoint layer."""
        return self._keypoint_layer

    @keypoint_layer.setter
    def keypoint_layer(self, layer_name):
        """Set the keypoint layer from the viewer."""
        for l in self._viewer.layers:
            if l.name == layer_name:
                self._keypoint_layer = l
                break

    def _build(self):
        """Create a TrackingModule plugin with :

        - A dropdown menu to select the keypoint layer
        - A set of keypoints to track
        - A button to start tracking
        - A Log that shows when starting. providing feedback on the tracking process
        """
        layout = QVBoxLayout()

        widgets = [
            self.video_layer_dropdown,
            self.keypoint_layer_dropdown,
            self.start_button,
        ]
        add_widgets(layout, widgets)
        self.setLayout(layout)

    def check_ready(self):
        """Check if the inputs are ready for tracking."""
        if self.video_layer_dropdown.layer is None:
            return False
        if self.keypoint_layer_dropdown.layer is None:
            return False
        return True

    def _display_status_report(self):
        """Adds a text log, a progress bar and a "save log" button on the left side of the viewer (usually when starting a worker)."""

        if self.container_docked:
            self.log.clear()
        elif not self.container_docked:
            add_widgets(
                self.report_container.layout,
                [self.progress, self.log],
            )
            self.report_container.setLayout(self.report_container.layout)
            report_dock = self._viewer.window.add_dock_widget(
                self.report_container,
                name="Status report",
                area="left",
                allowed_areas=["left", "right"],
            )
            report_dock._close_btn = False

            # self.docked_widgets.append(report_dock)
            self.container_docked = True

        self.log.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)

    def _update_progress_bar(self, current_frame, total_frame):
        """Update the progress bar."""
        pbar_value = (current_frame / total_frame) * 100
        if pbar_value > 100:
            pbar_value = 100

        self.progress.setValue(pbar_value)

    def _start(self):
        """Start the tracking process."""
        print("Started tracking")

        ### Below is code to start the worker and update the button for the use to start/stop the tracking process
        # if not self._check_ready():
        #     err = "Aborting, please choose valid inputs"
        #     self.log.print_and_log(err)
        #     raise ValueError(err)

        if self._worker is not None:
            if self._worker.is_running:
                pass
            else:
                self._worker.start()
                self.start_button.setText("Running... Click to stop")

        else:
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)
            self._setup_worker()

        if self._worker.is_running:  # if worker is running, tries to stop
            self.log.print_and_log(
                "Stop request, waiting for next inference..."
            )
            self.start_button.setText("Stopping...")
            self._worker.quit()
        else:  # once worker is started, update buttons
            self._worker.start()
            self.start_button.setText("Running...  Click to stop")

    def _setup_worker(self):
        self._worker = TrackingWorker()

        self._worker.started.connect(self._on_start)

        self._worker.log_signal.connect(self.log.print_and_log)
        self._worker.log_w_replace_signal.connect(self.log.replace_last_line)
        self._worker.warn_signal.connect(self.log.warn)
        self._worker.error_signal.connect(self.log.error)

        self._worker.yielded.connect(partial(self._on_yield))
        self._worker.errored.connect(partial(self._on_error))
        self._worker.finished.connect(self._on_finish)

        keypoint_cord = self.keypoint_layer_dropdown.layer_data()
        frames = self.video_layer_dropdown.layer_data()

        self.log.print_and_log(f"keypoint started at {keypoint_cord}")
        self.log.print_and_log(f"frames started at {frames}")


    def _on_yield(self, results):
        # TODO : display the results in the viewer
        # Testing version where an int i is yielded
        ############################
        self.log.print_and_log(f"Yielded {results}")
        self._update_progress_bar(results, 10)
        ############################

    def _on_start(self):
        """Catches start signal from worker to call :py:func:`~display_status_report`."""
        self._display_status_report()
        self.log.print_and_log(f"Worker started at {get_time()}")
        self.log.print_and_log("Worker is running...")

    def _on_error(self, error):
        """Catches errors and tries to clean up."""
        self.log.print_and_log("!" * 20)
        self.log.print_and_log("Worker errored...")
        self.log.error(error)
        self._worker.quit()
        self.on_finish()

    def _on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {get_time()}")
        self.log.print_and_log("*" * 20)
        self.start_button.setText("Start")

        self._worker = None

        return True  # signal clean exit


### -------- Tracking worker -------- ###


class LogSignal(WorkerBaseSignals):
    """Signal to send messages to be logged from another thread.

    Separate from Worker instances as indicated `on this post`_

    .. _on this post: https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect
    """  # TODO link ?

    log_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged"""
    log_w_replace_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged, replacing the last line"""
    warn_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some warning should be emitted in main thread"""
    error_signal = Signal(Exception, str)
    """qtpy.QtCore.Signal: signal to be sent when some error should be emitted in main thread"""

    # Should not be an instance variable but a class variable, not defined in __init__, see
    # https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect

    def __init__(self, parent=None):
        """Creates a LogSignal."""
        super().__init__(parent=parent)


class TrackingWorker(GeneratorWorker):
    """A custom worker to run tracking in."""

    def __init__(self, config=None):
        """Creates a TrackingWorker."""
        # super().__init__(self.run_tracking) #### TODO MUST BE CHANGED WHEN REAL TRACKING IS IMPLEMENTED
        super().__init__(self.fake_tracking)
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.log_w_replace_signal = self._signals.log_w_replace_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

        self.config = config  # use if needed

    def log(self, msg):
        """Log a message."""
        self.log_signal.emit(msg)

    def log_w_replace(self, msg):
        """Log a message, replacing the last line. For us with progress bars mainly."""
        self.log_w_replace_signal.emit(msg)

    def warn(self, msg):
        """Log a warning."""
        self.warn_signal.emit(msg)

    def run_tracking(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ):
        """Run the tracking."""
        self.log("Started tracking")

    def fake_tracking(self):
        """Fake tracking for testing purposes."""
        for i in range(10):
            self.log(f"Tracking frame {i}")
            yield i + 1



def track_mock(
    video: np.ndarray,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Mocks what a tracker would do.

    This method's signature should be re-used by all trackers (PIPS++ and CoTracker).

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    return np.repeat(keypoints, (len(video), 1, 1, 1))


def track_cotracker(
    video: Path,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Tracks keypoints in a video using CoTracker.

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    # TODO: Implement your code here!


def track_pips(
    video: Path,
    keypoints: np.ndarray,
) -> np.ndarray:
    """Tracks keypoints in a video using PIPS++.

    Args:
        video: The path to a video in which the points should be tracked.
        keypoints: The position of keypoints to track in the video. This array should
            have shape (n_animals, n_keypoints, 2), where
                n_animals: the number of animals to track
                n_keypoints: the number of keypoints to track for each individual
                2: as each point is defined by its (x, y) coordinates

    Returns:
        an array of shape (num_frames, n_animals, n_keypoints, 2) corresponding to the
        position of each keypoint in each frame of the video
    """
    # TODO: Implement your code here!
