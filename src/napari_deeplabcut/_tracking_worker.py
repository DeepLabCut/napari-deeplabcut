# TODO :
# - Implement the tracking worker with multithreading
# - Implement a Log widget to display the tracking progress + a progress bar
# - Prepare I/O with the actual tracking backend
from functools import partial
from pathlib import Path
from dataclasses import dataclass
import napari
import numpy as np
import pandas as pd
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
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

@dataclass
class TrackingConfig:
    ### Data ###
    video: np.ndarray
    keypoints: np.ndarray 
    # result_layer: napari.layers.Points 
    ### Metadata ###
    root: str  # path to the video
    paths: list # list of paths to the video frames
    bodyparts: list # list of bodyparts
    individuals_ids: list # list of individuals
    ### Config from data ###
    n_frames: int
    n_animals: int
    n_keypoints: int
    ### User config ###
    retrack_frame_id: int = None
    method : str = "CoTracker" # change when adding PIPS++
    device: str = "cpu" if not torch.cuda.is_available() else "cuda"

@dataclass
class TrackingResults: # Add anything relevant to be yielded by the worker here
    """Used to update the results and progress bar. Is yielded by the worker."""
    result_keypoints: np.ndarray = None
    layer_metadata: dict = None
    hdf_path : str = None
    pbar_update : tuple = None

class TrackingModule(QWidget, metaclass=QWidgetSingleton):
    """Plugin for tracking."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer", parent=None):
        """Creates a widget with links to documentation and about page."""
        super().__init__(parent=parent)
        self._viewer = napari_viewer
        self._worker = None
        self._video_layer : napari.layers.Image = None
        self._keypoint_layer : napari.layers.Points = None
        self.result_layer : napari.layers.Points = None
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
        self.enable_retracking = False
        
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
        self._viewer.dims.events.current_step.connect(self._update_start_button_display)

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

    def _check_ready(self):
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
        # self.progress.setVisible(True)
        # self.progress.setValue(0)
        
    def _update_start_button_display(self):
        """Update the start button display."""
        if self._worker.is_running:
            return
    
        if not self._worker.is_running and self.result_layer is not None:
            current_frame = self._viewer.dims.current_step[0]
            if current_frame == 0:
                self.start_button.setText("Start")
                return
            self.start_button.setText(f"Retrack from frame {current_frame}")
        else:
            self.start_button.setText("Start")

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
        if not self._check_ready():
            err = "Aborting, please choose valid inputs"
            self.log.print_and_log(err)
            raise ValueError(err)

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
        metadata = self.keypoint_layer_dropdown.layer().metadata
        keypoint_cord = self.keypoint_layer_dropdown.layer_data()
        frames = self.video_layer_dropdown.layer_data()

        header = metadata["header"]
        bodyparts = header.bodyparts
        individuals_ids = header.individuals
        
        current_frame = self._viewer.dims.current_step[0]
        retrack_frame_id = None
        if current_frame != 0 and self.enable_retracking:
            retrack_frame_id = current_frame

        self.worker_config = TrackingConfig(
            video=frames,
            keypoints=keypoint_cord,
            root=metadata["root"],
            paths=metadata["paths"],
            bodyparts=bodyparts,
            individuals_ids=individuals_ids,
            n_frames=len(frames),
            n_animals=len(individuals_ids),
            n_keypoints=len(bodyparts),
            retrack_frame_id=retrack_frame_id,
        )
        self._worker = TrackingWorker(self.worker_config)
            
        
        # self._worker = TrackingWorker(
        #     # metadata["metadata"]["root"],
        #     # metadata["metadata"]["images"],
        #     metadata["root"],
        #     metadata["paths"],
        #     bodyparts,
        #     individuals_ids,
        #     frames,
        #     keypoint_cord,
        # )

        self._worker.started.connect(self._on_start)

        self._worker.log_signal.connect(self.log.print_and_log)
        self._worker.log_w_replace_signal.connect(self.log.replace_last_line)
        self._worker.warn_signal.connect(self.log.warn)
        self._worker.error_signal.connect(self.log.error)

        self._worker.yielded.connect(partial(self._on_yield))
        self._worker.errored.connect(partial(self._on_error))
        self._worker.finished.connect(self._on_finish)

    def _display_results(self, results):
        """Display the results in the viewer, using the method already implemented in the viewer."""
        # path_test = "C:/Users/Cyril/Desktop/Code/DeepLabCut/examples/openfield-Pranav-2018-10-30/labeled-data/m4s1/CollectedData_Pranav.h5"
        from napari_deeplabcut._reader import read_hdf

        path_test = str(results)
        keypoint_data, metadata, _ = read_hdf(path_test)[0]
        # hdf data contains : keypoint data, metadata, and "points"
        # we want to create a points layer from the keypoint data
        # layer properties (dict) should be populated with metadata
        print(metadata)
        return self._viewer.add_points(
            ### data ###
            keypoint_data,
            name=f"Tracked keypoints - frame {self._worker.config.retrack_frame_id}",
            metadata=metadata["metadata"],
            # features=metadata["properties"],
            properties=metadata["properties"],
            ### display properties ###
            face_color=metadata["face_color"],
            face_color_cycle=metadata["face_color_cycle"],
            face_colormap=metadata["face_colormap"],
            edge_color=metadata["edge_color"],
            edge_color_cycle=metadata["edge_color_cycle"],
            edge_width=metadata["edge_width"],
            edge_width_is_relative=metadata["edge_width_is_relative"],
            size=metadata["size"],
        )
        

    def _on_yield(self, results):
        # TODO : display the results in the viewer
        # Testing version where an int i is yielded
        self.result_layer = self._display_results(results)
        ############################
        self.log.print_and_log(f"Yielded {results}")
        # self._update_progress_bar(results, 10)
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
        self.enable_retracking = True

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

    def __init__(self, config: TrackingConfig):
        """Creates a TrackingWorker."""
        super().__init__(self.run_tracking) 
        
        self.config = config
        
        self._root = config.root
        self._image_paths = config.paths
        self._bodyparts = config.bodyparts
        self._individuals = config.individuals_ids
        self._video = config.video
        self._keypoints = config.keypoints
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.log_w_replace_signal = self._signals.log_w_replace_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

    def log(self, msg):
        """Log a message."""
        self.log_signal.emit(msg)

    def log_w_replace(self, msg):
        """Log a message, replacing the last line. For us with progress bars mainly."""
        self.log_w_replace_signal.emit(msg)

    def warn(self, msg):
        """Log a warning."""
        self.warn_signal.emit(msg)

    def run_tracking(self):
        """Run the tracking."""
        with open("log.txt", "w") as f:
            f.write(f"{self._video.shape}")
            f.write(f"{self._keypoints.shape}")

        init_frame = self.config.retrack_frame_id if self.config.retrack_frame_id is not None else 0
        self.log(f"Started tracking from frame {init_frame}")


        video_frames = self._video[init_frame:]

        keypoints = np.array(self._keypoints)
        keypoints = keypoints[keypoints[:, 0] == init_frame][:, [2, 1]]
        keypoints = keypoints.reshape((len(self._individuals), len(self._bodyparts), 2))

        tracks = cotrack_online(
            self.log,
            video_frames,
            keypoints,
            device=self.config.device,
        )
        with open("log_finished_tracking.txt", "w") as f:
            f.write(f"Done! {tracks.shape}")
        self.log("Finished tracking")
        track_path = Path(self._root) / f"TrackedData_frame_{init_frame}.h5"
        self.save_tracking_data(track_path, tracks, "CoTracker", frame=init_frame)
        self.log("Finished saving")
        yield track_path

    def save_tracking_data(self, path: Path, tracks: np.ndarray, scorer: str, frame: int = 0) -> None:
        levels = ["scorer", "individuals", "bodyparts", "coords"]
        kpt_entries = ["x", "y"]
        columns = []
        for i in self._individuals:
            for b in self._bodyparts:
                columns += [(scorer, i, b, entry) for entry in kpt_entries]
        # for i, b in zip(self._individuals[:8], self._bodyparts[:8]):
        #     columns += [(scorer, i, b, entry) for entry in kpt_entries]

        index = []
        for img_path in self._image_paths[frame:]:
            if isinstance(img_path, str):
                index.append(tuple(Path(img_path).parts))
            elif isinstance(img_path, tuple):
                index.append(img_path)
            else:
                raise ValueError(f"Incorrect image path format: {img_path}")

        with open("log_df.txt", "w") as f:
            f.write(f"{tracks.reshape((len(tracks), -1)).shape}\n")
            f.write(f"{len(index)}\n")
            f.write(f"{len(columns)}\n")
            f.write(f"{self._individuals}\n")
            f.write(f"{self._bodyparts}\n")

        dataframe = pd.DataFrame(
            data=tracks.reshape((len(tracks), -1)),
            index=pd.MultiIndex.from_tuples(index),
            columns=pd.MultiIndex.from_tuples(columns, names=levels),
        )
        dataframe.to_hdf(path, key="df_with_missing")

    def fake_tracking(self):
        """Fake tracking for testing purposes."""
        for i in range(1):
            self.log(f"Tracking frame {i}")
            yield i + 1


# TODO: REQUIRES TO RUN pip install src/co-tracker
def cotrack_online(
    log,
    video,
    keypoints,
    device: str = "cpu",
) -> np.ndarray:
    log("Running CoTracker")
    with open("log_cotrack.txt", "w") as f:
        f.write(f"video={video.shape}\n")
        f.write(f"keypoints={keypoints.shape}\n")
        f.write(f"{keypoints}\n")

    def _process_step(window_frames, is_first_step, queries):
        with open("log_window_frames.txt", "w") as f:
            f.write(f"{len(window_frames)}\n")
            f.write(f"{model.step}\n")
            f.write(f"{-model.step * 2}\n")
            f.write(f"is_first_step={is_first_step}\n")

        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2:]), device=device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(video_chunk, is_first_step=is_first_step, queries=queries[None])

    n_frames = len(video)
    n_animals, n_keypoints = keypoints.shape[:2]

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(device)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
    window_frames = []

    queries = np.zeros((n_animals * n_keypoints, 3))
    queries[:, 1:] = keypoints.reshape((-1, 2))
    queries = torch.from_numpy(queries).to(device).float()

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    i = 0
    for i, frame in enumerate(video[0]):
        frame = frame.permute(1, 2, 0)
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries,
            )
            is_first_step = False
        window_frames.append(frame)
        log(f"Finished frame {i}")

    # Processing final frames in case video length is not a multiple of model.step
    # TODO: Use visibility
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1:],
        is_first_step,
        queries=queries,
    )

    with open("log_pred_tracks.txt", "w") as f:
        f.write(f"{len(pred_tracks)}\n")
        f.write(f"{pred_tracks.shape}\n")

    tracks = pred_tracks.squeeze().cpu().numpy()
    with open("log_pred_tracks_2.txt", "w") as f:
        f.write(f"{len(tracks)}\n")
        f.write(f"{tracks.shape}\n")
        f.write(f"{(n_frames, n_animals, n_keypoints, 2)}\n")

    return tracks.reshape((n_frames, n_animals, n_keypoints, 2))


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
