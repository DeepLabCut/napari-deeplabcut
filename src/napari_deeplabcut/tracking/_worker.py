import sys
import pandas as pd
import debugpy
import logging
import time
from qtpy.QtCore import QObject, QThread, Signal, Slot, QCoreApplication
from qtpy.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import numpy as np

from dataclasses import dataclass
logging.basicConfig(level=logging.DEBUG)

@dataclass
class TrackingWorkerData:
    tracker: str
    video: np.ndarray
    keypoints: np.ndarray # (num_keypoint, 3) first col is frame numbe in `video` and the second and third are x, y
    keypoint_features: dict
    keypoint_range: tuple[int, int]
    backward_tracking: bool


class TrackingWorker(QObject):
    started = Signal()
    finished = Signal()
    progress = Signal(tuple)
    trackingStarted = Signal()
    trackingFinished = Signal(TrackingWorkerData)
    trackingStopped = Signal()

    def __init__(self):
        super().__init__()
        import torch
        self.is_stopped = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: object | None = None

    @Slot(TrackingWorkerData)
    def track(self, cfg: TrackingWorkerData):
        # TODO: if cfg.tracker == 'cotracker'
        import torch
        debugpy.debug_this_thread()
        if self.model is None:
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        def _process_step(window_frames, is_first_step, queries):
            video_chunk = (
                torch.tensor(np.stack(window_frames[-self.model.step * 2:]), device=self.device)
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            return self.model(video_chunk, is_first_step=is_first_step, queries=queries[None], add_support_grid=True)
        # video is originally of shape (num_frames, height, width, channels)
        video = np.array(cfg.video)
        window_frames = []

        # We need to swap x, y so that it matches what cotracker expects 
        cfg.keypoints[:, [1, 2]] = cfg.keypoints[:, [2, 1]]

        queries = torch.from_numpy(cfg.keypoints).to(self.device).float()
        
        # Iterating over video frames, processing one window at a time:
        is_first_step = True
        for i, frame in enumerate(video):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, _pred_visibility = _process_step(window_frames, is_first_step, queries=queries)
                is_first_step = False
            window_frames.append(frame)
            self.progress.emit((i, len(video)))
            if self._should_stop():
                return

        # Processing final frames in case video length is not a multiple of model.step
        # TODO: Use visibility
        pred_tracks, _pred_visibility = _process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1:],
            is_first_step,
            queries=queries,
        )
        self.progress.emit((len(video), len(video)))

        tracks = pred_tracks.squeeze().cpu().numpy()
        tracks = tracks[:, :cfg.keypoints.shape[0], :] # drop the support grid (necessary only for cotracker version < 3)
        tracks = tracks.reshape(-1, 2)
        if cfg.backward_tracking:
            tracks = tracks[::-1]
        frame_ids = np.repeat(np.arange(cfg.keypoint_range[0], cfg.keypoint_range[1]), cfg.keypoints.shape[0])
        tracks = np.column_stack((frame_ids, tracks))
        cfg.keypoint_features = pd.concat([cfg.keypoint_features] * len(np.unique(tracks[:, 0])), ignore_index=True)
        cfg.keypoints = tracks
        cfg.keypoints[:, [1, 2]] = cfg.keypoints[:, [2, 1]]
        self.trackingFinished.emit(cfg)

    def run(self):
        self.started.emit()

    def start(self):
        self.thread = QThread()
        self.moveToThread(self.thread)

        self.finished.connect(self.thread.quit)
        self.thread.started.connect(self.run)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    @Slot()
    def stop_tracking(self):
        self.is_stopped = True

    def _should_stop(self) -> bool:
        QCoreApplication.processEvents()
        if self.is_stopped:
            self.trackingStopped.emit()
            self.is_stopped = False
            return True
        return False