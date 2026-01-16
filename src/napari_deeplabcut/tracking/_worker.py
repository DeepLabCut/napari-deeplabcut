import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from qtpy.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

from napari_deeplabcut.tracking._data import (
    RawModelOutputs,
    TrackingModelInputs,
    TrackingWorkerData,
    TrackingWorkerOutput,
)
from napari_deeplabcut.tracking._models import AVAILABLE_TRACKERS

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False
if DEBUG:
    logger.setLevel(logging.DEBUG)


@dataclass
class TorchHubModel:
    org: str
    model: str


class TrackingWorker(QObject):
    started = Signal()
    finished = Signal()
    progress = Signal(tuple)
    trackingStarted = Signal()
    trackingFinished = Signal(TrackingWorkerOutput)
    trackingStopped = Signal()

    def __init__(self):
        super().__init__()
        self.is_stopped = False

    @Slot(TrackingWorkerData)
    def track(self, cfg: TrackingWorkerData):
        """
        Tracking core logic:
            1. Instantiate model from registry.
            2. prepare_inputs(cfg)
            3. run(inputs, progress_cb, stop_cb)
            4. prepare_outputs(raw, inputs)
            5. Emit results to the plugin.
        """
        model = None
        try:
            # Choose model by name from your TrackerType (cfg.tracker.value.name)
            # if DEBUG:
            #     debugpy.debug_this_thread()
            model_name = cfg.tracker_name
            try:
                model_cls = AVAILABLE_TRACKERS[model_name]["class"]
            except KeyError:
                logger.error(f"Unknown tracker: {model_name}")
                self.trackingStopped.emit()
                return

            model = model_cls(cfg)

            # Define callbacks to let the model report status
            def progress_callback(current: int, total: int):
                self.progress.emit((current, total))

            def stop_callback() -> bool:
                # Return early if requested
                return self._should_stop()

            try:
                # we let the model handle coordinate conventions internally, such that
                # the worker and plugin can remain agnostic to these details.
                inputs: TrackingModelInputs = model.prepare_inputs(cfg)

                # Run inference; models implement their own batching and chunking
                raw: RawModelOutputs = model.run(inputs, progress_callback, stop_callback)

                # Convert to canonical output (N,3) plus features
                output: TrackingWorkerOutput = model.prepare_outputs(raw, cfg)

                if hasattr(model, "validate_outputs"):
                    valid, msg = model.validate_outputs(inputs, output)
                    if not valid:
                        raise ValueError(f"Invalid model outputs: {msg}")
            except Exception as exc:
                logger.exception("Tracking failed", exc_info=exc)
                self.trackingStopped.emit()
                return

            # Old : update the worker cfg & keep existing signal type
            # cfg.keypoints = output.keypoints
            # cfg.keypoint_features = output.keypoint_features
            # self.trackingFinished.emit(cfg)

            # Use the new signal type only and keep cfg immutable
            self.trackingFinished.emit(output)
        finally:
            torch.cuda.empty_cache()
            if model is not None:
                del model

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
