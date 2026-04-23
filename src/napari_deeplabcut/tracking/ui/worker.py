import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QThread, Signal, Slot

from napari_deeplabcut.tracking.core.data import (
    RawModelOutputs,
    TrackingModelInputs,
    TrackingWorkerData,
    TrackingWorkerOutput,
)
from napari_deeplabcut.tracking.core.models import AVAILABLE_TRACKERS

try:
    import importlib

    # import torch
    importlib.import_module("torch")
except ImportError as e:
    raise ImportError(
        "TrackingWorker requires PyTorch to be installed.Please install with `pip install napari-deeplabcut[tracking]`."
    ) from e

"""Worker is not allowed to perform main thread operations, such as :
- viewer.add_*
- viewer.layers.*
- layer.data = ...
- QWidget updates
- QCoreApplication.processEvents()
- anything vispy / OpenGL / rendering-related

Please be careful when editing this file.
"""

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEBUG = False
if DEBUG:
    logger.setLevel(logging.DEBUG)
    import debugpy


@dataclass
class TorchHubModel:
    org: str
    model: str


class TrackingWorker(QObject):
    started = Signal()
    finished = Signal()
    progress = Signal(int, int)  # current, total
    trackingStarted = Signal()
    trackingFinished = Signal(object)  # emits TrackingWorkerOutput
    trackingStopped = Signal()

    def __init__(self):
        super().__init__()
        self._stop_requested = threading.Event()
        self.thread = None

    @Slot(object)
    def track(self, cfg: TrackingWorkerData):
        """
        Tracking core logic:
            1. Instantiate model from registry.
            2. prepare_inputs(cfg)
            3. run(inputs, progress_cb, stop_cb)
            4. prepare_outputs(raw, inputs)
            5. Emit results to the plugin.
        """
        logger.debug(
            "TrackingWorker.track | python_thread=%s | qt_current_thread=%r | worker_thread=%r",
            threading.current_thread().name,
            QThread.currentThread(),
            self.thread,
        )
        model = None
        self._stop_requested.clear()

        try:
            if DEBUG:
                debugpy.debug_this_thread()

            logger.debug(
                "TrackingWorker.track | python_thread=%s | qt_current_thread=%r | worker_thread=%r",
                threading.current_thread().name,
                QThread.currentThread(),
                self.thread,
            )

            self.trackingStarted.emit()

            model_name = cfg.tracker_name
            try:
                model_cls = AVAILABLE_TRACKERS[model_name]["class"]
            except KeyError:
                logger.error(f"Unknown tracker: {model_name}")
                self.trackingStopped.emit()
                self.finished.emit()
                return

            model = model_cls(cfg)

            def progress_callback(current: int, total: int):
                self.progress.emit(int(current), int(total))

            def stop_callback() -> bool:
                return self._should_stop()

            try:
                inputs: TrackingModelInputs = model.prepare_inputs(cfg)
                raw: RawModelOutputs = model.run(inputs, progress_callback, stop_callback)

                if self._should_stop():
                    self.trackingStopped.emit()
                    self.finished.emit()
                    return

                output: TrackingWorkerOutput = model.prepare_outputs(raw, cfg)

                if hasattr(model, "validate_outputs"):
                    valid, msg = model.validate_outputs(inputs, output)
                    if not valid:
                        raise ValueError(f"Invalid model outputs: {msg}")

            except Exception as exc:
                logger.exception("Tracking failed", exc_info=exc)
                self.trackingStopped.emit()
                self.finished.emit()
                return

            self.trackingFinished.emit(output)
            self.finished.emit()

        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                logger.debug("Could not clear CUDA cache", exc_info=True)

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

    def request_stop(self):
        self._stop_requested.set()

    @Slot()
    def stop_tracking(self):
        self.request_stop()

    def _should_stop(self) -> bool:
        return self._stop_requested.is_set()
