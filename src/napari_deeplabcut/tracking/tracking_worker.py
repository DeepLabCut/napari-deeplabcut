import sys
import time
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import numpy as np

from dataclasses import dataclass

@dataclass
class TrackingWorkerData:
    video: np.ndarray
    keypoints: np.ndarray


class TrackingWorker(QObject):
    started = Signal()
    progress = Signal(int)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.is_paused = False
        self.is_aborted = False

    def run(self):
        """ Long-running task """
        self.started.emit()
        for i in range(1, 101):
            if self.is_aborted:
                break
            while self.is_paused:
                time.sleep(0.1)
            self.progress.emit(i)
            time.sleep(0.1)

        self.finished.emit()

    def start(self):
        """ Create a new thread and move the worker to the thread """
        self.thread = QThread()
        self.moveToThread(self.thread)

        self.thread.started.connect(self.run)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def abort(self):
        self.is_aborted = True