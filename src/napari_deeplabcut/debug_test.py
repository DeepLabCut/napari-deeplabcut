"""
from functools import partial

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QProgressBar, QPushButton

from my_package import long_running_function


class PercentageWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    percentageChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percentage = 0

    def __add__(self, other):
        if isinstance(other, int):
            self._percentage += other
            self.percentageChanged.emit(self._percentage)
            return self
        return super().__add__(other)

    def __lt__(self, other):
        if isinstance(other, int):
            return self._percentage < other
        return super().__lt__(other)

    def start_task(self, callback, initial_percentage):
        self._percentage = initial_percentage
        wrapper = partial(callback, self)
        QTimer.singleShot(0, wrapper)

    @pyqtSlot(object)
    def launch_task(self, wrapper):
        self.started()
        wrapper()
        self.finished()


class Actions(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Progress Bar")
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        self.button = QPushButton("Start", self)
        self.button.move(0, 30)
        self.show()

        self.button.clicked.connect(self.onButtonClick)

        thread = QThread(self)
        thread.start()
        self.percentage_worker = PercentageWorker()
        self.percentage_worker.moveToThread(thread)
        self.percentage_worker.percentageChanged.connect(self.progress.setValue)
        self.percentage_worker.started.connect(self.onStarted)
        self.percentage_worker.finished.connect(self.onFinished)

    @pyqtSlot()
    def onStarted(self):
        self.button.setDisabled(True)

    @pyqtSlot()
    def onFinished(self):
        self.button.setDisabled(False)

    @pyqtSlot()
    def onButtonClick(self):
        self.percentage_worker.start_task(long_running_function, 0)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = Actions()
    sys.exit(app.exec_())

""" 

path = 'C:/Users/Sabrina/Documents/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/'

print(path.split('training-datasets'))
