from __future__ import annotations

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LayerStatusPanel(QGroupBox):
    """
    Small dock-widget panel showing:
    - current folder name
    - labeling progress
    - point size control (slider)

    Public API is intentionally small/stable so KeypointControls can use it
    without caring about the exact Qt widget type.
    """

    point_size_changed = Signal(int)
    point_size_commit_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Layer status", parent=parent)

        self._folder_value = QLabel("—")
        self._folder_value.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._progress_value = QLabel("No active keypoints layer")
        self._progress_value.setWordWrap(True)
        self._progress_value.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._size_slider = QSlider(Qt.Horizontal, self)
        self._size_slider.setRange(1, 100)
        self._size_slider.setSingleStep(1)
        self._size_slider.setPageStep(2)
        self._size_slider.setValue(6)

        self._size_value = QLabel("6")
        self._size_value.setToolTip(
            "Global point size for all keypoints layers. Saved to config.yaml as dotsize when changed."
        )
        self._size_value.setMinimumWidth(28)
        self._size_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        size_row = QHBoxLayout()
        size_row.addWidget(self._size_slider, stretch=1)
        size_row.addWidget(self._size_value, stretch=0)

        form = QFormLayout()
        form.addRow("Folder", self._folder_value)
        form.addRow("Progress", self._progress_value)
        form.addRow("Point size", size_row)

        wrapper = QVBoxLayout(self)
        wrapper.addLayout(form)

        self._size_slider.valueChanged.connect(self._on_slider_changed)
        self._size_slider.sliderReleased.connect(self._emit_commit)

        # initialize default enabled appearance
        self.set_point_size_enabled(False)

    def _on_slider_changed(self, value: int) -> None:
        self._size_value.setText(str(int(value)))
        self.point_size_changed.emit(int(value))

    def _emit_commit(self) -> None:
        self.point_size_commit_requested.emit(self.point_size())

    def point_size(self) -> int:
        return int(self._size_slider.value())

    def set_point_size(self, value: int) -> None:
        blocker = QSignalBlocker(self._size_slider)
        self._size_slider.setValue(int(value))
        del blocker
        self._size_value.setText(str(int(value)))

    def set_point_size_enabled(self, enabled: bool, *, reason: str | None = None) -> None:
        enabled = bool(enabled)
        self._size_slider.setEnabled(enabled)
        self._size_value.setEnabled(enabled)

        if enabled:
            tooltip = "Point size for the active DLC keypoints layer. Saved to config.yaml as dotsize when changed."
            self._size_slider.setToolTip(tooltip)
            self._size_value.setToolTip(tooltip)
            self._size_value.setStyleSheet("")
        else:
            tooltip = reason or "Select a DLC keypoints layer to edit point size."
            self._size_slider.setToolTip(tooltip)
            self._size_value.setToolTip(tooltip)
            self._size_value.setStyleSheet("color: palette(mid);")

    def set_folder_name(self, folder_name: str) -> None:
        self._folder_value.setText(folder_name or "—")

    def set_progress_summary(
        self,
        *,
        labeled_percent: float,
        remaining_percent: float,
        labeled_points: int,
        total_points: int,
        frame_count: int,
        bodypart_count: int,
        individual_count: int,
    ) -> None:
        if total_points <= 0:
            self._progress_value.setText("Not enough metadata to estimate progress yet")
            self._progress_value.setToolTip("")
            return

        if individual_count <= 1:
            breakdown = f"{frame_count} frames × {bodypart_count} bodyparts"
        else:
            breakdown = f"{frame_count} frames × {bodypart_count} bodyparts × {individual_count} individuals"

        self._progress_value.setText(f"{labeled_percent:.1f}% labeled")
        self._progress_value.setToolTip(f"{labeled_points}/{total_points} of all possible points labeled • {breakdown}")

    def set_no_active_points_layer(self) -> None:
        self._progress_value.setText("No active keypoints layer")
        self._progress_value.setToolTip("")
        self.set_point_size_enabled(False, reason="Select a DLC keypoints layer to edit point size.")

    def set_invalid_points_layer(self) -> None:
        self._progress_value.setText("Active layer is not a DLC keypoints layer")
        self._progress_value.setToolTip("")
        self.set_point_size_enabled(False, reason="This control only works for DLC keypoints layers.")
