from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from napari_deeplabcut.core.layers import LabelProgress


class LayerStatusPanel(QGroupBox):
    """
    Small dock-widget panel showing:
    - current folder name
    - labeling progress
    - point size control (slider)
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
        self._progress_details_text = ""
        self._progress_value.setContextMenuPolicy(Qt.CustomContextMenu)
        self._progress_value.customContextMenuRequested.connect(self._show_progress_context_menu)

        self._size_slider = QSlider(Qt.Horizontal, self)
        self._size_slider.setRange(1, 100)
        self._size_slider.setSingleStep(1)
        self._size_slider.setPageStep(2)
        self._size_slider.setValue(6)
        self._size_opacity = QGraphicsOpacityEffect(self._size_slider)
        self._size_slider.setGraphicsEffect(self._size_opacity)

        self._size_value = QLabel("6")
        self._size_value.setMinimumWidth(28)
        self._size_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Dedicated container for the whole size-control row
        self._size_controls = QWidget(self)
        size_row = QHBoxLayout(self._size_controls)
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.addWidget(self._size_slider, stretch=1)
        size_row.addWidget(self._size_value, stretch=0)

        size_tooltip = "Point size for the active DLC keypoints layer. Saved to config.yaml as dotsize when changed."
        self._size_slider.setToolTip(size_tooltip)
        self._size_value.setToolTip(size_tooltip)
        self._size_controls.setToolTip(size_tooltip)

        form = QFormLayout()
        form.addRow("Folder", self._folder_value)
        form.addRow("Progress", self._progress_value)
        form.addRow("Point size", self._size_controls)

        wrapper = QVBoxLayout(self)
        wrapper.addLayout(form)

        self._size_slider.setTracking(False)

        self._size_slider.sliderMoved.connect(self._on_slider_moved_preview)
        self._size_slider.valueChanged.connect(self._on_value_changed_commit)

        self.set_point_size_enabled(False, reason="Select a DLC keypoints layer to edit point size.")

    def _on_slider_moved_preview(self, value: int) -> None:
        self._size_value.setText(str(int(value)))
        self.point_size_changed.emit(int(value))  # visual only

    def _on_value_changed_commit(self, value: int) -> None:
        self._size_value.setText(str(int(value)))
        # Ensure non-mouse / programmatic changes also update the visual layer size
        self.point_size_changed.emit(int(value))  # visual update on commit
        self.point_size_commit_requested.emit(int(value))  # save / persist

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

        # Disable the entire container so both slider and label get proper native disabled styling
        self._size_controls.setEnabled(enabled)
        self._size_slider.setEnabled(enabled)
        self._size_value.setEnabled(enabled)

        opacity = 1.0 if enabled else 0.35
        self._size_opacity.setOpacity(opacity)

        tooltip = (
            "Point size for the active DLC keypoints layer. Saved to config.yaml as dotsize when changed."
            if enabled
            else (reason or "Select a DLC keypoints layer to edit point size.")
        )

        self._size_controls.setToolTip(tooltip)
        self._size_slider.setToolTip(tooltip)
        self._size_value.setToolTip(tooltip)

    def set_folder_name(self, folder_name: str) -> None:
        self._folder_value.setText(folder_name or "—")

    def _show_progress_context_menu(self, pos) -> None:
        if not self._progress_details_text:
            return

        menu = QMenu(self)
        copy_action = menu.addAction("Copy progress details")
        chosen = menu.exec_(self._progress_value.mapToGlobal(pos))
        if chosen is copy_action:
            self._copy_progress_details_to_clipboard()

    def _copy_progress_details_to_clipboard(self) -> None:
        text = self._progress_details_text or self._progress_value.toolTip() or ""
        if not text:
            return
        QApplication.clipboard().setText(text)

    def set_progress_summary(
        self,
        *,
        progress: LabelProgress,
    ) -> None:

        p = progress

        if p.total_points <= 0:
            self._progress_value.setText("Not enough metadata to estimate progress yet")
            self._progress_value.setToolTip("")
            self._progress_details_text = ""
            return

        if p.individual_count <= 1:
            breakdown = f"{p.frame_count} frames × {p.bodypart_count} bodyparts"
        else:
            breakdown = f"{p.frame_count} frames × {p.bodypart_count} bodyparts × {p.individual_count} individuals"

        self._progress_value.setText(f"{p.labeled_percent:.1f}% labeled")

        incomplete_count = max(0, p.frame_count - p.completed_frames)
        first_incomplete = list(p.incomplete_frames[:10])

        details_lines = [
            f"{p.labeled_percent:.1f}% labeled, {p.remaining_percent:.1f}% remaining",
            f"{p.labeled_points}/{p.total_points} possible keypoint slots currently labeled • {breakdown}",
            f"{p.completed_frames}/{p.frame_count} frames complete ({p.completed_percent:.1f}%)",
            f"{incomplete_count} incomplete frames",
        ]

        if first_incomplete:
            details_lines.append("First incomplete frames: " + ", ".join(str(int(f)) for f in first_incomplete))

        if p.individual_count > 1:
            # Keep individual rows stable and compact
            per_individual_lines = []
            for individual, n_frames in sorted(p.incomplete_frames_by_individual.items()):
                if individual == "":
                    continue
                missing_points = int(p.missing_points_by_individual.get(individual, 0))
                per_individual_lines.append(
                    f"- {individual}: incomplete on {int(n_frames)} frame(s), {missing_points} missing keypoint(s)"
                )

            if per_individual_lines:
                details_lines.append("")
                details_lines.append("By individual:")
                details_lines.extend(per_individual_lines)

        details_lines.append("")
        details_lines.append("Tip: right-click this progress label to copy the full summary.")

        details_text = "\n".join(details_lines)
        self._progress_details_text = details_text
        self._progress_value.setToolTip(details_text)

    def set_no_active_points_layer(self) -> None:
        self._progress_value.setText("No active keypoints layer")
        self._progress_value.setToolTip("")
        self._progress_details_text = ""
        self.set_point_size_enabled(False, reason="Select a DLC keypoints layer to edit point size.")

    def set_invalid_points_layer(self) -> None:
        self._progress_value.setText("Active layer is not a DLC keypoints layer")
        self._progress_value.setToolTip("")
        self._progress_details_text = ""
        self.set_point_size_enabled(False, reason="This control only works for DLC keypoints layers.")
