# src/napari_deeplabcut/tracking/ui/merger.py
from __future__ import annotations

import logging
from html import escape

from napari.layers import Points
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.layer_versioning import mark_layer_presentation_changed
from ...tracking.core.merge import (
    TrackingMergePreview,
    apply_tracking_merge,
    preview_tracking_merge,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Small dialog/view helpers
# -----------------------------------------------------------------------------#


def _layer_display_name(layer: Points | None) -> str:
    if layer is None:
        return "—"
    try:
        return str(getattr(layer, "name", "Unnamed layer"))
    except Exception:
        return "Unnamed layer"


def _preview_summary_text(preview: TrackingMergePreview | None) -> str:
    if preview is None:
        return "Choose a tracking-result source layer and a DLC target layer to preview the merge."

    if not preview.is_valid:
        return f"Merge unavailable: {preview.invalid_reason or 'invalid merge preview'}"

    lines: list[str] = []

    if preview.n_appendable > 0:
        lines.append(
            f"{preview.n_appendable} tracked point"
            f"{'' if preview.n_appendable == 1 else 's'} can be added to "
            f'"{preview.target_layer_name}".'
        )
    else:
        lines.append("No new tracked points can be added to the selected target layer.")

    if preview.n_conflicts > 0:
        lines.append(
            f"{preview.n_conflicts} conflicting point{'' if preview.n_conflicts == 1 else 's'} will be skipped."
        )

    if preview.n_identical > 0:
        lines.append(
            f"{preview.n_identical} point{'' if preview.n_identical == 1 else 's'} already match the target exactly."
        )

    if preview.n_invalid_source > 0:
        lines.append(
            f"{preview.n_invalid_source} invalid source row"
            f"{'' if preview.n_invalid_source == 1 else 's'} will be ignored."
        )

    return "\n".join(lines)


def _preview_conflict_details_text(preview: TrackingMergePreview) -> str:
    if not preview.conflicts:
        return "No conflicts."

    lines: list[str] = []
    for entry in preview.conflicts:
        lines.append(
            f"{entry.frame_label} → {entry.keypoint_label}\n"
            f"  source: {entry.source_coords_text}\n"
            f"  target: {entry.target_coords_text}"
        )

    if preview.truncated_conflicts > 0:
        lines.append("")
        lines.append(f"… and {preview.truncated_conflicts} more conflict(s).")

    return "\n".join(lines)


# -----------------------------------------------------------------------------#
# Conflict report dialog
# -----------------------------------------------------------------------------#


class TrackingMergeConflictsDialog(QDialog):
    """
    Confirmation dialog shown when merge conflicts exist.

    This dialog is intentionally merge-specific rather than reusing the save
    overwrite report contract. Merge conflicts are live-layer conflicts, not
    persisted-file overwrite conflicts.
    """

    def __init__(self, parent: QWidget | None, *, preview: TrackingMergePreview):
        super().__init__(parent)
        self.preview = preview
        self.setWindowTitle("Merge conflicts detected")
        self.setModal(True)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        summary = QLabel(
            "Some tracked points conflict with existing annotations in the target layer.\n\n"
            "Only non-conflicting tracked points will be merged."
        )
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(summary)

        src_label = QLabel(
            f"<b>Source:</b> {escape(_layer_display_name(None) if preview is None else preview.source_layer_name)}"
        )
        src_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        src_label.setWordWrap(True)
        layout.addWidget(src_label)

        tgt_label = QLabel(
            f"<b>Target:</b> {escape(_layer_display_name(None) if preview is None else preview.target_layer_name)}"
        )
        tgt_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tgt_label.setWordWrap(True)
        layout.addWidget(tgt_label)

        affected_label = QLabel(
            f"<b>Affected:</b> {preview.n_conflicts} conflict{'' if preview.n_conflicts == 1 else 's'}."
        )
        affected_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        affected_label.setWordWrap(True)
        layout.addWidget(affected_label)

        details_label = QLabel("Conflicts (frame → keypoint):")
        details_label.setWordWrap(True)
        layout.addWidget(details_label)

        text = QPlainTextEdit(self)
        text.setReadOnly(True)
        text.setPlainText(_preview_conflict_details_text(preview))
        text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        fm = text.fontMetrics()
        line_h = fm.lineSpacing()
        text.setMinimumHeight(line_h * 6 + 16)
        text.setMaximumHeight(line_h * 16 + 24)
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.merge_btn = QPushButton("Merge non-conflicting points")
        self.merge_btn.setDefault(True)
        self.merge_btn.setAutoDefault(True)
        self.merge_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.merge_btn)

        layout.addLayout(btn_row)

    @staticmethod
    def confirm(parent: QWidget | None, *, preview: TrackingMergePreview) -> bool:
        dlg = TrackingMergeConflictsDialog(parent, preview=preview)
        return dlg.exec_() == QDialog.Accepted


# -----------------------------------------------------------------------------#
# Main merge dialog
# -----------------------------------------------------------------------------#


class TrackingMergeDialog(QDialog):
    """
    Small transactional dialog for tracking-result -> DLC points merge selection.

    Notes
    -----
    - If a valid `fixed_source_layer` is provided, the source is shown read-only.
    - Otherwise, the user selects both source and target from dropdowns.
    - The dialog builds a live preview using `preview_tracking_merge(...)`.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        layer_manager,
        source_candidates: tuple[Points, ...],
        target_candidates: tuple[Points, ...],
        initial_source: Points | None = None,
        initial_target: Points | None = None,
        fixed_source_layer: Points | None = None,
    ):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self._source_candidates = tuple(source_candidates)
        self._target_candidates = tuple(target_candidates)
        self._fixed_source_layer = fixed_source_layer

        self._latest_preview: TrackingMergePreview | None = None

        self.setWindowTitle("Merge tracked points")
        self.setModal(True)
        self.setSizeGripEnabled(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = QLabel(
            "Merge tracked points from a tracking-result layer into a regular DLC Points layer.\n"
            "Only missing points will be added. Existing conflicting annotations will be left unchanged."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Source row
        if fixed_source_layer is not None:
            self._source_label = QLabel(_layer_display_name(fixed_source_layer))
            self._source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._source_label.setWordWrap(True)
            form.addRow("Source tracking layer", self._source_label)
            self._source_combo = None
        else:
            self._source_combo = QComboBox(self)
            for layer in self._source_candidates:
                self._source_combo.addItem(_layer_display_name(layer), layer)
            form.addRow("Source tracking layer", self._source_combo)

        # Target row
        self._target_combo = QComboBox(self)
        for layer in self._target_candidates:
            self._target_combo.addItem(_layer_display_name(layer), layer)
        form.addRow("Target DLC points layer", self._target_combo)

        # Policy row (fixed for v1)
        self._policy_label = QLabel("Fill missing points only (recommended)")
        self._policy_label.setWordWrap(True)
        form.addRow("Merge policy", self._policy_label)

        root.addLayout(form)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._summary_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid palette(mid);
                border-radius: 8px;
                padding: 8px;
                background: palette(alternate-base);
            }
            """
        )
        root.addWidget(self._summary_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._review_conflicts_btn = QPushButton("Review conflicts…")
        self._review_conflicts_btn.clicked.connect(self._review_conflicts)
        btn_row.addWidget(self._review_conflicts_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        self._merge_btn = QPushButton("Merge")
        self._merge_btn.setDefault(True)
        self._merge_btn.setAutoDefault(True)
        self._merge_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._merge_btn)

        root.addLayout(btn_row)

        if self._source_combo is not None:
            self._source_combo.currentIndexChanged.connect(self._refresh_preview)
        self._target_combo.currentIndexChanged.connect(self._refresh_preview)

        self._apply_initial_selection(
            initial_source=initial_source,
            initial_target=initial_target,
        )
        self._refresh_preview()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def latest_preview(self) -> TrackingMergePreview | None:
        return self._latest_preview

    def selected_source_layer(self) -> Points | None:
        if self._fixed_source_layer is not None:
            return self._fixed_source_layer

        if self._source_combo is None:
            return None

        data = self._source_combo.currentData()
        return data if isinstance(data, Points) else None

    def selected_target_layer(self) -> Points | None:
        data = self._target_combo.currentData()
        return data if isinstance(data, Points) else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_initial_selection(
        self,
        *,
        initial_source: Points | None,
        initial_target: Points | None,
    ) -> None:
        if self._fixed_source_layer is None and self._source_combo is not None and initial_source is not None:
            idx = self._find_combo_layer_index(self._source_combo, initial_source)
            if idx >= 0:
                self._source_combo.setCurrentIndex(idx)

        if initial_target is not None:
            idx = self._find_combo_layer_index(self._target_combo, initial_target)
            if idx >= 0:
                self._target_combo.setCurrentIndex(idx)

    @staticmethod
    def _find_combo_layer_index(combo: QComboBox, layer: Points) -> int:
        for i in range(combo.count()):
            if combo.itemData(i) is layer:
                return i
        return -1

    def _refresh_preview(self) -> None:
        source_layer = self.selected_source_layer()
        target_layer = self.selected_target_layer()

        preview: TrackingMergePreview | None = None
        try:
            if source_layer is not None and target_layer is not None:
                preview = preview_tracking_merge(source_layer, target_layer)
        except Exception as e:
            logger.exception("Failed to build tracking merge preview")
            self._summary_label.setText(f"Could not preview merge:\n{e}")
            self._merge_btn.setEnabled(False)
            self._review_conflicts_btn.setEnabled(False)
            self._latest_preview = None
            return

        self._latest_preview = preview
        self._summary_label.setText(_preview_summary_text(preview))

        if preview is None:
            self._merge_btn.setEnabled(False)
            self._review_conflicts_btn.setEnabled(False)
            return

        self._review_conflicts_btn.setEnabled(preview.n_conflicts > 0)

        can_merge = bool(preview.is_valid and preview.n_appendable > 0)
        self._merge_btn.setEnabled(can_merge)

    def _review_conflicts(self) -> None:
        preview = self._latest_preview
        if preview is None or preview.n_conflicts <= 0:
            return

        TrackingMergeConflictsDialog(parent=self, preview=preview).exec_()


# -----------------------------------------------------------------------------#
# Workflow
# -----------------------------------------------------------------------------#


class TrackingMergeWorkflow:
    """
    UI/controller orchestration for merging tracking-result points into DLC points.

    Responsibilities
    ----------------
    - ask the lifecycle manager for valid source/target candidates
    - choose sensible defaults
    - present a transactional merge dialog
    - present a conflict confirmation dialog if needed
    - apply a validated preview back into the target layer

    Non-responsibilities
    --------------------
    - merge semantics (owned by tracking.core.merge)
    - viewer layer classification rules (owned by LayerLifecycleManager)
    """

    def __init__(self, *, parent, viewer, layer_manager, logger_: logging.Logger | None = None):
        self.parent = parent
        self.viewer = viewer
        self.layer_manager = layer_manager
        self.logger = logger_ or logger

    def run(self, *, source_layer: Points | None = None) -> bool:
        """
        Run the tracking merge workflow.

        Parameters
        ----------
        source_layer
            Optional source hint. If this is a valid tracking-result layer according
            to the lifecycle manager, it is fixed as the source in the dialog.

        Returns
        -------
        bool
            True if a merge was successfully applied, False otherwise.
        """
        source_candidates = tuple(self.layer_manager.iter_tracking_result_layers())
        if not source_candidates:
            QMessageBox.warning(
                self.parent,
                "No tracking result layer",
                "No tracking result layer is currently available to merge.",
                QMessageBox.Ok,
            )
            return False

        target_candidates = tuple(self.layer_manager.iter_mergeable_dlc_points_layers(prefer_managed=True))
        if not target_candidates:
            QMessageBox.warning(
                self.parent,
                "No DLC target layer",
                "No regular DLC Points layer is currently available as a merge target.",
                QMessageBox.Ok,
            )
            return False

        fixed_source_layer = None
        initial_source = None

        if (
            source_layer is not None
            and isinstance(source_layer, Points)
            and self.layer_manager.is_tracking_result_layer(source_layer)
        ):
            fixed_source_layer = source_layer
            initial_source = source_layer
        else:
            active = getattr(self.viewer.layers.selection, "active", None)
            if (
                active is not None
                and isinstance(active, Points)
                and self.layer_manager.is_tracking_result_layer(active)
                and active in source_candidates
            ):
                initial_source = active
            else:
                initial_source = source_candidates[0]

        initial_target = self.layer_manager.suggest_merge_target(initial_source)

        dialog = TrackingMergeDialog(
            parent=self.parent,
            layer_manager=self.layer_manager,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            initial_source=initial_source,
            initial_target=initial_target,
            fixed_source_layer=fixed_source_layer,
        )

        if dialog.exec_() != QDialog.Accepted:
            return False

        preview = dialog.latest_preview
        source = dialog.selected_source_layer()
        target = dialog.selected_target_layer()

        if preview is None or source is None or target is None:
            QMessageBox.warning(
                self.parent,
                "Cannot merge",
                "The merge preview could not be completed.",
                QMessageBox.Ok,
            )
            return False

        if not preview.is_valid:
            QMessageBox.warning(
                self.parent,
                "Cannot merge",
                preview.invalid_reason or "This merge is not valid.",
                QMessageBox.Ok,
            )
            return False

        if preview.n_conflicts > 0:
            confirmed = TrackingMergeConflictsDialog.confirm(
                self.parent,
                preview=preview,
            )
            if not confirmed:
                return False

        try:
            new_data, new_features = apply_tracking_merge(
                source_layer=source,
                target_layer=target,
                preview=preview,
            )
        except Exception as e:
            self.logger.exception("Failed to apply tracking merge")
            QMessageBox.warning(
                self.parent,
                "Merge failed",
                f"Could not merge tracked points:\n{e}",
                QMessageBox.Ok,
            )
            return False

        try:
            target.data = new_data
            target.features = new_features
            self.viewer.layers.selection.active = target
            mark_layer_presentation_changed(target)
        except Exception as e:
            self.logger.exception("Failed to write merged data back to target layer")
            QMessageBox.warning(
                self.parent,
                "Merge failed",
                f"The merge was computed but could not be applied to the target layer:\n{e}",
                QMessageBox.Ok,
            )
            return False

        self.viewer.status = (
            f"Merged {preview.n_appendable} tracked point"
            f"{'' if preview.n_appendable == 1 else 's'} into "
            f'"{preview.target_layer_name}"'
        )
        return True
