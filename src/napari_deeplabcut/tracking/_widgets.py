import logging
from functools import partial

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, create_widget
from napari.layers import Image, Points
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
)

from napari_deeplabcut._widgets import KeypointControls
from napari_deeplabcut.config.keybinds import (
    MOVE_BACKWARD_FRAME,
    MOVE_FORWARD_FRAME,
    TRACK_BACKWARD,
    TRACK_BACKWARD_END,
    TRACK_FORWARD,
    TRACK_FORWARD_END,
)
from napari_deeplabcut.config.settings import TRACKING_SHORTCUTS_ENABLED
from napari_deeplabcut.core.keypoints import KeypointStore
from napari_deeplabcut.core.layer_lifecycle import get_or_create_layer_manager
from napari_deeplabcut.core.layer_versioning import mark_layer_presentation_changed
from napari_deeplabcut.ui.base_widget.singleton_widget import ViewerSingletonWidget
from napari_deeplabcut.ui.icons import apply_help_info_icon

from .core.data import (
    TrackingWorkerData,
    TrackingWorkerOutput,
    add_query_identity_columns,
    build_tracking_result_metadata,
)
from .core.models import AVAILABLE_TRACKERS
from .core.refine import apply_delete_tracking_points_in_future, preview_delete_tracking_points_in_future
from .core.utils import (
    make_tracking_iteration_name,
)
from .ui.merger import TrackingMergeWorkflow
from .ui.worker import TrackingWorker

logger = logging.getLogger(__name__)
# TODO @C-Achard: fix the sliders sync not firing (on existing layers ?)


class TrackingControls(ViewerSingletonWidget):
    trackingRequested = Signal(object)
    trackedKeypointsAdded = Signal()

    def __init__(self, viewer: "napari.viewer.Viewer"):
        if not self._singleton_prepare_init(napari_viewer=viewer):
            return
        super().__init__()
        self._singleton_finalize_init()
        self._viewer = self.canonical_viewer(viewer)
        self.lifecycle_manager = get_or_create_layer_manager(viewer)
        # self.setObjectName("napari-deeplabcut-tracking-controls")
        self.setProperty("ndlc_tracking_controls", True)

        # Layout
        ## Data and model selection
        self._tracking_method_combo = QComboBox()
        self._model_info_button = QToolButton()
        apply_help_info_icon(self._model_info_button, theme=getattr(self._viewer, "theme", None))
        self._keypoint_layer_combo: ComboBox = create_widget(annotation=Points)
        self._video_layer_combo: ComboBox = create_widget(annotation=Image)
        self._video_layer_combo.changed.connect(self._video_layer_changed)
        ## Frame selection controls
        self._set_ref_button = QPushButton()
        self._reference_spinbox = QSpinBox()
        self._reference_spinbox.setReadOnly(True)
        self._reference_spinbox.setButtonSymbols(QSpinBox.NoButtons)
        self._updating_controls = False
        ### Backward
        self._backward_slider = QSlider(Qt.Horizontal)
        self._backward_spinbox_absolute = QSpinBox()
        self._backward_spinbox_relative = QSpinBox()
        ### Forward
        self._forward_slider = QSlider(Qt.Horizontal)
        self._forward_spinbox_absolute = QSpinBox()
        self._forward_spinbox_relative = QSpinBox()
        ## Tracking controls
        self._tracking_stop_button = QPushButton()
        self._tracking_forward_button = QPushButton()
        self._tracking_forward_button.clicked.connect(self.track_forward)
        self._tracking_forward_end_button = QPushButton()
        self._tracking_forward_end_button.clicked.connect(self.track_forward_end)
        self._tracking_backward_button = QPushButton()
        self._tracking_backward_end_button = QPushButton()
        self._tracking_backward_end_button.clicked.connect(self.track_backward_end)
        self._tracking_backward_button.clicked.connect(self.track_backward)
        self._tracking_bothway_button = QPushButton()
        self._tracking_bothway_button.clicked.connect(self.track_bothway)
        self._tracking_progress_bar = QProgressBar()
        ## Refine/Merge results controls
        self._delete_selected_future_button = QPushButton("Delete selected points in future frames")
        self._delete_selected_future_button.clicked.connect(self._delete_selected_in_future_frames)
        self._merge_tracked_button = QPushButton("Merge tracked points")
        self._merge_tracked_button.clicked.connect(self._open_merge_workflow)

        # Controls
        ## Forward controls
        self._forward_slider.valueChanged.connect(partial(self._forward_update, from_absolute=False, from_slider=True))
        self._forward_spinbox_relative.valueChanged.connect(
            partial(self._forward_update, from_absolute=False, from_slider=False)
        )
        self._forward_spinbox_absolute.valueChanged.connect(
            partial(self._forward_update, from_absolute=True, from_slider=False)
        )
        ## Backward controls
        self._backward_slider.valueChanged.connect(
            partial(self._backward_update, from_absolute=False, from_slider=True)
        )
        self._backward_spinbox_relative.valueChanged.connect(
            partial(self._backward_update, from_absolute=False, from_slider=False)
        )
        self._backward_spinbox_absolute.valueChanged.connect(
            partial(self._backward_update, from_absolute=True, from_slider=False)
        )

        # when the range of viewer dims changes (e.g. on opening a new video), update the reference spinbox max
        self._viewer.dims.events.current_step.connect(self._on_current_step_changed)
        self._viewer.dims.events.current_step.connect(self._set_frame_controls_range)
        self._reference_spinbox.valueChanged.connect(self._set_frame_controls_range)

        # Worker
        self.is_tracking = False
        self.worker_started = False
        self.worker: TrackingWorker | None = None

        # Reference to the keypoint control widget.
        # this gets assigned after the user requests tracking for the first time.
        self.keypoint_widget: KeypointControls | None = None

        self._setup_keybindings(viewer=viewer)

        self._build_layout()
        self._schedule_once("tracking_controls_initial_sync", 0, self._sync_from_viewer_dims)

    def _set_model_info_tooltip(self, current_model_name: str = None):
        """Retrieves the display info for the selected model and sets it as tooltip for the model info button."""
        tracker_info = AVAILABLE_TRACKERS.get(current_model_name, None)
        tracker_info = tracker_info["class"].info_text if tracker_info is not None else None
        if tracker_info is not None:
            self._model_info_button.setToolTip(tracker_info)
        else:
            self._model_info_button.setToolTip("")

    def _set_tooltips(self):
        self._tracking_forward_button.setToolTip(TRACK_FORWARD.get_display())
        self._tracking_forward_end_button.setToolTip(TRACK_FORWARD_END.get_display())
        self._tracking_backward_button.setToolTip(TRACK_BACKWARD.get_display())
        self._tracking_backward_end_button.setToolTip(TRACK_BACKWARD_END.get_display())
        self._tracking_bothway_button.setToolTip("Track both ways")
        self._tracking_stop_button.setToolTip("Stop tracking")
        self._set_ref_button.setToolTip("Set reference frame")

    def _dock_widget(self):
        try:
            for dock in self._viewer.window._dock_widgets.values():
                if dock.widget() is self:
                    return dock
        except Exception:
            return None
        return None

    def _sync_from_viewer_dims(self) -> None:
        """Initialize/sync tracking controls from the current viewer dims."""
        try:
            current_step = tuple(self._viewer.dims.current_step)
            if len(current_step) > 0:
                self._reference_spinbox.setValue(int(current_step[0]))
            self._set_frame_controls_range()
        except Exception:
            logger.debug("Failed to sync tracking controls from viewer dims", exc_info=True)

    @Slot(object)
    def _on_current_step_changed(self, event):
        try:
            self._reference_spinbox.setValue(int(event.value[0]))
        except Exception:
            logger.debug("Failed to update reference frame from dims", exc_info=True)

    def _tracking_shortcuts_active(self) -> bool:
        if not TRACKING_SHORTCUTS_ENABLED:
            return False
        dock = self._dock_widget()
        return dock is not None and dock.isVisible()

    def _setup_keybindings(self, viewer: "napari.viewer.Viewer"):
        if not TRACKING_SHORTCUTS_ENABLED:
            self._set_tooltips()
            return

        @Points.bind_key(TRACK_FORWARD.key, overwrite=True)
        def track_forward(event):
            if not self._tracking_shortcuts_active():
                return
            self.track_forward()

        @Points.bind_key(TRACK_FORWARD_END.key, overwrite=True)
        def track_forward_end(event):
            if not self._tracking_shortcuts_active():
                return
            self.track_forward_end()

        @Points.bind_key(TRACK_BACKWARD.key, overwrite=True)
        def track_backward(event):
            if not self._tracking_shortcuts_active():
                return
            self.track_backward()

        @Points.bind_key(TRACK_BACKWARD_END.key, overwrite=True)
        def track_backward_end(event):
            if not self._tracking_shortcuts_active():
                return
            self.track_backward_end()

        @Points.bind_key(MOVE_FORWARD_FRAME.key, overwrite=True)
        def move_forward_frame(event):
            if not self._tracking_shortcuts_active():
                return
            viewer.dims.current_step = (
                viewer.dims.current_step[0] + 1,
                *viewer.dims.current_step[1:],
            )

        @Points.bind_key(MOVE_BACKWARD_FRAME.key, overwrite=True)
        def move_backward_frame(event):
            if not self._tracking_shortcuts_active():
                return
            viewer.dims.current_step = (
                viewer.dims.current_step[0] - 1,
                *viewer.dims.current_step[1:],
            )

        self._set_tooltips()

    def _update_frame_controls(
        self,
        slider,
        relative_spinbox,
        absolute_spinbox,
        reference_spinbox,
        value,
        direction,
        from_absolute=False,
        from_slider=False,
    ):
        """
        Generic function to update slider, relative spinbox, and absolute spinbox.

        Parameters:
        - slider: The slider widget.
        - relative_spinbox: The relative spinbox widget.
        - absolute_spinbox: The absolute spinbox widget.
        - reference_spinbox: The reference spinbox widget.
        - value: The new value to set.
        - direction: "forward" or "backward".
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            if from_absolute:
                # Update relative and slider from absolute spinbox
                relative_value = value - reference_spinbox.value()
                if direction == "forward":
                    relative_value = max(0, relative_value)
                else:  # backward
                    relative_value = min(0, relative_value)
                relative_spinbox.setValue(relative_value)
                slider.setValue(relative_value)
            elif from_slider:
                # Update relative and absolute spinboxes from slider
                relative_spinbox.setValue(value)
                absolute_spinbox.setValue(reference_spinbox.value() + value)
            else:
                # Update slider and absolute spinbox from relative spinbox
                slider.setValue(value)
                absolute_spinbox.setValue(reference_spinbox.value() + value)
        finally:
            self._updating_controls = False

    def _seed_query_points_and_features(
        self,
        ref_frame_idx: int,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Extract the points/features from the chosen reference frame and attach
        stable query identity columns before sending them to the model.
        """
        layer = self.keypoint_layer
        if layer is None:
            raise ValueError("No keypoint layer selected.")

        mask = np.asarray(layer.data[:, 0]).astype(int) == int(ref_frame_idx)

        keypoints = np.asarray(layer.data[mask], dtype=float).copy()
        if len(keypoints) == 0:
            raise ValueError(
                f"No keypoints found on reference frame {ref_frame_idx}. "
                "Did you select the right layer and frame in the tracking controls?"
            )

        layer_features = layer.features
        if isinstance(layer_features, pd.DataFrame):
            seed_features = layer_features.loc[mask].reset_index(drop=True).copy()
        else:
            seed_features = pd.DataFrame(layer_features).loc[mask].reset_index(drop=True).copy()

        if len(seed_features) != len(keypoints):
            raise ValueError(
                f"Seed feature row count mismatch: got {len(seed_features)} feature rows "
                f"for {len(keypoints)} keypoints."
            )

        seed_features = add_query_identity_columns(
            seed_features,
            query_frame=ref_frame_idx,
            source_layer_name=layer.name,
        )

        # In the sliced tracking video, the query frame is always time 0
        keypoints[:, 0] = 0.0

        return keypoints, seed_features

    def _create_tracking_result_layer(
        self,
        keypoints: np.ndarray,
        features: pd.DataFrame,
        *,
        tracker_name: str,
        ref_frame_idx: int,
    ) -> Points:
        """
        Create a new Points layer holding the tracking result.
        This must NOT modify the original DLC annotation layer.
        """
        source = self.keypoint_layer
        if source is None:
            raise ValueError("No source keypoint layer selected.")

        # Preserve original DLC source provenance even if tracking is launched
        # from an existing tracking-result layer.
        source_layer_name = (
            self.lifecycle_manager.tracking_result_source_layer_name(source)
            if self.lifecycle_manager.is_tracking_result_layer(source)
            else None
        ) or source.name

        metadata = build_tracking_result_metadata(
            source.metadata,
            tracker_name=tracker_name,
            source_layer_name=source_layer_name,
            query_frame=ref_frame_idx,
        )

        layer_name = make_tracking_iteration_name(
            viewer=self._viewer,
            tracker_name=tracker_name,
            ref_frame_idx=ref_frame_idx,
            source=source,
        )

        layer = self._viewer.add_points(
            data=keypoints,
            features=features,
            name=layer_name,
            metadata=metadata,
        )

        # Distinguish tracking results visually
        self.lifecycle_manager.apply_points_display_settings(layer, source=source)

        return layer

    def _forward_update(self, value: int, from_absolute: bool, from_slider: bool):
        """Helper to update forward controls.

        Parameters:
        - value: The new value to set.
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        self._update_frame_controls(
            slider=self._forward_slider,
            relative_spinbox=self._forward_spinbox_relative,
            absolute_spinbox=self._forward_spinbox_absolute,
            reference_spinbox=self._reference_spinbox,
            value=value,
            direction="forward",
            from_absolute=from_absolute,
            from_slider=from_slider,
        )

    def _backward_update(self, value: int, from_absolute: bool, from_slider: bool):
        """Helper to update backward controls.

        Parameters:
        - value: The new value to set.
        - from_absolute: Whether the update is triggered by the absolute spinbox.
        - from_slider: Whether the update is triggered by the slider.
        """
        self._update_frame_controls(
            slider=self._backward_slider,
            relative_spinbox=self._backward_spinbox_relative,
            absolute_spinbox=self._backward_spinbox_absolute,
            reference_spinbox=self._reference_spinbox,
            value=value,
            direction="backward",
            from_absolute=from_absolute,
            from_slider=from_slider,
        )

    @Slot()
    def _set_frame_controls_range(self):
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            if self.video_layer is None:
                return

            max_frames = self.video_layer.data.shape[0] - 1
            # logger.debug(f"Updating tracking controls for video with {max_frames + 1} frames.")
            current_frame = max(0, min(self._viewer.dims.current_step[0], max_frames))
            # logger.debug(f"Current frame: {current_frame}")
            self._reference_spinbox.setRange(0, max_frames)
            self._reference_spinbox.setValue(current_frame)

            forward_delta = self._forward_spinbox_relative.value()
            self._forward_slider.setRange(0, max_frames - current_frame)
            self._forward_slider.setValue(forward_delta)
            self._forward_spinbox_relative.setRange(0, max_frames - current_frame)
            self._forward_spinbox_relative.setValue(forward_delta)
            self._forward_spinbox_absolute.setRange(current_frame, max_frames)
            self._forward_spinbox_absolute.setValue(current_frame + forward_delta)  # see _forward_update

            backward_delta = self._backward_spinbox_relative.value()
            self._backward_slider.setRange(-current_frame, 0)
            self._backward_slider.setValue(backward_delta)
            self._backward_spinbox_relative.setRange(-current_frame, 0)
            self._backward_spinbox_relative.setValue(backward_delta)
            self._backward_spinbox_absolute.setRange(0, current_frame)
            self._backward_spinbox_absolute.setValue(
                current_frame + self._backward_spinbox_relative.value()
            )  # see _backward_update

        finally:
            self._updating_controls = False

    def _start_worker(self):
        self.is_tracking = False
        self.worker_started = False
        self.worker = TrackingWorker()

        # Explicit queued connections for cross-thread delivery back to this QWidget
        self.worker.trackingStarted.connect(self.tracking_started, Qt.QueuedConnection)
        self.worker.started.connect(self._on_worker_started, Qt.QueuedConnection)
        self.worker.finished.connect(self._on_worker_finished, Qt.QueuedConnection)
        self.worker.progress.connect(self._on_worker_progress, Qt.QueuedConnection)
        self.worker.trackingFinished.connect(self.tracking_finished, Qt.QueuedConnection)
        self.worker.trackingStopped.connect(self.tracking_stopped, Qt.QueuedConnection)

        # Main thread -> worker thread
        self.trackingRequested.connect(self.worker.track, Qt.QueuedConnection)

        # Main-thread button click, no UI work in worker
        self._tracking_stop_button.clicked.connect(self._request_worker_stop)

        self.worker.start()

    def _request_worker_stop(self):
        if self.worker is not None:
            self.worker.request_stop()

    def _debug_thread(self, where: str) -> None:
        import threading

        from qtpy.QtCore import QThread

        logger.debug(
            "%s | python_thread=%s | qt_current_thread=%r | widget_thread=%r",
            where,
            threading.current_thread().name,
            QThread.currentThread(),
            self.thread,
        )

    @Slot()
    def _on_worker_started(self):
        self._debug_thread("_on_worker_started")
        self.worker_started = True

    @Slot()
    def _on_worker_finished(self):
        self._debug_thread("_on_worker_finished")
        self.worker_started = False

    @Slot(int, int)
    def _on_worker_progress(self, current: int, total: int):
        self._debug_thread("_on_worker_progress")
        if self._tracking_progress_bar.maximum() != total:
            self._tracking_progress_bar.setMaximum(total)
        self._tracking_progress_bar.setValue(current)

    @property
    def keypoint_layer(self) -> Points | None:
        return self._keypoint_layer_combo.value

    @property
    def keypoint_store(self) -> KeypointStore | None:
        return self.keypoint_widget._stores[self.keypoint_layer] if self.keypoint_widget else None

    @property
    def video_layer(self) -> Image | None:
        return self._video_layer_combo.value

    @Slot()
    def _video_layer_changed(self):
        if self._viewer.dims.ndim != 3:
            return
        self._set_frame_controls_range()

    @Slot()
    def tracking_started(self):
        self.is_tracking = True
        # self._tracking_progress_bar.setValue(0) # let model handle this
        self._tracking_progress_bar.setMaximum(0)  # indeterminate progress until model sends an update

    @Slot(object)
    def tracking_finished(self, out: TrackingWorkerOutput):
        self.is_tracking = False
        try:
            viewer_step = tuple(self._viewer.dims.current_step)

            new_features_df = (
                out.keypoint_features
                if isinstance(out.keypoint_features, pd.DataFrame)
                else pd.DataFrame(out.keypoint_features)
            )

            ref_frame_idx = (
                int(new_features_df["tracking_query_frame"].iloc[0])
                if "tracking_query_frame" in new_features_df.columns and len(new_features_df)
                else int(self._reference_spinbox.value())
            )

            layer = self._create_tracking_result_layer(
                out.keypoints,
                new_features_df,
                tracker_name=self._tracking_method_combo.currentText(),
                ref_frame_idx=ref_frame_idx,
            )

            self._viewer.layers.selection.active = layer
            self._viewer.dims.current_step = viewer_step
            self._select_keypoint_combo_layer(layer, restore_step=viewer_step)
            self._viewer.status = f'Created tracking result layer "{layer.name}"'
        except Exception as e:
            logger.exception("Error creating tracking result layer", exc_info=e)

        self._tracking_progress_bar.setValue(self._tracking_progress_bar.maximum())
        self._tracking_progress_bar.setFormat("%p% Done")
        self.trackedKeypointsAdded.emit()

    @Slot()
    def tracking_stopped(self):
        self.is_tracking = False
        self._tracking_progress_bar.setValue(self._tracking_progress_bar.maximum())
        self._tracking_progress_bar.setFormat("%p% Stopped")

    @Slot()
    def track_forward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        forward_frame_idx: int = self._forward_spinbox_absolute.value()
        if forward_frame_idx <= ref_frame_idx:
            return
        self.track(
            (ref_frame_idx, forward_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=False,
        )

    @Slot()
    def track_forward_end(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        forward_frame_idx: int = self.video_layer.data.shape[0] - 1
        if forward_frame_idx <= ref_frame_idx:
            return
        self.track(
            (ref_frame_idx, forward_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=False,
        )

    @Slot()
    def track_backward(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        backward_frame_idx: int = self._backward_spinbox_absolute.value()
        if backward_frame_idx >= ref_frame_idx:
            return
        logger.debug(f"Tracking backward from {backward_frame_idx} to {ref_frame_idx}")
        self.track(
            (backward_frame_idx, ref_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=True,
        )

    @Slot()
    def track_backward_end(self):
        ref_frame_idx: int = self._reference_spinbox.value()
        backward_frame_idx: int = 0
        if backward_frame_idx >= ref_frame_idx:
            return
        self.track(
            (backward_frame_idx, ref_frame_idx + 1),
            ref_frame_idx,
            backward_tracking=True,
        )

    @Slot()
    def track_bothway(self):
        # if forward target is invalid, go directly backward
        ref = self._reference_spinbox.value()
        fwd = self._forward_spinbox_absolute.value()
        if fwd <= ref:
            self.track_backward()
            return

        self.track_forward()
        self.trackedKeypointsAdded.connect(self.track_backward, type=Qt.ConnectionType.SingleShotConnection)

    def track(self, keypoint_range: tuple[int, int], ref_frame_idx, backward_tracking=False):
        if not self.worker_started:
            self._start_worker()

        if not self.keypoint_widget:
            for k, v in self._viewer.window._dock_widgets.items():
                if "Keypoint controls" in k and "napari-deeplabcut" in k:
                    self.keypoint_widget = v.widget()
                    break

        if self.is_tracking:
            return

        if self.video_layer is None:
            logger.warning("No video layer selected.")
            return

        if self.keypoint_layer is None:
            logger.warning("No keypoint layer selected.")
            return

        self._tracking_progress_bar.setFormat("%p%")

        if backward_tracking:
            video_slice = self.video_layer.data[keypoint_range[0] : keypoint_range[1]][::-1]
        else:
            video_slice = self.video_layer.data[keypoint_range[0] : keypoint_range[1]]

        try:
            seed_keypoints, seed_features = self._seed_query_points_and_features(ref_frame_idx)
        except ValueError as e:
            logger.warning(str(e))
            napari.utils.notifications.show_warning(str(e))
            return

        tracking_data = TrackingWorkerData(
            tracker_name=self._tracking_method_combo.currentText(),
            video=video_slice,
            keypoints=seed_keypoints,
            keypoint_features=seed_features,
            keypoint_range=keypoint_range,
            backward_tracking=backward_tracking,
            reference_frame_index=int(ref_frame_idx),
        )
        self.trackingRequested.emit(tracking_data)

    def _select_keypoint_combo_layer(self, layer: Points, *, restore_step: tuple[int, ...] | None = None) -> None:
        """
        Select a Points layer in the keypoint combo, deferring one event-loop turn
        so the combo choices are fully refreshed after layer insertion.
        """
        try:
            self._keypoint_layer_combo.reset_choices()
        except Exception:
            logger.debug("Failed to reset keypoint layer combo choices", exc_info=True)

        def _apply():
            try:
                self._keypoint_layer_combo.value = layer
                if restore_step is not None:
                    self._viewer.dims.current_step = restore_step
            except Exception:
                logger.debug(
                    "Failed to select tracked result layer in keypoint combo",
                    exc_info=True,
                )

        # Layer insertion / choice refresh may still be settling.
        self._schedule_once("update_tracking_layer_combobox", 0, _apply)

    def _delete_selected_in_future_frames(self) -> None:
        active = self._viewer.layers.selection.active
        if not isinstance(active, Points) or not self.lifecycle_manager.is_tracking_result_layer(active):
            napari.utils.notifications.show_warning("Select an active tracking-result Points layer first.")
            return

        selected = getattr(active, "selected_data", None) or set()
        if not selected:
            napari.utils.notifications.show_warning("Select one or more tracked points on the current frame first.")
            return

        anchor_frame = int(self._viewer.dims.current_step[0])

        try:
            preview = preview_delete_tracking_points_in_future(
                active,
                selected_indices=selected,
                anchor_frame=anchor_frame,
            )
        except Exception as e:
            logger.exception("Failed to build future-delete preview")
            napari.utils.notifications.show_warning(f"Could not prepare future deletion:\n{e}")
            return

        if not preview.is_valid:
            message = preview.invalid_reason or "Cannot delete future tracked points."

            if preview.ambiguous_slot_frames:
                lines = []
                for frame, slot_id, label in preview.ambiguous_slot_frames[:8]:
                    if slot_id:
                        lines.append(f"frame {frame}: {label} (id: {slot_id})")
                    else:
                        lines.append(f"frame {frame}: {label}")
                if len(preview.ambiguous_slot_frames) > 8:
                    lines.append(f"… and {len(preview.ambiguous_slot_frames) - 8} more")

                message = f"{message}\n\nAmbiguous future rows:\n" + "\n".join(lines)

            napari.utils.notifications.show_warning(message)
            return

        if preview.n_rows_to_delete <= 0:
            napari.utils.notifications.show_info("No matching tracked points were found in future frames.")
            return

        message = (
            f"Delete {preview.n_rows_to_delete} tracked point"
            f"{'s' if preview.n_rows_to_delete != 1 else ''} "
            f'from future frames in "{preview.layer_name}"?\n\n'
            f"{preview.n_selected_slot_keys} selected keypoint "
            f"on frame {preview.anchor_frame} will be used as the reference.\n"
            f"Only exact matches in future frames (after {preview.anchor_frame}) will be removed.\n"
            "Selected points on the current frame will be kept so you may correct them and run tracking again."
        )

        answer = QMessageBox.question(
            self,
            "Delete tracked points in future frames",
            message,
            QMessageBox.Cancel | QMessageBox.Ok,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Ok:
            return

        try:
            new_data, new_features = apply_delete_tracking_points_in_future(
                active,
                preview=preview,
            )
        except Exception as e:
            logger.exception("Failed to apply future-delete action")
            napari.utils.notifications.show_warning(f"Could not delete future tracked points:\n{e}")
            return

        try:
            active.data = new_data
            active.features = new_features
            self._viewer.layers.selection.active = active
            mark_layer_presentation_changed(active)
        except Exception as e:
            logger.exception("Failed to write refined tracking data back to layer")
            napari.utils.notifications.show_warning(
                f"The tracked points were computed for deletion but could not be applied:\n{e}"
            )
            return

        self._viewer.status = (
            f"Removed {preview.n_rows_to_delete} tracked point"
            f"{'s' if preview.n_rows_to_delete != 1 else ''} "
            f'from future frames in "{preview.layer_name}"'
        )

    def _open_merge_workflow(self):
        active = self._viewer.layers.selection.active
        hinted_source = (
            active if isinstance(active, Points) and self.lifecycle_manager.is_tracking_result_layer(active) else None
        )

        workflow = TrackingMergeWorkflow(
            parent=self,
            viewer=self._viewer,
            layer_manager=self.lifecycle_manager,
            logger_=logger,
        )
        workflow.run(source_layer=hinted_source)

    def _build_layout(self):
        # Layout
        self.setLayout(QVBoxLayout())
        ## Model selection
        self._tracking_method_combo.addItems(AVAILABLE_TRACKERS.keys())
        self._tracking_method_combo.setCurrentIndex(0)

        _model_info_layout = QHBoxLayout()
        _model_info_layout.addWidget(QLabel("Tracker"))
        _model_info_layout.addWidget(self._model_info_button)

        _tracking_method_layout = QHBoxLayout()
        _tracking_method_layout.addLayout(_model_info_layout)
        _tracking_method_layout.addWidget(self._tracking_method_combo)
        self._tracking_method_combo.currentTextChanged.connect(self._set_model_info_tooltip)
        self._set_model_info_tooltip(self._tracking_method_combo.currentText())
        ## Layer selection
        ### Keypoint layer
        self._viewer.layers.events.inserted.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._keypoint_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._keypoint_layer_combo.reset_choices)
        _keypoint_layer_method_layout = QHBoxLayout()
        _keypoint_layer_method_layout.addWidget(QLabel("Keypoints"))
        _keypoint_layer_method_layout.addWidget(self._keypoint_layer_combo.native)
        ### Video layer
        self._viewer.layers.events.inserted.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._video_layer_combo.reset_choices)
        self._viewer.layers.events.reordered.connect(self._video_layer_combo.reset_choices)
        _video_layer_method_layout = QHBoxLayout()
        _video_layer_method_layout.addWidget(QLabel("Video"))
        _video_layer_method_layout.addWidget(self._video_layer_combo.native)

        # Stack previous layouts
        self.layout().addLayout(_tracking_method_layout)
        self.layout().addLayout(_keypoint_layer_method_layout)
        self.layout().addLayout(_video_layer_method_layout)

        ## Frame range controls
        range_controls_layout = QGridLayout()  # 3 by 5
        self._backward_slider.setRange(-100, 0)
        # NOTE : check why this is not inverting the slider appearance as expected
        # self._backward_slider.setInvertedAppearance(True)
        range_controls_layout.addWidget(self._backward_slider, 0, 0, 1, 2)
        self._backward_spinbox_absolute.setRange(0, 100)
        self._backward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_absolute.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(self._backward_spinbox_absolute, 1, 1)
        range_controls_layout.addWidget(QLabel("<< Abs"), 1, 0)
        self._backward_spinbox_relative.setRange(-100, 0)
        self._backward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._backward_spinbox_relative.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("<< Rel"), 2, 0)
        range_controls_layout.addWidget(self._backward_spinbox_relative, 2, 1)
        _ref_label = QLabel("Current")
        self._reference_spinbox.setRange(0, 100)
        self._reference_spinbox.setAlignment(Qt.AlignCenter)
        self._reference_spinbox.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(self._reference_spinbox, 1, 2)
        _ref_label.setAlignment(Qt.AlignCenter)
        range_controls_layout.addWidget(_ref_label, 0, 2)
        self._forward_slider.setRange(0, 100)
        range_controls_layout.addWidget(self._forward_slider, 0, 3, 1, 2)
        self._forward_spinbox_absolute.setRange(0, 100)
        self._forward_spinbox_absolute.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_absolute.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("Abs >>"), 1, 4)
        range_controls_layout.addWidget(self._forward_spinbox_absolute, 1, 3)
        self._forward_spinbox_relative.setRange(0, 100)
        self._forward_spinbox_relative.setAlignment(Qt.AlignCenter)
        self._forward_spinbox_relative.setStyleSheet(
            """
            QSpinBox {
                padding: 0;
            }
        """
        )
        range_controls_layout.addWidget(QLabel("Rel >>"), 2, 4)
        range_controls_layout.addWidget(self._forward_spinbox_relative, 2, 3)

        self.layout().addLayout(range_controls_layout)

        ## Start/stop tracking controls

        def themed_icon(name: str, fallback: QStyle.StandardPixmap) -> QIcon:
            """Use napari's QApplication instance to get style and fallback icons."""
            # More consistent than using Unicode characters that may vary in appearance across platforms
            # Esp. on high-res displays they are very small (4K laptop screens etc)
            style = QApplication.instance().style()  # reuse existing app
            return QIcon.fromTheme(name, style.standardIcon(fallback))

        tracking_controls_layout = QGridLayout()  # 2 by 3
        # NOTE : leaving previous unicode characters as comments for reference
        # self._tracking_backward_button.setText("⇤")
        # self._tracking_backward_end_button.setText("⇤⇤")
        # self._tracking_stop_button.setText("□")
        # self._tracking_forward_button.setText("⇥")
        # self._tracking_forward_end_button.setText("⇥⇥")
        # self._tracking_bothway_button.setText("↹")
        self._tracking_backward_button.setIcon(themed_icon("go-previous", QStyle.SP_ArrowLeft))
        tracking_controls_layout.addWidget(self._tracking_backward_button, 0, 0)
        self._tracking_backward_end_button.setIcon(themed_icon("media-seek-backward", QStyle.SP_MediaSeekBackward))
        tracking_controls_layout.addWidget(self._tracking_backward_end_button, 1, 0)

        self._tracking_stop_button.setIcon(themed_icon("media-playback-stop", QStyle.SP_MediaStop))
        tracking_controls_layout.addWidget(self._tracking_stop_button, 0, 1)

        self._tracking_forward_button.setIcon(themed_icon("go-next", QStyle.SP_ArrowRight))
        tracking_controls_layout.addWidget(self._tracking_forward_button, 0, 2)
        self._tracking_forward_end_button.setIcon(themed_icon("media-seek-forward", QStyle.SP_MediaSeekForward))
        tracking_controls_layout.addWidget(self._tracking_forward_end_button, 1, 2)

        # NOTE: Find a better icon ? Not really any standard icon for "both way"
        self._tracking_bothway_button.setIcon(themed_icon("view-refresh", QStyle.SP_BrowserReload))
        tracking_controls_layout.addWidget(self._tracking_bothway_button, 1, 1)

        self._tracking_progress_bar.setRange(0, 100)
        self.layout().addLayout(tracking_controls_layout)
        self.layout().addWidget(self._tracking_progress_bar)

        # Refine / merge controls
        self.layout().addSpacing(16)
        self._delete_selected_future_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 6px;
                padding: 6px 12px;
                margin-top: 4px;
            }
            """
        )
        self.layout().addWidget(self._delete_selected_future_button)

        self._merge_tracked_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 6px;
                font-weight: bold;
                padding: 6px 12px;
                margin-top: 8px;
            }
            """
        )
        self.layout().addWidget(self._merge_tracked_button)
