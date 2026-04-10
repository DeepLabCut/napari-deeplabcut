"""Main widget and controls for napari-deeplabcut, including the tutorial and shortcuts windows.

NOTE: This file is generally already too long. For future development, please consider:
- Moving existing responsibilities out into separate modules (existing or new)
- Avoiding adding anything that is not strictly related to :
    - Building the final UI (blocks can be moved to ui/ for better organization)
    - Wiring to the core plugin functionality (e.g. via signals/slots, method calls, etc.)
    - Anything that requires the full widget+viewer+signal/event context to function properly
    - Similarly, test_widgets.py is a bit of a default drawer right now, please create new tests in _tests/ui
- Lifecycle of UI elements and Qt wiring should ideally:
    - Use parent child widgets/controllers to KeypointControls
    - Use child QTimers instead of fire-and-forget QTimer.singleShot for deferred UI work
    - Use normal Qt signal connections for Qt-owned objects
    - Keep explicit cleanup only for non-Qt subscriptions/resources
    (e.g. napari event connections, observer install/uninstall, monkey-patch restoration)

TODO: And general dev notes:
- The saving workflow is crammed into save_layers_dialog() right now,
  and should move to a dedicated  e.g. PointsLayerSaveFactory class in a dedicated file.
- Project/root/paths/image-meta/points-meta synchronization should be centralized.
  It is too distributed right now, and it can be unclear "which truth" is authoritative.
  Some sort of context-manager class would likely help.
- Maybe a dedicated layer lifecycle system would help, for layer adoption and setup.
- Something that owns UI sync state (what to refresh, why, when) could help with the heavy wiring.
- I'd suggest keeping in this file:
    - color/label mode, menu/help actions, widget visibility and user interaction hooks.
"""

# src/napari_deeplabcut/_widgets.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
from napari.layers import Image, Points, Tracks
from napari.utils.events import Event
from napari.utils.history import get_save_history, update_save_history
from pydantic import ValidationError
from qtpy.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

import napari_deeplabcut.core.io as io
from napari_deeplabcut import misc
from napari_deeplabcut.config import settings
from napari_deeplabcut.config.keybinds import (
    install_global_points_keybindings,
    install_points_layer_keybindings,
)
from napari_deeplabcut.config.models import DLCHeaderModel, ImageMetadata, PointsMetadata
from napari_deeplabcut.core import keypoints
from napari_deeplabcut.core.config_sync import (
    load_point_size_from_config,
    resolve_config_path_from_layer,
    save_point_size_to_config,
)
from napari_deeplabcut.core.conflicts import compute_overwrite_report_for_points_save
from napari_deeplabcut.core.layer_versioning import mark_layer_presentation_changed
from napari_deeplabcut.core.layers import (
    PointsInteractionEvent,
    PointsInteractionObserver,
    compute_label_progress,
    find_relevant_image_layer,
    get_first_points_layer,
    get_points_layer_with_tables,
    get_uniform_point_size,
    infer_folder_display_name,
    is_machine_layer,
    set_uniform_point_size,
)
from napari_deeplabcut.core.metadata import (
    MergePolicy,
    apply_project_paths_override_to_points_meta,
    infer_image_root,
    migrate_points_layer_metadata,
    read_points_meta,
    sync_points_from_image,
    write_points_meta,
)
from napari_deeplabcut.core.project_paths import (
    PathMatchPolicy,
    coerce_paths_to_dlc_row_keys,
    dataset_folder_has_files,
    find_nearest_config,
    resolve_project_root_from_config,
    target_dataset_folder_for_config,
)
from napari_deeplabcut.core.provenance import (
    apply_gt_save_target,
    is_projectless_folder_association_candidate,
    requires_gt_promotion,
    suggest_human_placeholder,
)
from napari_deeplabcut.core.remap import remap_layer_data_by_paths
from napari_deeplabcut.core.sidecar import (
    get_default_scorer,
    set_default_scorer,
)
from napari_deeplabcut.core.trails import TrailsController, safe_folder_anchor_from_points_layer
from napari_deeplabcut.napari_compat import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
    patch_color_manager_guess_continuous,
    register_points_action,
)
from napari_deeplabcut.napari_compat.points_layer import make_paste_data
from napari_deeplabcut.ui import dialogs as ui_dialogs
from napari_deeplabcut.ui.color_scheme_display import ColorSchemePanel
from napari_deeplabcut.ui.cropping import (
    build_video_action_menu,
    handle_apply_crop_toggled,
    resolve_project_path_from_image_layer,
    run_extract_current_frame,
    run_store_crop_coordinates,
    update_video_panel_context,
)
from napari_deeplabcut.ui.dialogs import Shortcuts, Tutorial
from napari_deeplabcut.ui.labels_and_dropdown import (
    DropdownMenu,
    KeypointsDropdownMenu,
)
from napari_deeplabcut.ui.layer_stats import LayerStatusPanel
from napari_deeplabcut.ui.plots.trajectory import TrajectoryMatplotlibCanvas

logger = logging.getLogger("napari-deeplabcut._widgets")
# logger.setLevel(logging.DEBUG)  # FIXME @C-Achard temp remove before merging


def _prompt_for_scorer(parent_widget, *, anchor: str, suggested: str) -> str | None:
    """Prompt user for a scorer name. Returns non-empty string or None if cancelled."""
    text, ok = QInputDialog.getText(
        parent_widget,
        "Choose scorer",
        "No DLC config.yaml scorer found.\n"
        "Please enter a scorer name for the CollectedData file.\n\n"
        "Tip: Use your name or a stable lab identifier.\n"
        "(We strongly discourage keeping the generic 'human_xxxxxx'.)",
        text=suggested,
    )
    if not ok:
        return None
    scorer = (text or "").strip()
    if not scorer:
        return None
    return scorer


@contextmanager
def _temporary_layer_metadata(layer: Points, metadata: dict):
    old_metadata = dict(layer.metadata or {})
    layer.metadata = metadata
    try:
        yield
    finally:
        layer.metadata = old_metadata


class KeypointControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        # Monkey-patch napari continuous variable type guess
        patch_color_manager_guess_continuous()

        self._is_saved = False

        self.viewer = napari_viewer

        self.viewer.layers.events.inserted.connect(self.on_insert)
        self.viewer.layers.events.removed.connect(self.on_remove)

        # self.viewer.window.qt_viewer._get_and_try_preferred_reader = MethodType(
        #     _get_and_try_preferred_reader,
        #     self.viewer.window.qt_viewer,
        # )
        # Project data
        self._project_path: str | None = None

        status_bar = self.viewer.window._qt_window.statusBar()
        self.last_saved_label = QLabel("")
        self.last_saved_label.hide()
        status_bar.addPermanentWidget(self.last_saved_label)

        self._color_mode = keypoints.ColorMode.default()
        self._label_mode = keypoints.LabelMode.default()

        # Hold references to the KeypointStores
        self._stores = {}
        # Intercept close event if data were not saved
        qt_win = self.viewer.window._qt_window
        orig_close_event = qt_win.closeEvent

        # Wrap event without overriding the original
        # for future-proofing
        def _close_event(event):
            self.on_close(event)
            points_inter = getattr(self, "_points_interactions", None)
            if points_inter is not None:
                points_inter.close()
            # if accepted, call original
            if event.isAccepted():
                orig_close_event(event)

        qt_win.closeEvent = _close_event

        # Storage for extra image metadata that are relevant to other layers.
        # These are updated anytime images are added to the Viewer
        # and passed on to the other layers upon creation.
        self._image_meta = ImageMetadata()
        # Storage for layers requiring recoloring
        self._recolor_pending = set()

        # Add some more controls
        self._layout = QVBoxLayout(self)
        self._menus = []
        self._layer_to_menu = {}
        self.viewer.layers.selection.events.active.connect(self.on_active_layer_change)

        self._video_group = build_video_action_menu(
            on_extract_frame=self._extract_single_frame,
            on_store_crop=self._store_crop_coordinates,
        )
        self.video_widget = self.viewer.window.add_dock_widget(self._video_group, name="video", area="right")
        self.video_widget.setVisible(False)
        self._video_group.export_labels_cb.toggled.connect(lambda _checked: self._refresh_video_panel_context())
        self._video_group.apply_crop_cb.toggled.connect(self._on_apply_crop_toggled)
        self.viewer.dims.events.current_step.connect(lambda event: self._refresh_video_panel_context())
        self.viewer.layers.selection.events.active.connect(lambda event: self._refresh_video_panel_context())
        QTimer.singleShot(0, self._refresh_video_panel_context)

        # form helper display
        self._keypoint_mapping_button = None
        self._load_superkeypoints_action = None
        help_buttons = self._form_help_buttons()
        self._layout.addLayout(help_buttons)

        grid = QGridLayout()

        self._confirm_overwrite_cb = QCheckBox("Warn on overwrite", parent=self)
        self._confirm_overwrite_cb.setToolTip(
            "When enabled, saving a layer that would overwrite existing keypoints will show a confirmation dialog."
        )
        self._confirm_overwrite_cb.setChecked(settings.get_overwrite_confirmation_enabled())
        self._confirm_overwrite_cb.stateChanged.connect(self._toggle_overwrite_confirmation)

        self._trail_cb = QCheckBox("Show trails", parent=self)
        self._trail_cb.setToolTip("Show the trails for each keypoint over time, in the main video viewer")
        self._trail_cb.setChecked(False)
        self._trail_cb.setEnabled(False)
        self._trail_cb.stateChanged.connect(self._on_show_trails_toggled)
        self._trails_controller = TrailsController(
            self.viewer,
            managed_points_layers_getter=lambda: tuple(self._stores.keys()),
            color_mode_getter=lambda: self.color_mode,
            resolved_cycle_getter=self._resolved_cycle_for_layer,
        )

        self._mpl_docked = False

        self._traj_mpl_canvas = TrajectoryMatplotlibCanvas(
            self.viewer,
            get_color_mode=lambda: self.color_mode,
        )
        self._show_traj_plot_cb = QCheckBox("Show trajectories", parent=self)
        self._show_traj_plot_cb.setToolTip("Toggle to see trajectories in a t-y plot outside of the main video viewer")
        self._show_traj_plot_cb.stateChanged.connect(self._show_traj_canvas)
        self._show_traj_plot_cb.setChecked(False)
        self._show_traj_plot_cb.setEnabled(False)
        self._view_scheme_cb = QCheckBox("Show color scheme", parent=self)

        grid.addWidget(self._confirm_overwrite_cb, 0, 0)
        grid.addWidget(self._show_traj_plot_cb, 1, 0)
        grid.addWidget(self._trail_cb, 2, 0)
        grid.addWidget(self._view_scheme_cb, 3, 0)

        # UX / status panel (folder, progress, point size)
        self._layer_status_panel = LayerStatusPanel(self)
        self._layer_status_panel.point_size_changed.connect(self._on_active_points_size_changed)
        self._layer_status_panel.point_size_commit_requested.connect(self._commit_active_points_size_to_config)
        self._layout.addWidget(self._layer_status_panel)

        self._layout.addLayout(grid)

        # form buttons for selection of annotation mode
        self._radio_box, self._radio_group = self._form_mode_radio_buttons()
        self._radio_box.setEnabled(False)

        # form color scheme display + color mode selector
        self._color_grp, self._color_mode_selector = self._form_color_mode_selector()
        self._color_grp.setEnabled(False)

        # Color scheme display panel
        self._color_scheme_panel = ColorSchemePanel(
            viewer=self.viewer,
            get_color_mode=lambda: self.color_mode,
            get_header_model=self._get_header_model_from_metadata,
            parent=self,
        )
        self._color_scheme_display = self.viewer.window.add_dock_widget(
            self._color_scheme_panel,
            name="Color scheme reference",
            area="left",
        )
        self._view_scheme_cb.setChecked(True)
        self._view_scheme_cb.toggled.connect(self._show_color_scheme)
        self._show_color_scheme()
        # self._color_scheme_panel.display.added.connect(
        #     lambda w: w.part_label.clicked.connect(self._on_color_scheme_label_clicked),
        # )

        self._points_interactions = PointsInteractionObserver(
            self.viewer,
            self._on_points_interaction,
            debounce_ms=0,
            watch_content=False,
        )
        self._points_interactions.install()
        ### UI setup ends here

        # Modes init
        self.color_mode = self._color_mode
        self.label_mode = self._label_mode

        # Substitute default menu action with custom one
        for action in self.viewer.window.file_menu.actions()[::-1]:
            action_name = action.text().lower()
            if "save selected layer" in action_name:
                action.triggered.disconnect()
                action.triggered.connect(
                    lambda: self._save_layers_dialog(
                        selected=True,
                    )
                )
            elif "save all layers" in action_name:
                self.viewer.window.file_menu.removeAction(action)

        # Add action to show the walkthrough again
        launch_tutorial = QAction("&Launch Tutorial", self)
        launch_tutorial.triggered.connect(self.start_tutorial)
        self.viewer.window.view_menu.addAction(launch_tutorial)

        # Add action to view keyboard shortcuts
        display_shortcuts_action = QAction("&Shortcuts", self)
        display_shortcuts_action.triggered.connect(self.display_shortcuts)
        self.viewer.window.help_menu.addAction(display_shortcuts_action)
        # Install global keybinds
        install_global_points_keybindings()

        # Hide some unused viewer buttons
        # NOTE (future) do we truly want to disable these ? Tracking util may need to create new points layers
        # NOTE fix direct access to qt_viewer private members when napari releases 0.7.0 (or later if they delay again)
        # self.viewer.window._qt_viewer.viewerButtons.gridViewButton.hide()
        self.viewer.window._qt_viewer.viewerButtons.rollDimsButton.hide()
        self.viewer.window._qt_viewer.viewerButtons.transposeDimsButton.hide()
        # self.viewer.window._qt_viewer.layerButtons.newPointsButton.setDisabled(True)
        self.viewer.window._qt_viewer.layerButtons.newLabelsButton.setDisabled(True)

        # Disable tutorial on first launch for now. Can be accessed any time from the button.
        # if self.settings.value("first_launch", True) and not os.environ.get(
        #     "NAPARI_DLC_HIDE_TUTORIAL", False
        # ):
        #     QTimer.singleShot(10, self.start_tutorial)
        #     self.settings.setValue("first_launch", False)

        # Slightly delay docking so it is shown underneath the KeypointsControls widget
        # NOTE while a timer may seem hacky, it is a simple, one-line solution that minimizes intrusion
        # There are to my knowledge no other way that is as concise and clean
        # (Of course this will be a problem if we start using it everywhere so do not reuse lightly)
        QTimer.singleShot(10, self.silently_dock_canvas)

        # If layers already exist (user loaded data before opening this widget),
        # adopt them so keypoint controls take ownership immediately.
        QTimer.singleShot(0, self._adopt_existing_layers)

        # Refresh layers stats widget
        QTimer.singleShot(0, self._refresh_layer_status_panel)

    @cached_property
    def settings(self):
        return QSettings()

    # ######################## #
    # Layer setup core methods #
    # ######################## #

    def _setup_image_layer(self, layer: Image, index: int | None = None, *, reorder: bool = True) -> None:
        md = layer.metadata or {}
        paths = md.get("paths")
        if paths is None and io.is_video(layer.name):
            self.video_widget.setVisible(True)

        self._update_image_meta_from_layer(layer)

        if not self._project_path:
            self._cache_project_path_from_image_layer(layer)
            if self._project_path is not None:
                try:
                    layer.metadata = dict(layer.metadata or {})
                    layer.metadata.setdefault("project", self._project_path)
                except Exception:
                    logger.debug(
                        "Failed to set project path metadata on image layer %r",
                        getattr(layer, "name", layer),
                        exc_info=True,
                    )

        self._sync_points_layers_from_image_meta()
        self._refresh_video_panel_context()

        if reorder and index is not None:
            QTimer.singleShot(10, partial(self._move_image_layer_to_bottom, index))

    def _maybe_merge_config_points_layer(self, layer: Points) -> bool:
        if not layer.metadata.get("project", "") or not self._stores:
            return False

        new_metadata = layer.metadata.copy()

        keypoints_menu = self._menus[0].menus["label"]
        current_keypoint_set = {keypoints_menu.itemText(i) for i in range(keypoints_menu.count())}
        hdr = self._get_header_model_from_metadata(new_metadata)
        if hdr is None:
            return False
        new_keypoint_set = set(hdr.bodyparts)
        diff = new_keypoint_set.difference(current_keypoint_set)

        if diff:
            answer = QMessageBox.question(self, "", "Do you want to display the new keypoints only?")
            if answer == QMessageBox.Yes:
                self.viewer.layers[-2].shown = False

            self.viewer.status = f"New keypoint{'s' if len(diff) > 1 else ''} {', '.join(diff)} found."
            for _layer, store in self._stores.items():
                pts = read_points_meta(_layer, migrate_legacy=True, drop_controls=True, drop_header=False)
                if not hasattr(pts, "errors"):
                    updated = pts.model_copy(update={"header": hdr})
                    write_points_meta(_layer, updated, merge_policy=MergePolicy.MERGE, fields={"header"})
                store.layer = _layer

            for menu in self._menus:
                menu._map_individuals_to_bodyparts()
                menu._update_items()

        QTimer.singleShot(10, self.viewer.layers.pop)

        # apply the new color cycles + recolor safely
        for _layer, store in self._stores.items():
            _layer.metadata["config_colormap"] = new_metadata.get(
                "config_colormap", _layer.metadata.get("config_colormap")
            )
            _layer.metadata["face_color_cycles"] = new_metadata["face_color_cycles"]
            _layer.metadata["colormap_name"] = new_metadata.get("colormap_name", _layer.metadata.get("colormap_name"))
            mark_layer_presentation_changed(_layer)
            self._apply_points_coloring_from_metadata(_layer)
            store.layer = _layer

        self._update_color_scheme()
        return True

    def _get_header_model_from_metadata(self, md: dict) -> DLCHeaderModel | None:
        """Return DLCHeaderModel regardless of whether md['header'] is a model, dict payload, or MultiIndex."""
        if not isinstance(md, dict):
            return None
        hdr = md.get("header", None)
        if hdr is None:
            return None

        if isinstance(hdr, DLCHeaderModel):
            logger.debug("Header is already a DLCHeaderModel instance.")
            return hdr

        if isinstance(hdr, dict):
            try:
                return DLCHeaderModel.model_validate(hdr)
            except Exception:
                return None

        # fallback: allow MultiIndex / list-of-tuples / Index inputs
        try:
            return DLCHeaderModel(columns=hdr)
        except Exception:
            return None

    @staticmethod
    def get_layer_controls(layer: Points) -> KeypointControls | None:
        return getattr(layer, "_dlc_controls", None)

    @staticmethod
    def get_layer_store(layer: Points) -> keypoints.KeypointStore | None:
        return getattr(layer, "_dlc_store", None)

    def _wire_points_layer(self, layer: Points) -> keypoints.KeypointStore | None:
        if not self._validate_header(layer):
            return None
        existing = getattr(layer, "_dlc_store", None)
        if existing is not None:
            self._stores[layer] = existing
            layer._dlc_controls = self
            return existing

        # ensure presence of IO metadata for saving & routing
        mig = migrate_points_layer_metadata(layer)
        if hasattr(mig, "errors"):
            logger.warning(
                "Points metadata validation failed during wiring for layer=%r: %s",
                getattr(layer, "name", layer),
                mig,
            )

        store = keypoints.KeypointStore(self.viewer, layer)
        self._stores[layer] = store
        layer._dlc_store = store
        layer._dlc_controls = self

        # default root/paths from current image meta if missing
        if not layer.metadata.get("root") and self._image_meta.root:
            layer.metadata["root"] = self._image_meta.root
        if not layer.metadata.get("paths") and self._image_meta.paths:
            layer.metadata["paths"] = self._image_meta.paths

        # save history
        if root := layer.metadata.get("root"):
            update_save_history(root)

        store._get_label_mode = lambda: self._label_mode
        layer.text.visible = False

        paste_func = make_paste_data(self, store=store)
        install_paste_patch(layer, paste_func=paste_func)

        add_impl = MethodType(keypoints._add, store)  # bind store to add implementation
        install_add_wrapper(layer, add_impl=add_impl, schedule_recolor=self._schedule_recolor)

        # store events / navigation
        layer.events.add(query_next_frame=Event)
        layer.events.query_next_frame.connect(store._advance_step)

        # Install keybinds
        install_points_layer_keybindings(layer, self, store)

        if len(self._stores) == 1 and self._is_multianimal(layer):
            # set internal mode without triggering recolor storms
            self._color_mode = keypoints.ColorMode.INDIVIDUAL
            # update button state so UI matches (optional but good)
            for btn in self._color_mode_selector.buttons():
                if btn.text().lower() == str(self._color_mode).lower():
                    btn.setChecked(True)
                    break

        # apply cycles (works even if empty; see method)
        self._apply_points_coloring_from_metadata(layer)
        self._maybe_initialize_layer_point_size_from_config(layer)
        self._connect_layer_status_events(layer)
        # refresh trails if enabled (e.g. when merging a config points layer with trails metadata)
        self._trails_controller.on_points_layer_added_or_rewired(checkbox_checked=self._trail_cb.isChecked())

        # menus
        self._form_dropdown_menus(store)

        # project path
        proj = layer.metadata.get("project")
        if proj:
            self._project_path = proj

        # enable GUI groups
        self._radio_box.setEnabled(True)
        self._color_grp.setEnabled(True)
        self._trail_cb.setEnabled(True)
        self._show_traj_plot_cb.setEnabled(True)

        return store

    def _setup_points_layer(self, layer: Points, *, allow_merge: bool = True) -> None:
        if not self._validate_header(layer):
            return

        if allow_merge and self._maybe_merge_config_points_layer(layer):
            return

        if layer.metadata.get("tables", ""):
            self._keypoint_mapping_button.show()

        store = self._wire_points_layer(layer)
        if store is None:
            return

        selector = apply_points_layer_ui_tweaks(self.viewer, layer, dropdown_cls=DropdownMenu, plt_module=plt)
        if selector is not None:
            try:
                selector.currentTextChanged.connect(self._update_colormap)
            except Exception:
                pass

        self._update_color_scheme()

    def _adopt_existing_layers(self) -> None:
        """
        When the widget is opened after layers already exist, we need to
        run the same initialization as if they had been inserted.
        """
        # Iterate over a snapshot, because on_insert may modify layer order
        layers_snapshot = list(self.viewer.layers)

        for idx, layer in enumerate(layers_snapshot):
            self._adopt_layer(layer, idx)

        # After adoption, refresh UI state
        try:
            active = self.viewer.layers.selection.active
            if active is not None:
                # Force the GUI to update visibility of menus, etc.
                self.on_active_layer_change(Event(type_name="active", value=active))
        except Exception:
            pass

        # Important: refresh the trajectory plot from the final adopted state.
        # This fixes the case where layers were loaded before the plugin opened
        # (e.g. drag-and-drop DLC data triggering the reader automatically).
        self._refresh_trajectory_plot_from_layers()

    def _adopt_layer(self, layer, index: int) -> None:
        """
        Run the relevant portion of on_insert() for an already-existing layer.
        This avoids duplicating your logic and prevents reliance on napari's Event object.
        """
        if isinstance(layer, Image):
            self._setup_image_layer(layer, index, reorder=True)
        elif isinstance(layer, Points):
            if layer not in self._stores:
                self._setup_points_layer(layer, allow_merge=False)  # typically don’t merge during adopt
        if not isinstance(layer, Image):
            self._remap_frame_indices(layer)

    def _validate_header(self, layer) -> bool:
        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if isinstance(res, ValidationError) or res.header is None:
            self.viewer.status = (
                "This Points layer does not look like a DLC keypoints layer. Missing a valid DLC header."
            )
            return False
        return True

    def _schedule_recolor(self, layer: Points) -> None:
        if not hasattr(self, "_recolor_pending"):
            self._recolor_pending = set()

        if layer in self._recolor_pending:
            return

        self._recolor_pending.add(layer)

        def _do():
            try:
                self._apply_points_coloring_from_metadata(layer)
            finally:
                self._recolor_pending.discard(layer)

        QTimer.singleShot(0, _do)

    def _ensure_traj_canvas_docked(self) -> None:
        """
        Dock the Matplotlib canvas as a napari dock widget, exactly once,
        and only if the Qt window exists. Safe no-op in headless/proxy teardown.
        """
        if self._mpl_docked:
            return

        window = getattr(self.viewer, "window", None)
        if window is None:
            return

        if getattr(window, "_qt_window", None) is None:
            return

        try:
            window.add_dock_widget(self._traj_mpl_canvas, name="Trajectory plot", area="right", tabify=False)
            self._traj_mpl_canvas.canvas.draw_idle()
            self._traj_mpl_canvas.hide()
            self._mpl_docked = True
        except Exception as e:
            logging.debug("Skipping docking canvas (not ready / teardown): %r", e)
            return

    def _safe_get_traj_canvas(self):
        canvas = getattr(self, "_traj_mpl_canvas", None)
        if canvas is None:
            return None

        try:
            # Any Qt call is enough to verify the underlying C++ object still exists
            canvas.isVisible()
        except RuntimeError:
            # Underlying Qt object was already deleted
            self._traj_mpl_canvas = None
            return None

        return canvas

    def silently_dock_canvas(self) -> None:
        """Dock the Matplotlib canvas without showing it."""
        self._ensure_traj_canvas_docked()
        if self._mpl_docked:
            self._traj_mpl_canvas.hide()

    def _show_traj_canvas(self, state):
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            self._ensure_traj_canvas_docked()
            if self._mpl_docked:
                self._traj_mpl_canvas._apply_napari_theme()
                self._traj_mpl_canvas.update_plot_range(
                    Event(type_name="", value=[self.viewer.dims.current_step[0]]),
                    force=True,
                )
                self._traj_mpl_canvas.sync_visible_lines_to_points_selection()
                self._traj_mpl_canvas.show()
        else:
            if self._mpl_docked:
                self._traj_mpl_canvas.hide()

    def _on_points_interaction(self, event: PointsInteractionEvent) -> None:
        """
        Keep the trajectory plot in sync with the active points-layer selection.

        This is intentionally selection-driven:
        - no selected points -> all trajectories
        - selected points    -> only selected labels' trajectories
        """
        traj_canvas = self._safe_get_traj_canvas()
        if traj_canvas is None or not traj_canvas.isVisible():
            return
        if {"selection", "active_layer", "layers"} & set(event.reasons):
            traj_canvas.sync_visible_lines_to_points_selection()

    def _refresh_trajectory_plot_from_layers(self) -> None:
        """
        Refresh trajectory plot from the current viewer state.

        Deferred through QTimer so it runs after layer adoption/remap settles.
        """
        traj_canvas = self._safe_get_traj_canvas()
        if traj_canvas is not None:
            QTimer.singleShot(0, traj_canvas.refresh_from_viewer_layers)

    def load_superkeypoints_diagram(self):
        points_layer = get_first_points_layer(self.viewer)
        if points_layer is None:
            return

        tables = deepcopy(points_layer.metadata.get("tables", {}))
        if not tables:
            return

        super_animal, table = tables.popitem()
        image = io.load_superkeypoints_diagram(super_animal)
        self.viewer.add_image(image, name=f"{super_animal} keypoint diagram", metadata={"super_animal": super_animal})
        superkpts_dict = io.load_superkeypoints(super_animal)
        xy = []
        labels = []
        for kpt_ref, kpt_super in table.items():
            xy.append([0.0, *superkpts_dict[kpt_super]])
            labels.append(kpt_ref)
        points_layer.data = np.array(xy)
        properties = deepcopy(points_layer.properties)
        properties["label"] = np.array(labels)
        points_layer.properties = properties
        self._apply_points_coloring_from_metadata(points_layer)
        self._keypoint_mapping_button.setText("Map keypoints")
        try:
            self._keypoint_mapping_button.clicked.disconnect(self._load_superkeypoints_action)
        except TypeError:
            pass
        self._keypoint_mapping_button.clicked.connect(lambda: self._map_keypoints(super_animal))

    def _map_keypoints(self, super_animal: str):
        # NOTE: This implementation makes several assumptions that may need review:
        # - Assumes points_layer.metadata contains "project" and "header" keys with the expected structure.
        # - Assumes _load_superkeypoints and _load_config succeed
        #   and return well-formed data; I/O errors are not handled.
        # - Silently ignores keypoints that have no nearest neighbor in the superkeypoint set (no user feedback).
        points_layer = get_points_layer_with_tables(self.viewer)
        if points_layer is None or not np.any(points_layer.data):
            return

        xy = points_layer.data[:, 1:3]
        superkpts_dict = io.load_superkeypoints(super_animal)
        xy_ref = np.asarray(list(superkpts_dict.values()), dtype=float)
        neighbors = keypoints._find_nearest_neighbors(xy, xy_ref)
        found = neighbors != -1
        if not np.any(found):
            return

        project_path = points_layer.metadata["project"]
        config_path = str(Path(project_path) / "config.yaml")
        cfg = io.load_config(config_path)
        conversion_tables = cfg.get("SuperAnimalConversionTables", {})
        hdr = self._get_header_model_from_metadata(points_layer.metadata or {})
        if hdr is None:
            return
        bdprts_map = map(str, hdr.bodyparts)
        conversion_tables[super_animal] = dict(
            zip(
                bdprts_map,  # Needed to fix an ugly yaml RepresenterError
                map(str, list(np.array(list(superkpts_dict))[neighbors[found]])),
                strict=False,
            )
        )
        cfg["SuperAnimalConversionTables"] = conversion_tables
        io.write_config(config_path, cfg)
        self.viewer.status = "Mapping to superkeypoint set successfully saved"

    def start_tutorial(self):
        Tutorial(self.viewer.window._qt_window.current()).show()

    def display_shortcuts(self):
        Shortcuts(self.viewer.window._qt_window.current(), viewer=self.viewer).show()

    def _move_image_layer_to_bottom(self, index):
        if (ind := index) != 0:
            self.viewer.layers.move_selected(ind, 0)
            self.viewer.layers.select_next()  # Auto-select the Points layer

    # ------------------------------------------------------------------
    # Metadata helpers (authoritative models + napari-friendly dict sync)
    # ------------------------------------------------------------------
    @staticmethod
    def _layer_source_path(layer) -> str | None:
        """Best-effort access to napari layer source path (may not exist)."""
        try:
            src = getattr(layer, "source", None)
            p = getattr(src, "path", None) if src is not None else None
            return str(p) if p else None
        except Exception:
            return None

    def _update_image_meta_from_layer(self, layer: Image) -> None:
        """
        Update authoritative self._images_meta using an Image layer.
        Also keep a dict-like subset synced for other layers (non-breaking).
        """
        md = layer.metadata or {}

        paths = md.get("paths")
        shape = None
        try:
            shape = layer.level_shapes[0]
        except Exception:
            shape = None

        root = infer_image_root(
            explicit_root=md.get("root"),
            paths=paths,
            source_path=self._layer_source_path(layer),
        )

        incoming = ImageMetadata(
            paths=list(paths) if paths else None,
            root=str(root) if root else None,
            shape=tuple(shape) if shape is not None else None,
            name=getattr(layer, "name", None),
        )

        # Merge without clobbering already-known values
        # (same behavior as old "only set if non-empty")
        base = self._image_meta
        merged = base.model_copy(deep=True)
        for field, value in incoming.model_dump().items():
            if getattr(merged, field) in (None, "", []) and value not in (None, "", []):
                setattr(merged, field, value)

        self._image_meta = merged

    def _sync_points_layers_from_image_meta(self) -> None:
        """
        Ensure all Points layers have core fields required for saving.

        Adapter-based flow:
        - read validated points meta (visible failures)
        - apply sync logic against authoritative self._image_meta
        - write back validated dict via gateway
        """
        if self._image_meta is None:
            return

        for ly in list(self.viewer.layers):
            if not isinstance(ly, Points):
                continue

            if ly.metadata is None:
                ly.metadata = {}

            # 1) Read + migrate legacy (io from source_h5, header coercion, etc.)
            res = read_points_meta(ly, migrate_legacy=True, drop_controls=False, drop_header=False)
            if hasattr(res, "errors"):  # ValidationError duck-typing
                logger.warning(
                    "Points metadata validation failed during sync for layer=%r: %s",
                    getattr(ly, "name", ly),
                    res,
                )
                continue

            pts_model: PointsMetadata = res

            # 2) Sync missing fields from image meta (pure model transform)
            synced = sync_points_from_image(self._image_meta, pts_model)

            # 3) Write back through gateway (fill missing only; never clobber)
            out = write_points_meta(
                ly,
                synced,
                merge_policy=MergePolicy.MERGE_MISSING,
                migrate_legacy=True,
                validate=True,
            )
            if hasattr(out, "errors"):
                logger.warning(
                    "Failed to write synced points metadata for layer=%r: %s",
                    getattr(ly, "name", ly),
                    out,
                )

    def _resolve_config_path_for_layer(self, layer: Points | None) -> Path | None:
        if layer is None:
            return None

        image_layer = find_relevant_image_layer(self.viewer)

        return resolve_config_path_from_layer(
            layer,
            fallback_project=self._project_path,
            fallback_root=self._image_meta.root,
            image_layer=image_layer,
            prefer_project_root=True,
            max_levels=5,
        )

    def _maybe_prepare_project_path_override_metadata(self, layer: Points) -> tuple[dict | None, bool]:
        """
        Optionally prepare save-time metadata by associating a project-less labeled
        folder with an explicit DLC project chosen via config.yaml.

        Returns
        -------
        tuple[dict | None, bool]
            (overridden_metadata, abort_save)

            - (None, False): feature not applicable; continue normal save
            - (metadata, False): apply metadata override and continue
            - (None, True): user cancelled or operation was refused; abort save
        """
        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if isinstance(res, ValidationError):
            return None, False

        pts_meta: PointsMetadata = res
        paths = pts_meta.paths or []
        if not paths:
            return None, False

        if not is_projectless_folder_association_candidate(pts_meta):
            return None, False

        source_root = pts_meta.root
        if not source_root:
            return None, False

        try:
            source_root_path = Path(source_root).expanduser().resolve(strict=False)
        except Exception:
            source_root_path = Path(source_root)

        # NOTE: @C-Achard 2026-03-27 Currently does not let user choose
        # a different dataset name than the source folder,
        # to keep a lightweight workflow.
        # This could be allowed in the future if there is demand.
        dataset_name = source_root_path.name
        if not dataset_name:
            return None, False

        initial_dir = self._project_path or pts_meta.project or str(source_root_path)
        dialog_result = ui_dialogs.prompt_for_project_config_for_save(self, initial_dir=initial_dir)

        if dialog_result.action is ui_dialogs.ProjectConfigPromptAction.CANCEL:
            logger.debug("User cancelled project association prompt.")
            return None, True  # abort save

        if dialog_result.action is ui_dialogs.ProjectConfigPromptAction.SKIP:
            logger.debug("User chose to continue without project association.")
            return None, False  # continue normal save path

        if dialog_result.action is not ui_dialogs.ProjectConfigPromptAction.ASSOCIATE:
            logger.warning("Unexpected project association dialog result: %r", dialog_result)
            return None, True  # fail safe: abort save

        config_path = dialog_result.config_path
        if not config_path:
            logger.warning("Project association result was ASSOCIATE but config_path was empty.")
            return None, True  # fail safe: abort save

        project_root = resolve_project_root_from_config(config_path)
        if project_root is None:
            QMessageBox.warning(
                self,
                "Invalid project configuration",
                "The selected file is not a valid DeepLabCut config.yaml or project root. "
                "The save operation has been cancelled.",
            )
            return None, True

        target_folder = target_dataset_folder_for_config(config_path, dataset_name=dataset_name)
        if dataset_folder_has_files(target_folder):
            ui_dialogs.warn_existing_dataset_folder_conflict(self, target_folder=target_folder)
            return None, True  # refuse the save

        rewritten_paths, unresolved = coerce_paths_to_dlc_row_keys(
            paths,
            source_root=source_root_path,
            dataset_name=dataset_name,
        )

        if not ui_dialogs.maybe_confirm_dataset_path_rewrite(
            self,
            project_root=project_root,
            dataset_name=dataset_name,
            n_paths=len(paths),
            n_unresolved=len(unresolved),
        ):
            return None, True  # user declined

        overridden = apply_project_paths_override_to_points_meta(
            pts_meta,
            project_root=project_root,
            rewritten_paths=rewritten_paths,
        )

        return overridden.model_dump(mode="python", exclude_none=True), False

    def _show_color_scheme(self):
        show = self._view_scheme_cb.isChecked()
        self._color_scheme_display.setVisible(show)

    def _current_dlc_points_layer(self) -> Points | None:
        active = self.viewer.layers.selection.active
        if not isinstance(active, Points):
            return None

        try:
            res = read_points_meta(active, migrate_legacy=True, drop_controls=True, drop_header=False)
        except Exception:
            return None

        if isinstance(res, ValidationError):
            return None

        if getattr(res, "header", None) is None:
            return None

        return active

    def _refresh_layer_status_panel(self) -> None:
        active_layer = self.viewer.layers.selection.active
        active_dlc_points = self._current_dlc_points_layer()
        active_image = find_relevant_image_layer(self.viewer)

        folder_name = infer_folder_display_name(
            active_image if active_image is not None else active_layer,
            fallback_root=self._image_meta.root,
        )
        self._layer_status_panel.set_folder_name(folder_name)

        # No active layer or not a Points layer at all
        if active_layer is None or not isinstance(active_layer, Points):
            self._layer_status_panel.set_no_active_points_layer()
            return

        # Active layer is a Points layer, but not a valid DLC points layer
        if active_dlc_points is None:
            self._layer_status_panel.set_invalid_points_layer()
            return

        self._layer_status_panel.set_point_size_enabled(True)
        self._layer_status_panel.set_point_size(get_uniform_point_size(active_dlc_points))

        progress = compute_label_progress(active_dlc_points, fallback_paths=self._image_meta.paths)
        self._layer_status_panel.set_progress_summary(progress=progress)

    def _on_active_points_size_changed(self, size: int) -> None:
        layer = self._current_dlc_points_layer()
        if layer is None:
            return

        set_uniform_point_size(layer, size)
        mark_layer_presentation_changed(layer)

    def _commit_active_points_size_to_config(self, size: int) -> None:
        layer = self._current_dlc_points_layer()
        if layer is None:
            return

        config_path = self._resolve_config_path_for_layer(layer)
        if config_path is None:
            logger.debug(
                "No config.yaml could be resolved at commit time for active layer %r",
                getattr(layer, "name", layer),
            )
            return

        try:
            changed = save_point_size_to_config(config_path, int(size))
            if changed:
                self.viewer.status = f"Updated config dotsize to {int(size)}"
        except Exception:
            logger.debug("Failed to sync point size to config", exc_info=True)

    def _maybe_initialize_layer_point_size_from_config(self, layer: Points) -> None:
        config_path = self._resolve_config_path_for_layer(layer)
        if config_path is None:
            return

        config_size = load_point_size_from_config(config_path)
        if config_size is None:
            return

        current_size = get_uniform_point_size(layer)

        # Conservative initialization
        if current_size <= 8:
            try:
                set_uniform_point_size(layer, config_size)
                mark_layer_presentation_changed(layer)
            except Exception:
                logger.debug("Could not initialize layer point size from config", exc_info=True)

    def _connect_layer_status_events(self, layer: Points) -> None:
        """
        Keep the UX panel live without adding heavy watchers.
        """
        try:
            layer.events.data.connect(lambda event=None, _layer=layer: self._refresh_layer_status_panel())
        except Exception:
            pass

        try:
            layer.events.size.connect(lambda event=None, _layer=layer: self._refresh_layer_status_panel())
        except Exception:
            pass

        try:
            layer.events.properties.connect(lambda event=None, _layer=layer: self._refresh_layer_status_panel())
        except Exception:
            pass

    def _form_help_buttons(self):
        layout = QVBoxLayout()
        help_buttons_layout = QHBoxLayout()
        self.show_shortcuts_btn = QPushButton("View shortcuts")
        self.show_shortcuts_btn.clicked.connect(self.display_shortcuts)
        help_buttons_layout.addWidget(self.show_shortcuts_btn)
        self.tutorial_btn = QPushButton("Start tutorial")
        self.tutorial_btn.clicked.connect(self.start_tutorial)
        help_buttons_layout.addWidget(self.tutorial_btn)
        layout.addLayout(help_buttons_layout)
        self._keypoint_mapping_button = QPushButton("Load superkeypoints diagram")
        self._load_superkeypoints_action = self._keypoint_mapping_button.clicked.connect(
            self.load_superkeypoints_diagram
        )
        self._keypoint_mapping_button.hide()
        layout.addWidget(self._keypoint_mapping_button)
        return layout

    def _refresh_video_panel_context(self) -> None:
        update_video_panel_context(self.viewer, self._video_group)

    def _cache_project_path_from_image_layer(self, layer: Image) -> None:
        """Best-effort cache of project path from an image/video layer."""
        project_path = resolve_project_path_from_image_layer(layer)
        if project_path is None:
            return

        self._project_path = project_path
        try:
            layer.metadata = dict(layer.metadata or {})
            layer.metadata.setdefault("project", self._project_path)
        except Exception:
            logger.debug(
                "Failed to set project path metadata on image layer %r",
                getattr(layer, "name", layer),
                exc_info=True,
            )

        self._refresh_video_panel_context()

    def _extract_single_frame(self, *args):
        ok, msg = run_extract_current_frame(
            self.viewer,
            self._video_group,
            validate_points_layer=self._validate_header,
        )
        self.viewer.status = msg
        self._refresh_video_panel_context()

    def _on_apply_crop_toggled(self, checked) -> None:
        handle_apply_crop_toggled(self.viewer, self._video_group, bool(checked))
        self._refresh_video_panel_context()

    def _store_crop_coordinates(self, *args):
        ok, msg, project_path = run_store_crop_coordinates(
            self.viewer,
            self._video_group,
            explicit_project_path=self._project_path,
            fallback_video_name=self._image_meta.name,
        )
        if project_path is not None:
            self._project_path = project_path
        self.viewer.status = msg
        self._refresh_video_panel_context()

    def _form_dropdown_menus(self, store):
        menu = KeypointsDropdownMenu(store)
        self.viewer.dims.events.current_step.connect(
            menu.smart_reset,
            position="last",
        )
        menu.smart_reset(event=None)
        self._menus.append(menu)
        self._layer_to_menu[store.layer] = len(self._menus) - 1
        layout = QVBoxLayout()
        layout.addWidget(menu)
        self._layout.addLayout(layout)

    def _form_mode_radio_buttons(self):
        group_box = QGroupBox("Labeling mode")
        layout = QHBoxLayout()
        group = QButtonGroup(self)
        for i, mode in enumerate(keypoints.LabelMode.__members__, start=1):
            btn = QRadioButton(mode.capitalize())
            btn.setToolTip(keypoints.TOOLTIPS[mode])
            group.addButton(btn, i)
            layout.addWidget(btn)
        group.button(1).setChecked(True)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)

        def _func():
            self.label_mode = group.checkedButton().text().lower()

        group.buttonClicked.connect(_func)
        return group_box, group

    def _form_color_mode_selector(self):
        group_box = QGroupBox("Keypoint coloring mode")
        layout = QHBoxLayout()
        group = QButtonGroup(self)
        for i, mode in enumerate(keypoints.ColorMode.__members__, start=1):
            btn = QRadioButton(mode.capitalize())
            group.addButton(btn, i)
            layout.addWidget(btn)
        group.button(1).setChecked(True)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)

        def _func():
            self.color_mode = group.checkedButton().text().lower()

        group.buttonClicked.connect(_func)
        return group_box, group

    def _update_color_scheme(self):
        if hasattr(self, "_color_scheme_panel"):
            self._color_scheme_panel.schedule_update()

    def _apply_points_coloring_from_metadata(self, layer: Points) -> None:
        """Apply categorical coloring using centralized resolver policy."""
        resolver = self._color_scheme_panel._resolver
        cycles = resolver.get_face_color_cycles(layer)
        if not cycles:
            try:
                layer.face_color_mode = "direct"
            except Exception:
                pass
            return

        prop = resolver.get_active_color_property(layer)
        if prop not in cycles or not cycles[prop]:
            return

        props = getattr(layer, "properties", {}) or {}
        values = props.get(prop, None)

        # id mode on single-animal / blank ids -> fallback to label
        if prop == "id":
            try:
                vals = np.asarray(values, dtype=object).ravel() if values is not None else np.array([], dtype=object)
                if len(vals) == 0 or all(v in ("", None) or misc._is_nan_value(v) for v in vals):
                    prop = "label"
                    values = props.get("label", None)
            except Exception:
                prop = "label"
                values = props.get("label", None)

        if values is None or len(values) == 0 or misc._array_has_nan(values):
            try:
                layer.face_color_mode = "direct"
            except Exception:
                pass
            return

        try:
            layer.face_color = prop
            layer.face_color_cycle = cycles[prop]
            layer.face_color_mode = "cycle"
            layer.events.face_color()
        except Exception:
            try:
                layer.face_color_mode = "direct"
            except Exception:
                pass

    def _remap_frame_indices(self, layer):
        """
        Best-effort remap of time/frame indices in non-Image layers to match current Image order.

        Safety principles
        -----------------
        - Never delete or silently corrupt user data.
        - Only write back to layer.data after a remap has been accepted as safe.
        - Always sync non-path image metadata when possible.
        - Do NOT replace metadata["paths"] unless remap is accepted as safe.
        - Specifically reject ambiguous basename-only remaps (depth=1 with duplicate /
        non-bijective warnings), which commonly happen when data are moved out of the
        standard DLC labeled-data layout.
        """
        try:
            new_paths = self._image_meta.paths
            if not new_paths:
                return

            if layer.metadata is None:
                layer.metadata = {}

            md = layer.metadata
            old_paths = md.get("paths") or []

            # Always sync safe non-path metadata from image meta.
            # Do NOT sync "paths" yet; that is only safe after we decide remap is acceptable.
            try:
                safe_image_meta = self._image_meta.model_dump(exclude_none=True)
                safe_image_meta.pop("paths", None)
                layer.metadata.update(safe_image_meta)
            except Exception:
                logger.debug(
                    "Failed to sync non-path image metadata for layer=%r",
                    getattr(layer, "name", str(layer)),
                    exc_info=True,
                )

            if not old_paths:
                logger.debug(
                    "Skipping remap for layer=%r: no existing layer metadata paths.",
                    getattr(layer, "name", str(layer)),
                )
                return

            # Determine time column (napari-specific choice)
            time_col = 1 if isinstance(layer, Tracks) else 0

            if logger.isEnabledFor(logging.DEBUG):
                arr_before = np.asarray(layer.data)
                logger.debug(
                    "Remap start layer=%r old_paths_len=%s new_paths_len=%s data_shape=%s frame_min=%s frame_max=%s",
                    getattr(layer, "name", str(layer)),
                    len(old_paths),
                    len(new_paths or []),
                    getattr(arr_before, "shape", None),
                    int(np.nanmin(arr_before[:, time_col])) if arr_before.size else None,
                    int(np.nanmax(arr_before[:, time_col])) if arr_before.size else None,
                )

            res = remap_layer_data_by_paths(
                data=layer.data,
                old_paths=old_paths,
                new_paths=new_paths,
                time_col=time_col,
                policy=PathMatchPolicy.ORDERED_DEPTHS,
            )

            logger.debug(
                "Remap result layer=%r changed=%s mapped_count=%s depth=%s message=%s warnings=%s",
                getattr(layer, "name", str(layer)),
                res.changed,
                res.mapped_count,
                res.depth_used,
                res.message,
                res.warnings,
            )

            if res.applied and res.data is not None:
                layer.data = res.data

            if res.accept_paths_update:
                layer.metadata["paths"] = list(new_paths)
                if isinstance(layer, Points):
                    mark_layer_presentation_changed(layer)

            # Final debug logging
            if res.depth_used is None:
                logger.debug("Remap skipped for %s: %s", getattr(layer, "name", str(layer)), res.message)
            else:
                logger.debug(
                    "Remap %s for %s (depth=%s, mapped=%s): %s",
                    "applied" if res.changed else "accepted-noop",
                    getattr(layer, "name", str(layer)),
                    res.depth_used,
                    res.mapped_count,
                    res.message,
                )

        except Exception:
            logger.exception("Failed to remap frame indices for layer %s", getattr(layer, "name", str(layer)))
            return

    def on_insert(self, event):
        layer = event.source[-1]
        if isinstance(layer, Image):
            self._setup_image_layer(layer, event.index, reorder=True)
        elif isinstance(layer, Points):
            self._setup_points_layer(layer, allow_merge=True)

        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self._remap_frame_indices(layer_)
        self._refresh_video_panel_context()
        self._refresh_layer_status_panel()

    def on_remove(self, event):
        layer = event.value
        n_points_layer = sum(isinstance(l, Points) for l in self.viewer.layers)

        if isinstance(layer, Points):
            self._stores.pop(layer, None)

            # Refresh color scheme panel regardless; it will clear itself if no valid target remains.
            self._update_color_scheme()
            self._trails_controller.on_points_layer_removed(layer)

            if n_points_layer == 0:
                while self._menus:
                    menu = self._menus.pop()
                    self._layout.removeWidget(menu)
                    menu.deleteLater()
                    menu.destroy()

                self._layer_to_menu = {}
                self._trail_cb.setEnabled(False)
                self._show_traj_plot_cb.setEnabled(False)
                self.last_saved_label.hide()

        elif isinstance(layer, Image):
            self._image_meta = ImageMetadata()
            paths = layer.metadata.get("paths")
            if paths is None:
                self.video_widget.setVisible(False)

        elif isinstance(layer, Tracks):
            was_trails = self._trails_controller.on_tracks_layer_removed(layer)
            if was_trails:
                with QSignalBlocker(self._trail_cb):
                    self._trail_cb.setChecked(False)

        self._refresh_video_panel_context()
        self._refresh_layer_status_panel()

    def _on_show_trails_toggled(self, state):
        self._trails_controller.toggle(Qt.CheckState(state) == Qt.CheckState.Checked)

    def _ensure_promotion_save_target(self, layer: Points) -> bool:
        """Ensure a prediction/machine source layer has a GT save_target set.

        Returns True if save_target is set (or already existed), False if user cancels.
        """
        if not is_machine_layer(layer):
            return True

        mig = migrate_points_layer_metadata(layer)
        if hasattr(mig, "errors"):
            logger.warning(
                "Failed to migrate points layer metadata for layer=%r: %s",
                getattr(layer, "name", layer),
                mig,
            )

        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if isinstance(res, ValidationError):
            logger.warning(
                "Points metadata validation failed for layer=%r during save target check: %s",
                getattr(layer, "name", layer),
                res,
            )
            QMessageBox.warning(self, "Cannot save", "Layer metadata is invalid; see logs for details.")
            return False

        pts: PointsMetadata = res

        if not requires_gt_promotion(pts):
            return True

        anchor = safe_folder_anchor_from_points_layer(layer)
        if not anchor:
            QMessageBox.warning(self, "Cannot save", "Could not determine a folder anchor for saving.")
            return False

        scorer = None

        # 1) Auto-discovered config.yaml always wins
        cfg_path = None
        try:
            cfg_path = find_nearest_config(anchor)
        except Exception:
            logger.debug("Automatic config discovery failed for anchor=%r", anchor, exc_info=True)

        if cfg_path:
            try:
                scorer = ui_dialogs.load_scorer_from_config(cfg_path)
            except Exception:
                logger.exception("Failed to load auto-discovered config.yaml: %s", cfg_path)
                ui_dialogs.warn_invalid_config_for_scorer(
                    self,
                    config_path=cfg_path,
                    reason="unreadable",
                    auto_found=True,
                )
                return False

            if not scorer:
                ui_dialogs.warn_invalid_config_for_scorer(
                    self,
                    config_path=cfg_path,
                    reason="missing_scorer",
                    auto_found=True,
                )
                return False

        else:
            # 2) No config found automatically -> let the user choose one
            dialog_result = ui_dialogs.prompt_for_project_config_for_save(
                self,
                initial_dir=self._project_path or anchor,
                window_title="Locate DLC config for scorer resolution",
                message=(
                    "No DeepLabCut config.yaml could be found automatically for this machine-labeled layer.\n\n"
                    "If this layer belongs to a DLC project, choose its config.yaml so the save uses the "
                    "project scorer and standard naming.\n\n"
                    "If no config.yaml exists, you can continue without one."
                ),
                choose_button_text="Choose config.yaml",
                skip_button_text="Continue without config",
                resolve_scorer=True,
            )

            if dialog_result.action is ui_dialogs.ProjectConfigPromptAction.CANCEL:
                return False

            if dialog_result.action is ui_dialogs.ProjectConfigPromptAction.ASSOCIATE:
                scorer = dialog_result.scorer

            else:
                # 3) Only if no config is available at all may sidecar be consulted
                scorer = get_default_scorer(anchor)

                # 4) Final fallback: prompt manually
                if not scorer:
                    suggested = suggest_human_placeholder(anchor)
                    while True:
                        s = _prompt_for_scorer(self, anchor=anchor, suggested=suggested)
                        if s is None:
                            return False
                        if s.startswith("human_"):
                            choice = QMessageBox.question(
                                self,
                                "Generic scorer name",
                                "You entered a generic scorer name starting with 'human_'.\n\n"
                                "We strongly recommend using a real name or stable identifier.\n"
                                "Do you want to keep this generic scorer anyway?",
                                QMessageBox.Yes | QMessageBox.No,
                            )
                            if choice == QMessageBox.No:
                                suggested = s
                                continue
                        scorer = s
                        break
                    try:
                        set_default_scorer(anchor, scorer)
                    except Exception:
                        logger.debug("Failed to persist default scorer to sidecar", exc_info=True)

        updated = apply_gt_save_target(
            pts,
            anchor=anchor,
            scorer=scorer,
            dataset_key="keypoints",
        )

        out = write_points_meta(
            layer,
            updated,
            merge_policy=MergePolicy.MERGE,
            fields={"save_target"},
            migrate_legacy=True,
            validate=True,
        )

        if hasattr(out, "errors"):
            logger.warning("Failed to write save_target for layer=%r: %s", getattr(layer, "name", layer), out)
            QMessageBox.warning(self, "Cannot save", "Failed to write save target metadata; see logs for details.")
            return False

        return True

    def _toggle_overwrite_confirmation(self, state) -> None:
        enabled = Qt.CheckState(state) == Qt.CheckState.Checked
        settings.set_overwrite_confirmation_enabled(enabled)
        self.viewer.status = "Overwrite confirmation enabled" if enabled else "Overwrite confirmation disabled"

    # Hack to save a KeyPoints layer without showing the Save dialog
    def _save_layers_dialog(self, selected=False):
        """Save layers (all or selected) to disk, using ``LayerList.save()``.
        Parameters
        ----------
        selected : bool
            If True, only layers that are selected in the viewer will be saved.
            By default, all layers are saved.
        """

        selected_layers = list(self.viewer.layers.selection)
        msg = ""
        if not len(self.viewer.layers):
            msg = "There are no layers in the viewer to save."
        elif selected and not len(selected_layers):
            msg = "Please select a Points layer to save."
        if msg:
            QMessageBox.warning(self, "Nothing to save", msg, QMessageBox.Ok)
            return
        if len(selected_layers) == 1 and isinstance(selected_layers[0], Points):
            layer = selected_layers[0]

            # Promotion-to-GT policy: never write back to machine/prediction sources.
            ok = self._ensure_promotion_save_target(layer)
            if not ok:
                return

            logger.debug(
                "About to save. io.kind=%r save_target=%r",
                layer.metadata.get("io", {}).get("kind"),
                layer.metadata.get("save_target"),
            )
            try:
                overridden_metadata, abort_save = self._maybe_prepare_project_path_override_metadata(layer)
                if abort_save:
                    logger.debug("Save aborted during project-association path handling.")
                    return

                attributes = {
                    "name": layer.name,
                    "metadata": overridden_metadata if overridden_metadata is not None else dict(layer.metadata or {}),
                    "properties": dict(layer.properties or {}),
                }
                report = compute_overwrite_report_for_points_save(layer.data, attributes)
            except Exception as e:
                logger.exception("Failed to compute overwrite preflight for layer %r", getattr(layer, "name", layer))
                QMessageBox.warning(
                    self,
                    "Cannot save",
                    f"Failed to prepare save preflight:\n{e}",
                    QMessageBox.Ok,
                )
                return

            if report is not None:
                if not ui_dialogs.maybe_confirm_overwrite(
                    parent=self,
                    report=report,
                ):
                    logger.debug("Save cancelled.")
                    return

            if overridden_metadata is not None:
                with _temporary_layer_metadata(layer, overridden_metadata):
                    self.viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
                # Persist the successful override into live metadata after save
                layer.metadata = dict(overridden_metadata)
            else:
                self.viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")
            # hook to persist UI state on successful save
            try:
                self._trails_controller.persist_folder_ui_state_for_points_layer(
                    layer,
                    checkbox_checked=self._trail_cb.isChecked(),
                )
            except Exception:
                logger.debug(
                    "Failed to persist folder UI state after save for layer=%r",
                    getattr(layer, "name", layer),
                    exc_info=True,
                )
            self.viewer.status = "Data successfully saved"
        else:
            dlg = QFileDialog()
            hist = get_save_history()
            dlg.setHistory(hist)
            filename, _ = dlg.getSaveFileName(
                caption=f"Save {'selected' if selected else 'all'} layers",
                dir=hist[0],  # home dir by default
            )
            if filename:
                self.viewer.layers.save(filename, selected=selected)
                #  hook to persist UI state on successful save
                try:
                    if selected:
                        candidate_layers = [ly for ly in selected_layers if isinstance(ly, Points)]
                    else:
                        candidate_layers = list(self._stores.keys())

                    for ly in candidate_layers:
                        if ly in self.viewer.layers:
                            self._trails_controller.persist_folder_ui_state_for_points_layer(
                                ly,
                                checkbox_checked=self._trail_cb.isChecked(),
                            )
                except Exception:
                    logger.debug("Failed to persist sidecar UI state after multi-layer save", exc_info=True)

            else:
                return
        self._is_saved = True
        self.last_saved_label.setText(f"Last saved at {str(datetime.now().time()).split('.')[0]}")
        self.last_saved_label.show()

    def on_close(self, event):
        if self._stores and not self._is_saved:
            choice = QMessageBox.warning(
                self,
                "Warning",
                "Data were not saved. Are you certain you want to leave?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def on_active_layer_change(self, event) -> None:
        """Updates the GUI when the active layer changes
        * Hides all KeypointsDropdownMenu that aren't for the selected layer
        * Sets the visibility of the "Color mode" box to True if the selected layer
            is a multi-animal one, or False otherwise
        """
        self._color_grp.setVisible(self._is_multianimal(event.value))
        # self._update_color_scheme() # if needed
        menu_idx = -1
        if event.value is not None and isinstance(event.value, Points):
            menu_idx = self._layer_to_menu.get(event.value, -1)

        for idx, menu in enumerate(self._menus):
            if idx == menu_idx:
                menu.setHidden(False)
            else:
                menu.setHidden(True)

        self._refresh_video_panel_context()
        self._refresh_layer_status_panel()

    def _update_colormap(self, colormap_name: str):
        for layer in self.viewer.layers.selection:
            if not isinstance(layer, Points) or not layer.metadata:
                continue

            layer.metadata["config_colormap"] = colormap_name
            mark_layer_presentation_changed(layer)
            self._apply_points_coloring_from_metadata(layer)

            self._update_color_scheme()
            self._trails_controller.on_points_visual_inputs_changed(checkbox_checked=self._trail_cb.isChecked())

    @register_points_action("Change labeling mode")
    def cycle_through_label_modes(self, *args):
        self.label_mode = next(keypoints.LabelMode)

    @register_points_action("Change color mode")
    def cycle_through_color_modes(self, *args):
        if self._active_layer_is_multianimal() or self.color_mode != str(keypoints.ColorMode.BODYPART):
            self.color_mode = next(keypoints.ColorMode)

    @property
    def label_mode(self):
        return str(self._label_mode)

    @label_mode.setter
    def label_mode(self, mode: str | keypoints.LabelMode):
        self._label_mode = keypoints.LabelMode(mode)
        self.viewer.status = self.label_mode
        mode_ = str(mode).lower()
        if mode_ == keypoints.LabelMode.LOOP.value.lower():
            for menu in self._menus:
                menu._locked = True
        else:
            for menu in self._menus:
                menu._locked = False
        for btn in self._radio_group.buttons():
            if btn.text().lower() == mode_:
                btn.setChecked(True)
                break

    @property
    def color_mode(self):
        return str(self._color_mode)

    @color_mode.setter
    def color_mode(self, mode: str | keypoints.ColorMode):
        self._color_mode = keypoints.ColorMode(mode)

        for layer in list(self._stores.keys()):
            if isinstance(layer, Points) and layer.metadata:
                self._apply_points_coloring_from_metadata(layer)

        for btn in self._color_mode_selector.buttons():
            if btn.text().lower() == str(mode).lower():
                btn.setChecked(True)
                break

        traj_canvas = self._safe_get_traj_canvas()
        if traj_canvas is not None:
            try:
                traj_canvas.refresh_from_viewer_layers()
            except Exception:
                logger.debug("Failed to refresh trajectory plot after color mode change", exc_info=True)

        self._update_color_scheme()
        self._trails_controller.on_points_visual_inputs_changed(checkbox_checked=self._trail_cb.isChecked())

    def _is_multianimal(self, layer) -> bool:
        if layer is None or not isinstance(layer, Points):
            return False

        md = layer.metadata or {}
        hdr = self._get_header_model_from_metadata(md)
        if hdr is None:
            return False

        try:
            inds = hdr.individuals
            return bool(inds and len(inds) > 0 and str(inds[0]) != "")
        except Exception:
            return False

    def _active_layer_is_multianimal(self) -> bool:
        """Returns: whether the active layer is a multi-animal points layer"""
        for layer in self.viewer.layers.selection:
            if self._is_multianimal(layer):
                return True

        return False

    def _resolved_cycle_for_layer(self, layer: Points) -> dict:
        """
        Return the resolved category->color mapping used by the points layer,
        so trails match the exact displayed colors.
        """
        resolver = self._color_scheme_panel._resolver
        cycles = resolver.get_face_color_cycles(layer) or {}

        prop = resolver.get_active_color_property(layer)
        props = getattr(layer, "properties", {}) or {}
        values = props.get(prop)

        if prop == "id":
            try:
                vals = np.asarray(values, dtype=object).ravel() if values is not None else np.array([], dtype=object)
                if len(vals) == 0 or all(v in ("", None) or misc._is_nan_value(v) for v in vals):
                    prop = "label"
            except Exception:
                prop = "label"

        cycle = cycles.get(prop, {}) or {}
        return {str(k): np.asarray(v, dtype=float) for k, v in cycle.items()}
