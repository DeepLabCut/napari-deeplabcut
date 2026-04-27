# src/napari_deeplabcut/ui/dialogs/save.py
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from napari.layers import Image, Points
from napari.utils.history import get_save_history
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ...core.conflicts import compute_overwrite_report_for_points_save
from ...core.errors import MissingProvenanceError
from ...core.io import is_video
from ...core.layers import is_machine_layer
from ...core.metadata import (
    MergePolicy,
    apply_project_paths_override_to_points_meta,
    migrate_points_layer_metadata,
    read_points_meta,
    write_points_meta,
)
from ...core.project_paths import (
    coerce_paths_to_dlc_row_keys,
    dataset_folder_has_files,
    find_nearest_config,
    looks_like_dlc_labeled_folder,
    normalize_anchor_candidate,
    resolve_project_root_from_config,
    target_dataset_folder_for_config,
)
from ...core.provenance import (
    apply_gt_save_target,
    is_projectless_folder_association_candidate,
    requires_gt_promotion,
    suggest_human_placeholder,
)
from ...core.sidecar import get_default_scorer, set_default_scorer
from ...core.trails import safe_folder_anchor_from_points_layer
from ..dialogs import (
    ProjectConfigPromptAction,
    load_scorer_from_config,
    maybe_confirm_dataset_path_rewrite,
    maybe_confirm_overwrite,
    prompt_for_project_config_for_save,
    warn_existing_dataset_folder_conflict,
    warn_invalid_config_for_scorer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...config.models import ImageMetadata
    from ...core.layer_lifecycle.manager import LayerLifecycleManager
    from ...core.trails import TrailsController


@dataclass(slots=True)
class SaveOutcome:
    saved: bool
    status_message: str | None = None


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


class PointsLayerSaveWorkflow:
    """Orchestrate save flows for napari-deeplabcut points layers.

    This class owns:
    - single-layer save routing
    - promotion-to-GT checks
    - project association / metadata override flow
    - overwrite preflight + confirmation
    - save-time provenance enrichment
    - post-save sidecar UI-state persistence

    It intentionally does NOT own widget-specific UI state updates such as:
    - KeypointControls._is_saved
    - last-saved timestamp label
    """

    def __init__(
        self,
        *,
        parent,
        viewer,
        layer_manager: LayerLifecycleManager,
        trails_controller: TrailsController,
        trail_checkbox_getter: Callable[[], bool],
        resolve_config_path_for_layer: Callable[[Points | None], Path | None],
        current_project_path_getter: Callable[[], str | None],
        current_image_meta_getter: Callable[[], ImageMetadata],
        logger: logging.Logger,
    ) -> None:
        self.parent = parent
        self.viewer = viewer
        self.layer_manager = layer_manager
        self.trails_controller = trails_controller
        self.trail_checkbox_getter = trail_checkbox_getter
        self.resolve_config_path_for_layer = resolve_config_path_for_layer
        self.current_project_path_getter = current_project_path_getter
        self.current_image_meta_getter = current_image_meta_getter
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def save_layers(self, *, selected: bool = False) -> SaveOutcome:
        selected_layers = list(self.viewer.layers.selection)

        msg = ""
        if not len(self.viewer.layers):
            msg = "There are no layers in the viewer to save."
        elif selected and not len(selected_layers):
            msg = "Please select a Points layer to save."

        if msg:
            QMessageBox.warning(self.parent, "Nothing to save", msg, QMessageBox.Ok)
            return SaveOutcome(saved=False)

        if len(selected_layers) == 1 and isinstance(selected_layers[0], Points):
            return self._save_single_points_layer(selected_layers[0])

        return self._save_multiple_layers(selected=selected, selected_layers=selected_layers)

    # ------------------------------------------------------------------ #
    # Single-layer points save                                           #
    # ------------------------------------------------------------------ #

    def _save_single_points_layer(self, layer: Points) -> SaveOutcome:
        ok = self._ensure_promotion_save_target(layer)
        if not ok:
            return SaveOutcome(saved=False)

        self.logger.debug(
            "About to save. io.kind=%r save_target=%r",
            layer.metadata.get("io", {}).get("kind"),
            layer.metadata.get("save_target"),
        )

        save_metadata: dict = dict(layer.metadata or {})

        try:
            overridden_metadata, abort_save = self._maybe_prepare_project_path_override_metadata(layer)
            if abort_save:
                self.logger.debug("Save aborted during project-association path handling.")
                return SaveOutcome(saved=False)

            base_metadata = overridden_metadata if overridden_metadata is not None else dict(layer.metadata or {})
            save_metadata = self._enrich_points_metadata_for_save(layer, base_metadata)

            if self._is_unsupported_direct_video_label_save(layer, save_metadata):
                self.logger.debug(
                    "Save aborted due to unsupported direct video + config.yaml label save case. Layer=%r",
                    getattr(layer, "name", layer),
                )
                self._warn_unsupported_direct_video_label_save(layer, save_metadata)
                return SaveOutcome(saved=False)

            attributes = {
                "name": layer.name,
                "metadata": save_metadata,
                "properties": dict(layer.properties or {}),
            }

            report = compute_overwrite_report_for_points_save(layer.data, attributes)

        except MissingProvenanceError:
            self.logger.exception(
                "Missing save provenance for layer %r",
                getattr(layer, "name", layer),
            )
            QMessageBox.warning(
                self.parent,
                "Cannot save keypoints",
                self._format_missing_provenance_save_message(layer, save_metadata),
                QMessageBox.Ok,
            )
            return SaveOutcome(saved=False)

        except Exception as e:
            self.logger.exception(
                "Failed to prepare save checks for layer %r",
                getattr(layer, "name", layer),
            )
            QMessageBox.warning(
                self.parent,
                "Cannot save keypoints",
                f"Something went wrong while preparing this layer for saving:\n{e}",
                QMessageBox.Ok,
            )
            return SaveOutcome(saved=False)

        if report is not None:
            if not maybe_confirm_overwrite(
                parent=self.parent,
                report=report,
            ):
                self.logger.debug("Save cancelled.")
                return SaveOutcome(saved=False)

        metadata_changed = save_metadata != dict(layer.metadata or {})

        with _temporary_layer_metadata(layer, save_metadata):
            self.viewer.layers.save("__dlc__.h5", selected=True, plugin="napari-deeplabcut")

        # Persist successful save-time metadata improvements into the live layer.
        if metadata_changed:
            layer.metadata = dict(save_metadata)

        self._persist_folder_ui_state_for_layers([layer])

        return SaveOutcome(saved=True, status_message="Data successfully saved")

    # ------------------------------------------------------------------ #
    # Multi-layer / generic save                                         #
    # ------------------------------------------------------------------ #

    def _save_multiple_layers(self, *, selected: bool, selected_layers: list) -> SaveOutcome:
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)

        filename, _ = dlg.getSaveFileName(
            caption=f"Save {'selected' if selected else 'all'} layers",
            dir=hist[0],  # home dir by default
        )

        if not filename:
            return SaveOutcome(saved=False)

        self.viewer.layers.save(filename, selected=selected)

        if selected:
            candidate_layers = [ly for ly in selected_layers if isinstance(ly, Points)]
        else:
            candidate_layers = list(self.layer_manager.managed_points_layers())

        self._persist_folder_ui_state_for_layers(candidate_layers)

        return SaveOutcome(saved=True, status_message="Data successfully saved")

    # ------------------------------------------------------------------ #
    # Save-time metadata enrichment                                      #
    # ------------------------------------------------------------------ #

    def _best_image_context_layer(self) -> Image | None:
        """Return the best available image layer for save-time provenance inference."""
        active = self.layer_manager.active_dlc_image_layer()
        if active is not None:
            return active

        selected = self.viewer.layers.selection.active
        if isinstance(selected, Image):
            return selected

        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer

        return None

    def _enrich_points_metadata_for_save(self, layer: Points, metadata: dict) -> dict:
        """Best-effort save-time metadata enrichment for DLC routing.

        Conservative policy:
        - never overwrite explicit metadata already present
        - first reuse lifecycle-owned image context
        - then try nearby config.yaml + image/source hints
        """
        md = dict(metadata or {})

        if md.get("root"):
            return md

        if self.layer_manager.image_root:
            md.setdefault("root", self.layer_manager.image_root)
        if self.layer_manager.image_paths:
            md.setdefault("paths", self.layer_manager.image_paths)

        if md.get("root"):
            return md

        config_path = self.resolve_config_path_for_layer(layer)
        if config_path is None:
            return md

        project_root = resolve_project_root_from_config(config_path)
        if project_root is None:
            return md

        md.setdefault("project", str(project_root))

        image_layer = self._best_image_context_layer()
        if image_layer is None:
            return md

        try:
            src = getattr(getattr(image_layer, "source", None), "path", None)
        except Exception:
            src = None

        src_anchor = normalize_anchor_candidate(src) if src else None
        if src_anchor is not None and looks_like_dlc_labeled_folder(src_anchor):
            md.setdefault("root", str(src_anchor))
            return md

        image_name = getattr(image_layer, "name", None)
        if image_name:
            candidate = project_root / "labeled-data" / str(image_name)
            if candidate.is_dir():
                md.setdefault("root", str(candidate))

        return md

    def _is_video_context_layer(self, layer: Image | None) -> bool:
        if layer is None:
            return False

        md = getattr(layer, "metadata", {}) or {}
        dlc_md = md.get("dlc") or {}
        if dlc_md.get("session_role") == "video":
            return True

        try:
            src = getattr(getattr(layer, "source", None), "path", None)
        except Exception:
            src = None

        for candidate in (src, getattr(layer, "name", None)):
            if candidate and is_video(str(candidate)):
                return True

        return False

    def _is_unsupported_direct_video_label_save(self, layer: Points, metadata: dict) -> bool:
        """
        Unsupported case:
        - points layer has no extracted-frame row keys (`paths`)
        - current save context is a video session

        This corresponds to labeling directly on a loaded video after adding
        a config.yaml / placeholder config, which bypasses DLC frame extraction.
        """
        paths = metadata.get("paths") or []
        if paths:
            return False

        image_layer = self._best_image_context_layer()
        return self._is_video_context_layer(image_layer)

    def _warn_unsupported_direct_video_label_save(self, layer: Points, metadata: dict) -> None:
        image_layer = self._best_image_context_layer()
        image_name = getattr(image_layer, "name", None) if image_layer is not None else "current video"
        config_path = self.resolve_config_path_for_layer(layer)

        image_html = escape(image_name or "not found")
        config_html = escape(str(config_path) if config_path is not None else "not found")

        msg = QMessageBox(self.parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Cannot save labels from video directly")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "<p>"
            "The currently loaded image layer looks like it is a video, and the keypoints layer "
            "has no paths to individual image files."
            "</p>"
            "<p>"
            "Saving labels created directly on a loaded video by adding <code>config.yaml</code> "
            "is <b>not supported</b>."
            "</p>"
            "<p>"
            "This bypasses DeepLabCut's frame extraction workflow, and the plugin cannot write a valid "
            "<code>CollectedData_&lt;scorer&gt;.h5</code> file from video frame indices alone."
            "</p>"
            f"<p><b>Video:</b> {image_html}<br>"
            f"<b>Config:</b> {config_html}</p>"
            "<p><b>What to do instead:</b></p>"
            "<ul>"
            "<li>First, use <b>Video panel &gt; 'Extract current frame'</b> (or use DeepLabCut) for this video</li>"
            "<li>Load the resulting <code>labeled-data</code> folder / extracted images</li>"
            "<li>If needed, drag and drop the <code>config.yaml</code> to create a placeholder points layer</li>"
            "<li>Start annotating in the created Points layer and save it</li>"
            "</ul>"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def _format_missing_provenance_save_message(self, layer: Points, metadata: dict) -> str:
        config_path = self.resolve_config_path_for_layer(layer)
        image_layer = self._best_image_context_layer()

        layer_name = getattr(layer, "name", "Unnamed layer")
        project_hint = metadata.get("project") or self.current_project_path_getter() or "not available"
        root_hint = metadata.get("root") or "not available"
        n_paths = len(metadata.get("paths") or [])

        image_name = getattr(image_layer, "name", None) if image_layer is not None else None

        return (
            "Couldn't determine where to save this keypoints layer in a DeepLabCut project.\n\n"
            f"Layer: {layer_name}\n"
            f"Project hint: {project_hint}\n"
            f"Dataset folder (root): {root_hint}\n"
            f"Paths metadata: {n_paths} item(s)\n"
            f"Nearby config.yaml: {str(config_path) if config_path is not None else 'not found'}\n"
            f"Image/video context: {image_name or 'not available'}\n\n"
            "To save this layer, the plugin needs either:\n"
            "• a dataset folder (metadata['root']), or\n"
            "• enough project/image context to infer one automatically.\n\n"
            "Try one of the following:\n"
            "• open the image/video from the DLC project first,\n"
            "• load the project's config.yaml or labeled-data folder, or\n"
            "• make sure this layer is associated with the correct dataset folder."
        )

    # ------------------------------------------------------------------ #
    # Promotion / provenance helpers                                     #
    # ------------------------------------------------------------------ #

    def _ensure_promotion_save_target(self, layer: Points) -> bool:
        """Ensure a prediction/machine source layer has a GT save_target set.

        Returns True if save_target is set (or already existed), False if user cancels.
        """
        if not getattr(layer, "metadata", None):
            layer.metadata = {}

        from ...config.models import PointsMetadata  # local import to avoid unnecessary module load in import path

        if not safe_folder_anchor_from_points_layer(layer) and not is_machine_layer(layer := layer):
            # Preserve old fast path for non-machine layers
            return True

        if not is_machine_layer(layer):
            return True

        mig = migrate_points_layer_metadata(layer)
        if hasattr(mig, "errors"):
            self.logger.warning(
                "Failed to migrate points layer metadata for layer=%r: %s",
                getattr(layer, "name", layer),
                mig,
            )

        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if hasattr(res, "errors"):
            self.logger.warning(
                "Points metadata validation failed for layer=%r during save target check: %s",
                getattr(layer, "name", layer),
                res,
            )
            QMessageBox.warning(self.parent, "Cannot save", "Layer metadata is invalid; see logs for details.")
            return False

        pts: PointsMetadata = res

        if not requires_gt_promotion(pts):
            return True

        anchor = safe_folder_anchor_from_points_layer(layer)
        if not anchor:
            QMessageBox.warning(self.parent, "Cannot save", "Could not determine a folder anchor for saving.")
            return False

        scorer = None

        cfg_path = None
        try:
            cfg_path = find_nearest_config(anchor)
        except Exception:
            self.logger.debug("Automatic config discovery failed for anchor=%r", anchor, exc_info=True)

        if cfg_path:
            try:
                scorer = load_scorer_from_config(cfg_path)
            except Exception:
                self.logger.exception("Failed to load auto-discovered config.yaml: %s", cfg_path)
                warn_invalid_config_for_scorer(
                    self.parent,
                    config_path=cfg_path,
                    reason="unreadable",
                    auto_found=True,
                )
                return False

            if not scorer:
                warn_invalid_config_for_scorer(
                    self.parent,
                    config_path=cfg_path,
                    reason="missing_scorer",
                    auto_found=True,
                )
                return False

        else:
            dialog_result = prompt_for_project_config_for_save(
                self.parent,
                initial_dir=self.current_project_path_getter() or anchor,
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

            if dialog_result.action is ProjectConfigPromptAction.CANCEL:
                return False

            if dialog_result.action is ProjectConfigPromptAction.ASSOCIATE:
                scorer = dialog_result.scorer

            else:
                scorer = get_default_scorer(anchor)

                if not scorer:
                    suggested = suggest_human_placeholder(anchor)
                    while True:
                        s = _prompt_for_scorer(self.parent, anchor=anchor, suggested=suggested)
                        if s is None:
                            return False
                        if s.startswith("human_"):
                            choice = QMessageBox.question(
                                self.parent,
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
                        self.logger.debug("Failed to persist default scorer to sidecar", exc_info=True)

        updated = apply_gt_save_target(
            pts,
            anchor=anchor,
            scorer=scorer,
            dataset_key="df_with_missing",
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
            self.logger.warning("Failed to write save_target for layer=%r: %s", getattr(layer, "name", layer), out)
            QMessageBox.warning(
                self.parent,
                "Cannot save",
                "Failed to write save target metadata; see logs for details.",
            )
            return False

        return True

    def _maybe_prepare_project_path_override_metadata(self, layer: Points) -> tuple[dict | None, bool]:
        """Optionally prepare save-time metadata by associating a project-less labeled
        folder with an explicit DLC project chosen via config.yaml.
        """
        from ...config.models import PointsMetadata  # local import to avoid unnecessary module load in import path

        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if hasattr(res, "errors"):
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

        dataset_name = source_root_path.name
        if not dataset_name:
            return None, False

        initial_dir = self.current_project_path_getter() or pts_meta.project or str(source_root_path)
        dialog_result = prompt_for_project_config_for_save(self.parent, initial_dir=initial_dir)

        if dialog_result.action is ProjectConfigPromptAction.CANCEL:
            self.logger.debug("User cancelled project association prompt.")
            return None, True

        if dialog_result.action is ProjectConfigPromptAction.SKIP:
            self.logger.debug("User chose to continue without project association.")
            return None, False

        if dialog_result.action is not ProjectConfigPromptAction.ASSOCIATE:
            self.logger.warning("Unexpected project association dialog result: %r", dialog_result)
            return None, True

        config_path = dialog_result.config_path
        if not config_path:
            self.logger.warning("Project association result was ASSOCIATE but config_path was empty.")
            return None, True

        project_root = resolve_project_root_from_config(config_path)
        if project_root is None:
            QMessageBox.warning(
                self.parent,
                "Invalid project configuration",
                "The selected file is not a valid DeepLabCut config.yaml or project root. "
                "The save operation has been cancelled.",
            )
            return None, True

        target_folder = target_dataset_folder_for_config(config_path, dataset_name=dataset_name)
        if dataset_folder_has_files(target_folder):
            warn_existing_dataset_folder_conflict(self.parent, target_folder=target_folder)
            return None, True

        rewritten_paths, unresolved = coerce_paths_to_dlc_row_keys(
            paths,
            source_root=source_root_path,
            dataset_name=dataset_name,
        )

        if not maybe_confirm_dataset_path_rewrite(
            self.parent,
            project_root=project_root,
            dataset_name=dataset_name,
            n_paths=len(paths),
            n_unresolved=len(unresolved),
        ):
            return None, True

        overridden = apply_project_paths_override_to_points_meta(
            pts_meta,
            project_root=project_root,
            rewritten_paths=rewritten_paths,
        )

        return overridden.model_dump(mode="python", exclude_none=True), False

    # ------------------------------------------------------------------ #
    # Post-save state persistence                                        #
    # ------------------------------------------------------------------ #

    def _persist_folder_ui_state_for_layers(self, layers: Iterable[Points]) -> None:
        try:
            checked = bool(self.trail_checkbox_getter())
            for layer in layers:
                if layer in self.viewer.layers:
                    self.trails_controller.persist_folder_ui_state_for_points_layer(
                        layer,
                        checkbox_checked=checked,
                    )
        except Exception:
            self.logger.debug("Failed to persist folder UI state after save", exc_info=True)
