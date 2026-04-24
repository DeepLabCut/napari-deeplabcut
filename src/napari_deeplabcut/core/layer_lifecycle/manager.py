from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Image, Layer, Points, Tracks
from napari.utils.events import Event
from napari.utils.history import update_save_history
from qtpy.QtCore import QObject, QTimer, Signal

from ...config.keybinds import install_points_layer_keybindings
from ...config.models import DLCHeaderModel, ImageMetadata, PointsMetadata
from ...core import keypoints
from ...core.io import is_video
from ...core.layer_versioning import mark_layer_presentation_changed
from ...core.metadata import (
    MergePolicy,
    infer_image_root,
    migrate_points_layer_metadata,
    read_points_meta,
    sync_points_from_image,
    write_points_meta,
)
from ...core.project_paths import PathMatchPolicy
from ...core.remap import remap_layer_data_by_paths
from ...napari_compat import install_add_wrapper, install_paste_patch
from ...napari_compat.points_layer import make_paste_data
from ...tracking.core.data import TRACKING_LAYER_METADATA_KEY, is_tracking_result_points_layer
from ...ui.cropping import resolve_project_path_from_image_layer
from ...utils.debug import log_timing
from .merge import MergeDecisionProvider, MergeDecisionRequest, MergeDecisionResult, MergeDisposition
from .registry import (
    ClearedRegistryEntry,
    ManagedPointsRuntime,
    PointsLayerSetupRequest,
    PointsRuntimeResources,
    RegistryAuditReport,
    RuntimeRegistry,
)

if TYPE_CHECKING:
    import napari

    from ..keypoints import KeypointStore

logger = logging.getLogger("napari-deeplabcut.lifecycle")


class LayerLifecycleManager(QObject):
    """Lifecycle wrapper around existing widget behavior.

    Goals
    -----
    - centralize viewer layer event entry points
    - own the runtime registry for managed Points layers
    - centralize layer liveness / store resolution
    - keep current behavior by delegating back into widget-owned hooks

    Non-goals (for now)
    -------------------
    - full reconciliation engine
    - moving all logic out of the widget
    - changing merge/remap policies
    """

    # UI signals for widget hooks
    refresh_video_panel_requested = Signal()
    refresh_layer_status_requested = Signal()
    video_widget_visibility_requested = Signal(bool)
    move_image_layer_to_bottom_requested = Signal(object)

    # Layer setup/teardown
    points_layer_setup_requested = Signal(object)  # PointsLayerSetupRequest
    points_layers_merged_requested = Signal(object)  # tuple[Points, ...]
    points_layer_removed_requested = Signal(object, int)  # layer, remaining_points_layers
    tracks_layer_removed_requested = Signal(object)

    # Layer insertion/adoption
    adopted_existing_layers = Signal()
    layer_insert_processed = Signal(object)
    layer_remove_processed = Signal(object)

    # Session management
    session_conflict_rejected = Signal(str)  # if a new DLC folder is loaded on top of the current one

    def __init__(self, viewer: napari.Viewer, *, parent: QObject | None = None) -> None:
        super().__init__(parent=parent)

        self.viewer = viewer
        self.registry: RuntimeRegistry[Any] = RuntimeRegistry()
        self._merge_decision_provider: MergeDecisionProvider | None = None

        # Lifecycle-owned viewer/image context
        self._active_dlc_image_layer_id: int | None = None
        self._image_meta = ImageMetadata()
        self._project_path: str | None = None

        # Layers management
        ## Layer insertion
        self._initial_adopt_timer = QTimer(self)
        self._initial_adopt_timer.setSingleShot(True)
        self._initial_adopt_timer.timeout.connect(self.adopt_existing_layers)
        ## Layer removal
        self._post_remove_refresh_timer = QTimer(self)
        self._post_remove_refresh_timer.setSingleShot(True)
        self._post_remove_refresh_timer.timeout.connect(self._flush_post_remove_refresh)

        self._attached = False

    # ------------------------------------------------------------------ #
    # Centralized access API                                             #
    # ------------------------------------------------------------------ #

    def resolve_live_layer(self, layer_or_id: Any) -> Layer | None:
        layer = self.registry.resolve_live_layer(layer_or_id)
        return layer if isinstance(layer, Layer) else None

    def get_live_runtime(self, layer_or_id: Any) -> ManagedPointsRuntime[Any] | None:
        return self.registry.get_live_runtime(layer_or_id)

    def get_store(self, layer_or_id: Any) -> KeypointStore | None:
        store = self.registry.get_store(layer_or_id)
        return store  # typed via TYPE_CHECKING

    def require_store(self, layer_or_id: Any) -> KeypointStore:
        store = self.registry.require_store(layer_or_id)
        return store  # typed via TYPE_CHECKING

    def iter_managed_points(self) -> Iterator[tuple[Points, KeypointStore]]:
        """Iterate only live managed Points layers and their stores."""
        for layer, runtime in self.registry.iter_live_items():
            if isinstance(layer, Points):
                yield layer, runtime.store

    def managed_points_layers(self) -> tuple[Points, ...]:
        return tuple(layer for layer, _ in self.iter_managed_points())

    def managed_points_count(self) -> int:
        return sum(1 for _ in self.iter_managed_points())

    def has_managed_points(self) -> bool:
        return any(True for _ in self.iter_managed_points())

    def clear_dead_entries(self, *, log: bool = True) -> tuple[ClearedRegistryEntry[Any], ...]:
        return self.registry.clear_dead_entries(log=log)

    def audit_registry(self) -> RegistryAuditReport:
        return self.registry.audit()

    def set_merge_decision_provider(self, provider: MergeDecisionProvider | None) -> None:
        self._merge_decision_provider = provider

    # ------------------------------------------------------------------ #
    # lifecycle-owned image/project context                              #
    # ------------------------------------------------------------------ #

    @property
    def image_meta(self) -> ImageMetadata:
        return self._image_meta

    @property
    def project_path(self) -> str | None:
        return self._project_path

    @project_path.setter
    def project_path(self, value: str | None) -> None:
        self._project_path = value

    @property
    def image_root(self) -> str | None:
        return self._image_meta.root

    @property
    def image_paths(self) -> list[str] | None:
        return self._image_meta.paths

    @property
    def image_name(self) -> str | None:
        return self._image_meta.name

    # ------------------------------------------------------------------ #
    # Lifecycle wiring                                                   #
    # ------------------------------------------------------------------ #

    def attach(self) -> None:
        """Attach to viewer layer events."""
        if self._attached:
            return

        self.viewer.layers.events.inserted.connect(self.on_insert)
        self.viewer.layers.events.removed.connect(self.on_remove)
        self._attached = True

    def detach(self) -> None:
        """Detach from viewer layer events."""
        if not self._attached:
            return

        try:
            self.viewer.layers.events.inserted.disconnect(self.on_insert)
        except Exception:
            pass

        try:
            self.viewer.layers.events.removed.disconnect(self.on_remove)
        except Exception:
            pass

        self._attached = False

    def schedule_initial_adoption(self) -> None:
        """Schedule adoption of existing layers after the event loop starts."""
        self._initial_adopt_timer.start(0)

    def _dlc_meta_for_layer(self, layer: Layer) -> dict | None:
        md = layer.metadata or {}
        if not isinstance(md, dict):
            return None
        payload = md.get("dlc", None)
        return payload if isinstance(payload, dict) else None

    def is_dlc_session_image_layer(self, layer: Image) -> bool:
        payload = self._dlc_meta_for_layer(layer)
        if not payload:
            return False

        role = payload.get("session_role", None)
        ctx = payload.get("project_context", None)

        return role in {"image", "video"} and isinstance(ctx, dict) and bool(ctx)

    def active_dlc_image_layer(self) -> Image | None:
        if self._active_dlc_image_layer_id is None:
            return None

        for layer in self.viewer.layers:
            if id(layer) == self._active_dlc_image_layer_id and isinstance(layer, Image):
                return layer

        return None

    def can_accept_dlc_session_image(self, layer: Image) -> tuple[bool, str | None]:
        active = self.active_dlc_image_layer()
        if active is None:
            return True, None
        if active is layer:
            return True, None
        return (
            False,
            "A DLC project/video is already open.\n"
            "The plugin will attempt to load annotations from the new project, "
            "but will not load the video.\n\n"
            "If you meant to load extra annotations for the current video, "
            "please only load the corresponding h5 files.\n"
            "If you meant to switch to a different project/video, "
            "please save and clear the current layers before loading the new labeled data folder.",
        )

    def _reject_conflicting_dlc_image_layer(self, layer: Image, reason: str) -> None:
        """Reject a conflicting DLC session image safely.

        Do not remove synchronously inside the insert callback:
        napari may still be finalizing list insertion / selection.
        """
        self.viewer.status = reason
        self.session_conflict_rejected.emit(reason)

        def _remove_later(ly=layer):
            try:
                if ly in self.viewer.layers:
                    self.viewer.layers.remove(ly)
            except Exception:
                logger.debug(
                    "Failed to remove conflicting DLC image layer %r",
                    getattr(ly, "name", ly),
                    exc_info=True,
                )

        QTimer.singleShot(0, _remove_later)

    # ------------------------------------------------------------------ #
    # Layer setup managers                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _layer_source_path(layer) -> str | None:
        """Best-effort access to napari layer source path (may not exist)."""
        try:
            src = getattr(layer, "source", None)
            p = getattr(src, "path", None) if src is not None else None
            return str(p) if p else None
        except Exception:
            return None

    @staticmethod
    def get_header_model_from_metadata(md: dict) -> DLCHeaderModel | None:
        """Return DLCHeaderModel from metadata payload, if possible."""
        if not isinstance(md, dict):
            return None

        hdr = md.get("header", None)
        if hdr is None:
            return None

        if isinstance(hdr, DLCHeaderModel):
            return hdr

        if isinstance(hdr, dict):
            try:
                return DLCHeaderModel.model_validate(hdr)
            except Exception:
                return None

        try:
            return DLCHeaderModel(columns=hdr)
        except Exception:
            return None

    @staticmethod
    def is_multianimal(layer) -> bool:
        """Return True if this layer looks like a multi-animal Points layer."""
        if layer is None or not isinstance(layer, Points):
            return False

        md = layer.metadata or {}
        hdr = LayerLifecycleManager.get_header_model_from_metadata(md)
        if hdr is None:
            return False

        try:
            inds = hdr.individuals
            # return bool(inds and len(inds) > 0 and str(inds[0]) != "")
            return any(str(ind) != "" for ind in inds)
        except Exception:
            return False

    @staticmethod
    def is_config_placeholder_points_layer(layer: Points) -> bool:
        """Return True if this looks like the temporary config placeholder layer.

        - must be a Points layer
        - must carry a project hint
        - must not already be tied to image/root/paths context
        - must not contain actual point data
        """
        if layer is None or not isinstance(layer, Points):
            return False

        md = layer.metadata or {}
        if not md.get("project"):
            return False

        # Real labeled/prediction layers usually carry image/root/paths context.
        if md.get("root") or md.get("paths"):
            return False

        try:
            data = np.asarray(layer.data) if layer.data is not None else np.empty((0, 3))
        except Exception:
            data = np.empty((0, 3))

        return data.size == 0

    @staticmethod
    def validate_header(layer: Points) -> bool:
        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if hasattr(res, "errors") or getattr(res, "header", None) is None:
            # self.viewer.status = (
            #     "This Points layer does not look like a DLC keypoints layer. Missing a valid DLC header."
            # )
            logger.debug("Points layer %r failed DLC header validation: %s", getattr(layer, "name", layer), res)
            return False
        return True

    def _update_image_meta_from_layer(self, layer: Image) -> bool:
        """Update lifecycle-owned image metadata from an Image layer.

        Returns
        -------
        bool
            True if the authoritative image context changed, else False.
        """
        md = layer.metadata or {}

        paths = md.get("paths")
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

        base = self._image_meta
        merged = base.model_copy(deep=True)
        for field, value in incoming.model_dump().items():
            if getattr(merged, field) in (None, "", []) and value not in (None, "", []):
                setattr(merged, field, value)

        changed = merged != self._image_meta
        self._image_meta = merged
        return changed

    def _sync_points_layers_from_image_meta(self) -> None:
        """Sync managed/all points metadata from lifecycle-owned image context."""
        if self._image_meta is None:
            return

        for ly in list(self.viewer.layers):
            if not isinstance(ly, Points):
                continue

            if ly.metadata is None:
                ly.metadata = {}

            res = read_points_meta(ly, migrate_legacy=True, drop_controls=False, drop_header=False)
            if hasattr(res, "errors"):
                logger.warning(
                    "Points metadata validation failed during sync for layer=%r: %s",
                    getattr(ly, "name", ly),
                    res,
                )
                continue

            pts_model: PointsMetadata = res
            synced = sync_points_from_image(self._image_meta, pts_model)

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

    def _cache_project_path_from_image_layer(self, layer: Image) -> None:
        """Best-effort lifecycle-owned cache of project path from an image/video layer."""
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

        self.refresh_video_panel_requested.emit()

    def _setup_image_layer(self, layer: Image, index: int | None = None, *, reorder: bool = True) -> None:
        """Lifecycle-owned setup for an inserted/adopted Image layer."""
        md = layer.metadata or {}
        paths = md.get("paths")
        if paths is None:
            try:
                if is_video(layer.name):
                    self.video_widget_visibility_requested.emit(True)
            except Exception:
                pass

        self._active_dlc_image_layer_id = id(layer)
        context_changed = self._update_image_meta_from_layer(layer)

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

        if context_changed:
            self._sync_points_layers_from_image_meta()

        self.refresh_video_panel_requested.emit()

        logger.debug(
            "Setup image layer=%r index=%s reorder=%s paths_count=%s root=%r context_changed=%s",
            getattr(layer, "name", layer),
            index,
            reorder,
            len(md.get("paths") or []),
            md.get("root"),
            context_changed,
        )

        if reorder and index is not None:
            QTimer.singleShot(10, lambda ly=layer: self.move_image_layer_to_bottom_requested.emit(ly))

    def _wire_points_layer(self, layer: Points) -> KeypointStore | None:
        """Lifecycle-owned wiring of a managed Points layer.

        The manager owns registration and lifecycle sequencing.
        Runtime/UI completion is delegated via points_layer_setup_requested.
        """
        if not self.validate_header(layer):
            return None

        existing = getattr(layer, "_dlc_store", None)
        if existing is not None:
            self.register_managed_points_layer(layer, existing)

            runtime = self.get_live_runtime(layer)
            existing_resources = None
            if runtime is not None:
                existing_resources = runtime.resources.get("points_runtime", None)

            req = PointsLayerSetupRequest(
                layer=layer,
                store=existing,
                existing_resources=existing_resources,
            )
            self.points_layer_setup_requested.emit(req)

            runtime = self.get_live_runtime(layer)
            if runtime is not None and req.runtime_resources is not None:
                runtime.resources["points_runtime"] = req.runtime_resources

            return existing

        mig = migrate_points_layer_metadata(layer)
        if hasattr(mig, "errors"):
            logger.warning(
                "Points metadata validation failed during wiring for layer=%r: %s",
                getattr(layer, "name", layer),
                mig,
            )

        store = keypoints.KeypointStore(self.viewer, layer)
        self.register_managed_points_layer(layer, store)

        layer._dlc_store = store

        proj = layer.metadata.get("project")
        if proj:
            self._project_path = proj

        if not layer.metadata.get("root") and self._image_meta.root:
            layer.metadata["root"] = self._image_meta.root
        if not layer.metadata.get("paths") and self._image_meta.paths:
            layer.metadata["paths"] = self._image_meta.paths

        if root := layer.metadata.get("root"):
            update_save_history(root)

        layer.text.visible = False

        req = PointsLayerSetupRequest(layer=layer, store=store)
        self.points_layer_setup_requested.emit(req)

        runtime = self.get_live_runtime(layer)
        if runtime is not None and req.runtime_resources is not None:
            runtime.resources["points_runtime"] = req.runtime_resources

        md = layer.metadata or {}
        logger.debug(
            "Wire points layer=%r existing_store=%s project=%s root=%s len_paths=%s",
            getattr(layer, "name", layer),
            getattr(layer, "_dlc_store", None) is not None,
            md.get("project"),
            md.get("root"),
            len(md.get("paths", [])),
        )

        return store

    def _remove_layer_if_present(self, layer: Layer) -> None:
        try:
            if layer in self.viewer.layers:
                with log_timing(
                    logger,
                    f"viewer.layers.remove layer={getattr(layer, 'name', layer)!r}",
                    threshold_ms=0.01,
                ):
                    self.viewer.layers.remove(layer)
        except Exception:
            logger.debug("Failed to remove layer=%r", getattr(layer, "name", layer), exc_info=True)

    @staticmethod
    def _set_layer_visible(layer: Layer, visible: bool) -> None:
        try:
            layer.visible = visible
        except Exception:
            try:
                layer.shown = visible
            except Exception:
                logger.debug("Failed to set visibility for layer=%r", getattr(layer, "name", layer), exc_info=True)

    def _setup_points_layer(self, layer: Points, *, allow_merge: bool = True) -> None:
        """Lifecycle-owned setup for an inserted/adopted Points layer."""
        if not self.validate_header(layer):
            return

        if allow_merge:
            consumed = self._maybe_merge_config_points_layer(layer)
            if consumed:
                logger.debug(
                    "Consumed temporary config placeholder layer=%r during merge path",
                    getattr(layer, "name", layer),
                )
                return

        store = self._wire_points_layer(layer)
        if store is None:
            return

        logger.debug(
            "Setup points layer=%r allow_merge=%s metadata_keys=%s",
            getattr(layer, "name", layer),
            allow_merge,
            sorted((layer.metadata or {}).keys()),
        )

    def _schedule_post_remove_refresh(self) -> None:
        """Coalesce repeated UI refreshes during layer removal bursts."""
        self._post_remove_refresh_timer.start(0)

    def _flush_post_remove_refresh(self) -> None:
        with log_timing(logger, "_flush_post_remove_refresh total", threshold_ms=0.01):
            self.refresh_video_panel_requested.emit()
            self.refresh_layer_status_requested.emit()

    def _handle_removed_layer(self, layer: Any) -> None:
        """Lifecycle-owned remove handling.

        Transitional note:
        ------------------
        UI/menu cleanup still delegates to a widget-owned UI hook.
        """
        with log_timing(
            logger,
            f"_handle_removed_layer total layer={getattr(layer, 'name', layer)!r}",
            threshold_ms=0.01,
        ):
            n_points_layer = sum(isinstance(l, Points) for l in self.viewer.layers)

            if isinstance(layer, Points):
                with log_timing(
                    logger,
                    f"unregister_managed_layer layer={getattr(layer, 'name', layer)!r}",
                    threshold_ms=0.01,
                ):
                    store = self.unregister_managed_layer(layer)

                if store is not None:
                    with log_timing(
                        logger,
                        f"points_layer_removed_requested layer={getattr(layer, 'name', layer)!r}",
                        threshold_ms=0.01,
                    ):
                        self.points_layer_removed_requested.emit(layer, n_points_layer)

            elif isinstance(layer, Image):
                if self._active_dlc_image_layer_id == id(layer):
                    self._active_dlc_image_layer_id = None
                    self._image_meta = ImageMetadata()
                    self._project_path = None

                    paths = layer.metadata.get("paths")
                    if paths is None:
                        self.video_widget_visibility_requested.emit(False)
                else:
                    logger.debug(
                        "Removed non-session or inactive image layer=%r; keeping current DLC session context.",
                        getattr(layer, "name", layer),
                    )

            elif isinstance(layer, Tracks):
                self.tracks_layer_removed_requested.emit(layer)

            self._schedule_post_remove_refresh()

    def _remap_frame_indices(self, layer: Any) -> None:
        """Lifecycle-owned remap of non-Image layer time/frame indices."""
        try:
            new_paths = self._image_meta.paths
            if not new_paths:
                return

            if layer.metadata is None:
                layer.metadata = {}

            md = layer.metadata
            old_paths = md.get("paths") or []

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

    def _maybe_merge_config_points_layer(self, layer: Points) -> bool:
        """Merge a temporary config placeholder layer into existing managed layers.

        Returns
        -------
        bool
            True if the placeholder layer was consumed by the merge flow
            (including explicit HIDE_NEW / CANCEL handling), else False.
        """
        if not self.is_config_placeholder_points_layer(layer):
            return False

        managed = list(self.iter_managed_points())
        if not managed:
            return False

        md = layer.metadata or {}
        logger.debug(
            "Maybe merge config placeholder layer=%r project=%r managed_layers=%d",
            getattr(layer, "name", layer),
            md.get("project"),
            len(managed),
        )

        new_metadata = md.copy()
        new_header = self.get_header_model_from_metadata(new_metadata)
        if new_header is None:
            logger.debug(
                "Skipping config placeholder merge for layer=%r: missing/invalid header",
                getattr(layer, "name", layer),
            )
            return False

        reference_layer, _reference_store = managed[0]
        reference_header = self.get_header_model_from_metadata(reference_layer.metadata or {})
        if reference_header is None:
            logger.debug(
                "Skipping config placeholder merge for layer=%r: reference managed layer has no valid header",
                getattr(layer, "name", layer),
            )
            return False

        current_keypoint_set = set(reference_header.bodyparts)
        new_keypoint_set = set(new_header.bodyparts)
        diff = tuple(sorted(new_keypoint_set.difference(current_keypoint_set)))

        visible_existing_layer = None
        for managed_layer, _store in managed:
            if managed_layer is layer:
                continue
            try:
                if getattr(managed_layer, "visible", True):
                    visible_existing_layer = managed_layer
                    break
            except Exception:
                visible_existing_layer = managed_layer
                break

        message = f"New keypoint{'s' if len(diff) > 1 else ''} {', '.join(diff)} found." if diff else ""

        decision = self._resolve_merge_decision(
            new_layer=layer,
            existing_layers=tuple(ly for ly, _ in managed if ly is not layer),
            added_keypoints=diff,
            message=message,
        )

        disposition = decision.disposition

        # Optional visibility policy before merge
        if disposition is MergeDisposition.HIDE_EXISTING and visible_existing_layer is not None:
            self._set_layer_visible(visible_existing_layer, False)
            logger.debug(
                "Config placeholder merge layer=%r hid_existing_layer=%r",
                getattr(layer, "name", layer),
                getattr(visible_existing_layer, "name", visible_existing_layer),
            )

        if disposition is MergeDisposition.HIDE_NEW:
            QTimer.singleShot(0, lambda ly=layer: self._remove_layer_if_present(ly))
            return True

        if disposition is MergeDisposition.CANCEL:
            # Conservative behavior: stop lifecycle setup and leave the placeholder
            # layer untouched/unmanaged for now.
            logger.debug(
                "Config placeholder merge cancelled for layer=%r",
                getattr(layer, "name", layer),
            )
            return True

        if diff:
            self.viewer.status = message

        # Merge header into all managed layers.
        affected_layers: list[Points] = []
        for managed_layer, store in managed:
            pts = read_points_meta(
                managed_layer,
                migrate_legacy=True,
                drop_controls=True,
                drop_header=False,
            )
            if not hasattr(pts, "errors"):
                updated = pts.model_copy(update={"header": new_header})
                write_points_meta(
                    managed_layer,
                    updated,
                    merge_policy=MergePolicy.MERGE,
                    fields={"header"},
                )
            store.layer = managed_layer
            affected_layers.append(managed_layer)

        # Apply updated presentation metadata to existing managed layers.
        for managed_layer, store in managed:
            managed_layer.metadata["config_colormap"] = new_metadata.get(
                "config_colormap",
                managed_layer.metadata.get("config_colormap"),
            )
            if "face_color_cycles" in new_metadata:
                managed_layer.metadata["face_color_cycles"] = new_metadata["face_color_cycles"]
            managed_layer.metadata["colormap_name"] = new_metadata.get(
                "colormap_name",
                managed_layer.metadata.get("colormap_name"),
            )

            mark_layer_presentation_changed(managed_layer)
            store.layer = managed_layer

        # Ask UI consumers to refresh menus/colors based on updated managed layers.
        self.points_layers_merged_requested.emit(tuple(affected_layers))

        # Remove the temporary placeholder layer explicitly by identity.
        QTimer.singleShot(0, lambda ly=layer: self._remove_layer_if_present(ly))

        # General panel refreshes
        self.refresh_layer_status_requested.emit()

        return True

    def _resolve_merge_decision(
        self,
        *,
        new_layer: Any,
        existing_layers: tuple[Any, ...],
        added_keypoints: tuple[str, ...],
        message: str,
    ) -> MergeDecisionResult:
        provider = self._merge_decision_provider
        if provider is None:
            return MergeDecisionResult(disposition=MergeDisposition.KEEP_BOTH)

        req = MergeDecisionRequest(
            new_layer=new_layer,
            existing_layers=existing_layers,
            added_keypoints=added_keypoints,
            message=message,
        )

        try:
            result = provider.resolve_merge(req)
        except Exception:
            logger.debug("Merge decision provider failed; defaulting to KEEP_BOTH", exc_info=True)
            return MergeDecisionResult(disposition=MergeDisposition.KEEP_BOTH)

        if result is None:
            return MergeDecisionResult(disposition=MergeDisposition.KEEP_BOTH)

        return result

    def attach_points_layer_runtime(
        self,
        *,
        layer: Points,
        store: keypoints.KeypointStore,
        controls: Any,
        resolve_layer_by_id: Callable[[int], Points | None],
        get_label_mode: Callable[[], Any],
        schedule_recolor: Callable[[Points], None],
        existing_resources: PointsRuntimeResources | None = None,
    ) -> PointsRuntimeResources:
        """Attach managed runtime behavior to a Points layer.

        Responsibilities
        ----------------
        - bind lifecycle-backed layer resolution to the store
        - bind label-mode getter to the store
        - install paste patch
        - install add wrapper
        - add/connect query_next_frame event
        - install points-layer keybindings

        This helper does NOT:
        - register the runtime in the lifecycle registry
        - own UI/menu setup
        - decide merge/remap/save policy
        """
        resources = existing_resources or PointsRuntimeResources()

        # Narrow lifecycle dependencies injected explicitly.
        store.attach_layer_resolver(resolve_layer_by_id)
        store.set_label_mode_getter(get_label_mode)

        # Copy/paste patch
        if not resources.paste_patch_installed:
            paste_func = make_paste_data(controls, store=store)
            install_paste_patch(layer, paste_func=paste_func)
            resources.paste_patch_installed = True

        # Add layer to store
        if not resources.add_wrapper_installed:
            add_impl = MethodType(keypoints.KeypointStore.add, store)
            install_add_wrapper(layer, add_impl=add_impl, schedule_recolor=schedule_recolor)
            resources.add_wrapper_installed = True

        # layer-specific navigation event
        if not hasattr(layer.events, "query_next_frame"):
            layer.events.add(query_next_frame=Event)
            resources.query_next_frame_event_added = True

        if not resources.query_next_frame_connected:
            try:
                layer.events.query_next_frame.connect(store._advance_step)
                resources.query_next_frame_connected = True
            except Exception:
                pass

        if not resources.keybindings_installed:
            install_points_layer_keybindings(layer, controls, store)
            resources.keybindings_installed = True

        return resources

    # ------------------------------------------------------------------ #
    # Tracking-result layer support                                      #
    # ------------------------------------------------------------------ #

    def tracking_result_metadata(self, layer: Any) -> dict | None:
        """Return tracking-result metadata payload for a Points layer, if present."""
        if layer is None or not isinstance(layer, Points):
            return None

        md = getattr(layer, "metadata", {}) or {}
        payload = md.get(TRACKING_LAYER_METADATA_KEY, None)
        return payload if isinstance(payload, dict) else None

    def is_tracking_result_layer(self, layer: Any) -> bool:
        """
        Authoritative viewer/session-facing predicate for tracking-result layers.

        Notes
        -----
        This delegates the raw metadata check to the tracking module but keeps
        the semantic classification entry point centralized in the lifecycle manager.
        """
        return bool(isinstance(layer, Points) and is_tracking_result_points_layer(layer))

    def tracking_result_source_layer_name(self, layer: Any) -> str | None:
        """Return the recorded source DLC layer name for a tracking-result layer, if known."""
        payload = self.tracking_result_metadata(layer)
        if not payload:
            return None

        name = payload.get("source_layer_name", None)
        if name is None:
            return None

        text = str(name).strip()
        return text or None

    def is_mergeable_dlc_points_layer(self, layer: Any, *, require_managed: bool = False) -> bool:
        """
        Return True if a Points layer is a valid merge target for tracking results.

        Rules
        -----
        - must be a Points layer
        - must not be a tracking-result layer
        - must not be a temporary config placeholder
        - must have a valid DLC header
        - optionally must already be managed by the lifecycle manager
        """
        if layer is None or not isinstance(layer, Points):
            return False

        if self.is_tracking_result_layer(layer):
            return False

        if self.is_config_placeholder_points_layer(layer):
            return False

        if not self.validate_header(layer):
            return False

        if require_managed and not self.is_managed(layer):
            return False

        return True

    def iter_tracking_result_layers(self) -> Iterator[Points]:
        """Iterate live tracking-result Points layers in current viewer order."""
        for layer in self.viewer.layers:
            if isinstance(layer, Points) and self.is_tracking_result_layer(layer):
                yield layer

    def iter_mergeable_dlc_points_layers(
        self,
        *,
        prefer_managed: bool = True,
        managed_only: bool = False,
    ) -> Iterator[Points]:
        """
        Iterate Points layers that are valid tracking-merge targets.

        Parameters
        ----------
        prefer_managed
            If True, yield managed mergeable layers first, then any additional live
            mergeable layers in viewer order.
        managed_only
            If True, only yield managed mergeable layers.
        """
        seen: set[int] = set()

        if prefer_managed or managed_only:
            for layer, _store in self.iter_managed_points():
                if self.is_mergeable_dlc_points_layer(layer, require_managed=True):
                    layer_id = id(layer)
                    if layer_id not in seen:
                        seen.add(layer_id)
                        yield layer

        if managed_only:
            return

        for layer in self.viewer.layers:
            if not isinstance(layer, Points):
                continue
            if id(layer) in seen:
                continue
            if self.is_mergeable_dlc_points_layer(layer, require_managed=False):
                seen.add(id(layer))
                yield layer

    def suggest_merge_target(self, source_layer: Points | None) -> Points | None:
        """
        Suggest the best default DLC merge target for a tracking-result layer.

        Priority
        --------
        1. mergeable managed layer whose name matches tracking source_layer_name
        2. any mergeable live layer whose name matches tracking source_layer_name
        3. currently active mergeable DLC points layer
        4. first mergeable managed DLC points layer
        5. first mergeable live DLC points layer
        """
        if source_layer is None or not isinstance(source_layer, Points):
            return None

        preferred_name = self.tracking_result_source_layer_name(source_layer)

        if preferred_name:
            for layer in self.iter_mergeable_dlc_points_layers(prefer_managed=True, managed_only=True):
                if layer is not source_layer and getattr(layer, "name", None) == preferred_name:
                    return layer

            for layer in self.iter_mergeable_dlc_points_layers(prefer_managed=False, managed_only=False):
                if layer is not source_layer and getattr(layer, "name", None) == preferred_name:
                    return layer

        active = getattr(self.viewer.layers.selection, "active", None)
        if active is not source_layer and self.is_mergeable_dlc_points_layer(active, require_managed=False):
            return active

        for layer in self.iter_mergeable_dlc_points_layers(prefer_managed=True, managed_only=True):
            if layer is not source_layer:
                return layer

        for layer in self.iter_mergeable_dlc_points_layers(prefer_managed=False, managed_only=False):
            if layer is not source_layer:
                return layer

        return None

    # ------------------------------------------------------------------ #
    # Registry facade                                                    #
    # ------------------------------------------------------------------ #

    def is_managed(self, layer: Any) -> bool:
        """Whether this exact currently live layer is registered."""
        return self.registry.is_managed(layer)

    def register_managed_layer(self, layer: Layer, store: KeypointStore, **resources: Any) -> None:
        if isinstance(layer, Points):
            self.register_managed_points_layer(layer, store, **resources)
        else:
            raise ValueError(f"Unsupported layer type for management: {type(layer).__name__}")

    def register_managed_points_layer(self, layer: Points, store: KeypointStore, **resources: Any) -> None:
        """Register a managed Points layer if not already registered."""
        if self.registry.is_managed(layer):
            return

        self.registry.register(
            layer,
            ManagedPointsRuntime(
                layer_id=id(layer),
                store=store,
                resources=dict(resources),
            ),
        )

    def unregister_managed_layer(self, layer_or_id: Any) -> Any | None:
        """Unregister a managed layer by layer object or layer id."""
        runtime = self.registry.unregister(layer_or_id)
        return None if runtime is None else runtime.store

    # ------------------------------------------------------------------ #
    # Event entry points                                                 #
    # ------------------------------------------------------------------ #

    def adopt_existing_layers(self) -> None:
        logger.debug("Lifecycle manager adopting existing layers count=%d", len(self.viewer.layers))

        layers_snapshot = list(self.viewer.layers)
        for idx, layer in enumerate(layers_snapshot):
            self._adopt_layer(layer, idx)

        self.adopted_existing_layers.emit()

    def on_insert(self, event: Any) -> None:
        layer = self._resolve_inserted_layer(event)
        if layer is None:
            logger.debug("Lifecycle manager could not resolve inserted layer for event=%r", event)
            return

        logger.debug(
            "Lifecycle manager processing insert layer=%r type=%s index=%s",
            getattr(layer, "name", layer),
            type(layer).__name__,
            getattr(event, "index", None),
        )

        should_remap = False

        if isinstance(layer, Image):
            should_remap = self._maybe_accept_and_setup_image_layer(
                layer,
                getattr(event, "index", None),
            )
        elif isinstance(layer, Points):
            self._setup_points_layer(layer, allow_merge=True)
            should_remap = True

        if should_remap:
            for layer_ in self.viewer.layers:
                if not isinstance(layer_, Image):
                    self._remap_frame_indices(layer_)

        self.refresh_video_panel_requested.emit()
        self.refresh_layer_status_requested.emit()
        self.layer_insert_processed.emit(layer)

    def on_remove(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        if layer is None:
            logger.debug("Lifecycle manager received remove event without value: %r", event)
            return

        logger.debug(
            "Lifecycle manager processing remove layer=%r type=%s",
            getattr(layer, "name", layer),
            type(layer).__name__,
        )

        with log_timing(
            logger,
            f"on_remove total layer={getattr(layer, 'name', layer)!r}",
            threshold_ms=0.01,
        ):
            self._handle_removed_layer(layer)
            self.layer_remove_processed.emit(layer)

    def _maybe_accept_and_setup_image_layer(self, layer: Image, index: int | None) -> bool:
        if not self.is_dlc_session_image_layer(layer):
            logger.debug(
                "Ignoring non-DLC image layer during lifecycle setup: %r",
                getattr(layer, "name", layer),
            )
            return False

        ok, reason = self.can_accept_dlc_session_image(layer)
        if not ok:
            self._reject_conflicting_dlc_image_layer(
                layer,
                reason or "Conflicting DLC project/video layer",
            )
            return False

        self._setup_image_layer(layer, index, reorder=True)
        return True

    def _adopt_layer(self, layer: Any, index: int) -> None:
        logger.debug(
            "Lifecycle manager adopt layer=%r type=%s index=%s",
            getattr(layer, "name", layer),
            type(layer).__name__,
            index,
        )

        if isinstance(layer, Image):
            self._maybe_accept_and_setup_image_layer(layer, index)
        elif isinstance(layer, Points):
            if not self.registry.is_managed(layer):
                self._setup_points_layer(layer, allow_merge=False)

        if not isinstance(layer, Image):
            self._remap_frame_indices(layer)

    def _resolve_inserted_layer(self, event: Any) -> Any | None:
        # Best case: event carries the inserted value directly
        layer = getattr(event, "value", None)
        if layer is not None:
            return layer

        # Prefer explicit event index over “last item in source”
        index = getattr(event, "index", None)
        if isinstance(index, int):
            try:
                return self.viewer.layers[index]
            except Exception:
                pass

        # Conservative fallback only
        source = getattr(event, "source", None)
        try:
            if source:
                return source[-1]
        except Exception:
            pass

        return None
