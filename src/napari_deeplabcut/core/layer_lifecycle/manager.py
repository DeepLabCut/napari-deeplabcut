from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Image, Layer, Points, Tracks
from napari.utils.events import Event
from napari.utils.history import update_save_history
from qtpy.QtCore import QObject, QSignalBlocker, QTimer, Signal

from ...config.keybinds import install_points_layer_keybindings
from ...config.models import ImageMetadata, PointsMetadata
from ...core import keypoints
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
from ...ui.cropping import resolve_project_path_from_image_layer
from .registry import (
    ClearedRegistryEntry,
    ManagedPointsRuntime,
    PointsRuntimeResources,
    RegistryAuditReport,
    RuntimeRegistry,
)

if TYPE_CHECKING:
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

    adopted_existing_layers = Signal()
    layer_insert_processed = Signal(object)
    layer_remove_processed = Signal(object)

    # Session management
    session_conflict_rejected = Signal(str)  # if a new DLC folder is loaded on top of the current one

    def __init__(self, owner: Any) -> None:
        if isinstance(owner, QObject):
            super().__init__(parent=owner)
        else:
            super().__init__()

        self.owner = owner
        self.viewer = owner.viewer
        self.registry: RuntimeRegistry[Any] = RuntimeRegistry()

        # Lifecycle-owned viewer/image context
        self._active_dlc_image_layer_id: int | None = None
        self._image_meta = ImageMetadata()
        self._project_path: str | None = None

        self._initial_adopt_timer = QTimer(self)
        self._initial_adopt_timer.setSingleShot(True)
        self._initial_adopt_timer.timeout.connect(self.adopt_existing_layers)

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

    def _update_image_meta_from_layer(self, layer: Image) -> None:
        """Update lifecycle-owned image metadata from an Image layer."""
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

        self._image_meta = merged

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

        self.owner._refresh_video_panel_context()

    def _setup_image_layer(self, layer: Image, index: int | None = None, *, reorder: bool = True) -> None:
        """Lifecycle-owned setup for an inserted/adopted Image layer."""
        md = layer.metadata or {}
        paths = md.get("paths")
        if paths is None:
            try:
                if self.owner.io.is_video(layer.name):
                    self.owner.video_widget.setVisible(True)
            except Exception:
                pass

        self._active_dlc_image_layer_id = id(layer)
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
        self.owner._refresh_video_panel_context()

        logger.debug(
            "Setup image layer=%r index=%s reorder=%s paths_count=%s root=%r",
            getattr(layer, "name", layer),
            index,
            reorder,
            len(md.get("paths") or []),
            md.get("root"),
        )

        if reorder and index is not None:
            QTimer.singleShot(10, lambda ly=layer: self.owner._move_image_layer_to_bottom(ly))

    def _wire_points_layer(self, layer: Points) -> KeypointStore | None:
        """Lifecycle-owned wiring of a managed Points layer.

        Transitional note:
        ------------------
        The manager owns runtime attachment; the widget still owns UI completion hooks.
        """
        if not self.owner._validate_header(layer):
            return None

        existing = getattr(layer, "_dlc_store", None)
        if existing is not None:
            self.register_managed_points_layer(layer, existing)

            runtime = self.get_live_runtime(layer)
            existing_resources = None
            if runtime is not None:
                existing_resources = runtime.resources.get("points_runtime", None)

            resources = self.attach_points_layer_runtime(
                layer=layer,
                store=existing,
                controls=self.owner,
                resolve_layer_by_id=self.resolve_live_layer,
                get_label_mode=lambda: self.owner._label_mode,
                schedule_recolor=self.owner._schedule_recolor,
                existing_resources=existing_resources,
            )

            runtime = self.get_live_runtime(layer)
            if runtime is not None:
                runtime.resources["points_runtime"] = resources

            layer._dlc_controls = self.owner
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

        resources = self.attach_points_layer_runtime(
            layer=layer,
            store=store,
            controls=self.owner,
            resolve_layer_by_id=self.resolve_live_layer,
            get_label_mode=lambda: self.owner._label_mode,
            schedule_recolor=self.owner._schedule_recolor,
        )

        runtime = self.get_live_runtime(layer)
        if runtime is not None:
            runtime.resources["points_runtime"] = resources

        layer._dlc_store = store
        layer._dlc_controls = self.owner

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

        if self.managed_points_count() == 1 and self.owner._is_multianimal(layer):
            self.owner._color_mode = keypoints.ColorMode.INDIVIDUAL
            for btn in self.owner._color_mode_selector.buttons():
                if btn.text().lower() == str(self.owner._color_mode).lower():
                    btn.setChecked(True)
                    break

        # Transitional: widget still owns these UI-adjacent helpers.
        self.owner._maybe_initialize_layer_point_size_from_config(layer)
        self.owner._connect_layer_status_events(layer)

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
        if not self.owner._validate_header(layer):
            return

        # Transitional: merge policy still lives in the widget for now.
        if allow_merge:
            consumed = self.owner._maybe_merge_config_points_layer(layer)
            if consumed:
                logger.debug(
                    "Consumed temporary config placeholder layer=%r during merge path",
                    getattr(layer, "name", layer),
                )
                return

        store = self._wire_points_layer(layer)
        if store is None:
            return

        # Widget owns UI completion only.
        self.owner._complete_points_layer_ui_setup(layer, store)

        logger.debug(
            "Setup points layer=%r allow_merge=%s metadata_keys=%s",
            getattr(layer, "name", layer),
            allow_merge,
            sorted((layer.metadata or {}).keys()),
        )

    def _handle_removed_layer(self, layer: Any) -> None:
        """Lifecycle-owned remove handling.

        Transitional note:
        ------------------
        UI/menu cleanup still delegates to a widget-owned UI hook.
        """
        n_points_layer = sum(isinstance(l, Points) for l in self.viewer.layers)

        if isinstance(layer, Points):
            self.unregister_managed_layer(layer)
            self.owner._on_points_layer_removed_ui(layer, remaining_points_layers=n_points_layer)

        elif isinstance(layer, Image):
            if self._active_dlc_image_layer_id == id(layer):
                self._active_dlc_image_layer_id = None
                self._image_meta = ImageMetadata()
                self._project_path = None

                paths = layer.metadata.get("paths")
                if paths is None:
                    self.owner.video_widget.setVisible(False)
            else:
                logger.debug(
                    "Removed non-session or inactive image layer=%r; keeping current DLC session context.",
                    getattr(layer, "name", layer),
                )

        elif isinstance(layer, Tracks):
            was_trails = self.owner._trails_controller.on_tracks_layer_removed(layer)
            if was_trails:
                with QSignalBlocker(self.owner._trail_cb):
                    self.owner._trail_cb.setChecked(False)

        self.owner._refresh_video_panel_context()
        self.owner._refresh_layer_status_panel()

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

        if isinstance(layer, Image):
            self._maybe_accept_and_setup_image_layer(layer, getattr(event, "index", None))
        elif isinstance(layer, Points):
            self._setup_points_layer(layer, allow_merge=True)

        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self._remap_frame_indices(layer_)

        self.owner._refresh_video_panel_context()
        self.owner._refresh_layer_status_panel()
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
