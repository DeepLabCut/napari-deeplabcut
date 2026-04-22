from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from napari.layers import Image, Layer, Points
from qtpy.QtCore import QObject, QTimer, Signal

from .registry import (
    ClearedRegistryEntry,
    ManagedPointsRuntime,
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

    def __init__(self, owner: Any) -> None:
        if isinstance(owner, QObject):
            super().__init__(parent=owner)
        else:
            super().__init__()

        self.owner = owner
        self.viewer = owner.viewer
        self.registry: RuntimeRegistry[Any] = RuntimeRegistry()

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
            self.owner._setup_image_layer(layer, getattr(event, "index", None), reorder=True)
        elif isinstance(layer, Points):
            self.owner._setup_points_layer(layer, allow_merge=True)

        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self.owner._remap_frame_indices(layer_)

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

        self.owner._handle_removed_layer(layer)
        self.layer_remove_processed.emit(layer)

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _adopt_layer(self, layer: Any, index: int) -> None:
        logger.debug(
            "Lifecycle manager adopt layer=%r type=%s index=%s",
            getattr(layer, "name", layer),
            type(layer).__name__,
            index,
        )

        if isinstance(layer, Image):
            self.owner._setup_image_layer(layer, index, reorder=True)
        elif isinstance(layer, Points):
            if not self.registry.is_managed(layer):
                self.owner._setup_points_layer(layer, allow_merge=False)

        if not isinstance(layer, Image):
            self.owner._remap_frame_indices(layer)

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
