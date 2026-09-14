# src/napari_deeplabcut/core/layer_lifecycle/registry.py
from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ...napari_compat.proxy import unwrap

if TYPE_CHECKING:
    from napari.layers import Points

    from ..keypoints import KeypointStore

logger = logging.getLogger(__name__)
StoreT = TypeVar("StoreT")


@dataclass(slots=True)
class ManagedPointsRuntime(Generic[StoreT]):
    """Runtime attachment for a managed Points layer.

    Important:
    ----------
    This runtime should not strongly store the layer object itself.

    It stores:
    - the stable layer identity used by the registry (`layer_id`)
    - the store (KeypointStore) associated with this layer
    - generic resources for future cleanup hooks / attachments

    Layer liveness / resolution is owned by the registry.
    Everything downstream should use it to resolve whether the layer is still live.
    This is meant to let napari own true layer lifecycles without interference,
    while still enabling robust cleanup of plugin-managed
    runtime attachments when layers are removed.
    """

    layer_id: int
    store: StoreT
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PointsRuntimeResources:
    """Non-Qt runtime attachments installed on a managed Points layer.

    It gives the lifecycle manager one place to record what it attached and later clean up or audit.
    Intended to fit in ManagedPointsRuntime.resources.
    """

    query_next_frame_connected: bool = False
    keybindings_installed: bool = False


@dataclass
class PointsLayerSetupRequest:
    layer: Points
    store: KeypointStore
    existing_resources: PointsRuntimeResources | None = None
    runtime_resources: PointsRuntimeResources | None = None


@dataclass(slots=True)
class _RegistryEntry(Generic[StoreT]):
    layer_id: int
    layer_ref: weakref.ReferenceType[Any]
    runtime: ManagedPointsRuntime[StoreT]

    def resolve_layer(self) -> Any:
        return self.layer_ref()


class RuntimeRegistry(Generic[StoreT]):
    """Single owner of managed runtime attachments.

    Invariants
    ----------
    - At most one runtime bundle per registered layer identity.
    - Registration is explicit.
    - Layer liveness is resolved here.
    - Runtime attachments do not need to strongly own the layer object.
    """

    def __init__(self) -> None:
        self._entries_by_id: dict[int, _RegistryEntry[StoreT]] = {}

    # ------------------------------------------------------------------ #
    # core identity / registration                                       #
    # ------------------------------------------------------------------ #

    def layer_ids(self) -> tuple[int, ...]:
        """All currently registered entry ids, including stale/dead ones."""
        return tuple(self._entries_by_id.keys())

    def is_managed(self, layer: Any) -> bool:
        """Whether this exact live layer object is currently registered and live."""
        target = unwrap(layer)
        entry = self._entries_by_id.get(id(target))
        if entry is None:
            return False
        resolved = entry.resolve_layer()
        return resolved is target

    def register(self, layer: Any, runtime: ManagedPointsRuntime[StoreT]) -> None:
        # A PublicOnlyProxy and the layer it wraps have different id()s,
        # so a caller holding either must register and resolve the same entry.
        layer = unwrap(layer)
        layer_id = id(layer)

        existing = self._entries_by_id.get(layer_id)
        if existing is not None:
            if existing.resolve_layer() is not None:
                raise ValueError(f"Layer already registered and live: id={layer_id}")
            logger.warning(f"Removed dead layer entry with id={layer_id}")
            del self._entries_by_id[layer_id]

        if runtime.layer_id != layer_id:
            raise ValueError(f"Runtime layer_id mismatch: runtime.layer_id={runtime.layer_id}, actual={layer_id}")

        layer_ref: weakref.ReferenceType[Any] | None = weakref.ref(layer)

        self._entries_by_id[layer_id] = _RegistryEntry(
            layer_id=layer_id,
            layer_ref=layer_ref,
            runtime=runtime,
        )

    def unregister(self, layer_or_id: Any) -> ManagedPointsRuntime[StoreT] | None:
        entry = self._entries_by_id.pop(self._coerce_layer_id(layer_or_id), None)
        return None if entry is None else entry.runtime

    # ------------------------------------------------------------------ #
    # centralized live resolution                                        #
    # ------------------------------------------------------------------ #

    def resolve_live_layer(self, layer_or_id: Any) -> Any | None:
        """Resolve a currently live layer object from a layer or layer id."""
        entry = self._entries_by_id.get(self._coerce_layer_id(layer_or_id))
        if entry is None:
            return None
        return entry.resolve_layer()

    def get_live_runtime(self, layer_or_id: Any) -> ManagedPointsRuntime[StoreT] | None:
        """Return runtime only if the corresponding layer is currently live."""
        entry = self._entries_by_id.get(self._coerce_layer_id(layer_or_id))
        if entry is None:
            return None
        if entry.resolve_layer() is None:
            return None
        return entry.runtime

    def get_store(self, layer_or_id: Any) -> StoreT | None:
        runtime = self.get_live_runtime(layer_or_id)
        return None if runtime is None else runtime.store

    def require_store(self, layer_or_id: Any) -> StoreT:
        runtime = self.get_live_runtime(layer_or_id)
        if runtime is None:
            raise KeyError(f"Managed live runtime not found: {layer_or_id!r}")
        return runtime.store

    # ------------------------------------------------------------------ #
    # live iteration                                                     #
    # ------------------------------------------------------------------ #

    def iter_live_items(self) -> Iterator[tuple[Any, ManagedPointsRuntime[StoreT]]]:
        """Yield only currently live (layer, runtime) pairs."""
        for entry in list(self._entries_by_id.values()):
            layer = entry.resolve_layer()
            if layer is not None:
                yield layer, entry.runtime

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_layer_id(layer_or_id: Any) -> int:
        if isinstance(layer_or_id, int):
            return layer_or_id
        return id(unwrap(layer_or_id))
