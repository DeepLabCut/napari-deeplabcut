from __future__ import annotations

import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

StoreT = TypeVar("StoreT")


@dataclass(slots=True)
class ManagedPointsRuntime(Generic[StoreT]):
    """Runtime attachment for a managed Points layer.

    This wraps:
    - the layer
    - its store
    - a generic resources dict for future cleanup hooks / attachments
    """

    layer: Any
    store: StoreT
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RegistryEntry(Generic[StoreT]):
    layer_id: int
    layer_ref: weakref.ReferenceType[Any] | None
    strong_layer: Any | None
    runtime: ManagedPointsRuntime[StoreT]

    def resolve_layer(self) -> Any | None:
        if self.layer_ref is not None:
            return self.layer_ref()
        return self.strong_layer


class RuntimeRegistry(Generic[StoreT]):
    """Single owner of managed Points-layer runtime attachments.

    Invariants
    ----------
    - At most one runtime bundle per live layer object.
    - Registration is explicit.
    - Querying is by object identity.
    """

    def __init__(self) -> None:
        self._entries_by_id: dict[int, _RegistryEntry[StoreT]] = {}

    def __len__(self) -> int:
        self.purge_dead()
        return len(self._entries_by_id)

    def is_managed(self, layer: Any) -> bool:
        self.purge_dead()
        return id(layer) in self._entries_by_id

    def register(self, layer: Any, runtime: ManagedPointsRuntime[StoreT]) -> None:
        self.purge_dead()
        layer_id = id(layer)
        if layer_id in self._entries_by_id:
            raise ValueError(f"Layer already registered: id={layer_id}")

        try:
            layer_ref: weakref.ReferenceType[Any] | None = weakref.ref(layer)
            strong_layer = None
        except TypeError:
            layer_ref = None
            strong_layer = layer

        self._entries_by_id[layer_id] = _RegistryEntry(
            layer_id=layer_id,
            layer_ref=layer_ref,
            strong_layer=strong_layer,
            runtime=runtime,
        )

    def unregister(self, layer: Any) -> ManagedPointsRuntime[StoreT] | None:
        self.purge_dead()
        entry = self._entries_by_id.pop(id(layer), None)
        return None if entry is None else entry.runtime

    def get(self, layer: Any) -> ManagedPointsRuntime[StoreT] | None:
        self.purge_dead()
        entry = self._entries_by_id.get(id(layer))
        return None if entry is None else entry.runtime

    def require(self, layer: Any) -> ManagedPointsRuntime[StoreT]:
        runtime = self.get(layer)
        if runtime is None:
            raise KeyError(f"Layer is not managed: id={id(layer)}")
        return runtime

    def items(self) -> Iterator[tuple[Any, ManagedPointsRuntime[StoreT]]]:
        self.purge_dead()
        for entry in list(self._entries_by_id.values()):
            layer = entry.resolve_layer()
            if layer is not None:
                yield layer, entry.runtime

    def layers(self) -> Iterator[Any]:
        for layer, _runtime in self.items():
            yield layer

    def runtimes(self) -> Iterator[ManagedPointsRuntime[StoreT]]:
        self.purge_dead()
        for entry in self._entries_by_id.values():
            yield entry.runtime

    def layer_ids(self) -> tuple[int, ...]:
        self.purge_dead()
        return tuple(self._entries_by_id.keys())

    def purge_dead(self) -> None:
        dead_ids: list[int] = []
        for layer_id, entry in self._entries_by_id.items():
            if entry.resolve_layer() is None:
                dead_ids.append(layer_id)

        for layer_id in dead_ids:
            self._entries_by_id.pop(layer_id, None)

    def clear(self) -> None:
        self._entries_by_id.clear()

    def assert_consistent(self) -> None:
        self.purge_dead()
        seen_ids: set[int] = set()

        for layer, runtime in self.items():
            assert runtime.layer is layer, "Runtime bundle layer reference is inconsistent"
            layer_id = id(layer)
            assert layer_id not in seen_ids, "Duplicate managed layer identity detected"
            seen_ids.add(layer_id)
