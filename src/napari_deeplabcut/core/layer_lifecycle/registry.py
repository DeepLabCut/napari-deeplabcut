# src/napari_deeplabcut/core/layer_lifecycle/registry.py
from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logger = logging.getLogger("napari-deeplabcut.lifecycle.registry")
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

    query_next_frame_event_added: bool = False
    query_next_frame_connected: bool = False
    add_wrapper_installed: bool = False
    paste_patch_installed: bool = False
    keybindings_installed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClearedRegistryEntry(Generic[StoreT]):
    """A registry entry that was removed because its layer was no longer live."""

    layer_id: int
    runtime: ManagedPointsRuntime[StoreT]


@dataclass(frozen=True, slots=True)
class RegistryAuditIssue:
    code: str
    message: str
    layer_id: int | None = None


@dataclass(frozen=True, slots=True)
class RegistryAuditReport:
    live_count: int
    dead_count: int
    issues: tuple[RegistryAuditIssue, ...]


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

    def __len__(self) -> int:
        """Number of currently live entries."""
        return sum(1 for _layer, _runtime in self.iter_live_items())

    def layer_ids(self) -> tuple[int, ...]:
        """All currently registered entry ids, including stale/dead ones."""
        return tuple(self._entries_by_id.keys())

    def is_managed(self, layer: Any) -> bool:
        """Whether this exact live layer object is currently registered and live."""
        entry = self._entries_by_id.get(id(layer))
        if entry is None:
            return False
        resolved = entry.resolve_layer()
        return resolved is layer

    def contains_layer_id(self, layer_id: int) -> bool:
        """Whether a registry entry exists for this id (live or stale)."""
        return layer_id in self._entries_by_id

    def register(self, layer: Any, runtime: ManagedPointsRuntime[StoreT]) -> None:
        layer_id = id(layer)
        if layer_id in self._entries_by_id:
            raise ValueError(f"Layer already registered: id={layer_id}")

        if runtime.layer_id != layer_id:
            raise ValueError(f"Runtime layer_id mismatch: runtime.layer_id={runtime.layer_id}, actual={layer_id}")

        try:
            layer_ref: weakref.ReferenceType[Any] | None = weakref.ref(layer)
            strong_layer = None
        except TypeError:
            logger.error("Could not cleanly register layer as a weakref; storing strong reference instead: %r", layer)
            # Fallback for objects that do not support weakref.
            # This means the registry *will* strongly hold such layers.
            layer_ref = None
            strong_layer = layer

        self._entries_by_id[layer_id] = _RegistryEntry(
            layer_id=layer_id,
            layer_ref=layer_ref,
            strong_layer=strong_layer,
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

    def require_live_runtime(self, layer_or_id: Any) -> ManagedPointsRuntime[StoreT]:
        runtime = self.get_live_runtime(layer_or_id)
        if runtime is None:
            raise KeyError(f"Managed live runtime not found: {layer_or_id!r}")
        return runtime

    def get_store(self, layer_or_id: Any) -> StoreT | None:
        runtime = self.get_live_runtime(layer_or_id)
        return None if runtime is None else runtime.store

    def require_store(self, layer_or_id: Any) -> StoreT:
        runtime = self.require_live_runtime(layer_or_id)
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

    def iter_live_layers(self) -> Iterator[Any]:
        for layer, _runtime in self.iter_live_items():
            yield layer

    def iter_live_runtimes(self) -> Iterator[ManagedPointsRuntime[StoreT]]:
        for _layer, runtime in self.iter_live_items():
            yield runtime

    # ------------------------------------------------------------------ #
    # dead-entry handling / reporting                                    #
    # ------------------------------------------------------------------ #

    def dead_layer_ids(self) -> tuple[int, ...]:
        """Return ids whose registered layer object is no longer live."""
        dead: list[int] = []
        for layer_id, entry in self._entries_by_id.items():
            if entry.resolve_layer() is None:
                dead.append(layer_id)
        return tuple(dead)

    def clear_dead_entries(self, *, log: bool = True) -> tuple[ClearedRegistryEntry[StoreT], ...]:
        """Remove dead entries and return what was reaped.

        This is intentionally observable so lifecycle cleanup bugs are not silently hidden.
        """
        reaped: list[ClearedRegistryEntry[StoreT]] = []

        for layer_id in list(self.dead_layer_ids()):
            entry = self._entries_by_id.pop(layer_id, None)
            if entry is None:
                continue

            item = ClearedRegistryEntry(layer_id=layer_id, runtime=entry.runtime)
            reaped.append(item)

            if log:
                logger.warning(
                    "Cleared dead managed layer entry without explicit unregister: layer_id=%s",
                    layer_id,
                )

        return tuple(reaped)

    # ------------------------------------------------------------------ #
    # diagnostics / auditing                                             #
    # ------------------------------------------------------------------ #

    def audit(self) -> RegistryAuditReport:
        issues: list[RegistryAuditIssue] = []
        live_count = 0
        dead_count = 0
        seen_ids: set[int] = set()

        for layer_id, entry in self._entries_by_id.items():
            if layer_id in seen_ids:
                issues.append(
                    RegistryAuditIssue(
                        code="duplicate-layer-id",
                        message="Duplicate registry layer id detected",
                        layer_id=layer_id,
                    )
                )
            seen_ids.add(layer_id)

            resolved = entry.resolve_layer()
            if resolved is None:
                dead_count += 1
                issues.append(
                    RegistryAuditIssue(
                        code="dead-entry",
                        message="Managed entry has no live layer",
                        layer_id=layer_id,
                    )
                )
                continue

            live_count += 1

            if entry.runtime.layer_id != layer_id:
                issues.append(
                    RegistryAuditIssue(
                        code="runtime-layer-id-mismatch",
                        message=(
                            f"Runtime layer_id ({entry.runtime.layer_id}) does not match registry entry id ({layer_id})"
                        ),
                        layer_id=layer_id,
                    )
                )

        return RegistryAuditReport(
            live_count=live_count,
            dead_count=dead_count,
            issues=tuple(issues),
        )

    def assert_consistent(self) -> None:
        report = self.audit()
        assert not report.issues, f"Registry consistency issues: {report.issues!r}"

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        self._entries_by_id.clear()

    @staticmethod
    def _coerce_layer_id(layer_or_id: Any) -> int:
        if isinstance(layer_or_id, int):
            return layer_or_id
        return id(layer_or_id)
