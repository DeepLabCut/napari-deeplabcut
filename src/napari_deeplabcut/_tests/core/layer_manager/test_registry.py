from __future__ import annotations

import gc

import pytest

from napari_deeplabcut.core.layer_lifecycle.registry import (
    ManagedPointsRuntime,
    RuntimeRegistry,
)


class DummyLayer:
    pass


def test_registry_register_resolve_require_unregister_roundtrip():
    registry = RuntimeRegistry()
    layer = DummyLayer()
    runtime = ManagedPointsRuntime(layer_id=id(layer), store="store-1")

    registry.register(layer, runtime)

    assert registry.is_managed(layer) is True
    assert registry.resolve_live_layer(layer) is layer
    assert registry.get_live_runtime(layer) is runtime
    assert registry.get_store(layer) == "store-1"
    assert registry.require_store(layer) == "store-1"

    removed = registry.unregister(layer)

    assert removed is runtime
    assert registry.is_managed(layer) is False
    assert registry.resolve_live_layer(layer) is None
    assert registry.get_live_runtime(layer) is None
    assert registry.get_store(layer) is None

    with pytest.raises(KeyError, match="Managed live runtime not found"):
        registry.require_store(layer)


def test_registry_duplicate_registration_raises():
    registry = RuntimeRegistry()
    layer = DummyLayer()

    registry.register(
        layer,
        ManagedPointsRuntime(layer_id=id(layer), store="a"),
    )

    with pytest.raises(ValueError):
        registry.register(
            layer,
            ManagedPointsRuntime(layer_id=id(layer), store="b"),
        )


def test_registry_iter_live_items_and_stores():
    registry = RuntimeRegistry()
    layer1 = DummyLayer()
    layer2 = DummyLayer()

    runtime1 = ManagedPointsRuntime(layer_id=id(layer1), store="s1")
    runtime2 = ManagedPointsRuntime(layer_id=id(layer2), store="s2")

    registry.register(layer1, runtime1)
    registry.register(layer2, runtime2)

    assert list(registry.iter_live_items()) == [(layer1, runtime1), (layer2, runtime2)]

    assert registry.get_store(layer1) == "s1"
    assert registry.get_store(layer2) == "s2"


def test_registry_entry_stops_resolving_when_its_layer_is_collected():
    registry = RuntimeRegistry()

    layer = DummyLayer()
    registry.register(layer, ManagedPointsRuntime(layer_id=id(layer), store="store"))

    layer_id = id(layer)

    # Remove the only strong reference held by the test.
    del layer
    gc.collect()

    # The stale id stays in the index, but stops counting as live.
    assert layer_id in registry.layer_ids()
    assert registry.resolve_live_layer(layer_id) is None
    assert registry.get_live_runtime(layer_id) is None
    assert list(registry.iter_live_items()) == []


def test_registry_register_removes_dead_entry_at_reused_id():
    registry = RuntimeRegistry()

    stale = DummyLayer()
    registry.register(stale, ManagedPointsRuntime(layer_id=id(stale), store="stale"))

    entry = registry._entries_by_id.pop(id(stale))
    layer_b = DummyLayer()
    entry.layer_id = id(layer_b)
    registry._entries_by_id[id(layer_b)] = entry

    del stale
    gc.collect()

    assert registry.is_managed(layer_b) is False

    registry.register(layer_b, ManagedPointsRuntime(layer_id=id(layer_b), store="new"))

    assert registry.is_managed(layer_b) is True
    assert registry.get_store(layer_b) == "new"
