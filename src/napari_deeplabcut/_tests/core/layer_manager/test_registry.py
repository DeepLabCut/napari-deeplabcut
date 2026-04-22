from __future__ import annotations

import gc

import pytest

from napari_deeplabcut.core.layer_lifecycle.registry import (
    ManagedPointsRuntime,
    RuntimeRegistry,
)


class DummyLayer:
    pass


def test_registry_register_get_require_unregister_roundtrip():
    registry = RuntimeRegistry()
    layer = DummyLayer()
    runtime = ManagedPointsRuntime(layer=layer, store="store-1")

    registry.register(layer, runtime)

    assert registry.is_managed(layer) is True
    assert registry.get(layer) is runtime
    assert registry.require(layer) is runtime
    assert len(registry) == 1

    removed = registry.unregister(layer)

    assert removed is runtime
    assert registry.is_managed(layer) is False
    assert registry.get(layer) is None
    assert len(registry) == 0


def test_registry_duplicate_registration_raises():
    registry = RuntimeRegistry()
    layer = DummyLayer()

    registry.register(layer, ManagedPointsRuntime(layer=layer, store="a"))

    with pytest.raises(ValueError):
        registry.register(layer, ManagedPointsRuntime(layer=layer, store="b"))


def test_registry_items_layers_runtimes_are_consistent():
    registry = RuntimeRegistry()
    layer1 = DummyLayer()
    layer2 = DummyLayer()

    runtime1 = ManagedPointsRuntime(layer=layer1, store="s1")
    runtime2 = ManagedPointsRuntime(layer=layer2, store="s2")

    registry.register(layer1, runtime1)
    registry.register(layer2, runtime2)

    items = list(registry.items())
    layers = list(registry.layers())
    runtimes = list(registry.runtimes())

    assert items == [(layer1, runtime1), (layer2, runtime2)]
    assert layers == [layer1, layer2]
    assert runtimes == [runtime1, runtime2]
    assert set(registry.layer_ids()) == {id(layer1), id(layer2)}

    registry.assert_consistent()


def test_registry_purges_dead_weakrefs():
    registry = RuntimeRegistry()

    layer = DummyLayer()
    runtime = ManagedPointsRuntime(layer=layer, store="store")
    registry.register(layer, runtime)

    assert len(registry) == 1

    layer_id = id(layer)
    del layer
    gc.collect()

    registry.purge_dead()

    assert layer_id not in registry.layer_ids()
    assert len(registry) == 0
