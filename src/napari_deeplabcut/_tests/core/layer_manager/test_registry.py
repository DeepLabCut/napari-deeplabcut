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
    assert registry.require_live_runtime(layer) is runtime
    assert registry.get_store(layer) == "store-1"
    assert registry.require_store(layer) == "store-1"
    assert len(registry) == 1

    removed = registry.unregister(layer)

    assert removed is runtime
    assert registry.is_managed(layer) is False
    assert registry.resolve_live_layer(layer) is None
    assert registry.get_live_runtime(layer) is None
    assert registry.get_store(layer) is None
    assert len(registry) == 0


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


def test_registry_iter_live_items_layers_runtimes_are_consistent():
    registry = RuntimeRegistry()
    layer1 = DummyLayer()
    layer2 = DummyLayer()

    runtime1 = ManagedPointsRuntime(layer_id=id(layer1), store="s1")
    runtime2 = ManagedPointsRuntime(layer_id=id(layer2), store="s2")

    registry.register(layer1, runtime1)
    registry.register(layer2, runtime2)

    items = list(registry.iter_live_items())
    layers = list(registry.iter_live_layers())
    runtimes = list(registry.iter_live_runtimes())

    assert items == [(layer1, runtime1), (layer2, runtime2)]
    assert layers == [layer1, layer2]
    assert runtimes == [runtime1, runtime2]
    assert set(registry.layer_ids()) == {id(layer1), id(layer2)}

    assert registry.get_store(layer1) == "s1"
    assert registry.get_store(layer2) == "s2"

    registry.assert_consistent()


def test_registry_reap_dead_entries_removes_stale_entry_and_reports_it():
    registry = RuntimeRegistry()

    layer = DummyLayer()
    runtime = ManagedPointsRuntime(layer_id=id(layer), store="store")
    registry.register(layer, runtime)

    assert len(registry) == 1
    layer_id = id(layer)

    # Remove the only strong reference held by the test.
    del layer
    gc.collect()

    # Before reaping, the stale id may still be present in the registry index,
    # but it should no longer count as live.
    assert len(registry) == 0
    assert layer_id in registry.layer_ids()

    reaped = registry.clear_dead_entries(log=False)

    assert len(reaped) == 1
    assert reaped[0].layer_id == layer_id
    assert reaped[0].runtime is runtime

    assert layer_id not in registry.layer_ids()
    assert len(registry) == 0
    assert list(registry.iter_live_items()) == []


def test_registry_audit_reports_dead_entry_before_reap():
    registry = RuntimeRegistry()

    layer = DummyLayer()
    runtime = ManagedPointsRuntime(layer_id=id(layer), store="store")
    registry.register(layer, runtime)

    layer_id = id(layer)
    del layer
    gc.collect()

    report = registry.audit()

    assert report.live_count == 0
    assert report.dead_count == 1
    assert any(issue.code == "dead-entry" and issue.layer_id == layer_id for issue in report.issues)

    reaped = registry.clear_dead_entries(log=False)
    assert len(reaped) == 1

    report_after = registry.audit()
    assert report_after.live_count == 0
    assert report_after.dead_count == 0
    assert report_after.issues == ()
