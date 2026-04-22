from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image, Points

from napari_deeplabcut.core.layer_lifecycle.manager import LayerLifecycleManager


class DummySignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    @property
    def callbacks(self):
        return list(self._callbacks)


class DummyLayerEvents:
    def __init__(self):
        self.inserted = DummySignal()
        self.removed = DummySignal()


class DummyLayerList(list):
    def __init__(self, layers=()):
        super().__init__(layers)
        self.events = DummyLayerEvents()


class DummyViewer:
    def __init__(self, layers=()):
        self.layers = DummyLayerList(layers)


class DummyOwner:
    def __init__(self, viewer):
        self.viewer = viewer
        self.calls = []

    def _setup_image_layer(self, layer, index, reorder=True):
        self.calls.append(("setup_image", layer, index, reorder))

    def _setup_points_layer(self, layer, allow_merge=True):
        self.calls.append(("setup_points", layer, allow_merge))

    def _remap_frame_indices(self, layer):
        self.calls.append(("remap", layer))

    def _refresh_video_panel_context(self):
        self.calls.append(("refresh_video",))

    def _refresh_layer_status_panel(self):
        self.calls.append(("refresh_status",))

    def _handle_removed_layer(self, layer):
        self.calls.append(("handle_removed", layer))


def make_image(name="img"):
    layer = Image(np.zeros((5, 5)))
    layer.name = name
    return layer


def make_points(name="pts"):
    layer = Points(np.zeros((0, 3)))
    layer.name = name
    return layer


def test_manager_attach_and_detach_are_idempotent(qtbot):
    viewer = DummyViewer()
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    manager.attach()
    manager.attach()

    assert viewer.layers.events.inserted.callbacks == [manager.on_insert]
    assert viewer.layers.events.removed.callbacks == [manager.on_remove]

    manager.detach()
    manager.detach()

    assert viewer.layers.events.inserted.callbacks == []
    assert viewer.layers.events.removed.callbacks == []


def test_manager_register_and_query_managed_points(qtbot):
    viewer = DummyViewer()
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    pts = make_points()
    store = object()

    assert manager.has_managed_points() is False
    assert manager.managed_points_count() == 0
    assert manager.managed_points_layers() == ()

    manager.register_managed_points_layer(pts, store)

    assert manager.is_managed(pts) is True
    assert manager.has_managed_points() is True
    assert manager.managed_points_count() == 1
    assert manager.managed_points_layers() == (pts,)
    assert list(manager.iter_managed_points()) == [(pts, store)]

    removed = manager.unregister_managed_layer(pts)
    assert removed is store
    assert manager.has_managed_points() is False


def test_manager_on_insert_points_delegates_to_owner_and_remaps_non_images(qtbot):
    img = make_image()
    pts_existing = make_points("existing")
    pts_inserted = make_points("inserted")

    viewer = DummyViewer([img, pts_existing, pts_inserted])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = SimpleNamespace(value=pts_inserted, index=2, source=viewer.layers)

    manager.on_insert(event)

    assert ("setup_points", pts_inserted, True) in owner.calls
    assert ("remap", pts_existing) in owner.calls
    assert ("remap", pts_inserted) in owner.calls
    assert ("refresh_video",) in owner.calls
    assert ("refresh_status",) in owner.calls


def test_manager_on_insert_image_delegates_to_owner(qtbot):
    img = make_image("inserted-image")
    pts = make_points()

    viewer = DummyViewer([img, pts])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = SimpleNamespace(value=img, index=0, source=viewer.layers)

    manager.on_insert(event)

    assert ("setup_image", img, 0, True) in owner.calls
    assert ("remap", pts) in owner.calls
    assert ("refresh_video",) in owner.calls
    assert ("refresh_status",) in owner.calls


def test_manager_adopt_existing_layers_skips_already_managed_points(qtbot):
    img = make_image()
    pts_managed = make_points("managed")
    pts_unmanaged = make_points("unmanaged")

    viewer = DummyViewer([img, pts_managed, pts_unmanaged])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    manager.register_managed_points_layer(pts_managed, object())

    manager.adopt_existing_layers()

    assert ("setup_image", img, 0, True) in owner.calls
    assert ("setup_points", pts_unmanaged, False) in owner.calls
    assert ("setup_points", pts_managed, False) not in owner.calls
    assert ("remap", pts_managed) in owner.calls
    assert ("remap", pts_unmanaged) in owner.calls


def test_manager_on_remove_delegates_to_owner(qtbot):
    pts = make_points()
    viewer = DummyViewer([pts])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = SimpleNamespace(value=pts)

    manager.on_remove(event)

    assert owner.calls == [("handle_removed", pts)]


@pytest.mark.parametrize(
    ("event_factory", "expected_name"),
    [
        (lambda viewer, img, pts: SimpleNamespace(value=pts, index=1, source=viewer.layers), "pts"),
        (lambda viewer, img, pts: SimpleNamespace(index=1, source=viewer.layers), "pts"),
        (lambda viewer, img, pts: SimpleNamespace(source=[img, pts]), "pts"),
    ],
)
def test_manager_resolve_inserted_layer_prefers_value_then_index_then_source(qtbot, event_factory, expected_name):
    img = make_image("img")
    pts = make_points("pts")

    viewer = DummyViewer([img, pts])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = event_factory(viewer, img, pts)

    layer = manager._resolve_inserted_layer(event)

    assert layer is pts
    assert layer.name == expected_name
