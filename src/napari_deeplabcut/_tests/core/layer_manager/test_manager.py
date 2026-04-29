from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image, Points

from napari_deeplabcut.core.layer_lifecycle import LayerLifecycleManager


def mark_as_dlc_session_image(layer, *, role="image"):
    layer.metadata = dict(layer.metadata or {})
    layer.metadata["dlc"] = {
        "session_role": role,
        "project_context": {
            "root_anchor": "C:/project/labeled-data/test",
            "project_root": "C:/project",
            "config_path": "C:/project/config.yaml",
            "dataset_folder": "C:/project/labeled-data/test",
        },
        "session_key": "C:/project",
    }
    return layer


# ---------------------------------------------------------------------------
# Minimal fake viewer/layer event infrastructure
# ---------------------------------------------------------------------------
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


class DummyImageMeta:
    def __init__(self):
        self.root = None
        self.paths = None

    def model_dump(self, **kwargs):
        return {}


class SignalRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)

    @property
    def count(self):
        return len(self.calls)


def connect_signal_recorders(manager):
    rec = SimpleNamespace(
        refresh_video=SignalRecorder(),
        refresh_status=SignalRecorder(),
        setup_points=SignalRecorder(),
        merged_points=SignalRecorder(),
        removed_points=SignalRecorder(),
        removed_tracks=SignalRecorder(),
        move_image_bottom=SignalRecorder(),
        video_visibility=SignalRecorder(),
        adopted=SignalRecorder(),
        inserted=SignalRecorder(),
        removed=SignalRecorder(),
        conflicts=SignalRecorder(),
    )

    manager.refresh_video_panel_requested.connect(rec.refresh_video)
    manager.refresh_layer_status_requested.connect(rec.refresh_status)
    manager.points_layer_setup_requested.connect(rec.setup_points)
    manager.points_layers_merged_requested.connect(rec.merged_points)
    manager.points_layer_removed_requested.connect(rec.removed_points)
    manager.tracks_layer_removed_requested.connect(rec.removed_tracks)
    manager.move_image_layer_to_bottom_requested.connect(rec.move_image_bottom)
    manager.video_widget_visibility_requested.connect(rec.video_visibility)
    manager.adopted_existing_layers.connect(rec.adopted)
    manager.layer_insert_processed.connect(rec.inserted)
    manager.layer_remove_processed.connect(rec.removed)
    manager.session_conflict_rejected.connect(rec.conflicts)
    return rec


# ---------------------------------------------------------------------------
# Fake store used so manager tests do not depend on real KeypointStore/viewer
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, viewer, layer):
        self.viewer = viewer
        self._layer = layer
        self._layer_id = id(layer)
        self._resolver = None
        self._get_label_mode = None

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer
        self._layer_id = id(layer)

    @property
    def layer_id(self):
        return self._layer_id

    def attach_layer_resolver(self, resolver):
        self._resolver = resolver

    def set_label_mode_getter(self, getter):
        self._get_label_mode = getter

    def _advance_step(self, event=None):
        return None

    def add(self, coord):
        return None


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def make_image(name="img"):
    layer = Image(np.zeros((5, 5)))
    layer.name = name
    return layer


def make_points(name="pts"):
    layer = Points(np.zeros((0, 3)))
    layer.name = name
    return layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def immediate_qtimer(monkeypatch):
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    monkeypatch.setattr(
        manager_module.QTimer,
        "singleShot",
        staticmethod(lambda _ms, fn: fn()),
    )


@pytest.fixture
def fake_store(monkeypatch):
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    monkeypatch.setattr(manager_module.keypoints, "KeypointStore", FakeStore)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_manager_attach_and_detach_are_idempotent(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

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
    manager = LayerLifecycleManager(viewer=viewer)

    pts = make_points()
    store = object()

    assert manager.has_managed_points() is False
    assert manager.managed_points_count() == 0
    assert manager.managed_points_layers() == ()
    assert manager.resolve_live_layer(pts) is None
    assert manager.get_live_runtime(pts) is None
    assert manager.get_store(pts) is None

    manager.register_managed_points_layer(pts, store)

    assert manager.is_managed(pts) is True
    assert manager.has_managed_points() is True
    assert manager.managed_points_count() == 1
    assert manager.managed_points_layers() == (pts,)
    assert list(manager.iter_managed_points()) == [(pts, store)]

    assert manager.resolve_live_layer(pts) is pts
    runtime = manager.get_live_runtime(pts)
    assert runtime is not None
    assert runtime.layer_id == id(pts)
    assert runtime.store is store
    assert manager.get_store(pts) is store
    assert manager.require_store(pts) is store

    removed = manager.unregister_managed_layer(pts)
    assert removed is store

    assert manager.is_managed(pts) is False
    assert manager.has_managed_points() is False
    assert manager.managed_points_count() == 0
    assert manager.managed_points_layers() == ()
    assert manager.resolve_live_layer(pts) is None
    assert manager.get_live_runtime(pts) is None
    assert manager.get_store(pts) is None


def test_manager_on_insert_points_sets_up_points_and_refreshes_ui(
    qtbot,
    fake_store,
    monkeypatch,
):
    img = make_image()
    pts_existing = make_points("existing")
    pts_inserted = make_points("inserted")

    viewer = DummyViewer([img, pts_existing, pts_inserted])
    manager = LayerLifecycleManager(viewer=viewer)
    rec = connect_signal_recorders(manager)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    event = SimpleNamespace(value=pts_inserted, index=2, source=viewer.layers)

    manager.on_insert(event)

    assert manager.is_managed(pts_inserted) is True

    assert rec.setup_points.count == 1
    req = rec.setup_points.calls[0][0]
    assert req.layer is pts_inserted
    assert req.store is manager.get_store(pts_inserted)

    assert rec.refresh_video.count >= 1
    assert rec.refresh_status.count >= 1
    assert rec.inserted.count == 1
    assert rec.inserted.calls[0][0] is pts_inserted

    assert pts_existing in remap_calls
    assert pts_inserted in remap_calls


def test_manager_on_insert_image_updates_context_and_refreshes_ui(qtbot):
    img = mark_as_dlc_session_image(make_image("inserted-image"))
    pts = make_points()

    viewer = DummyViewer([img, pts])
    manager = LayerLifecycleManager(viewer=viewer)
    rec = connect_signal_recorders(manager)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    event = SimpleNamespace(value=img, index=0, source=viewer.layers)

    manager.on_insert(event)

    assert manager.active_dlc_image_layer() is img
    assert manager.image_meta.name == "inserted-image"

    assert rec.refresh_video.count >= 1
    assert rec.refresh_status.count >= 1
    assert rec.move_image_bottom.count == 1
    assert rec.move_image_bottom.calls[0][0] is img

    assert pts in remap_calls


def test_manager_adopt_existing_layers_skips_already_managed_points(
    qtbot,
    fake_store,
    monkeypatch,
):
    img = mark_as_dlc_session_image(make_image())
    pts_managed = make_points("managed")
    pts_unmanaged = make_points("unmanaged")

    viewer = DummyViewer([img, pts_managed, pts_unmanaged])
    manager = LayerLifecycleManager(viewer=viewer)
    rec = connect_signal_recorders(manager)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    manager.register_managed_points_layer(pts_managed, object())

    manager.adopt_existing_layers()

    assert manager.active_dlc_image_layer() is img
    assert manager.image_meta.name == img.name

    assert rec.move_image_bottom.count == 1
    assert rec.move_image_bottom.calls[0][0] is img

    assert rec.setup_points.count == 1
    req = rec.setup_points.calls[0][0]
    assert req.layer is pts_unmanaged

    assert rec.adopted.count == 1

    assert pts_managed in remap_calls
    assert pts_unmanaged in remap_calls


def test_manager_on_remove_triggers_ui_cleanup_and_refresh(qtbot):
    pts = make_points()
    viewer = DummyViewer([pts])
    manager = LayerLifecycleManager(viewer=viewer)
    rec = connect_signal_recorders(manager)

    manager.register_managed_points_layer(pts, object())

    event = SimpleNamespace(value=pts)

    manager.on_remove(event)

    assert rec.removed_points.count == 1
    removed_layer, remaining = rec.removed_points.calls[0]
    assert removed_layer is pts
    assert remaining == 1

    manager._flush_post_remove_refresh()

    assert rec.refresh_video.count >= 1
    assert rec.refresh_status.count >= 1
    assert rec.removed.count == 1
    assert rec.removed.calls[0][0] is pts


def test_manager_reap_dead_entries_removes_stale_entry(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    pts = make_points()
    store = object()

    manager.register_managed_points_layer(pts, store)

    layer_id = id(pts)
    del pts
    gc.collect()

    report_before = manager.audit_registry()
    assert report_before.dead_count == 1
    assert any(issue.code == "dead-entry" and issue.layer_id == layer_id for issue in report_before.issues)

    reaped = manager.clear_dead_entries(log=False)

    assert len(reaped) == 1
    assert reaped[0].layer_id == layer_id
    assert reaped[0].runtime.store is store

    report_after = manager.audit_registry()
    assert report_after.dead_count == 0
    assert report_after.issues == ()


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
    manager = LayerLifecycleManager(viewer=viewer)

    event = event_factory(viewer, img, pts)

    layer = manager._resolve_inserted_layer(event)

    assert layer is pts
    assert layer.name == expected_name
