from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from napari.layers import Image, Points

from napari_deeplabcut.core.layer_lifecycle.manager import LayerLifecycleManager
from napari_deeplabcut.core.layer_lifecycle.registry import PointsRuntimeResources


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


# ---------------------------------------------------------------------------
# Transitional owner stub matching the current manager boundary
# ---------------------------------------------------------------------------
class DummyOwner:
    """Owner stub for the refactored lifecycle manager.

    The owner provides:
    - small validation/context hooks
    - UI completion hooks
    - refresh hooks
    """

    def __init__(self, viewer):
        self.viewer = viewer

        # Transitional context still stored on owner for now
        self._image_meta = DummyImageMeta()
        self._project_path = None

        # Transitional runtime/UI bits manager still consults
        self._label_mode = "sequential"
        self._recolor_pending = set()

        # Lifecycle/helper hooks
        self._validate_header = MagicMock(return_value=True)
        self._maybe_merge_config_points_layer = MagicMock(return_value=False)
        self._is_multianimal = MagicMock(return_value=False)

        self._maybe_initialize_layer_point_size_from_config = MagicMock()
        self._connect_layer_status_events = MagicMock()
        self._update_image_meta_from_layer = MagicMock()
        self._sync_points_layers_from_image_meta = MagicMock()
        self._cache_project_path_from_image_layer = MagicMock()
        self._schedule_recolor = MagicMock()

        # UI-only hooks
        self._complete_points_layer_ui_setup = MagicMock()
        self._on_points_layer_removed_ui = MagicMock()
        self._refresh_video_panel_context = MagicMock()
        self._refresh_layer_status_panel = MagicMock()
        self._move_image_layer_to_bottom = MagicMock()


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
    """Run QTimer.singleShot callbacks immediately for deterministic tests."""
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    monkeypatch.setattr(
        manager_module.QTimer,
        "singleShot",
        staticmethod(lambda _ms, fn: fn()),
    )


@pytest.fixture
def fake_store(monkeypatch):
    """Patch manager-side KeypointStore construction to a lightweight fake."""
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    monkeypatch.setattr(manager_module.keypoints, "KeypointStore", FakeStore)


@pytest.fixture
def fake_runtime_attachment(monkeypatch):
    """Patch away napari patch/keybinding attachment details.

    These tests verify lifecycle orchestration, not compat patch internals.
    """
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module
    from napari_deeplabcut.core.layer_lifecycle.manager import LayerLifecycleManager

    def _fake_attach_points_layer_runtime(self, **kwargs):
        return PointsRuntimeResources(
            query_next_frame_event_added=False,
            query_next_frame_connected=False,
            add_wrapper_installed=True,
            paste_patch_installed=True,
            keybindings_installed=True,
        )

    # Support either implementation shape:
    # - class/instance method: self.attach_points_layer_runtime(...)
    # - module-level function: attach_points_layer_runtime(...)
    if hasattr(LayerLifecycleManager, "attach_points_layer_runtime"):
        monkeypatch.setattr(
            LayerLifecycleManager,
            "attach_points_layer_runtime",
            _fake_attach_points_layer_runtime,
        )

    if hasattr(manager_module, "attach_points_layer_runtime"):
        monkeypatch.setattr(
            manager_module,
            "attach_points_layer_runtime",
            lambda **kwargs: PointsRuntimeResources(
                query_next_frame_event_added=False,
                query_next_frame_connected=False,
                add_wrapper_installed=True,
                paste_patch_installed=True,
                keybindings_installed=True,
            ),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
    fake_runtime_attachment,
):
    img = make_image()
    pts_existing = make_points("existing")
    pts_inserted = make_points("inserted")

    viewer = DummyViewer([img, pts_existing, pts_inserted])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    event = SimpleNamespace(value=pts_inserted, index=2, source=viewer.layers)

    manager.on_insert(event)

    assert manager.is_managed(pts_inserted) is True

    owner._validate_header.assert_any_call(pts_inserted)
    owner._maybe_merge_config_points_layer.assert_called_once_with(pts_inserted)
    owner._complete_points_layer_ui_setup.assert_called_once()
    ui_args, ui_kwargs = owner._complete_points_layer_ui_setup.call_args
    assert ui_args[0] is pts_inserted

    owner._refresh_video_panel_context.assert_called()
    owner._refresh_layer_status_panel.assert_called()

    assert pts_existing in remap_calls
    assert pts_inserted in remap_calls


def test_manager_on_insert_image_updates_context_and_refreshes_ui(qtbot):
    img = mark_as_dlc_session_image(make_image("inserted-image"))
    pts = make_points()

    viewer = DummyViewer([img, pts])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    event = SimpleNamespace(value=img, index=0, source=viewer.layers)

    manager.on_insert(event)

    assert manager.active_dlc_image_layer() is img
    assert manager.image_meta.name == "inserted-image"
    owner._refresh_video_panel_context.assert_called()
    owner._refresh_layer_status_panel.assert_called()
    owner._move_image_layer_to_bottom.assert_called_once_with(img)
    assert pts in remap_calls

    owner._refresh_video_panel_context.assert_called()
    owner._refresh_layer_status_panel.assert_called()

    assert pts in remap_calls


def test_manager_adopt_existing_layers_skips_already_managed_points(
    qtbot,
    fake_store,
    fake_runtime_attachment,
):
    img = mark_as_dlc_session_image(make_image())
    pts_managed = make_points("managed")
    pts_unmanaged = make_points("unmanaged")

    viewer = DummyViewer([img, pts_managed, pts_unmanaged])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    remap_calls = []
    manager._remap_frame_indices = lambda layer: remap_calls.append(layer)

    manager.register_managed_points_layer(pts_managed, object())

    manager.adopt_existing_layers()

    assert manager.active_dlc_image_layer() is img
    assert manager.image_meta.name == img.name
    owner._move_image_layer_to_bottom.assert_called_once_with(img)
    owner._complete_points_layer_ui_setup.assert_called_once()
    ui_args, _ui_kwargs = owner._complete_points_layer_ui_setup.call_args
    assert ui_args[0] is pts_unmanaged

    assert pts_managed in remap_calls
    assert pts_unmanaged in remap_calls


def test_manager_on_remove_triggers_ui_cleanup_and_refresh(qtbot):
    pts = make_points()
    viewer = DummyViewer([pts])
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = SimpleNamespace(value=pts)

    manager.on_remove(event)

    owner._on_points_layer_removed_ui.assert_called_once_with(
        pts,
        remaining_points_layers=1,
    )

    # Let the deferred QTimer(0) refresh run
    qtbot.wait(1)

    owner._refresh_video_panel_context.assert_called()
    owner._refresh_layer_status_panel.assert_called()


def test_manager_reap_dead_entries_removes_stale_entry(qtbot):
    viewer = DummyViewer()
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

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
    owner = DummyOwner(viewer)
    manager = LayerLifecycleManager(owner=owner)

    event = event_factory(viewer, img, pts)

    layer = manager._resolve_inserted_layer(event)

    assert layer is pts
    assert layer.name == expected_name
