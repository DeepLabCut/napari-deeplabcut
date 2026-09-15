# src/napari_deeplabcut/_tests/core/layer_manager/test_manager.py
from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image, Points

from napari_deeplabcut.config.models import AnnotationKind, ImageMetadata
from napari_deeplabcut.core.layer_lifecycle import LayerLifecycleManager
from napari_deeplabcut.core.layer_lifecycle.display_settings import (
    MACHINE_LABELS_POINTS_DISPLAY,
    PointsDisplaySource,
)
from napari_deeplabcut.core.layer_lifecycle.manager import PointsRuntimeResources
from napari_deeplabcut.tracking.core.data import build_tracking_result_metadata


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
        dataset_mismatch=SignalRecorder(),
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
    manager.layer_dataset_mismatch.connect(rec.dataset_mismatch)
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


def make_nonempty_points(name="pts"):
    layer = Points(np.array([[0, 1, 2]], dtype=float))
    layer.name = name
    return layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def immediate_qtimer(monkeypatch):
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    monkeypatch.setattr(
        manager_module.LayerLifecycleManager,
        "_single_shot_owned",
        lambda self, _ms, fn: fn(),
    )
    monkeypatch.setattr(
        manager_module.LayerLifecycleManager,
        "_schedule_once",
        lambda self, _name, _ms, fn: fn(),
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


def test_manager_register_points_layer_survives_reused_layer_id(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    stale = make_points("stale")
    manager.register_managed_points_layer(stale, object())

    entry = manager.registry._entries_by_id.pop(id(stale))
    pts_b = make_points("pts-b")
    entry.layer_id = id(pts_b)
    manager.registry._entries_by_id[id(pts_b)] = entry

    del stale
    gc.collect()

    store_b = object()
    manager.register_managed_points_layer(pts_b, store_b)

    assert manager.get_store(pts_b) is store_b


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


# ----------------------------------------------------------------------------
# Display settings logic tests
# ---------------------------------------------------------------------------


def make_machine_points(name="machine", *, nonempty: bool = False):
    layer = make_nonempty_points(name) if nonempty else make_points(name)
    layer.metadata = {
        "io": {
            "kind": AnnotationKind.MACHINE,
        }
    }
    return layer


def make_machine_points_serialized_kind(name="machine", *, nonempty: bool = False):
    layer = make_nonempty_points(name) if nonempty else make_points(name)
    layer.metadata = {
        "io": {
            "kind": "machine",
        }
    }
    return layer


def make_tracking_points(name="tracking", *, nonempty: bool = False):
    layer = make_nonempty_points(name) if nonempty else make_points(name)
    layer.metadata = build_tracking_result_metadata(
        {},
        tracker_name="test-tracker",
        source_layer_name="source",
        query_frame=0,
    )
    return layer


def test_manager_detects_machine_label_layer_from_annotation_kind(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    layer = make_machine_points()

    assert LayerLifecycleManager.is_machine_label_layer(layer) is True
    assert manager.points_display_role(layer) is PointsDisplaySource.MACHINE_LABELS


def test_manager_detects_machine_label_layer_from_serialized_kind(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    layer = make_machine_points_serialized_kind()

    assert LayerLifecycleManager.is_machine_label_layer(layer) is True
    assert manager.points_display_role(layer) is PointsDisplaySource.MACHINE_LABELS


def test_manager_does_not_treat_regular_points_as_machine_labels(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    layer = make_points("regular")
    layer.metadata = {
        "io": {
            "kind": AnnotationKind.GT,
        }
    }

    assert LayerLifecycleManager.is_machine_label_layer(layer) is False
    assert manager.points_display_role(layer) is None


def test_manager_tracking_display_role_takes_precedence_over_machine_kind(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    layer = make_tracking_points()
    layer.metadata["io"] = {"kind": AnnotationKind.MACHINE}

    assert LayerLifecycleManager.is_tracking_result_layer(layer) is True
    assert LayerLifecycleManager.is_machine_label_layer(layer) is False
    assert manager.points_display_role(layer) is PointsDisplaySource.TRACKING_RESULT


def test_setup_points_layer_applies_display_settings(
    qtbot,
    fake_store,
    monkeypatch,
):
    pts = make_machine_points("machine")
    viewer = DummyViewer([pts])
    manager = LayerLifecycleManager(viewer=viewer)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    applied = []

    def fake_apply(layer, *, source=None):
        applied.append((layer, source))
        return layer

    monkeypatch.setattr(manager, "apply_points_display_settings", fake_apply)

    result = manager._setup_points_layer(pts, allow_merge=False)

    assert result is not None
    assert manager.is_managed(pts) is True
    assert applied == [(pts, None)]


def test_manager_apply_points_display_settings_delegates_machine_role(
    qtbot,
    monkeypatch,
):
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    pts = make_machine_points("machine")
    viewer = DummyViewer([pts])
    manager = LayerLifecycleManager(viewer=viewer)

    calls = []

    def fake_apply_points_display_role(layer, role, *, source=None):
        calls.append((layer, role, source))
        return layer

    monkeypatch.setattr(
        manager_module,
        "apply_points_display_role",
        fake_apply_points_display_role,
    )

    out = manager.apply_points_display_settings(pts)

    assert out is pts
    assert calls == [(pts, PointsDisplaySource.MACHINE_LABELS, None)]


def test_manager_apply_points_display_settings_delegates_tracking_role_with_source(
    qtbot,
    monkeypatch,
):
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    source = make_points("source")
    tracking = make_tracking_points("tracking")

    viewer = DummyViewer([source, tracking])
    manager = LayerLifecycleManager(viewer=viewer)

    calls = []

    def fake_apply_points_display_role(layer, role, *, source=None):
        calls.append((layer, role, source))
        return layer

    monkeypatch.setattr(
        manager_module,
        "apply_points_display_role",
        fake_apply_points_display_role,
    )

    out = manager.apply_points_display_settings(tracking, source=source)

    assert out is tracking
    assert calls == [(tracking, PointsDisplaySource.TRACKING_RESULT, source)]


def test_setup_points_layer_styles_machine_labels_using_config(
    qtbot,
    fake_store,
    monkeypatch,
):
    pts = make_machine_points("machine", nonempty=True)
    viewer = DummyViewer([pts])
    manager = LayerLifecycleManager(viewer=viewer)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    manager._setup_points_layer(pts, allow_merge=False)

    if MACHINE_LABELS_POINTS_DISPLAY.symbol is not None:
        assert MACHINE_LABELS_POINTS_DISPLAY.symbol in set(pts.symbol)

    if MACHINE_LABELS_POINTS_DISPLAY.opacity is not None:
        assert pts.opacity == MACHINE_LABELS_POINTS_DISPLAY.opacity

    if MACHINE_LABELS_POINTS_DISPLAY.border_width is not None:
        assert getattr(pts, "border_width", None) == MACHINE_LABELS_POINTS_DISPLAY.border_width

    if MACHINE_LABELS_POINTS_DISPLAY.border_color is not None:
        assert getattr(pts, "border_color", None) is not None


def test_attach_points_layer_runtime_reattach_rebinds_to_current_store(qtbot, monkeypatch):
    """Re-attaching rebinds the add wrapper and the shortcuts to the store passed."""
    from napari_deeplabcut.core.layer_lifecycle import manager as manager_module

    class RecordingStore(FakeStore):
        def __init__(self, viewer, layer):
            super().__init__(viewer, layer)
            self.added = []

        def add(self, coord):
            self.added.append(coord)

        def next_keypoint(self, *_args):
            return None

        def prev_keypoint(self, *_args):
            return None

        def _find_first_unlabeled_frame(self, *_args):
            return None

    class RecordingControls:
        def cycle_through_label_modes(self, *_args):
            return None

        def cycle_through_color_modes(self, *_args):
            return None

    monkeypatch.setattr(manager_module.keypoints, "KeypointStore", RecordingStore)

    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)
    manager.viewer_keybinds_installed = True

    layer = make_points()
    first = RecordingStore(viewer, layer)
    second = RecordingStore(viewer, layer)
    first_controls = RecordingControls()
    second_controls = RecordingControls()

    def attach(store, controls, resources):
        return manager.attach_points_layer_runtime(
            layer=layer,
            store=store,
            controls=controls,
            resolve_layer_by_id=lambda _layer_id: layer,
            schedule_recolor=lambda _layer: None,
            existing_resources=resources,
        )

    resources = attach(first, first_controls, PointsRuntimeResources())
    attach(second, second_controls, resources)

    layer.add(np.zeros((1, 3)))

    assert first.added == []
    assert len(second.added) == 1
    keymap = {str(key): callback for key, callback in layer.keymap.items()}

    for key in ("W", "Up", "S", "Down", "Shift+Left", "Shift+Right"):
        assert keymap[key].__self__ is second, f"{key} still bound to the previous store"

    for key in ("M", "F"):
        assert keymap[key].__self__ is second_controls, f"{key} still bound to the previous controls"


# ---------------------------------------------------------------------------
# _remap_frame_indices
# ---------------------------------------------------------------------------
def _points_bound_to(paths, *, root, dataset_key=None):
    """A Points layer as the readers build one: paths, root, and an immutable identity."""
    layer = make_nonempty_points("bound")
    layer.metadata = {
        "paths": list(paths),
        "root": root,
        "dataset_key": root if dataset_key is None else dataset_key,
    }
    return layer


def _manager_showing(layer, *, paths, root, dataset_key=None):
    """A manager whose image context is the given folder."""
    manager = LayerLifecycleManager(viewer=DummyViewer([layer]))
    manager._image_meta = ImageMetadata(paths=list(paths), root=root)
    manager._image_dataset_key = root if dataset_key is None else dataset_key
    return manager


def test_remap_frame_indices_refuses_a_layer_from_another_dataset(monkeypatch):
    """Identity is decided by dataset_key, whatever the frame names happen to be.

    These two folders share every frame name, which is the DLC norm rather than evidence
    that they hold the same footage.
    """
    old_paths = ["labeled-data/videoA/img000.png", "labeled-data/videoA/img001.png"]
    layer = _points_bound_to(old_paths, root="C:/project/labeled-data/videoA")

    manager = _manager_showing(
        layer,
        paths=["labeled-data/videoB/img000.png", "labeled-data/videoB/img001.png"],
        root="C:/project/labeled-data/videoB",
    )

    warned = []
    monkeypatch.setattr(manager, "_report_layer_left_on_previous_dataset", lambda ly: warned.append(ly))

    manager._remap_frame_indices(layer)

    assert layer.metadata["paths"] == old_paths
    assert layer.metadata["root"] == "C:/project/labeled-data/videoA"
    assert warned == [layer]


def test_remap_frame_indices_refuses_another_project_with_the_same_video_name(monkeypatch):
    """The keys are absolute, so two projects holding `labeled-data/mouse1` stay distinct."""
    old_paths = ["labeled-data/mouse1/img000.png"]
    layer = _points_bound_to(old_paths, root="C:/project-A/labeled-data/mouse1")

    manager = _manager_showing(
        layer,
        paths=["labeled-data/mouse1/img000.png"],
        root="C:/project-B/labeled-data/mouse1",
    )

    warned = []
    monkeypatch.setattr(manager, "_report_layer_left_on_previous_dataset", lambda ly: warned.append(ly))

    manager._remap_frame_indices(layer)

    assert layer.metadata["root"] == "C:/project-A/labeled-data/mouse1"
    assert warned == [layer]


def test_remap_frame_indices_leaves_metadata_alone_when_nothing_maps(monkeypatch):
    """Same dataset, but no frame overlap: adopt neither root nor paths."""
    old_paths = ["labeled-data/videoA/imgA000.png"]
    layer = _points_bound_to(old_paths, root="C:/project/labeled-data/videoA")

    manager = _manager_showing(
        layer,
        paths=["labeled-data/videoA/renamed000.png"],
        root="C:/project/labeled-data/videoA",
    )

    warned = []
    monkeypatch.setattr(manager, "_report_layer_left_on_previous_dataset", lambda ly: warned.append(ly))

    manager._remap_frame_indices(layer)

    assert layer.metadata["paths"] == old_paths
    assert layer.metadata["root"] == "C:/project/labeled-data/videoA"
    assert warned == [layer]


def test_remap_frame_indices_adopts_root_and_paths_together_when_frames_map():
    """The project moved: same dataset, new prefix, so both fields follow."""
    new_paths = ["labeled-data/videoA/imgA000.png"]
    layer = _points_bound_to(
        ["old/labeled-data/videoA/imgA000.png"],
        root="D:/moved/labeled-data/videoA",
        dataset_key="C:/project/labeled-data/videoA",
    )

    manager = _manager_showing(layer, paths=new_paths, root="C:/project/labeled-data/videoA")

    manager._remap_frame_indices(layer)

    assert layer.metadata["paths"] == new_paths
    assert layer.metadata["root"] == "C:/project/labeled-data/videoA"


def test_remap_frame_indices_adopts_an_unbound_layer():
    """A config placeholder has no dataset of its own, so it takes whatever is open."""
    new_paths = ["labeled-data/videoA/img000.png"]
    layer = make_points("placeholder")
    layer.metadata = {"project": "C:/project"}

    manager = _manager_showing(layer, paths=new_paths, root="C:/project/labeled-data/videoA")

    manager._remap_frame_indices(layer)

    assert layer.metadata["root"] == "C:/project/labeled-data/videoA"


def test_dataset_mismatch_is_reported_once_per_target_folder(qtbot):
    """The remap sweep revisits every layer per insert, so repeats must be suppressed."""
    layer = _points_bound_to(["labeled-data/videoA/imgA000.png"], root="C:/project/labeled-data/videoA")

    manager = LayerLifecycleManager(viewer=DummyViewer([layer]))
    rec = connect_signal_recorders(manager)
    manager._image_meta = ImageMetadata(
        paths=["labeled-data/videoB/imgB000.png"],
        root="C:/project/labeled-data/videoB",
    )

    manager._remap_frame_indices(layer)
    manager._remap_frame_indices(layer)

    assert rec.dataset_mismatch.count == 1
    assert "videoA" in rec.dataset_mismatch.calls[0][0]

    # A different folder is a new fact, so it is reported again.
    manager._image_meta = ImageMetadata(
        paths=["labeled-data/videoC/imgC000.png"],
        root="C:/project/labeled-data/videoC",
    )
    manager._remap_frame_indices(layer)

    assert rec.dataset_mismatch.count == 2


def test_an_unbound_layer_binds_to_the_dataset_it_adopts(monkeypatch):
    """A config placeholder must stop being unbound once it takes a folder."""
    layer = make_points("placeholder")
    layer.metadata = {"project": "C:/project"}

    manager = _manager_showing(
        layer,
        paths=["labeled-data/videoA/img000.png"],
        root="C:/project/labeled-data/videoA",
    )
    manager._remap_frame_indices(layer)

    assert layer.metadata["dataset_key"] == "C:/project/labeled-data/videoA"

    # Now a different folder reusing the same frame names must be refused.
    manager._image_meta = ImageMetadata(
        paths=["labeled-data/videoB/img000.png"],
        root="C:/project/labeled-data/videoB",
    )
    manager._image_dataset_key = "C:/project/labeled-data/videoB"

    warned = []
    monkeypatch.setattr(manager, "_report_layer_left_on_previous_dataset", lambda ly: warned.append(ly))

    manager._remap_frame_indices(layer)

    assert layer.metadata["root"] == "C:/project/labeled-data/videoA"
    assert warned == [layer]
