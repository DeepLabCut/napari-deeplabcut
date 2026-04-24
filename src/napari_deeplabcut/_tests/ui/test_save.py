from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image, Points

import napari_deeplabcut.ui.ui_dialogs.save as save_mod

# ---------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------


class DummySelection(list):
    def __init__(self, items=(), *, active=None):
        super().__init__(items)
        self.active = active

    def select_only(self, layer):
        self[:] = [layer]
        self.active = layer


class DummyLayers(list):
    def __init__(self, items=()):
        super().__init__(items)
        self.selection = DummySelection()
        self.save_calls = []

    def save(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))


class DummyViewer:
    def __init__(self, layers=()):
        self.layers = DummyLayers(layers)


class DummyLayerManager:
    def __init__(self):
        self.image_root = None
        self.image_paths = None
        self._active_image = None
        self._managed_points = ()

    def active_dlc_image_layer(self):
        return self._active_image

    def managed_points_layers(self):
        return tuple(self._managed_points)


class DummyTrailsController:
    def __init__(self):
        self.persist_calls = []

    def persist_folder_ui_state_for_points_layer(self, layer, *, checkbox_checked: bool):
        self.persist_calls.append((layer, checkbox_checked))


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


@pytest.fixture
def points_layer():
    return Points(
        np.empty((0, 3), dtype=float),
        name="points",
        metadata={},
        properties={},
    )


@pytest.fixture
def points_layer_2():
    return Points(
        np.empty((0, 3), dtype=float),
        name="points-2",
        metadata={},
        properties={},
    )


@pytest.fixture
def image_layer():
    return Image(
        np.zeros((5, 8, 8), dtype=np.uint8),
        name="img-stack",
        metadata={},
    )


@pytest.fixture
def workflow_factory():
    def _make(*, viewer=None, layer_manager=None, trails_controller=None, trail_checked=False, resolve_config=None):
        viewer = viewer or DummyViewer()
        layer_manager = layer_manager or DummyLayerManager()
        trails_controller = trails_controller or DummyTrailsController()

        wf = save_mod.PointsLayerSaveWorkflow(
            parent=None,
            viewer=viewer,
            layer_manager=layer_manager,
            trails_controller=trails_controller,
            trail_checkbox_getter=lambda: trail_checked,
            resolve_config_path_for_layer=resolve_config or (lambda _layer: None),
            current_project_path_getter=lambda: None,
            current_image_meta_getter=lambda: None,
            logger=logging.getLogger("test.save_workflow"),
        )
        return wf, viewer, layer_manager, trails_controller

    return _make


# ---------------------------------------------------------------------
# _save_multiple_layers
# ---------------------------------------------------------------------


def test_save_multiple_layers_returns_false_when_dialog_cancelled(
    monkeypatch,
    workflow_factory,
    points_layer,
):
    class FakeDialog:
        def __init__(self):
            self.history = None

        def setHistory(self, hist):
            self.history = hist

        def getSaveFileName(self, **kwargs):
            return "", ""

    monkeypatch.setattr(save_mod, "QFileDialog", FakeDialog)
    monkeypatch.setattr(save_mod, "get_save_history", lambda: [r"C:\tmp"])

    viewer = DummyViewer([points_layer])
    wf, viewer, _lm, trails = workflow_factory(viewer=viewer)

    outcome = wf._save_multiple_layers(selected=True, selected_layers=[points_layer])

    assert outcome.saved is False
    assert viewer.layers.save_calls == []
    assert trails.persist_calls == []


def test_save_multiple_layers_selected_persists_only_selected_points_layers(
    monkeypatch,
    workflow_factory,
    points_layer,
    image_layer,
):
    captured = {}

    class FakeDialog:
        def __init__(self):
            self.history = None

        def setHistory(self, hist):
            self.history = hist
            captured["history"] = hist

        def getSaveFileName(self, **kwargs):
            captured["dialog_kwargs"] = kwargs
            return r"C:\tmp\out.tif", "ignored"

    monkeypatch.setattr(save_mod, "QFileDialog", FakeDialog)
    monkeypatch.setattr(save_mod, "get_save_history", lambda: [r"C:\start"])

    viewer = DummyViewer([points_layer, image_layer])
    wf, viewer, _lm, trails = workflow_factory(viewer=viewer, trail_checked=True)

    selected_layers = [points_layer, image_layer]
    outcome = wf._save_multiple_layers(selected=True, selected_layers=selected_layers)

    assert outcome.saved is True
    assert outcome.status_message == "Data successfully saved"

    assert viewer.layers.save_calls == [((r"C:\tmp\out.tif",), {"selected": True})]

    # Only Points layers from the selected set should be persisted
    assert trails.persist_calls == [(points_layer, True)]

    assert captured["history"] == [r"C:\start"]
    assert captured["dialog_kwargs"]["caption"] == "Save selected layers"
    assert captured["dialog_kwargs"]["dir"] == r"C:\start"


def test_save_multiple_layers_all_uses_managed_points_layers(
    monkeypatch,
    workflow_factory,
    points_layer,
    points_layer_2,
):
    class FakeDialog:
        def setHistory(self, hist):
            pass

        def getSaveFileName(self, **kwargs):
            return r"C:\tmp\all_layers.npy", "ignored"

    monkeypatch.setattr(save_mod, "QFileDialog", FakeDialog)
    monkeypatch.setattr(save_mod, "get_save_history", lambda: [r"C:\start"])

    viewer = DummyViewer([points_layer, points_layer_2])
    lm = DummyLayerManager()
    lm._managed_points = (points_layer, points_layer_2)

    wf, viewer, _lm, trails = workflow_factory(
        viewer=viewer,
        layer_manager=lm,
        trail_checked=False,
    )

    outcome = wf._save_multiple_layers(selected=False, selected_layers=[])

    assert outcome.saved is True
    assert viewer.layers.save_calls == [((r"C:\tmp\all_layers.npy",), {"selected": False})]
    assert trails.persist_calls == [
        (points_layer, False),
        (points_layer_2, False),
    ]


def test_save_layers_dispatches_to_save_multiple_for_non_single_points_selection(
    monkeypatch,
    workflow_factory,
    points_layer,
    image_layer,
):
    viewer = DummyViewer([points_layer, image_layer])
    viewer.layers.selection[:] = [points_layer, image_layer]
    viewer.layers.selection.active = points_layer

    wf, _viewer, _lm, _trails = workflow_factory(viewer=viewer)

    called = {}

    def _fake_save_multiple(*, selected, selected_layers):
        called["selected"] = selected
        called["selected_layers"] = list(selected_layers)
        return save_mod.SaveOutcome(saved=True, status_message="ok")

    monkeypatch.setattr(wf, "_save_multiple_layers", _fake_save_multiple)

    outcome = wf.save_layers(selected=True)

    assert outcome.saved is True
    assert called["selected"] is True
    assert called["selected_layers"] == [points_layer, image_layer]


# ---------------------------------------------------------------------
# _best_image_context_layer
# ---------------------------------------------------------------------


def test_best_image_context_layer_prefers_lifecycle_owned_active_image(
    workflow_factory,
    image_layer,
):
    selected_image = Image(np.zeros((2, 2), dtype=np.uint8), name="selected")
    first_image = Image(np.zeros((2, 2), dtype=np.uint8), name="first")

    viewer = DummyViewer([first_image])
    viewer.layers.selection.active = selected_image

    lm = DummyLayerManager()
    lm._active_image = image_layer

    wf, *_ = workflow_factory(viewer=viewer, layer_manager=lm)

    assert wf._best_image_context_layer() is image_layer


def test_best_image_context_layer_falls_back_to_selected_image(workflow_factory):
    selected_image = Image(np.zeros((2, 2), dtype=np.uint8), name="selected")
    viewer = DummyViewer([selected_image])
    viewer.layers.selection.active = selected_image

    wf, *_ = workflow_factory(viewer=viewer)

    assert wf._best_image_context_layer() is selected_image


def test_best_image_context_layer_falls_back_to_first_image_in_viewer(workflow_factory):
    img1 = Image(np.zeros((2, 2), dtype=np.uint8), name="img1")
    img2 = Image(np.zeros((2, 2), dtype=np.uint8), name="img2")

    viewer = DummyViewer([img1, img2])
    viewer.layers.selection.active = None

    wf, *_ = workflow_factory(viewer=viewer)

    assert wf._best_image_context_layer() is img1


# ---------------------------------------------------------------------
# _enrich_points_metadata_for_save
# ---------------------------------------------------------------------


def test_enrich_metadata_returns_unchanged_if_root_already_present(
    workflow_factory,
    points_layer,
):
    wf, _viewer, lm, _trails = workflow_factory()
    lm.image_root = r"C:\project\labeled-data\ctx"
    lm.image_paths = ["img001.png", "img002.png"]

    md = {"root": r"C:\already\set", "paths": []}

    out = wf._enrich_points_metadata_for_save(points_layer, md)

    # Current behavior: early return if root exists
    assert out == md


def test_enrich_metadata_fills_from_layer_manager_image_context(
    workflow_factory,
    points_layer,
):
    wf, _viewer, lm, _trails = workflow_factory()
    lm.image_root = r"C:\project\labeled-data\session1"
    lm.image_paths = ["img001.png", "img002.png"]

    out = wf._enrich_points_metadata_for_save(points_layer, {})

    assert out["root"] == r"C:\project\labeled-data\session1"
    assert out["paths"] == ["img001.png", "img002.png"]


def test_enrich_metadata_returns_unchanged_when_no_context_and_no_config(
    workflow_factory,
    points_layer,
):
    wf, *_ = workflow_factory(resolve_config=lambda _layer: None)

    md = {}
    out = wf._enrich_points_metadata_for_save(points_layer, md)

    assert out == {}


def test_enrich_metadata_adds_project_when_config_resolves_but_no_image_layer(
    monkeypatch,
    workflow_factory,
    points_layer,
    tmp_path,
):
    project_root = tmp_path / "project"
    config_path = project_root / "config.yaml"
    project_root.mkdir()
    config_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(save_mod, "resolve_project_root_from_config", lambda p: project_root)

    wf, *_ = workflow_factory(resolve_config=lambda _layer: config_path)

    out = wf._enrich_points_metadata_for_save(points_layer, {})

    assert out["project"] == str(project_root)
    assert "root" not in out


def test_enrich_metadata_uses_source_anchor_when_it_looks_like_labeled_folder(
    monkeypatch,
    workflow_factory,
    points_layer,
    tmp_path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / "config.yaml"
    config_path.write_text("dummy", encoding="utf-8")

    inferred_root = project_root / "labeled-data" / "sessionA"

    monkeypatch.setattr(save_mod, "resolve_project_root_from_config", lambda p: project_root)
    monkeypatch.setattr(save_mod, "normalize_anchor_candidate", lambda src: inferred_root)
    monkeypatch.setattr(save_mod, "looks_like_dlc_labeled_folder", lambda p: True)

    lm = DummyLayerManager()
    lm._active_image = SimpleNamespace(
        source=SimpleNamespace(path=r"C:\whatever\img001.png"),
        name="ignored",
        metadata={},
    )

    wf, _viewer, _lm, _trails = workflow_factory(
        layer_manager=lm,
        resolve_config=lambda _layer: config_path,
    )

    out = wf._enrich_points_metadata_for_save(points_layer, {})

    assert out["project"] == str(project_root)
    assert out["root"] == str(inferred_root)


def test_enrich_metadata_falls_back_to_project_labeled_data_image_name_folder(
    monkeypatch,
    workflow_factory,
    points_layer,
    tmp_path,
):
    project_root = tmp_path / "project"
    dataset_dir = project_root / "labeled-data" / "session42"
    dataset_dir.mkdir(parents=True)
    config_path = project_root / "config.yaml"
    config_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(save_mod, "resolve_project_root_from_config", lambda p: project_root)
    monkeypatch.setattr(save_mod, "normalize_anchor_candidate", lambda src: None)

    lm = DummyLayerManager()
    lm._active_image = SimpleNamespace(
        source=SimpleNamespace(path=None),
        name="session42",
        metadata={},
    )

    wf, _viewer, _lm, _trails = workflow_factory(
        layer_manager=lm,
        resolve_config=lambda _layer: config_path,
    )

    out = wf._enrich_points_metadata_for_save(points_layer, {})

    assert out["project"] == str(project_root)
    assert out["root"] == str(dataset_dir)
