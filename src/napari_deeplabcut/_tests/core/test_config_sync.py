from __future__ import annotations

from types import SimpleNamespace

import napari_deeplabcut.core.config_sync as cs


class DummyLayer:
    def __init__(self, *, metadata=None, source_path=None):
        self.metadata = metadata or {}
        self.source = SimpleNamespace(path=source_path) if source_path is not None else None


def test_resolve_config_prefers_points_meta_inference(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={})

    monkeypatch.setattr(
        cs,
        "read_points_meta",
        lambda *args, **kwargs: SimpleNamespace(project=None, root=None, paths=[]),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project_from_points_meta",
        lambda *args, **kwargs: SimpleNamespace(config_path=config_path),
    )

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == config_path


def test_resolve_config_uses_image_layer_inference_when_points_meta_has_no_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={})
    image_layer = DummyLayer(metadata={"root": str(tmp_path)})

    monkeypatch.setattr(
        cs,
        "read_points_meta",
        lambda *args, **kwargs: SimpleNamespace(project=None, root=None, paths=[]),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project_from_points_meta",
        lambda *args, **kwargs: SimpleNamespace(config_path=None),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project_from_image_layer",
        lambda *args, **kwargs: SimpleNamespace(config_path=config_path),
    )

    resolved = cs.resolve_config_path_from_layer(layer, image_layer=image_layer)

    assert resolved == config_path


def test_resolve_config_uses_generic_fallback_hints(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={})

    monkeypatch.setattr(
        cs, "read_points_meta", lambda *args, **kwargs: SimpleNamespace(project=None, root=None, paths=[])
    )
    monkeypatch.setattr(
        cs, "infer_dlc_project_from_points_meta", lambda *args, **kwargs: SimpleNamespace(config_path=None)
    )
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=config_path))

    resolved = cs.resolve_config_path_from_layer(layer, fallback_project=str(tmp_path))

    assert resolved == config_path
