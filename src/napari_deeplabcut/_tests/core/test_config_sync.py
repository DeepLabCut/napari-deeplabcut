from __future__ import annotations

from types import SimpleNamespace

import napari_deeplabcut.core.config_sync as cs


class DummyLayer:
    def __init__(self, *, metadata=None, source_path=None):
        self.metadata = metadata or {}
        self.source = SimpleNamespace(path=source_path) if source_path is not None else None


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def test_coerce_point_size_rounds_and_clamps():
    assert cs._coerce_point_size(12) == 12
    assert cs._coerce_point_size(12.6) == 13
    assert cs._coerce_point_size("7") == 7
    assert cs._coerce_point_size(-5) == 1
    assert cs._coerce_point_size(999) == 100
    assert cs._coerce_point_size("not-a-number") == 6


def test_layer_source_path_returns_string_when_available():
    layer = DummyLayer(source_path="/tmp/some/file.png")
    assert cs._layer_source_path(layer) == "/tmp/some/file.png"


def test_layer_source_path_returns_none_when_source_missing():
    layer = DummyLayer()
    assert cs._layer_source_path(layer) is None


def test_layer_source_path_returns_none_when_source_path_access_fails():
    class BadSource:
        @property
        def path(self):
            raise RuntimeError("boom")

    layer = DummyLayer()
    layer.source = BadSource()
    assert cs._layer_source_path(layer) is None


# -----------------------------------------------------------------------------
# resolve_config_path_from_layer
# -----------------------------------------------------------------------------


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
        "infer_dlc_project",
        lambda *args, **kwargs: SimpleNamespace(config_path=config_path),
    )

    resolved = cs.resolve_config_path_from_layer(layer, fallback_project=str(tmp_path))

    assert resolved == config_path


def test_resolve_config_uses_find_nearest_config_as_last_resort(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={"root": str(tmp_path)})

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=None))
    monkeypatch.setattr(cs, "find_nearest_config", lambda *args, **kwargs: config_path)

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == config_path


def test_resolve_config_returns_none_when_everything_fails(monkeypatch):
    layer = DummyLayer(metadata={})

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=None))
    monkeypatch.setattr(cs, "find_nearest_config", lambda *args, **kwargs: None)

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved is None


def test_resolve_config_ignores_points_meta_when_read_points_meta_raises(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={"root": str(tmp_path)})

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=config_path))

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == config_path


def test_resolve_config_skips_points_meta_when_errors_attribute_present(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={"root": str(tmp_path)})

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: SimpleNamespace(errors=["bad meta"]))
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=config_path))

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == config_path


def test_resolve_config_skips_non_file_points_meta_config_and_falls_through(monkeypatch, tmp_path):
    missing_config = tmp_path / "missing_config.yaml"
    real_config = tmp_path / "config.yaml"
    real_config.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(metadata={"root": str(tmp_path)})

    monkeypatch.setattr(
        cs,
        "read_points_meta",
        lambda *args, **kwargs: SimpleNamespace(project=None, root=None, paths=[]),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project_from_points_meta",
        lambda *args, **kwargs: SimpleNamespace(config_path=missing_config),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project",
        lambda *args, **kwargs: SimpleNamespace(config_path=real_config),
    )

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == real_config


def test_resolve_config_passes_paths_into_generic_inference(monkeypatch, tmp_path):
    captured = {}

    layer = DummyLayer(
        metadata={
            "project": str(tmp_path / "proj"),
            "root": str(tmp_path / "root"),
            "paths": [
                "labeled-data/session_001/img001.png",
                "labeled-data/session_001/img002.png",
                "labeled-data/session_001/img003.png",
                "labeled-data/session_001/img004.png",
            ],
        },
        source_path=str(tmp_path / "video.mp4"),
    )

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: None)

    def fake_infer_dlc_project(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config_path=None)

    monkeypatch.setattr(cs, "infer_dlc_project", fake_infer_dlc_project)
    monkeypatch.setattr(cs, "find_nearest_config", lambda *args, **kwargs: None)

    resolved = cs.resolve_config_path_from_layer(
        layer,
        fallback_project=str(tmp_path / "fallback_project"),
        fallback_root=str(tmp_path / "fallback_root"),
        prefer_project_root=False,
        max_levels=7,
    )

    assert resolved is None
    assert captured["dataset_candidates"] == ["labeled-data/session_001/img001.png"]
    assert captured["anchor_candidates"] == [
        str(tmp_path / "proj"),
        str(tmp_path / "root"),
        str(tmp_path / "video.mp4"),
        str(tmp_path / "fallback_project"),
        str(tmp_path / "fallback_root"),
        "labeled-data/session_001/img001.png",
        "labeled-data/session_001/img002.png",
        "labeled-data/session_001/img003.png",
    ]
    assert captured["prefer_project_root"] is False
    assert captured["max_levels"] == 7


def test_resolve_config_uses_image_inference_after_points_inference_exception(monkeypatch, tmp_path):
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
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        cs,
        "infer_dlc_project_from_image_layer",
        lambda *args, **kwargs: SimpleNamespace(config_path=config_path),
    )

    resolved = cs.resolve_config_path_from_layer(layer, image_layer=image_layer)

    assert resolved == config_path


def test_resolve_config_continues_when_find_nearest_config_raises_for_one_candidate(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dotsize: 6\n", encoding="utf-8")

    layer = DummyLayer(
        metadata={
            "project": "bad-candidate",
            "root": str(tmp_path),
        }
    )

    monkeypatch.setattr(cs, "read_points_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs, "infer_dlc_project", lambda *args, **kwargs: SimpleNamespace(config_path=None))

    def fake_find(candidate, **kwargs):
        if candidate == "bad-candidate":
            raise RuntimeError("boom")
        return config_path

    monkeypatch.setattr(cs, "find_nearest_config", fake_find)

    resolved = cs.resolve_config_path_from_layer(layer)

    assert resolved == config_path


# -----------------------------------------------------------------------------
# load_point_size_from_config
# -----------------------------------------------------------------------------


def test_load_point_size_from_config_returns_none_for_missing_path():
    assert cs.load_point_size_from_config(None) is None


def test_load_point_size_from_config_returns_none_when_load_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cs.load_point_size_from_config(config_path) is None


def test_load_point_size_from_config_returns_none_when_key_missing(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"colormap": "rainbow"})

    assert cs.load_point_size_from_config(config_path) is None


def test_load_point_size_from_config_coerces_and_clamps(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"dotsize": "250"})

    assert cs.load_point_size_from_config(config_path) == 100


# -----------------------------------------------------------------------------
# save_point_size_to_config
# -----------------------------------------------------------------------------


def test_save_point_size_to_config_returns_false_when_path_missing():
    assert cs.save_point_size_to_config(None, 12) is False


def test_save_point_size_to_config_returns_false_when_load_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cs.save_point_size_to_config(config_path, 12) is False


def test_save_point_size_to_config_returns_false_when_value_unchanged(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    written = []

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"dotsize": 12})
    monkeypatch.setattr(cs.io, "write_config", lambda *args, **kwargs: written.append(True))

    assert cs.save_point_size_to_config(config_path, 12) is False
    assert written == []


def test_save_point_size_to_config_writes_updated_value(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    written = {}

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"dotsize": 6, "colormap": "rainbow"})

    def fake_write(path, cfg):
        written["path"] = path
        written["cfg"] = cfg

    monkeypatch.setattr(cs.io, "write_config", fake_write)

    assert cs.save_point_size_to_config(config_path, 12) is True
    assert written["path"] == str(config_path)
    assert written["cfg"]["dotsize"] == 12
    assert written["cfg"]["colormap"] == "rainbow"


def test_save_point_size_to_config_clamps_before_writing(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    written = {}

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {})

    def fake_write(path, cfg):
        written["cfg"] = cfg

    monkeypatch.setattr(cs.io, "write_config", fake_write)

    assert cs.save_point_size_to_config(config_path, 999) is True
    assert written["cfg"]["dotsize"] == 100


def test_save_point_size_to_config_still_writes_when_old_value_is_not_coercible(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    written = {}

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"dotsize": object()})

    def fake_write(path, cfg):
        written["cfg"] = cfg

    monkeypatch.setattr(cs.io, "write_config", fake_write)

    assert cs.save_point_size_to_config(config_path, 15) is True
    assert written["cfg"]["dotsize"] == 15


def test_save_point_size_to_config_returns_false_when_write_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(cs.io, "load_config", lambda *args, **kwargs: {"dotsize": 6})
    monkeypatch.setattr(cs.io, "write_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cs.save_point_size_to_config(config_path, 15) is False
