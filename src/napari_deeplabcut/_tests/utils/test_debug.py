from __future__ import annotations

import logging
from types import SimpleNamespace

from napari_deeplabcut.utils.debug import (
    InMemoryDebugRecorder,
    _safe_tail,
    collect_environment_summary,
    format_debug_report,
    get_debug_recorder,
    install_debug_recorder,
    summarize_layer,
    summarize_viewer,
)


class _BrokenStr:
    def __str__(self):
        raise RuntimeError("boom")


def _cleanup_logger(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    recorder = get_debug_recorder(logger_name=logger_name)
    if recorder is not None:
        try:
            logger.removeHandler(recorder)
        except Exception:
            pass
        try:
            delattr(logger, "_napari_dlc_debug_recorder")
        except Exception:
            pass


def test_install_debug_recorder_is_idempotent():
    logger_name = "napari-deeplabcut.test.idempotent"
    _cleanup_logger(logger_name)

    rec1 = install_debug_recorder(logger_name=logger_name, capacity=50)
    rec2 = install_debug_recorder(logger_name=logger_name, capacity=999)

    try:
        assert rec1 is rec2
        assert get_debug_recorder(logger_name=logger_name) is rec1
    finally:
        _cleanup_logger(logger_name)


def test_recorder_captures_debug_message():
    logger_name = "napari-deeplabcut.test.capture"
    _cleanup_logger(logger_name)

    recorder = install_debug_recorder(logger_name=logger_name, capacity=50)
    logger = logging.getLogger(logger_name)

    try:
        logger.debug("hello %s", "world")

        records = recorder.snapshot()
        assert len(records) == 1
        rec = records[0]
        assert rec.level == "DEBUG"
        assert rec.logger_name == logger_name
        assert rec.message == "hello world"
        assert rec.exc_text is None
    finally:
        _cleanup_logger(logger_name)


def test_recorder_captures_exception_text():
    logger_name = "napari-deeplabcut.test.exc"
    _cleanup_logger(logger_name)

    recorder = install_debug_recorder(logger_name=logger_name, capacity=50)
    logger = logging.getLogger(logger_name)

    try:
        try:
            raise ValueError("bad things")
        except ValueError:
            logger.exception("something failed")

        records = recorder.snapshot()
        assert len(records) == 1
        rec = records[0]
        assert rec.message == "something failed"
        assert rec.exc_text is not None
        assert "ValueError: bad things" in rec.exc_text
    finally:
        _cleanup_logger(logger_name)


def test_recorder_is_bounded():
    recorder = InMemoryDebugRecorder(capacity=3)
    logger = logging.getLogger("napari-deeplabcut.test.bounded")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)

    try:
        for i in range(10):
            logger.debug("msg-%d", i)

        records = recorder.snapshot()
        assert len(records) == 3
        assert [r.message for r in records] == ["msg-7", "msg-8", "msg-9"]
    finally:
        logger.removeHandler(recorder)


def test_clear_resets_records_and_dropped_count():
    recorder = InMemoryDebugRecorder(capacity=3)
    logger = logging.getLogger("napari-deeplabcut.test.clear")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)

    try:
        logger.debug("before clear")
        assert len(recorder.snapshot()) == 1

        recorder.clear()

        assert recorder.snapshot() == []
        assert recorder.dropped_count == 0
    finally:
        logger.removeHandler(recorder)


def test_render_text_includes_message_and_level():
    recorder = InMemoryDebugRecorder(capacity=10)
    logger = logging.getLogger("napari-deeplabcut.test.render")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)

    try:
        logger.info("hello render")
        text = recorder.render_text(limit=10)

        assert "INFO" in text
        assert "hello render" in text
        assert "napari-deeplabcut.test.render" in text
    finally:
        logger.removeHandler(recorder)


def test_render_text_respects_limit():
    recorder = InMemoryDebugRecorder(capacity=10)
    logger = logging.getLogger("napari-deeplabcut.test.limit")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)

    try:
        for i in range(5):
            logger.debug("line-%d", i)

        text = recorder.render_text(limit=2)
        assert "line-3" in text
        assert "line-4" in text
        assert "line-0" not in text
        assert "line-1" not in text
        assert "line-2" not in text
    finally:
        logger.removeHandler(recorder)


def test_emit_handles_unrenderable_message_without_raising():
    recorder = InMemoryDebugRecorder(capacity=10)

    record = logging.LogRecord(
        name="napari-deeplabcut.test.unrenderable",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=_BrokenStr(),
        args=(),
        exc_info=None,
    )

    # Should never raise
    recorder.emit(record)

    records = recorder.snapshot()
    assert len(records) == 1
    assert records[0].message == "<unrenderable log message>"


class FakeLayer:
    def __init__(
        self,
        *,
        name="layer",
        metadata=None,
        properties=None,
        data=None,
        visible=True,
    ):
        self.name = name
        self.metadata = metadata if metadata is not None else {}
        self.properties = properties if properties is not None else {}
        self.data = data
        self.visible = visible


class FakeSelection:
    def __init__(self, active=None):
        self.active = active


class FakeViewerLayers(list):
    def __init__(self, layers, active=None):
        super().__init__(layers)
        self.selection = FakeSelection(active=active)


class FakeViewer:
    def __init__(self, layers, active=None):
        self.layers = FakeViewerLayers(layers, active=active)


def test_safe_tail_redacts_to_last_two_parts():
    assert _safe_tail("/a/b/c/d.txt").endswith("c/d.txt")
    assert _safe_tail("single") == "single"


def test_collect_environment_summary_contains_expected_keys():
    env = collect_environment_summary()

    expected = {
        "python",
        "platform",
        "napari-deeplabcut",
        "napari",
        "qtpy",
        "PySide6",
        "PyQt6",
        "shiboken6",
        "numpy",
        "pydantic",
        "plugin_module_path",
    }
    assert expected.issubset(env.keys())


def test_summarize_layer_basic_fields():
    layer = FakeLayer(
        name="points",
        metadata={
            "project": "/tmp/myproj",
            "root": "/tmp/data/images",
            "paths": ["a.png", "b.png"],
            "header": {"dummy": True},
            "tables": {"x": "y"},
            "save_target": {"kind": "gt"},
        },
        properties={"label": ["nose"], "id": ["mouse1"]},
        data=SimpleNamespace(shape=(12, 3)),
        visible=False,
    )

    summary = summarize_layer(layer)

    assert summary["name"] == "points"
    assert summary["type"] == "FakeLayer"
    assert summary["visible"] is False
    assert summary["paths_count"] == 2
    assert summary["has_header"] is True
    assert summary["has_tables"] is True
    assert summary["has_save_target"] is True
    assert "project" in summary
    assert "root" in summary
    assert summary["property_keys"] == ["id", "label"]
    assert summary["data_shape"] == (12, 3)


def test_summarize_layer_handles_missing_or_weird_values():
    layer = FakeLayer(
        name="empty",
        metadata=None,
        properties=None,
        data=None,
        visible=True,
    )

    summary = summarize_layer(layer)

    assert summary["name"] == "empty"
    assert summary["metadata_keys"] == []
    assert summary["property_keys"] == []
    assert summary["data_shape"] is None
    assert summary["has_header"] is False
    assert summary["has_tables"] is False
    assert summary["has_save_target"] is False


def test_summarize_viewer_basic():
    layer1 = FakeLayer(name="img")
    layer2 = FakeLayer(name="pts")
    viewer = FakeViewer([layer1, layer2], active=layer2)

    summary = summarize_viewer(viewer)

    assert summary["n_layers"] == 2
    assert summary["active_layer"] == "pts"
    assert len(summary["layers"]) == 2
    assert summary["layers"][0]["name"] == "img"
    assert summary["layers"][1]["name"] == "pts"


def test_format_debug_report_contains_sections():
    text = format_debug_report(
        env={"python": "3.x", "platform": "test-os"},
        viewer_summary={
            "n_layers": 2,
            "active_layer": "pts",
            "layers": [
                {
                    "name": "pts",
                    "type": "Points",
                    "visible": True,
                    "data_shape": (10, 3),
                    "metadata_keys": ["header", "paths"],
                    "property_keys": ["label"],
                    "project": "proj/config",
                    "root": "data/images",
                    "paths_count": 2,
                    "has_header": True,
                    "has_tables": False,
                    "has_save_target": False,
                }
            ],
        },
        logs_text="12:00:00 | DEBUG | napari-deeplabcut | hello",
    )

    assert "## Environment" in text
    assert "## Viewer" in text
    assert "## Layers" in text
    assert "## Recent plugin logs" in text
    assert "hello" in text
    assert "pts" in text
