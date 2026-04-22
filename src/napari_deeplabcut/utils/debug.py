# src/napari_deeplabcut/utils/debug.py
from __future__ import annotations

import logging
import platform
import sys
import threading
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter_ns

_DEBUG_HANDLER_ATTR = "_napari_dlc_debug_recorder"
LOG_QUEUE_MAXLEN = 1000

# FIXME disable for release to avoid any overhead, or make configurable via env var/settings
NAPARI_DLC_LOG_TIMING = True


@contextmanager
def log_timing(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.DEBUG,
    threshold_ms: float | None = None,
):
    """Lightweight scoped timer for debug instrumentation.

    Uses perf_counter_ns() for monotonic timing.
    Logs only if logger is enabled for the requested level.
    Optionally suppresses tiny timings below threshold_ms.
    """
    if not logger.isEnabledFor(level) or not NAPARI_DLC_LOG_TIMING:
        yield
        return

    t0 = perf_counter_ns()
    try:
        yield
    finally:
        dt_ms = (perf_counter_ns() - t0) / 1e6
        if threshold_ms is None or dt_ms >= threshold_ms:
            logger.log(level, "%s took %.3f ms", label, dt_ms)


def install_debug_recorder(
    *,
    logger_name: str = "napari-deeplabcut",
    capacity: int = LOG_QUEUE_MAXLEN,
) -> InMemoryDebugRecorder:
    """
    Attach a single in-memory recorder to the plugin logger namespace.
    Idempotent: repeated calls return the same recorder.
    """
    root_logger = logging.getLogger(logger_name)

    existing = getattr(root_logger, _DEBUG_HANDLER_ATTR, None)
    if isinstance(existing, InMemoryDebugRecorder):
        return existing

    recorder = InMemoryDebugRecorder(capacity=capacity, level=logging.DEBUG)
    recorder.set_name("napari-deeplabcut-debug-recorder")

    # Important:
    # - attach only to plugin namespace, not global root logger
    # - set logger level to DEBUG so plugin debug calls are emitted
    # - keep propagation unchanged
    root_logger.addHandler(recorder)
    root_logger.setLevel(logging.DEBUG)

    setattr(root_logger, _DEBUG_HANDLER_ATTR, recorder)
    return recorder


def get_debug_recorder(
    *,
    logger_name: str = "napari-deeplabcut",
) -> InMemoryDebugRecorder | None:
    logger = logging.getLogger(logger_name)
    recorder = getattr(logger, _DEBUG_HANDLER_ATTR, None)
    return recorder if isinstance(recorder, InMemoryDebugRecorder) else None


# --------------------------
#  Debug infrastructure
# --------------------------

# Recorder


@dataclass(frozen=True)
class RecordedLog:
    created: float
    level: str
    logger_name: str
    message: str
    exc_text: str | None = None


class InMemoryDebugRecorder(logging.Handler):
    """
    Lightweight, fail-open in-memory log recorder.

    Safety properties:
    - bounded memory via deque(maxlen=...)
    - no file/network I/O
    - swallow-all-errors in emit()
    - does not log from inside itself
    - stores only small text snapshots
    """

    def __init__(self, *, capacity: int = LOG_QUEUE_MAXLEN, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._records: deque[RecordedLog] = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Never call logging from here.
            # Never inspect plugin objects.
            msg = self._safe_message(record)
            exc_text = self._safe_exception_text(record)

            snap = RecordedLog(
                created=float(getattr(record, "created", 0.0) or 0.0),
                level=str(getattr(record, "levelname", "UNKNOWN")),
                logger_name=str(getattr(record, "name", "")),
                message=msg,
                exc_text=exc_text,
            )

            with self._lock:
                self._records.append(snap)

        except Exception:
            # Fail open: never let diagnostics interfere with runtime behavior.
            try:
                self._dropped += 1
            except Exception:
                pass

    def clear(self) -> None:
        try:
            with self._lock:
                self._records.clear()
                self._dropped = 0
        except Exception:
            pass

    def snapshot(self) -> list[RecordedLog]:
        try:
            with self._lock:
                return list(self._records)
        except Exception:
            return []

    def render_text(self, *, limit: int = 200) -> str:
        lines: list[str] = []
        try:
            records = self.snapshot()[-max(1, int(limit)) :]
            if not records:
                return ""

            base = records[0].created
            for rec in records:
                ts = datetime.fromtimestamp(rec.created).strftime("%H:%M:%S.%f")[:-3]
                rel_ms = (rec.created - base) * 1000.0
                lines.append(f"{ts} (+{rel_ms:8.1f} ms) | {rec.level:<8} | {rec.logger_name} | {rec.message}")
                if rec.exc_text:
                    lines.append(rec.exc_text.rstrip())

            if self._dropped:
                lines.append(f"[debug-recorder] dropped internal failures: {self._dropped}")
        except Exception:
            return "[debug-recorder] failed to render logs"
        return "\n".join(lines)

    @staticmethod
    def _safe_message(record: logging.LogRecord) -> str:
        try:
            return record.getMessage()
        except Exception:
            try:
                return str(record.msg)
            except Exception:
                return "<unrenderable log message>"

    @staticmethod
    def _safe_exception_text(record: logging.LogRecord) -> str | None:
        try:
            if not record.exc_info:
                return None
            return "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            return "<exception details unavailable>"


# Env info


def _version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except Exception:
        return "not-installed"


def _module_path(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        p = getattr(mod, "__file__", None)
        return str(Path(p).resolve()) if p else "unknown"
    except Exception:
        return "unknown"


def _safe_tail(pathlike) -> str:
    """
    Redact user-specific absolute paths:
    keep only the last 2 path components when possible.
    """
    try:
        p = Path(str(pathlike))
        parts = p.parts
        if len(parts) >= 2:
            return str(Path(*parts[-2:]).as_posix())
        return str(p.as_posix())
    except Exception:
        return str(pathlike)


def collect_environment_summary() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "napari-deeplabcut": _version("napari-deeplabcut"),
        "napari": _version("napari"),
        "qtpy": _version("QtPy"),
        "PySide6": _version("PySide6"),
        "PyQt6": _version("PyQt6"),
        "shiboken6": _version("shiboken6"),
        "numpy": _version("numpy"),
        "pydantic": _version("pydantic"),
        "plugin_module_path": _module_path("napari_deeplabcut"),
    }


def summarize_layer(layer) -> dict[str, object]:
    try:
        md = dict(getattr(layer, "metadata", {}) or {})
    except Exception:
        md = {}

    def _len_or_none(x):
        try:
            return len(x)
        except Exception:
            return None

    summary = {
        "name": getattr(layer, "name", "<unnamed>"),
        "type": type(layer).__name__,
        "visible": bool(getattr(layer, "visible", True)),
        "metadata_keys": sorted(md.keys()),
        "project": _safe_tail(md.get("project")) if md.get("project") else None,
        "root": _safe_tail(md.get("root")) if md.get("root") else None,
        "paths_count": _len_or_none(md.get("paths")),
        "has_header": md.get("header") is not None,
        "has_tables": bool(md.get("tables")),
        "has_save_target": md.get("save_target") is not None,
    }

    # Optional lightweight properties summary
    try:
        props = getattr(layer, "properties", {}) or {}
        summary["property_keys"] = sorted(props.keys())
    except Exception:
        summary["property_keys"] = []

    # Optional lightweight data shape
    try:
        data = getattr(layer, "data", None)
        summary["data_shape"] = getattr(data, "shape", None)
    except Exception:
        summary["data_shape"] = None

    return summary


def summarize_viewer(viewer) -> dict[str, object]:
    try:
        layers = list(viewer.layers)
    except Exception:
        layers = []

    try:
        active = viewer.layers.selection.active
        active_name = getattr(active, "name", None) if active is not None else None
    except Exception:
        active_name = None

    return {
        "n_layers": len(layers),
        "active_layer": active_name,
        "layers": [summarize_layer(layer) for layer in layers],
    }


def format_debug_report(
    *,
    env: dict[str, str],
    viewer_summary: dict[str, object],
    logs_text: str,
) -> str:
    lines: list[str] = []

    lines.append("## Environment")
    for k, v in env.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Viewer")
    lines.append(f"- n_layers: {viewer_summary.get('n_layers')}")
    lines.append(f"- active_layer: {viewer_summary.get('active_layer')}")

    lines.append("")
    lines.append("## Layers")
    for idx, layer in enumerate(viewer_summary.get("layers", []), start=1):
        lines.append(f"- [{idx}] name={layer.get('name')!r}, type={layer.get('type')}, visible={layer.get('visible')}")
        lines.append(f"  - data_shape: {layer.get('data_shape')}")
        lines.append(f"  - metadata_keys: {layer.get('metadata_keys')}")
        lines.append(f"  - property_keys: {layer.get('property_keys')}")
        lines.append(f"  - project: {layer.get('project')}")
        lines.append(f"  - root: {layer.get('root')}")
        lines.append(f"  - paths_count: {layer.get('paths_count')}")
        lines.append(f"  - has_header: {layer.get('has_header')}")
        lines.append(f"  - has_tables: {layer.get('has_tables')}")
        lines.append(f"  - has_save_target: {layer.get('has_save_target')}")

    lines.append("")
    lines.append("## Recent plugin logs")
    lines.append("```text")
    lines.append(logs_text or "<no captured logs>")
    lines.append("```")

    return "\n".join(lines)


def build_debug_report(
    *,
    viewer,
    recorder: InMemoryDebugRecorder | None,
    log_limit: int = 300,
) -> str:
    logs_text = recorder.render_text(limit=log_limit) if recorder is not None else "<debug recorder unavailable>"

    env = collect_environment_summary()
    viewer_summary = summarize_viewer(viewer)

    return format_debug_report(
        env=env,
        viewer_summary=viewer_summary,
        logs_text=logs_text,
    )
