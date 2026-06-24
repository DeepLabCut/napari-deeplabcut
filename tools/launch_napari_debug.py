# tools/launch_napari_debug.py
"""This script launches napari with aggressive logging and crash diagnostics for debugging.

While the logging is far too low level for most dev work,
when segmentation faults or Qt-related C++ errors start surfacing,
this script can help capture the events leading up to the crash,
and potentially the stack trace of the crash directly.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Environment variables: set these BEFORE importing napari/Qt/vispy.
# ---------------------------------------------------------------------

os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Useful Qt/VisPy diagnostics. Keep these non-invasive by default.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=true")
os.environ.setdefault("VISPY_LOG_LEVEL", "debug")

# Optional diagnostics you can enable from CLI flags below:
#   QT_OPENGL=software
#   QT_FATAL_WARNINGS=1


class Tee:
    """Write stdout/stderr to terminal and log file at the same time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _install_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"napari-debug-{timestamp}.log"

    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    # Tee print output and tracebacks.
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    faulthandler.enable(file=log_file, all_threads=True)

    # Periodic stack dumps help if the UI hangs before crashing.
    faulthandler.dump_traceback_later(
        15,
        repeat=True,
        file=log_file,
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d : %(levelname)s : %(threadName)s : %(name)s : %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )

    for logger_name in [
        "napari",
        "vispy",
        "qt",
        "napari_deeplabcut",
        "napari_deeplabcut.core.layer_lifecycle.manager",
    ]:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    def excepthook(exc_type, exc, tb):
        logging.critical("Uncaught Python exception", exc_info=(exc_type, exc, tb))
        traceback.print_exception(exc_type, exc, tb)
        try:
            log_file.flush()
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    logging.critical("Debug logging installed")
    logging.critical("PID: %s", os.getpid())
    logging.critical("Log file: %s", log_path)

    return log_path


def _install_qt_message_handler():
    from qtpy.QtCore import qInstallMessageHandler

    def qt_message_handler(mode, context, message):
        logging.getLogger("qt").warning(
            "Qt message mode=%r file=%s line=%s function=%s: %s",
            mode,
            getattr(context, "file", None),
            getattr(context, "line", None),
            getattr(context, "function", None),
            message,
        )

    qInstallMessageHandler(qt_message_handler)
    logging.critical("Qt message handler installed")


def _install_viewer_event_trace(viewer):
    """Trace layer/selection events leading up to the crash."""
    logger = logging.getLogger("napari_deeplabcut.crash_trace")

    def lname(layer):
        if layer is None:
            return None
        return f"{type(layer).__name__}({getattr(layer, 'name', None)!r}, id={id(layer)})"

    def log_event(label):
        def _handler(event=None):
            try:
                value = getattr(event, "value", None)
                index = getattr(event, "index", None)
                active = viewer.layers.selection.active

                logger.critical(
                    "%s event_type=%r index=%r value=%s active=%s selected=%s layers=%s",
                    label,
                    getattr(event, "type", None),
                    index,
                    lname(value),
                    lname(active),
                    [lname(layer) for layer in viewer.layers.selection],
                    [lname(layer) for layer in viewer.layers],
                )
            except Exception:
                logger.exception("Failed to log viewer event: %s", label)

        return _handler

    layer_events = viewer.layers.events

    for name in [
        "inserting",
        "inserted",
        "removing",
        "removed",
        "reordered",
        "changed",
    ]:
        if hasattr(layer_events, name):
            getattr(layer_events, name).connect(log_event(f"layers.{name}"))

    selection_events = viewer.layers.selection.events

    if hasattr(selection_events, "active"):
        selection_events.active.connect(log_event("selection.active"))

    if hasattr(selection_events, "changed"):
        selection_events.changed.connect(log_event("selection.changed"))

    logger.critical("Viewer event tracing installed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch napari with aggressive crash/event logging for napari-deeplabcut debugging."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files/folders to open after napari starts.",
    )
    parser.add_argument(
        "--log-dir",
        default="crash-logs",
        help="Directory where debug logs are written. Default: crash-logs",
    )
    parser.add_argument(
        "--software-gl",
        action="store_true",
        help="Set QT_OPENGL=software before importing napari. Useful to test GPU/driver involvement.",
    )
    parser.add_argument(
        "--fatal-qt-warnings",
        action="store_true",
        help="Set QT_FATAL_WARNINGS=1. Useful but may abort on unrelated Qt warnings.",
    )
    parser.add_argument(
        "--no-event-trace",
        action="store_true",
        help="Disable napari layer/selection event tracing.",
    )

    args = parser.parse_args()

    if args.software_gl:
        os.environ["QT_OPENGL"] = "software"

    if args.fatal_qt_warnings:
        os.environ["QT_FATAL_WARNINGS"] = "1"

    log_path = _install_logging(Path(args.log_dir))

    logging.critical("Environment snapshot:")
    for key in [
        "PYTHONFAULTHANDLER",
        "PYTHONUNBUFFERED",
        "QT_OPENGL",
        "QT_FATAL_WARNINGS",
        "QT_LOGGING_RULES",
        "VISPY_LOG_LEVEL",
        "QT_API",
    ]:
        logging.critical("  %s=%r", key, os.environ.get(key))

    # Import Qt/napari only after env/logging setup.
    _install_qt_message_handler()

    import napari

    logging.critical("Imported napari from: %s", getattr(napari, "__file__", None))
    logging.critical("napari version: %s", getattr(napari, "__version__", None))

    viewer = napari.Viewer(show=True)

    if not args.no_event_trace:
        _install_viewer_event_trace(viewer)

    for path in args.paths:
        try:
            logging.critical("Opening path via viewer.open: %s", path)
            viewer.open(path)
        except Exception:
            logging.exception("Failed to open path: %s", path)

    logging.critical("Starting napari event loop")
    logging.critical("If this process crashes, inspect log file: %s", log_path)

    try:
        napari.run()
    finally:
        logging.critical("napari.run() returned")
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
