from __future__ import annotations

import logging
import warnings

# ---- Package exports ---------------------------------------------------------

# Re-export selected symbols so linters recognize them as used.
__all__ = (
    "get_config_reader",
    "get_folder_parser",
    "get_hdf_reader",
    "get_image_reader",
    "get_video_reader",
    "write_hdf",
    "write_masks",
    "__version__",
)

# Import internal modules to populate the public API.
from ._reader import (  # noqa: F401 (explicit re-export via __all__)
    get_config_reader,
    get_folder_parser,
    get_hdf_reader,
    get_image_reader,
    get_video_reader,
)
from ._writer import (  # noqa: F401 (explicit re-export via __all__)
    write_hdf,
    write_masks,
)

try:
    from ._version import version as __version__  # noqa: F401
except Exception:  # pragma: no cover
    __version__ = "unknown"

# ---- Warnings & logging setup ------------------------------------------------

# FIXME: Circumvent the need to access window.qt_viewer
warnings.filterwarnings("ignore", category=FutureWarning)


class VispyWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ignore_messages = (
            "delivering touch release to same window QWindow(0x0) not QWidgetWindow",
            "skipping QEventPoint",
        )
        msg = record.getMessage()
        return not any(needle in msg for needle in ignore_messages)


vispy_logger = logging.getLogger("vispy")
vispy_logger.addFilter(VispyWarningFilter())
