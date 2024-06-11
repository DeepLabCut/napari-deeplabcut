import logging
import warnings

# FIXME: Circumvent the need to access window.qt_viewer
warnings.filterwarnings("ignore", category=FutureWarning)


class VispyWarningFilter(logging.Filter):
    def filter(self, record):
        ignore_messages = [
            "delivering touch release to same window QWindow(0x0) not QWidgetWindow",
            "skipping QEventPoint",
        ]
        return not any(msg in record.getMessage() for msg in ignore_messages)


vispy_logger = logging.getLogger("vispy")
vispy_logger.addFilter(VispyWarningFilter())

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"


from ._reader import (
    get_hdf_reader,
    get_image_reader,
    get_video_reader,
    get_folder_parser,
    get_config_reader,
)
from ._writer import write_hdf, write_masks
