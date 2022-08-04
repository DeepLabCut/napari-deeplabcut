import warnings

# FIXME: Circumvent the need to access window.qt_viewer
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import (
    get_hdf_reader,
    get_image_reader,
    get_video_reader,
    get_folder_parser,
    get_config_reader,
)
from ._writer import write_hdf, write_masks
