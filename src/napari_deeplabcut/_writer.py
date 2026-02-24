"""Writers for DeepLabCut data formats."""

# src/napari_deeplabcut/_writer.py
import logging
from pathlib import Path

from skimage.io import imsave
from skimage.util import img_as_ubyte

from napari_deeplabcut.core.io import write_hdf

logger = logging.getLogger(__name__)


def write_hdf_napari_dlc(path: str, data, attributes: dict) -> list[str]:
    if not path:
        path = "__dlc__.h5"  # dummy path to trigger napari-deeplabcut-specific handling in write_hdf
    if path != "__dlc__.h5":
        logger.info(
            "This function should not be used with a user-specified path."
            "Layer metadata from the reader (in attributes) is used to decide where to save rather than user input."
            "One path that requires user input is when machine labels"
            "are refined by a human (as we do not want to overwrite machine labels),"
            "but that case is handled separately."
        )
    return write_hdf(path, data, attributes)


# TODO rewrite explicitly as napari-facing func
def _write_image(data, output_path, plugin=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imsave(
        output_path,
        img_as_ubyte(data).squeeze(),
        plugin=plugin,
        check_contrast=False,
    )
