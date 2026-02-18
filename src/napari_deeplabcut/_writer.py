"""Writers for DeepLabCut data formats."""

# src/napari_deeplabcut/_writer.py
import logging
import os
from itertools import groupby
from pathlib import Path

from napari.layers import Shapes
from napari_builtins.io import napari_write_shapes
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
    return write_hdf(None, data, attributes)


# TODO rewrite explicitly as napari-facing func
def _write_image(data, output_path, plugin=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imsave(
        output_path,
        img_as_ubyte(data).squeeze(),
        plugin=plugin,
        check_contrast=False,
    )


# TODO rewrite explicitly as napari-facing func
def write_masks(foldername, data, metadata):
    folder, _ = os.path.splitext(foldername)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, "{}_obj_{}.png")
    shapes = Shapes(data, shape_type="polygon")
    meta = metadata["metadata"]
    frame_inds = [int(array[0, 0]) for array in data]
    shape_inds = []
    for _, group in groupby(frame_inds):
        shape_inds += range(sum(1 for _ in group))
    masks = shapes.to_masks(mask_shape=meta["shape"][1:])
    for n, mask in enumerate(masks):
        image_name = os.path.basename(meta["paths"][frame_inds[n]])
        output_path = filename.format(os.path.splitext(image_name)[0], shape_inds[n])
        _write_image(mask, output_path)
    napari_write_shapes(os.path.join(folder, "vertices.csv"), data, metadata)
    return folder
