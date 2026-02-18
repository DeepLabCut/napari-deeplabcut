"""Readers for DeepLabCut data formats."""

# src/napari_deeplabcut/_reader.py
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import pandas as pd
import yaml
from dask import delayed
from dask_image.imread import imread
from napari.types import LayerData
from natsort import natsorted

from napari_deeplabcut import misc
from napari_deeplabcut.config.models import AnnotationKind, IOProvenance
from napari_deeplabcut.core.io import discover_annotations, read_hdf_single
from napari_deeplabcut.core.paths import canonicalize_path, looks_like_dlc_labeled_folder

SUPPORTED_IMAGES = ".jpg", ".jpeg", ".png"
SUPPORTED_VIDEOS = ".mp4", ".mov", ".avi"

logger = logging.getLogger(__name__)


def is_video(filename: str):
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEOS)


def get_hdf_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".h5"):
        return None

    return read_hdf


def get_image_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
        return None

    return read_images


def get_video_reader(path):
    if isinstance(path, str) and any(path.lower().endswith(ext) for ext in SUPPORTED_VIDEOS):
        return read_video
    return None


def get_config_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".yaml"):
        return None

    return read_config


def _filter_extensions(
    image_paths: list[str | Path],
    valid_extensions: tuple[str] = SUPPORTED_IMAGES,
) -> list[Path]:
    """
    Filter image paths by valid extensions.
    """
    return [Path(p) for p in image_paths if Path(p).suffix.lower() in valid_extensions]


def get_folder_parser(path):
    if not path or not Path(path).is_dir():
        return None

    if not looks_like_dlc_labeled_folder(path):
        return None
    layers = []

    images = _filter_extensions(Path(path).iterdir(), valid_extensions=SUPPORTED_IMAGES)

    if not images:
        has_video = any(Path(path).glob(f"*{ext}") for ext in SUPPORTED_VIDEOS)
        if has_video:
            logger.info(
                "No supported images found in '%s' (extensions: %s). "
                "Annotations will not be loaded from images. A supported video appears to be present; "
                "try opening the video directly to view frames and overlay annotations.",
                path,
                SUPPORTED_IMAGES,
            )
        else:
            logger.warning(
                "No supported images found in '%s' (extensions: %s), and no supported videos found (extensions: %s). "
                "Annotations cannot be loaded. Add images or open a supported video.",
                path,
                SUPPORTED_IMAGES,
                SUPPORTED_VIDEOS,
            )
        return None

    image_layer = read_images(images)
    layers.extend(image_layer)

    # ---------------------------------------------------------------------
    # Deterministic, safe discovery: load ALL relevant H5 files in the folder.
    # We explicitly avoid "first match wins" behavior.
    # ---------------------------------------------------------------------
    artifacts = discover_annotations(path)
    h5_artifacts = [(Path(a.h5_path), getattr(a, "kind", None)) for a in artifacts if a.h5_path is not None]

    if not h5_artifacts:
        # No H5 artifacts found. Keep existing behavior: only images will load.
        # (We do not attempt to load CSV-only here because policy is "H5 only".)
        return lambda _: layers

    errors: list[tuple[type, str, Exception]] = []

    for h5_path, k in h5_artifacts:
        try:
            kind = k if isinstance(k, AnnotationKind) else (AnnotationKind(k) if isinstance(k, str) and k else None)
            layers.extend(read_hdf_single(h5_path, kind=kind))
        except Exception as e:
            # Don't silently skip: collect errors and fail deterministically
            # if nothing else loads successfully.
            logger.debug("Could not read annotation data from %s", h5_path, exc_info=True)
            exc = (RuntimeError, f"Could not read annotation data from {h5_path}", e)
            errors.append(exc)

    # If we found H5 files but failed to load all of them, raise only if none loaded.
    # This keeps folder-open robust when one artifact is corrupt.
    n_points_layers = sum(1 for _, _, layer_type in layers if layer_type == "points")
    if n_points_layers == 0 and errors:
        exc_type, msg, cause = errors[0]
        raise exc_type(msg) from cause

    return lambda _: layers


# Helper functions for lazy image reading and normalization
# NOTE : forced keyword-only arguments for clarity
def _read_and_normalize(*, filepath: Path, normalize_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise OSError(f"Could not read image: {filepath}")
    return normalize_func(arr)


def _normalize_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _expand_image_paths(path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    # Normalize input to list[Path]
    raw_paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]

    expanded: list[Path] = []
    for p in raw_paths:
        if p.is_dir() and p.suffix.lower() != ".zarr":
            file_matches: list[Path] = []
            for ext in SUPPORTED_IMAGES:
                file_matches.extend(p.glob(f"*{ext}"))
            expanded.extend(x for x in natsorted(file_matches, key=str) if x.is_file())
        else:
            matches = list(p.parent.glob(p.name))
            expanded.extend(matches or [p])

    return [p for p in expanded if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES]


# Lazy image reader that supports directories and lists of files
def _lazy_imread(
    filenames: str | Path | list[str | Path],
    use_dask: bool | None = None,
    stack: bool = True,
) -> np.ndarray | da.Array | list[np.ndarray | da.Array]:
    """Lazily reads one or more images with optional Dask support.

    Resolves file paths using `_expand_image_paths`, ensuring consistent
    handling of directories, glob patterns, and lists/tuples of paths.
    Images are normalized to RGB and may be wrapped in Dask delayed
    objects for lazy loading.

    Behavior:
        * If a single image is resolved:
            - The image is read eagerly and returned as a NumPy array.
        * If multiple images are resolved:
            - The first image is read eagerly to determine shape and dtype.
            - Subsequent images are loaded lazily via Dask unless
              `use_dask=False`.
            - Stacking behavior is controlled by `stack`.

    Args:
        filenames (str | Path | list[str | Path]):
            File path(s), directory, or glob pattern(s) to load.
        use_dask (bool | None, optional):
            Whether to load images lazily using Dask.
            Defaults to `True` when multiple files are found, otherwise
            `False`.
        stack (bool, optional):
            If True, stack images along axis 0 into a single array.
            If False, return a list of arrays or delayed arrays.
            Defaults to True.

    Returns:
        np.ndarray | da.Array | list[np.ndarray | da.Array]:
            Loaded image data. The return type depends on the number of
            images found, the `use_dask` flag, and the `stack` option.

    Raises:
        ValueError: If no supported images are found.
    """
    expanded = _expand_image_paths(filenames)

    if not expanded:
        raise ValueError(f"No supported images were found for input: {filenames}")

    if use_dask is None:
        use_dask = len(expanded) > 1

    images = []
    first_shape = None
    first_dtype = None

    def make_delayed_array(fp: Path, first_shape: tuple[int, ...], first_dtype: np.dtype) -> da.Array:
        """Create a dask array for a single file."""
        return da.from_delayed(
            delayed(_read_and_normalize)(filepath=fp, normalize_func=_normalize_to_rgb),
            shape=first_shape,
            dtype=first_dtype,
        )

    for fp in expanded:
        if first_shape is None:
            arr0 = _read_and_normalize(filepath=fp, normalize_func=_normalize_to_rgb)
            first_shape = arr0.shape
            first_dtype = arr0.dtype

            if use_dask:
                images.append(make_delayed_array(fp, first_shape, first_dtype))
            else:
                images.append(arr0)
            continue

        if use_dask:
            images.append(make_delayed_array(fp, first_shape, first_dtype))
        else:
            images.append(_read_and_normalize(filepath=fp, normalize_func=_normalize_to_rgb))

    if len(images) == 1:
        return images[0]

    try:
        return da.stack(images) if use_dask and stack else (np.stack(images) if stack else images)
    except ValueError as e:
        raise ValueError(
            "Cannot stack images with different shapes using NumPy. "
            "Ensure all images have the same shape or set stack=False."
        ) from e






# Read images from a list of files or a glob/string path
def read_images(path: str | Path | list[str | Path]):
    """Reads one or multiple images and returns a Napari Image layer.

    Uses `_expand_image_paths` to resolve the input into a list of valid
    image files. Supports single paths, glob expressions, directories,
    and lists or tuples of such paths.

    Behavior:
        * If one file is found:
            - Loaded using `dask_image.imread.imread`.
        * If multiple files are found:
            - Loaded lazily using `lazy_imread` into a stacked image
              layer.

    Args:
        path (str | Path | list[str | Path]):
            Input path(s), directory, or glob pattern(s) to expand into
            supported image files.

    Returns:
        list[LayerData]:
            A list containing one Napari layer tuple of the form
            `(data, metadata, "image")`.

    Raises:
        OSError: If no supported images are found after expansion.
    """
    filepaths = _expand_image_paths(path)

    if not filepaths:
        raise OSError(f"No supported images were found in {path}")

    filepaths = natsorted(filepaths, key=str)

    # Multiple images → lazy-imread stack
    if len(filepaths) > 1:
        relative_paths = [canonicalize_path(fp, 3) for fp in filepaths]
        params = {
            "name": "images",
            "metadata": {
                "paths": relative_paths,
                "root": str(filepaths[0].parent),
            },
        }
        data = _lazy_imread(filepaths, use_dask=True, stack=True)
        return [(data, params, "image")]

    # Single image → old behavior
    image_path = filepaths[0]
    params = {
        "name": "images",
        "metadata": {
            "paths": [canonicalize_path(image_path, 3)],
            "root": str(image_path.parent),
        },
    }
    return [(imread(str(image_path)), params, "image")]


# Helper to populate keypoint layer metadata
def _populate_metadata(
    header: misc.DLCHeader,
    *,
    labels: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    likelihood: Sequence[float] | None = None,
    paths: list[str] | None = None,
    size: int | None = 8,
    pcutoff: float | None = 0.6,
    colormap: str | None = "viridis",
) -> dict:
    if labels is None:
        labels = header.bodyparts
    if ids is None:
        ids = header.individuals
    if likelihood is None:
        likelihood = np.ones(len(labels))
    face_color_cycle_maps = misc.build_color_cycles(header, colormap)
    face_color_prop = "id" if ids[0] else "label"
    return {
        "name": "keypoints",
        "text": "{id}–{label}" if ids[0] else "label",
        "properties": {
            "label": list(labels),
            "id": list(ids),
            "likelihood": likelihood,
            "valid": likelihood > pcutoff,
        },
        "face_color_cycle": face_color_cycle_maps[face_color_prop],
        "face_color": face_color_prop,
        "face_colormap": colormap,
        "border_color": "valid",
        "border_color_cycle": ["black", "red"],
        "border_width": 0,
        "border_width_is_relative": False,
        "size": size,
        "metadata": {
            "header": header,
            "face_color_cycles": face_color_cycle_maps,
            "colormap_name": colormap,
            "paths": paths or [],
        },
    }


def _load_superkeypoints_diagram(super_animal: str):
    path = str(Path(__file__).parent / "assets" / f"{super_animal}.jpg")
    try:
        return imread(path), {"root": ""}, "images"
    except Exception as e:
        raise FileNotFoundError(f"Superkeypoints diagram not found for {super_animal}.") from e


def _load_superkeypoints(super_animal: str):
    path = str(Path(__file__).parent / "assets" / f"{super_animal}.json")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Superkeypoints JSON file not found for {super_animal}.")
    with open(path) as f:
        return json.load(f)


def _load_config(config_path: str):
    with open(config_path) as file:
        return yaml.safe_load(file)


# Read config file and create keypoint layer metadata
def read_config(configname: str) -> list[LayerData]:
    config = _load_config(configname)
    header = misc.DLCHeader.from_config(config)
    metadata = _populate_metadata(
        header,
        size=config["dotsize"],
        pcutoff=config["pcutoff"],
        colormap=config["colormap"],
        likelihood=np.array([1]),
    )
    metadata["name"] = f"CollectedData_{config['scorer']}"
    metadata["ndim"] = 3
    metadata["property_choices"] = metadata.pop("properties")
    metadata["metadata"]["project"] = str(Path(configname).parent)
    conversion_tables = config.get("SuperAnimalConversionTables")
    if conversion_tables is not None:
        super_animal, table = conversion_tables.popitem()
        metadata["metadata"]["tables"] = {super_animal: table}
    return [(None, metadata, "points")]


# Read HDF file and create keypoint layers
def read_hdf(filename: str) -> list[LayerData]:
    layers = []
    for file in Path(filename).parent.glob(Path(filename).name):
        layers.extend(read_hdf_single(file))
    return layers


# Video reader using OpenCV
class Video:
    def __init__(self, video_path):
        if not Path(video_path).is_file():
            raise ValueError(f'Video path "{video_path}" does not point to a file.')

        self.path = video_path
        self.stream = cv2.VideoCapture(video_path)
        if not self.stream.isOpened():
            raise OSError("Video could not be opened.")

        self._n_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame = cv2.UMat(self._height, self._width, cv2.CV_8UC3)

    def __len__(self):
        return self._n_frames

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def set_to_frame(self, ind):
        ind = min(ind, len(self) - 1)
        ind += 1  # Unclear why this is needed at all
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, ind)

    def read_frame(self):
        self.stream.retrieve(self._frame)
        cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB, self._frame, 3)
        return self._frame.get()

    def close(self):
        self.stream.release()


def read_video(filename: str, opencv: bool = True):
    if opencv:
        stream = Video(filename)
        # NOTE construct output shape tuple in (H, W, C) order to match read_frame() data
        shape = stream.height, stream.width, 3

        def _read_frame(ind):
            stream.set_to_frame(ind)
            return stream.read_frame()

        lazy_reader = delayed(_read_frame)
    else:  # pragma: no cover
        from pims import PyAVReaderIndexed

        try:
            stream = PyAVReaderIndexed(filename)
        except ImportError:
            raise ImportError("`pip install av` to use the PyAV video reader.") from None

        shape = stream.frame_shape
        lazy_reader = delayed(stream.get_frame)

    movie = da.stack([da.from_delayed(lazy_reader(i), shape=shape, dtype=np.uint8) for i in range(len(stream))])
    elems = list(Path(filename).parts)
    elems[-2] = "labeled-data"
    elems[-1] = Path(elems[-1]).stem  # + Path(filename).suffix
    root = str(Path(*elems))
    params = {
        "name": filename,
        "metadata": {
            "root": root,
        },
    }
    return [(movie, params, "image")]
