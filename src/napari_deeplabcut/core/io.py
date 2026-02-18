"""
Core IO utilities.

Includes:
- Config file reading/writing
- HDF reading with provenance attachment
- Lazy image reading with Dask support
- Video reading with OpenCV and optional PyAV fallback
- Superkeypoints diagram and JSON loading
"""
# src/napari_deeplabcut/core/io.py

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

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
from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.layers import populate_keypoint_layer_metadata
from napari_deeplabcut.core.metadata import attach_source_and_io
from napari_deeplabcut.core.paths import canonicalize_path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Supported formats (shared by image/video readers)
# -----------------------------------------------------------------------------
# FIXME move to config/data_formats.py or similar if more formats are added
SUPPORTED_IMAGES = (".jpg", ".jpeg", ".png")
SUPPORTED_VIDEOS = (".mp4", ".mov", ".avi")


# =============================================================================
# CONFIG (YAML)
# =============================================================================


def load_config(config_path: str):
    # NOTE: intentionally minimal; callers own error handling
    with open(config_path) as file:
        return yaml.safe_load(file)


# Read config file and create keypoint layer metadata
def read_config(configname: str) -> list[LayerData]:
    config = load_config(configname)
    # FIXME duplicated DLCHeader misc/models
    header = misc.DLCHeader.from_config(config)
    metadata = populate_keypoint_layer_metadata(
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


def write_config(config_path: str | Path, params: dict[str, Any]) -> None:
    """Write DeepLabCut config.yaml parameters."""
    with open(str(config_path), "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f)


# =============================================================================
# KEYPOINTS / ANNOTATIONS (HDF5)
# =============================================================================
# NOTE: This reader returns a napari Points layer (data + metadata + "points")
# and attaches provenance via attach_source_and_io.


def read_hdf(filename: str) -> list[LayerData]:
    layers = []
    for file in Path(filename).parent.glob(Path(filename).name):
        layers.extend(read_hdf_single(file))
    return layers


def read_hdf_single(file: Path, *, kind: AnnotationKind | None = None) -> list[LayerData]:
    """Read a single H5 file and attach provenance with optional explicit kind."""
    temp = pd.read_hdf(str(file))
    temp = misc.merge_multiple_scorers(temp)
    header = misc.DLCHeader(temp.columns)
    temp = temp.droplevel("scorer", axis=1)

    # Handle legacy/single-animal column layout by inserting empty "individuals" level.
    # Colormap selection also falls back to config when possible.
    if "individuals" not in temp.columns.names:
        old_idx = temp.columns.to_frame()
        old_idx.insert(0, "individuals", "")
        temp.columns = pd.MultiIndex.from_frame(old_idx)
        try:
            cfg = load_config(misc.find_project_config_path(str(file)))
            colormap = cfg["colormap"]
        except FileNotFoundError:
            colormap = "rainbow"
    else:
        colormap = "Set3"

    # If the on-disk index is a MultiIndex (path parts), collapse it to string paths.
    if isinstance(temp.index, pd.MultiIndex):
        temp.index = [str(Path(*row)) for row in temp.index]

    df = (
        temp.stack(["individuals", "bodyparts"])
        .reindex(header.individuals, level="individuals")
        .reindex(header.bodyparts, level="bodyparts")
        .reset_index()
    )

    nrows = df.shape[0]
    data = np.empty((nrows, 3))
    image_paths = df["level_0"]

    # Convert image keys to integer indices when they are already numeric,
    # otherwise encode category paths deterministically.
    if pd.api.types.is_numeric_dtype(getattr(image_paths, "dtype", np.asarray(image_paths).dtype)):
        image_inds = image_paths.values
        paths2inds = []
    else:
        image_inds, paths2inds = misc.encode_categories(
            image_paths,
            is_path=True,
            return_unique=True,
            do_sort=True,
        )

    data[:, 0] = image_inds
    data[:, 1:] = df[["y", "x"]].to_numpy()

    metadata = populate_keypoint_layer_metadata(
        header,
        labels=df["bodyparts"],
        ids=df["individuals"],
        likelihood=df.get("likelihood"),
        paths=list(paths2inds),
        colormap=colormap,
    )
    metadata["name"] = file.stem
    metadata["metadata"]["root"] = str(file.parent)
    metadata["metadata"]["name"] = metadata["name"]

    # Attach provenance. If explicit kind provided, we store it directly.
    if kind is not None:
        meta = metadata.setdefault("metadata", {})
        # Keep legacy source fields too
        attach_source_and_io(metadata, file)
        # Override kind in io to discovered kind
        if isinstance(meta.get("io"), dict):
            meta["io"]["kind"] = kind  # stored as enum value for JSON safety
    else:
        attach_source_and_io(metadata, file)

    return [(data, metadata, "points")]


# =============================================================================
# SUPERKEYPOINTS (assets: diagram + JSON)
# =============================================================================
# NOTE: These are used to support DLCHeader superkeypoints workflows.


def load_superkeypoints_json_from_path(json_path: str | Path):
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Superkeypoints JSON file not found at {json_path}.")
    with open(path) as f:
        return json.load(f)


def load_superkeypoints_diagram_from_path(image_path: str | Path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Superkeypoints diagram not found at {image_path}.")
    try:
        return imread(path).squeeze(), {"root": ""}, "images"
    except Exception as e:
        raise RuntimeError(f"Superkeypoints diagram could not be loaded from {image_path}.") from e


def load_superkeypoints_diagram(super_animal: str):
    path = resources.files("napari_deeplabcut") / "assets" / f"{super_animal}.jpg"
    return load_superkeypoints_diagram_from_path(path)


def load_superkeypoints(super_animal: str):
    path = resources.files("napari_deeplabcut") / "assets" / f"{super_animal}.json"
    return load_superkeypoints_json_from_path(path)


# =============================================================================
# IMAGES (lazy stack with Dask)
# =============================================================================
# NOTE: Image reading uses OpenCV for normalization and Dask for laziness.


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
        # NOTE: canonicalize_path(fp, 3) stores a stable relative-ish path for the UI/metadata.
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


# =============================================================================
# VIDEO (OpenCV; optional PyAV fallback)
# =============================================================================


def is_video(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEOS)


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
