import json
from collections.abc import Callable, Sequence
from functools import partial
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

SUPPORTED_IMAGES = ".jpg", ".jpeg", ".png"
SUPPORTED_VIDEOS = ".mp4", ".mov", ".avi"


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
    layers = []

    images = _filter_extensions(Path(path).iterdir(), valid_extensions=SUPPORTED_IMAGES)

    if not images:
        raise OSError(f"No supported images were found in {path} with extensions {SUPPORTED_IMAGES}.")

    image_layer = read_images(images)
    layers.extend(image_layer)
    for file in Path(path).iterdir():
        if file.name.endswith(".h5"):
            try:
                layers.extend(read_hdf(str(file)))
                break  # one h5 per annotated video
            except Exception as e:
                raise RuntimeError(f"Could not read annotation data from {file}") from e
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


# Lazy image reader that supports directories and lists of files
def lazy_imread(
    filenames: str | Path | list[str | Path],
    use_dask: bool | None = None,
    stack: bool = True,
):
    _raw = [Path(p) for p in filenames] if isinstance(filenames, (list, tuple)) else [Path(filenames)]
    if not _raw:
        raise ValueError("No files found")

    expanded: list[Path] = []
    for p in _raw:
        if p.is_dir() and not str(p).endswith(".zarr"):
            expanded.extend([x for x in natsorted(p.glob("*.*")) if not x.is_dir()])
        else:
            expanded.append(p)

    if use_dask is None:
        use_dask = len(expanded) > 1

    expanded = [p for p in expanded if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES]
    if not expanded:
        raise ValueError(f"No files found in {filenames} after removing subdirectories")

    images = []
    first_shape = None
    first_dtype = None

    def make_delayed_array(fp: Path):
        """Create a dask array for a single file."""
        delayed_reader = delayed(partial(_read_and_normalize, filepath=fp, normalize_func=_normalize_to_rgb))
        return da.from_delayed(delayed_reader(), shape=first_shape, dtype=first_dtype)

    for fp in expanded:
        if first_shape is None:
            arr0 = _read_and_normalize(filepath=fp, normalize_func=_normalize_to_rgb)
            if arr0 is None:
                raise OSError(f"Could not read image: {fp}")
            first_shape = arr0.shape
            first_dtype = arr0.dtype

            if use_dask:
                images.append(make_delayed_array(fp))
            else:
                images.append(arr0)
            continue

        if use_dask:
            images.append(make_delayed_array(fp))
        else:
            images.append(_read_and_normalize(filepath=fp, normalize_func=_normalize_to_rgb))

    if not images:
        return None
    if len(images) == 1:
        return images[0]

    return da.stack(images) if use_dask and stack else (np.stack(images) if stack else images)


# Read images from a list of files or a glob/string path
def read_images(path: str | Path | list[str | Path]) -> list[LayerData]:
    """
    Read images from a list of files or a glob/string path.
    - List: filter by SUPPORTED_IMAGES, build metadata, then lazily read via `lazy_imread`
      with padding to allow stacking images of different sizes.
    - Glob/string: preserve previous behavior using `dask_image.imread.imread`.
    """
    if isinstance(path, list):
        filepaths: list[Path] = _filter_extensions(path, valid_extensions=SUPPORTED_IMAGES)
        if not filepaths:
            raise OSError(f"No supported images were found in list with extensions {SUPPORTED_IMAGES}.")
        filepaths = natsorted(filepaths, key=str)

        relative_paths = [str(Path(*fp.parts[-3:])) for fp in filepaths]
        params = {
            "name": "images",
            "metadata": {
                "paths": relative_paths,
                "root": str(filepaths[0].parent),
            },
        }
        data = lazy_imread(filepaths, use_dask=True, stack=True)
        return [(data, params, "image")]

    # Original behavior for glob/string
    image_path = Path(path)
    matches = list(image_path.parent.glob(image_path.name))

    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {image_path}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files match the pattern '{image_path.name}', but only a single image is expected: {matches}"
        )

    # Exactly 1 match
    image_path = matches[0]

    filepaths = [str(Path(*image_path.parts[-3:]))]
    params = {
        "name": "images",
        "metadata": {
            "paths": filepaths,
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
    return imread(path), {"root": ""}, "image"


def _load_superkeypoints(super_animal: str):
    path = str(Path(__file__).parent / "assets" / f"{super_animal}.json")
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
    config_path = misc.find_project_config_path(filename)
    layers = []
    for file in Path(filename).parent.glob(Path(filename).name):
        temp = pd.read_hdf(str(file))
        temp = misc.merge_multiple_scorers(temp)
        header = misc.DLCHeader(temp.columns)
        temp = temp.droplevel("scorer", axis=1)
        if "individuals" not in temp.columns.names:
            # Append a fake level to the MultiIndex
            # to make it look like a multi-animal DataFrame
            old_idx = temp.columns.to_frame()
            old_idx.insert(0, "individuals", "")
            temp.columns = pd.MultiIndex.from_frame(old_idx)
            try:
                cfg = _load_config(config_path)
                colormap = cfg["colormap"]
            except FileNotFoundError:
                colormap = "rainbow"
        else:
            colormap = "Set3"
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
        if np.issubdtype(image_paths.dtype, np.number):
            image_inds = image_paths.values
            paths2inds = []
        else:
            image_inds, paths2inds = misc.encode_categories(
                image_paths,
                return_map=True,
            )
        data[:, 0] = image_inds
        data[:, 1:] = df[["y", "x"]].to_numpy()
        metadata = _populate_metadata(
            header,
            labels=df["bodyparts"],
            ids=df["individuals"],
            likelihood=df.get("likelihood"),
            paths=list(paths2inds),
            colormap=colormap,
        )
        metadata["name"] = Path(filename).stem
        metadata["metadata"]["root"] = str(Path(filename).parent)
        # Store file name in case the layer's name is edited by the user
        metadata["metadata"]["name"] = metadata["name"]
        layers.append((data, metadata, "points"))
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
        shape = stream.width, stream.height, 3

        def _read_frame(ind):
            stream.set_to_frame(ind)
            return stream.read_frame()

        lazy_imread = delayed(_read_frame)
    else:  # pragma: no cover
        from pims import PyAVReaderIndexed

        try:
            stream = PyAVReaderIndexed(filename)
        except ImportError:
            raise ImportError("`pip install av` to use the PyAV video reader.") from None

        shape = stream.frame_shape
        lazy_imread = delayed(stream.get_frame)

    movie = da.stack([da.from_delayed(lazy_imread(i), shape=shape, dtype=np.uint8) for i in range(len(stream))])
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
    return [(movie, params)]
