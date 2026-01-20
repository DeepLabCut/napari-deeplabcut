import json
from collections.abc import Sequence
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


def get_folder_parser(path):
    if not path or not Path(path).is_dir():
        return None

    layers = []
    files = Path(path).iterdir()
    images = ""
    for file in files:
        if any(file.name.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
            images = str(Path(path) / f"*{Path(file.name).suffix}")
            break
    if not images:
        raise OSError(f"No supported images were found in {path}.")

    layers.extend(read_images(images))
    for file in Path(path).iterdir():
        if file.name.endswith(".h5"):
            layers.extend(read_hdf(str(file)))
            break  # one h5 per annotated video

    return lambda _: layers


def read_images(path):
    if isinstance(path, list):
        first_path = Path(path[0])
        suffixes = first_path.suffixes
        ext = "".join(suffixes) if suffixes else ""
        pattern = f"*{ext}" if ext else "*"
        path = str(first_path.parent / pattern)
    # Retrieve filepaths exactly as parsed by pims
    filepaths = []
    for filepath in Path(path).parent.glob(Path(path).name):
        relpath = Path(filepath).parts[-3:]
        filepaths.append(str(Path(*relpath)))
    params = {
        "name": "images",
        "metadata": {
            "paths": natsorted(filepaths),
            "root": str(Path(path).parent),
        },
    }

    # https://github.com/soft-matter/pims/issues/452
    if len(filepaths) == 1:
        path = next(Path(path).parent.glob(Path(path).name), None)
        if path is None:
            raise FileNotFoundError(f"No files found for pattern: {path}")
    return [(imread(path), params, "image")]


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
