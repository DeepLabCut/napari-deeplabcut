import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import dask.array as da
import numpy as np
import pandas as pd
import yaml
from dask import delayed
from dask_image.imread import imread
from napari.types import LayerData

from napari_deeplabcut import misc

SUPPORTED_IMAGES = "jpg", "jpeg", "png"
SUPPORTED_VIDEOS = "mp4", "mov", "avi"


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
    if isinstance(path, str) and any(
        path.lower().endswith(ext) for ext in SUPPORTED_VIDEOS
    ):
        return read_video
    return None


def get_config_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".yaml"):
        return None

    return read_config


def get_folder_parser(path):
    if not os.path.isdir(path):
        return None

    layers = []
    files = os.listdir(path)
    images = ""
    for file in files:
        if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
            images = os.path.join(path, f"*{os.path.splitext(file)[1]}")
            break
    if not images:
        raise OSError("No supported images were found.")

    layers.extend(read_images(images))
    datafile = ""
    for file in os.listdir(path):
        if file.endswith(".h5"):
            datafile = os.path.join(path, "*.h5")
            break
    if datafile:
        layers.extend(read_hdf(datafile))

    return lambda _: layers


def read_images(path):
    if isinstance(path, list):
        root, ext = os.path.splitext(path[0])
        path = os.path.join(os.path.dirname(root), f"*{ext}")
    # Retrieve filepaths exactly as parsed by pims
    filepaths = []
    for filepath in glob.iglob(path):
        _, *relpath = filepath.rsplit(os.sep, 3)
        filepaths.append(os.path.join(*relpath))
    params = {
        "name": "images",
        "metadata": {
            "paths": sorted(filepaths),
            "root": os.path.split(path)[0],
        },
    }
    return [(imread(path), params, "image")]


def _populate_metadata(
    header: misc.DLCHeader,
    *,
    labels: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    likelihood: Optional[Sequence[float]] = None,
    paths: Optional[List[str]] = None,
    size: Optional[int] = 8,
    pcutoff: Optional[float] = 0.6,
    colormap: Optional[str] = "viridis",
) -> Dict:
    if labels is None:
        labels = header.bodyparts
    if ids is None:
        ids = header.individuals
    if likelihood is None:
        likelihood = np.ones(len(labels))
    label_colors = misc.build_color_cycle(len(header.bodyparts), colormap)
    id_colors = misc.build_color_cycle(len(header.individuals), colormap)
    face_color_cycle_maps = {
        "label": dict(zip(header.bodyparts, label_colors)),
        "id": dict(zip(header.individuals, id_colors)),
    }
    return {
        "name": "keypoints",
        "text": "{id}–{label}" if ids[0] else "label",
        "properties": {
            "label": list(labels),
            "id": list(ids),
            "likelihood": likelihood,
            "valid": likelihood > pcutoff,
        },
        "face_color_cycle": label_colors,
        "face_color": "label",
        "face_colormap": colormap,
        "edge_color": "valid",
        "edge_color_cycle": ["black", "red"],
        "edge_width": 0,
        "edge_width_is_relative": False,
        "size": size,
        "metadata": {
            "header": header,
            "face_color_cycles": face_color_cycle_maps,
            "paths": paths or [],
        },
    }


def _load_config(config_path: str):
    with open(config_path) as file:
        return yaml.safe_load(file)


def read_config(configname: str) -> List[LayerData]:
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
    metadata["metadata"]["project"] = os.path.dirname(configname)
    return [(None, metadata, "points")]


def read_hdf(filename: str) -> List[LayerData]:
    layers = []
    for filename in glob.iglob(filename):
        temp = pd.read_hdf(filename)
        header = misc.DLCHeader(temp.columns)
        temp = temp.droplevel("scorer", axis=1)
        if "individuals" not in temp.columns.names:
            # Append a fake level to the MultiIndex
            # to make it look like a multi-animal DataFrame
            old_idx = temp.columns.to_frame()
            old_idx.insert(0, "individuals", "")
            temp.columns = pd.MultiIndex.from_frame(old_idx)
        if isinstance(temp.index, pd.MultiIndex):
            temp.index = [os.path.join(*row) for row in temp.index]
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
        )
        metadata["name"] = os.path.split(filename)[1].split(".")[0]
        metadata["metadata"]["root"] = os.path.split(filename)[0]
        # Store file name in case the layer's name is edited by the user
        metadata["metadata"]["name"] = metadata["name"]
        layers.append((data, metadata, "points"))
    return layers


class Video:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
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
    else:
        from pims import PyAVReaderIndexed

        try:
            stream = PyAVReaderIndexed(filename)
        except ImportError:
            raise ImportError("`pip install av` to use the PyAV video reader.")

        shape = stream.frame_shape
        lazy_imread = delayed(stream.get_frame)

    movie = da.stack(
        [
            da.from_delayed(lazy_imread(i), shape=shape, dtype=np.uint8)
            for i in range(len(stream))
        ]
    )
    elems = list(Path(filename).parts)
    elems[-2] = "labeled-data"
    elems[-1] = elems[-1].split(".")[0]
    root = os.path.join(*elems)
    params = {
        "name": os.path.split(filename)[1],
        "metadata": {
            "root": root,
        },
    }
    return [(movie, params)]
