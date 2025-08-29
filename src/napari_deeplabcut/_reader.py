import glob
import json
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
from natsort import natsorted

from napari_deeplabcut import misc
from napari_plugin_engine import napari_hook_implementation
from decord import VideoReader, cpu

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
        raise OSError(f"No supported images were found in {path}.")

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
        relpath = Path(filepath).parts[-3:]
        filepaths.append(os.path.join(*relpath))
    params = {
        "name": "images",
        "metadata": {
            "paths": natsorted(filepaths),
            "root": os.path.split(path)[0],
        },
    }

    # https://github.com/soft-matter/pims/issues/452
    if len(filepaths) == 1:
        path = glob.glob(path)[0]

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
        "edge_color": "valid",
        "edge_color_cycle": ["black", "red"],
        "edge_width": 0,
        "edge_width_is_relative": False,
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
    return imread(path), {"root": ""}, "images"


def _load_superkeypoints(super_animal: str):
    path = str(Path(__file__).parent / "assets" / f"{super_animal}.json")
    with open(path) as f:
        return json.load(f)


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
    conversion_tables = config.get("SuperAnimalConversionTables")
    if conversion_tables is not None:
        super_animal, table = conversion_tables.popitem()
        metadata["metadata"]["tables"] = {super_animal: table}
    return [(None, metadata, "points")]


def read_hdf(filename: str) -> List[LayerData]:
    config_path = misc.find_project_config_path(filename)
    layers = []
    for filename in glob.iglob(filename):
        temp = pd.read_hdf(filename)
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
            colormap=colormap,
        )
        metadata["name"] = os.path.split(filename)[1].split(".")[0]
        metadata["metadata"]["root"] = os.path.split(filename)[0]
        # Store file name in case the layer's name is edited by the user
        metadata["metadata"]["name"] = metadata["name"]
        layers.append((data, metadata, "points"))
    return layers


class VideoReaderDecord(VideoReader):
    def __init__(self, video_path):
        super().__init__(video_path, ctx=cpu(0))
    
    def __getitem__(self, index):
        
        # The following __getitem__ code comes from napari-video, which is an 
        # OpenCV-based video reader that relies on pyvideoreader and allows
        # napari to run videos.
        # https://github.com/janclemenslab/napari-video
        # https://github.com/postpop/videoreader
        # This has been modified, so that the Decord video player can be used
        # to read videos within napari.
        
        frames = None
        if isinstance(index, int):  # single frame
            # MODIFIED LINES
            self.seek_accurate(index)
            frames = self.next().asnumpy()
            # ret, frames = self.read(index)
            # frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        elif isinstance(index, slice):  # slice of frames
            frames = np.stack([self[ii] for ii in range(*index.indices(len(self)))])
        elif isinstance(index, range):  # range of frames
            frames = np.stack([self[ii] for ii in index])
        elif isinstance(index, tuple):  # unpack tuple of indices
            if isinstance(index[0], slice):
                indices = range(*index[0].indices(len(self)))
                # ADDED LINE
                frames = self.get_batch(indices)
            elif isinstance(index[0], (np.integer, int)):
                indices = int(index[0])
                # ADDED LINE
                frames = self[indices]
            else:
                indices = None

            if indices is not None:
                # REMOVED LINE
                # frames = self[indices]
                # ADDED LINES
                if isinstance(frames, np.ndarray) == False:
                    frames = frames.asnumpy()

                # index into pixels and channels
                for cnt, idx in enumerate(index[1:]):
                    if isinstance(idx, slice):
                        ix = range(*idx.indices(self.shape[cnt+1]))
                    elif isinstance(idx, int):
                        ix = range(idx-1, idx)
                    else:
                        continue

                    if frames.ndim==4: # ugly indexing from the back (-1,-2 etc)
                        cnt = cnt+1
                    frames = np.take(frames, ix, axis=cnt)

        if frames is not None:
            if frames.shape[0] == 1:
                frames = frames[0]
        return frames

    @property
    def dtype(self):
        return np.uint8
    
    # MODIFIED
    @property
    def shape(self):
        return (self._num_frame,) + self[0].shape
    
    # MODIFIED
    @property
    def ndim(self):
        return len(self[0].shape)+1

    @property
    def size(self):
        return np.product(self.shape)


def video_file_reader(path):
    array = VideoReaderDecord(path)
    return [(array, {'name': path}, 'image')]


@napari_hook_implementation
def napari_get_reader(path):
    # remember, path can be a list, so we check it's type first...
    if isinstance(path, str) and any([path.endswith(ext) for ext in [".mp4", ".mov", ".avi"]]):
        # If we recognize the format, we return the actual reader function
        return video_file_reader
    # otherwise we return None.
    return None


def read_video(filename: str, opencv: bool = True):
    movie = VideoReaderDecord(filename)
    elems = list(Path(filename).parts)
    elems[-2] = "labeled-data"
    elems[-1] = elems[-1].split(".")[0]
    root = os.path.join(*elems)
    params = {
        "name": filename,
        "metadata": {
            "root": root,
        },
    }
    return [(movie, params)]
