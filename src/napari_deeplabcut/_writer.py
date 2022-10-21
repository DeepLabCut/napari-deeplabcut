import os
from itertools import groupby

import pandas as pd
import yaml
from napari.layers import Shapes
from napari_builtins.io import napari_write_shapes
from skimage.io import imsave
from skimage.util import img_as_ubyte

from napari_deeplabcut import misc
from napari_deeplabcut._reader import _load_config


def _write_config(config_path: str, params: dict):
    with open(config_path, "w") as file:
        yaml.safe_dump(params, file)


def write_hdf(filename, data, metadata):
    file, _ = os.path.splitext(filename)  # FIXME Unused currently
    temp = pd.DataFrame(data[:, -1:0:-1], columns=["x", "y"])
    properties = metadata["properties"]
    meta = metadata["metadata"]
    temp["bodyparts"] = properties["label"]
    temp["individuals"] = properties["id"]
    temp["inds"] = data[:, 0].astype(int)
    temp["likelihood"] = properties["likelihood"]
    temp["scorer"] = meta["header"].scorer
    df = temp.set_index(["scorer", "individuals", "bodyparts", "inds"]).stack()
    df.index.set_names("coords", level=-1, inplace=True)
    df = df.unstack(["scorer", "individuals", "bodyparts", "coords"])
    df.index.name = None
    if not properties["id"][0]:
        df = df.droplevel("individuals", axis=1)
    df = df.reindex(meta["header"].columns, axis=1)
    # Fill unannotated rows with NaNs
    # df = df.reindex(range(len(meta['paths'])))
    # df.index = meta['paths']
    if meta["paths"]:
        df.index = [meta["paths"][i] for i in df.index]
    misc.guarantee_multiindex_rows(df)

    name = metadata["name"]
    root = meta["root"]
    if "machine" in name:  # We are attempting to save refined model predictions
        df.drop("likelihood", axis=1, level="coords", inplace=True, errors="ignore")
        header = misc.DLCHeader(df.columns)
        gt_file = ""
        for file in os.listdir(root):
            if file.startswith("CollectedData") and file.endswith("h5"):
                gt_file = file
                break
        if gt_file:  # Refined predictions must be merged into the existing data
            df_gt = pd.read_hdf(os.path.join(root, gt_file))
            new_scorer = df_gt.columns.get_level_values("scorer")[0]
            header.scorer = new_scorer
            df.columns = header.columns
            df = pd.concat((df, df_gt))
            df = df[~df.index.duplicated(keep="first")]
            name = os.path.splitext(gt_file)[0]
        else:
            # Let us fetch the config.yaml file to get the scorer name...
            project_folder = root.rsplit(os.sep, 2)[0]
            config = _load_config(os.path.join(project_folder, "config.yaml"))
            new_scorer = config["scorer"]
            header.scorer = new_scorer
            df.columns = header.columns
            name = f"CollectedData_{new_scorer}"
    df.sort_index(inplace=True)
    filename = name + ".h5"
    path = os.path.join(root, filename)
    df.to_hdf(path, key="keypoints", mode="w")
    df.to_csv(path.replace(".h5", ".csv"))
    return filename


def _write_image(data, output_path, plugin=None):
    imsave(
        output_path,
        img_as_ubyte(data).squeeze(),
        plugin=plugin,
        check_contrast=False,
    )


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
