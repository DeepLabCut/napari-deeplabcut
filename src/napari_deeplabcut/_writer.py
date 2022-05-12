import os

import pandas as pd

from napari_deeplabcut import misc
from napari_deeplabcut._reader import _load_config


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
