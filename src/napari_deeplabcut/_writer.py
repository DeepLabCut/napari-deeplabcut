"""Writers for DeepLabCut data formats."""

# src/napari_deeplabcut/_writer.py
import os
from datetime import datetime
from itertools import groupby
from pathlib import Path

import pandas as pd
import yaml
from napari.layers import Shapes
from napari_builtins.io import napari_write_shapes
from skimage.io import imsave
from skimage.util import img_as_ubyte

from napari_deeplabcut import misc
from napari_deeplabcut._reader import _load_config
from napari_deeplabcut.ui.dialogs import _maybe_confirm_overwrite


def _write_config(config_path: str, params: dict):
    with open(config_path, "w") as file:
        yaml.safe_dump(params, file)


def _form_df(points_data, metadata):
    temp = pd.DataFrame(points_data[:, -1:0:-1], columns=["x", "y"])
    properties = metadata["properties"]
    meta = metadata["metadata"]
    temp["bodyparts"] = properties["label"]
    temp["individuals"] = properties["id"]
    temp["inds"] = points_data[:, 0].astype(int)
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
    return df


def write_hdf(filename, data, metadata):
    """
    Write DLC keypoints to disk.

    SAFETY POLICY:
    - For ground-truth files (CollectedData_*), never delete existing labels silently.
      If a file already exists, merge-on-save:
        * new non-NaN values overwrite old
        * old values are preserved when new are NaN/missing
    - For machine/refinement files, keep existing behavior (special merging logic).
    """
    file, _ = os.path.splitext(filename)  # currently unused
    df_new = _form_df(data, metadata)

    meta = metadata["metadata"]
    name = metadata["name"]
    root = meta["root"]

    # Determine output path early (may be updated below in "machine" branch)
    out_name = name
    out_path = os.path.join(root, out_name + ".h5")

    if "machine" in name:
        # ---- existing behavior for refined predictions ----
        df_new.drop("likelihood", axis=1, level="coords", inplace=True, errors="ignore")
        header = misc.DLCHeader(df_new.columns)

        gt_file = ""
        for file in os.listdir(root):
            if file.startswith("CollectedData") and file.endswith("h5"):
                gt_file = file
                break

        if gt_file:
            # Refined predictions must be merged into existing GT file
            gt_path = os.path.join(root, gt_file)
            df_gt = pd.read_hdf(gt_path, key="keypoints")
            new_scorer = df_gt.columns.get_level_values("scorer")[0]
            header.scorer = new_scorer
            df_new.columns = header.columns

            # Keep existing concat + de-dupe approach
            df_out = pd.concat((df_new, df_gt))
            df_out = df_out[~df_out.index.duplicated(keep="first")]
            out_name = os.path.splitext(gt_file)[0]
            out_path = os.path.join(root, out_name + ".h5")
        else:
            # Let us fetch the config.yaml file to get the scorer name...
            project_folder = Path(root).parents[1]
            config = _load_config(str(project_folder / "config.yaml"))
            new_scorer = config["scorer"]
            header.scorer = new_scorer
            df_new.columns = header.columns
            out_name = f"CollectedData_{new_scorer}"
            out_path = os.path.join(root, out_name + ".h5")

            df_out = df_new

    else:
        # ---- NEW: safe merge-on-save for ground truth ----
        if os.path.exists(out_path):
            try:
                df_old = pd.read_hdf(out_path, key="keypoints")
            except (KeyError, ValueError):
                # Fall back in case older files used a different key or structure
                df_old = pd.read_hdf(out_path)

            key_conflict = misc.keypoint_conflicts(df_old, df_new)
            if not _maybe_confirm_overwrite(metadata, key_conflict):
                return None  # user cancelled save

            # Merge in a non-destructive way:
            # df_new wins where it has values; df_old fills where df_new is NaN.
            df_out = df_new.combine_first(df_old)

            # Optional: ensure DLC header order is preserved (helps stability)
            try:
                header = misc.DLCHeader(df_out.columns)
                df_out = df_out.reindex(header.columns, axis=1)
            except Exception:
                # If header inference fails, still save merged data.
                pass
        else:
            df_out = df_new

    # Sort before writing
    df_out.sort_index(inplace=True)

    # Always write atomically-ish by overwriting the final outputs
    df_out.to_hdf(out_path, key="keypoints", mode="w")
    df_out.to_csv(out_path.replace(".h5", ".csv"))

    # Mark the session as saved if KeypointControls was attached to the layer metadata
    layer_meta = metadata.get("metadata", {})
    controls = layer_meta.get("controls")
    if controls is not None:
        controls._is_saved = True
        # Guard UI updates (tests/headless may not have Qt widgets)
        try:
            controls.last_saved_label.setText(f"Last saved at {str(datetime.now().time()).split('.')[0]}")
            controls.last_saved_label.show()
        except Exception:
            pass

    return os.path.basename(out_path)


def _write_image(data, output_path, plugin=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
