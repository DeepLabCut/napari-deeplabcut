"""Writers for DeepLabCut data formats."""

# src/napari_deeplabcut/_writer.py
import logging
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
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.metadata import parse_points_metadata, resolve_provenance_path
from napari_deeplabcut.ui.dialogs import _maybe_confirm_overwrite

logger = logging.getLogger(__name__)


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


def _resolve_output_path_from_metadata(metadata: dict) -> str | None:
    """
    Resolve the target .h5 path from PointsMetadata.io if available,
    otherwise fall back to legacy `source_h5` if present.

    Returns an absolute filesystem path string, or None if it cannot be resolved.
    """
    # `metadata` here is the napari writer metadata dict passed into write_hdf.
    # It contains a nested `metadata` dict (layer.metadata).
    layer_meta = metadata.get("metadata")
    if not isinstance(layer_meta, dict):
        layer_meta = {}

    # 1) Preferred: PointsMetadata.io
    pts = parse_points_metadata(layer_meta)
    io = pts.io
    if io is not None:
        try:
            p = resolve_provenance_path(io, root_anchor=io.project_root, allow_missing=True)
            return str(p)
        except (MissingProvenanceError, UnresolvablePathError):
            pass

    # 2) Legacy fallback: source_h5
    src = layer_meta.get("source_h5")
    if isinstance(src, str) and src:
        return src

    return None


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

    layer_meta = metadata.get("metadata")
    if not isinstance(layer_meta, dict):
        layer_meta = {}

    layer_name = metadata.get("name", "")
    # ------------------------------------------------------------
    # NEW: resolve output path using provenance (PointsMetadata.io)
    # ------------------------------------------------------------
    out_path = _resolve_output_path_from_metadata(metadata)
    # root may be nested or top-level depending on napari/version/reader path
    root = layer_meta.get("root") or metadata.get("root")

    # If provenance is unavailable, fall back to legacy root/name routing for now.
    # (Later we will enforce deterministic abort on ambiguity.)
    if not out_path:
        out_name = layer_name
        out_path = os.path.join(root, out_name + ".h5")

    # paths may be nested or top-level too
    paths = layer_meta.get("paths") or metadata.get("paths")

    # Fallback: infer root from paths if still missing
    if not root and paths:
        try:
            root = str(Path(paths[0]).expanduser().resolve().parent)
        except Exception:
            root = None

    if not root:
        raise KeyError(
            "root (missing from writer metadata; cannot determine where to write CollectedData*.h5). "
            "Expected either metadata['metadata']['root'] or metadata['root'] or inferable from paths."
        )

    # Determine output path early (may be updated below in "machine" branch)
    out_name = layer_name
    out_path = os.path.join(root, out_name + ".h5")

    # Determine kind from PointsMetadata.io if present (do NOT infer from layer name).
    pts = parse_points_metadata(layer_meta)
    kind = getattr(getattr(pts, "io", None), "kind", None)

    if str(kind) == "machine":
        # ------------------------------------------------------------
        # Machine outputs: save to their own target file only.
        # No automatic merge into GT (policy).
        # ------------------------------------------------------------
        df_out = df_new

    else:
        # ------------------------------------------------------------
        # Ground-truth: safe merge-on-save
        # ------------------------------------------------------------
        if os.path.exists(out_path):
            try:
                df_old = pd.read_hdf(out_path, key="keypoints")
            except (KeyError, ValueError):
                df_old = pd.read_hdf(out_path)

            key_conflict = misc.keypoint_conflicts(df_old, df_new)
            if not _maybe_confirm_overwrite(metadata, key_conflict):
                return None  # user cancelled save

            df_out = df_new.combine_first(df_old)

            try:
                header = misc.DLCHeader(df_out.columns)
                df_out = df_out.reindex(header.columns, axis=1)
            except Exception:
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
