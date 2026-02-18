"""Writers for DeepLabCut data formats."""

# src/napari_deeplabcut/_writer.py
import logging
import os
from datetime import datetime
from itertools import groupby
from pathlib import Path

import pandas as pd
from napari.layers import Shapes
from napari_builtins.io import napari_write_shapes
from skimage.io import imsave
from skimage.util import img_as_ubyte

from napari_deeplabcut import misc
from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.dataframes import harmonize_keypoint_row_index
from napari_deeplabcut.core.errors import MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.metadata import parse_points_metadata
from napari_deeplabcut.core.provenance import resolve_provenance_path
from napari_deeplabcut.ui.dialogs import _maybe_confirm_overwrite

logger = logging.getLogger(__name__)


def _kind_value(kind) -> str:
    """Normalize kind to a stable lowercase string."""
    if kind is None:
        return ""
    # Enum-like
    v = getattr(kind, "value", None)
    if isinstance(v, str):
        return v
    return str(kind).lower()


def _set_df_scorer(df: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Return df with scorer level set to the given scorer (if present)."""
    scorer = (scorer or "").strip()
    if not scorer:
        return df
    if not hasattr(df.columns, "names") or "scorer" not in df.columns.names:
        return df

    try:
        cols = df.columns.to_frame(index=False)
        cols["scorer"] = scorer
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(cols)
    except Exception:
        pass
    return df


# TODO @C-Achard move this to a helper in io and test more directly
# This is a crucial part of the data pipeline and it should really use the Metadata models instead of raw dicts
def _form_df(
    points_data,
    layer_metadata: dict,
    layer_properties: dict,
):
    """Form a DataFrame from the given points data and layer metadata, structured according to DLC conventions.

    Arguments:
    - points_data: numpy array of shape (N, 3) where columns are [frame_index, x, y]
    - layer_metadata: metadata dict from the napari layer
    - layer_properties: properties dict from the napari layer (optional, can be extracted from metadata if not provided)
    """
    temp_df = pd.DataFrame(points_data[:, -1:0:-1], columns=["x", "y"])
    temp_df["bodyparts"] = layer_properties["label"]
    temp_df["individuals"] = layer_properties["id"]
    temp_df["inds"] = points_data[:, 0].astype(int)
    temp_df["likelihood"] = layer_properties["likelihood"]
    temp_df["scorer"] = layer_metadata["header"].scorer

    df = temp_df.set_index(["scorer", "individuals", "bodyparts", "inds"]).stack()
    df.index.set_names("coords", level=-1, inplace=True)
    df = df.unstack(["scorer", "individuals", "bodyparts", "coords"])
    df.index.name = None
    if not layer_properties["id"][0]:
        df = df.droplevel("individuals", axis=1)
    df = df.reindex(layer_metadata["header"].columns, axis=1)

    # Fill unannotated rows with NaNs
    # df = df.reindex(range(len(meta['paths'])))
    # df.index = meta['paths']
    if layer_metadata["paths"]:
        df.index = [layer_metadata["paths"][i] for i in df.index]

    misc.guarantee_multiindex_rows(df)
    return df


def _resolve_output_path_from_metadata(metadata: dict) -> tuple[str | None, str | None, AnnotationKind | None]:
    """
    Resolve output path with promotion support.

    Returns:
      (out_path, target_scorer, source_kind)

    - Prefer PointsMetadata.save_target (promotion-to-GT).
    - For GT sources, fall back to io/source_h5.
    - For machine sources without save_target, return (None, None, "machine") to allow safe abort.
    """
    layer_meta = metadata.get("metadata")
    if not isinstance(layer_meta, dict):
        layer_meta = {}

    pts = parse_points_metadata(layer_meta)
    io = pts.io
    st = pts.save_target

    source_kind = getattr(io, "kind", None) if io is not None else None

    # Promotion target wins
    if st is not None:
        try:
            p = resolve_provenance_path(st, root_anchor=st.project_root, allow_missing=True)
            target_scorer = getattr(st, "scorer", None)
            if isinstance(target_scorer, str) and target_scorer.strip():
                return str(p), target_scorer.strip(), source_kind
            # Also accept scorer stored in dict extra
            if isinstance(layer_meta.get("save_target"), dict):
                s2 = layer_meta["save_target"].get("scorer")
                if isinstance(s2, str) and s2.strip():
                    return str(p), s2.strip(), source_kind
            return str(p), None, source_kind
        except (MissingProvenanceError, UnresolvablePathError):
            return None, None, source_kind

    # If source is machine/prediction and no save_target: never write back
    if source_kind is AnnotationKind.MACHINE:
        return None, None, source_kind

    # GT source: prefer io if available
    if io is not None:
        try:
            p = resolve_provenance_path(io, root_anchor=io.project_root, allow_missing=True)
            return str(p), None, source_kind
        except (MissingProvenanceError, UnresolvablePathError):
            pass

    # Legacy fallback: source_h5 (GT only)
    src = layer_meta.get("source_h5")
    if isinstance(src, str) and src:
        return src, None, source_kind

    return None, None, source_kind


def write_hdf(filename, data, metadata):
    """
    Write DLC keypoints to disk.
    Napari-facing function that must respect the napari writer signature:
    def write_function(path: str, data: Any, metadata: dict) -> Optional[str]:
    The returned string is the path that was actually written, or None if the write was aborted

    SAFETY POLICY:
    - For ground-truth files (CollectedData_*), never delete existing labels silently.
      If a file already exists, merge-on-save:
        * new non-NaN values overwrite old
        * old values are preserved when new are NaN/missing
    - For machine/refinement files, keep existing behavior (special merging logic).
    """
    root = metadata.get("root")
    paths = metadata.get("paths")
    layer_name = metadata.get("name", "")

    out_path, target_scorer, source_kind = _resolve_output_path_from_metadata(metadata)
    df_new = _form_df(data, layer_metadata=metadata["metadata"], layer_properties=metadata["properties"])
    # If promoting to GT and a target scorer is known, rewrite the scorer level.
    if target_scorer:
        df_new = _set_df_scorer(df_new, target_scorer)

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

    # ------------------------------------------------------------
    # Resolve output path using provenance (PointsMetadata.io) or source_h5.
    # ------------------------------------------------------------

    # Never write back to machine sources; promotion must supply save_target.
    if source_kind is AnnotationKind.MACHINE and not out_path:
        raise MissingProvenanceError(
            "Refined predictions are promoted to CollectedData on save. No save_target was set for this layer."
        )

    # If still missing, fall back for GT only (temporary): write beside root using layer_name
    if not out_path:
        out_path = os.path.join(root, f"{layer_name}.h5")

    # Decide write mode based on DESTINATION (promotion writes to GT target).
    pts = parse_points_metadata(metadata)
    has_save_target = pts.save_target is not None

    if has_save_target:
        destination_kind = AnnotationKind.GT
    else:
        destination_kind = getattr(getattr(pts, "io", None), "kind", None) or AnnotationKind.GT

    if destination_kind == AnnotationKind.GT:
        # ------------------------------------------------------------
        # Ground-truth destination: safe merge-on-save + overwrite confirm
        # ------------------------------------------------------------
        if os.path.exists(out_path):
            try:
                df_old = pd.read_hdf(out_path, key="keypoints")
            except (KeyError, ValueError):
                df_old = pd.read_hdf(out_path)

            key_conflict = misc.keypoint_conflicts(df_old, df_new)
            if not _maybe_confirm_overwrite(metadata, key_conflict):
                return None  # user cancelled save

            # Ensure both dataframes have compatible index types before merging
            try:
                misc.guarantee_multiindex_rows(df_new)
            except Exception:
                pass
            try:
                misc.guarantee_multiindex_rows(df_old)
            except Exception:
                pass

            df_new, df_old = harmonize_keypoint_row_index(df_new, df_old)
            df_out = df_new.combine_first(df_old)

            try:
                header = misc.DLCHeader(df_out.columns)
                df_out = df_out.reindex(header.columns, axis=1)
            except Exception:
                pass
        else:
            df_out = df_new
    else:
        # For now: any non-GT destination simply overwrites its target
        df_out = df_new

    # Guarantee consistent index again after merge (belt and suspenders)
    try:
        misc.guarantee_multiindex_rows(df_out)
    except Exception:
        pass

    # Sort before writing
    df_out.sort_index(inplace=True)

    # Always write atomically-ish by overwriting the final outputs
    df_out.to_hdf(out_path, key="keypoints", mode="w")
    df_out.to_csv(out_path.replace(".h5", ".csv"))

    # Mark the session as saved if KeypointControls was attached to the layer metadata
    plugin_metadata = metadata.get("metadata", {})
    controls = plugin_metadata.get("controls")
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
