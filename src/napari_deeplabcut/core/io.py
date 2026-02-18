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
from datetime import datetime
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
from pydantic import ValidationError

from napari_deeplabcut import misc
from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel, PointsMetadata
from napari_deeplabcut.core import schemas as dlc_schemas
from napari_deeplabcut.core.dataframes import (
    form_df_from_validated,
    guarantee_multiindex_rows,
    harmonize_keypoint_column_index,
    harmonize_keypoint_row_index,
    keypoint_conflicts,
    merge_multiple_scorers,
)
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError, UnresolvablePathError
from napari_deeplabcut.core.layers import populate_keypoint_layer_metadata
from napari_deeplabcut.core.metadata import attach_source_and_io, parse_points_metadata
from napari_deeplabcut.core.paths import canonicalize_path
from napari_deeplabcut.core.provenance import resolve_provenance_path
from napari_deeplabcut.ui.dialogs import maybe_confirm_overwrite

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
    """Read a single H5 file and attach provenance with optional explicit kind.

    - Produces one Points layer per H5 file
    - Points.data contains only finite coordinates
    - Unlabeled keypoints are omitted from Points.data
    - Empty Points layers are valid
    """
    temp = pd.read_hdf(str(file))
    temp = merge_multiple_scorers(temp)
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
    finite = np.isfinite(data).all(axis=1)
    df = df.loc[finite].reset_index(drop=True)

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
            meta["io"]["kind"] = kind  # stored as actual enum, not value
    else:
        attach_source_and_io(metadata, file)

    return [(data, metadata, "points")]


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

    # Never save back to machine sources
    if source_kind == AnnotationKind.MACHINE:
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


# TODO move to dataframes.py
def form_df(
    points_data,
    layer_metadata: dict,
    layer_properties: dict,
) -> pd.DataFrame:
    """
    Form a DataFrame from points data + layer metadata, structured according to DLC conventions.

    Arguments
    ---------
    points_data:
        array-like of shape (N, 3) in napari-style [frame, y, x]
    layer_metadata:
        dict that must contain at least: 'header' (DLCHeader-like or DLCHeaderModel), optional 'paths'
    layer_properties:
        dict that must contain: 'label', 'id', optional 'likelihood'
    """
    layer_metadata = layer_metadata or {}
    layer_properties = layer_properties or {}

    # -----------------------------
    # 1) Normalize/wrap header
    # -----------------------------
    header_obj = layer_metadata.get("header", None)
    if header_obj is None:
        raise KeyError("layer_metadata['header'] is required to write DLC keypoints.")

    if isinstance(header_obj, DLCHeaderModel):
        header_model = header_obj
    else:
        # Accept a misc.DLCHeader-like object (has .columns)
        cols = getattr(header_obj, "columns", None)
        if cols is None:
            raise TypeError("layer_metadata['header'] must be a DLCHeaderModel or an object with a .columns attribute.")
        header_model = DLCHeaderModel(columns=cols)

    # Build a PointsMetadata model from the layer_metadata dict,
    # but replace raw header with our DLCHeaderModel wrapper.
    meta_payload = dict(layer_metadata)
    meta_payload["header"] = header_model
    pts_meta = PointsMetadata.model_validate(meta_payload)

    # -----------------------------
    # 2) Fill missing likelihood (preserve old behavior)
    # -----------------------------
    # Your old code assumed likelihood always existed.
    # To remain backwards compatible, auto-fill with 1.0 if missing/None.
    n = np.asarray(points_data).shape[0] if points_data is not None else 0
    props_payload = dict(layer_properties)
    if props_payload.get("likelihood", None) is None:
        props_payload["likelihood"] = [1.0] * n

    # -----------------------------
    # 3) Validate with dedicated schemas
    # -----------------------------
    try:
        points = dlc_schemas.PointsDataModel.model_validate({"data": points_data})
        props = dlc_schemas.KeypointPropertiesModel.model_validate(props_payload)
        ctx = dlc_schemas.PointsWriteInputModel.model_validate({"points": points, "meta": pts_meta, "props": props})
    except ValidationError as e:
        # Give a concise error that points to the failing part.
        # The full `e` still has structured details if you want to log it.
        raise ValueError(f"Invalid keypoint write inputs: {e}") from e

    # -----------------------------
    # 4) Delegate transformation
    # -----------------------------
    df = form_df_from_validated(ctx)

    # Keep your belt-and-suspenders guarantee
    guarantee_multiindex_rows(df)
    return df


def _atomic_to_hdf(df: pd.DataFrame, out_path: Path, key: str = "keypoints") -> None:
    """Best-effort atomic write: write to temp and replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # Write temp
    df.to_hdf(tmp, key=key, mode="w")
    # Replace
    tmp.replace(out_path)


def write_hdf(path: str, data, attributes: dict) -> list[str]:
    """
    NPE2 single-layer writer.

    Signature required by napari (manifest-based writers):
        def writer(path: str, data: Any, attributes: dict) -> List[str]
    Writers must return a list of successfully-written paths.

    Contract:
    - Empty Points layers may be written only if promoted
    - Finite Points must always produce finite stored coordinates

    This function writes DLC keypoints to .h5 (and companion .csv).
    """
    attrs = dlc_schemas.PointsLayerAttributesModel.model_validate(attributes or {})
    pts_meta: PointsMetadata = parse_points_metadata(attrs.metadata, drop_header=False)
    if not pts_meta.header:
        raise ValueError("Layer metadata must include a valid DLC header to write keypoints.")

    points = dlc_schemas.PointsDataModel.model_validate({"data": data})
    props = dlc_schemas.KeypointPropertiesModel.model_validate(attrs.properties)

    # Bundle + validate cross-field invariants
    ctx = dlc_schemas.PointsWriteInputModel.model_validate(
        {
            "points": points,
            "meta": pts_meta,
            "props": props,
        }
    )

    logger.debug("HEADER nlevels: %s", ctx.meta.header.as_multiindex().nlevels)
    logger.debug("HEADER names: %s", ctx.meta.header.as_multiindex().names)

    df_new = form_df_from_validated(ctx)

    logger.debug("DF_NEW columns nlevels: %s", df_new.columns.nlevels)
    logger.debug("DF_NEW columns names: %s", df_new.columns.names)
    logger.debug("DF_NEW finite count: %s", np.isfinite(df_new.to_numpy()).sum())

    # Build df from points + plugin metadata + layer properties
    df_new = form_df_from_validated(ctx)

    # Decide output path:
    # 1) User-requested path should be ignored in favor of provenance when available
    #  This is a fallback only used when provenance is missing or unresolvable,
    #  and is never expected to be set for this plugin
    # requested_out = _normalize_requested_out_path(path, layer_name)

    # 2) provenance/save_target is always the source of truth for where to write
    out_path, target_scorer, source_kind = _resolve_output_path_from_metadata(attributes)

    # If promoting to GT and scorer is known, rewrite scorer level
    if target_scorer:
        df_new = _set_df_scorer(df_new, target_scorer)

    # Never write back to machine sources without an explicit promotion target
    if not out_path and source_kind == AnnotationKind.MACHINE:
        raise MissingProvenanceError("Cannot resolve provenance output path for MACHINE source.")

    # If provenance returned nothing, default to requested path
    if not out_path:
        # Strict only for MACHINE
        # Safety: never write back to machine sources unless promotion target exists
        if source_kind == AnnotationKind.MACHINE:
            raise MissingProvenanceError("Cannot resolve provenance output path for MACHINE source.")

        # GT fallback
        root = pts_meta.root
        if not root:
            raise MissingProvenanceError("GT fallback requires root.")

        root_path = Path(root)
        candidates = sorted(root_path.glob("CollectedData_*.h5"))
        if len(candidates) > 1:
            raise AmbiguousSaveError(
                f"Multiple CollectedData_*.h5 files found in {root}."
                " Cannot determine where to save."
                " Please specify a save_target with explicit path and scorer.",
                candidates=[str(c) for c in candidates],
            )
        elif len(candidates) == 1:
            out = candidates[0]
        else:
            scorer = target_scorer or pts_meta.header.scorer
            out = root_path / f"CollectedData_{scorer}.h5"
    else:
        out = Path(out_path)

    # Determine destination kind (promotion writes to GT target)
    has_save_target = pts_meta.save_target is not None
    destination_kind = (
        AnnotationKind.GT
        if has_save_target
        else ((pts_meta.io.kind if pts_meta.io is not None else None) or AnnotationKind.GT)
    )

    # Merge-on-save for GT
    if destination_kind == AnnotationKind.GT and out.exists():
        try:
            df_old = pd.read_hdf(out, key="keypoints")
        except (KeyError, ValueError):
            df_old = pd.read_hdf(out)

        key_conflict = keypoint_conflicts(df_old, df_new)
        if not maybe_confirm_overwrite(attributes, key_conflict):
            raise RuntimeError("User aborted save due to keypoint conflicts.")

        # Harmonize indices and merge
        try:
            guarantee_multiindex_rows(df_new)
            guarantee_multiindex_rows(df_old)
        except Exception:
            pass

        df_new, df_old = harmonize_keypoint_row_index(df_new, df_old)
        df_new = harmonize_keypoint_column_index(df_new)
        df_old = harmonize_keypoint_column_index(df_old)
        df_out = df_new.combine_first(df_old)

        # Normalize columns to DLC header if possible
        try:
            header = misc.DLCHeader(df_out.columns)
            df_out = df_out.reindex(header.columns, axis=1)
        except Exception:
            pass
    else:
        df_out = df_new

    # Final cleanup
    try:
        guarantee_multiindex_rows(df_out)
    except Exception:
        pass
    df_out.sort_index(inplace=True)

    # Write .h5 and .csv
    _atomic_to_hdf(df_out, out, key="keypoints")
    csv_path = out.with_suffix(".csv")
    df_out.to_csv(csv_path)

    # Update UI controls if present (safe in headless)
    controls = getattr(pts_meta, "controls", None)
    if controls is not None:
        controls._is_saved = True
        try:
            controls.last_saved_label.setText(f"Last saved at {str(datetime.now().time()).split('.')[0]}")
            controls.last_saved_label.show()
        except Exception:
            pass

    return [str(out), str(csv_path)]


# =============================================================================
# SUPERKEYPOINTS (assets: diagram + JSON)
# =============================================================================
# NOTE: These are used to support DLCHeader superkeypoints workflows.


def load_superkeypoints_json_from_path(json_path: str | Path):
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Superkeypoints JSON file not found at {json_path}.")
    with open(path) as f:
        payload = json.load(f)
        if payload:
            return payload
        else:
            raise ValueError(f"Superkeypoints JSON file at {json_path} is empty or invalid.")


def load_superkeypoints_diagram_from_path(image_path: str | Path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Superkeypoints diagram not found at {image_path}.")
    try:
        return imread(path).squeeze()
    except Exception as e:
        raise RuntimeError(f"Superkeypoints diagram could not be loaded from {image_path}.") from e


def load_superkeypoints_diagram(super_animal: str):
    path = resources.files("napari_deeplabcut") / "assets" / f"{super_animal}.jpg"
    return load_superkeypoints_diagram_from_path(path)


def load_superkeypoints(super_animal: str):
    path = resources.files("napari_deeplabcut") / "assets" / f"{super_animal}.json"
    payload = load_superkeypoints_json_from_path(path)
    return payload.get("data", payload) if isinstance(payload, dict) else payload


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
