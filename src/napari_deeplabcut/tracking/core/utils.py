from __future__ import annotations

import re
from typing import Any

import napari
import numpy as np
import pandas as pd
from napari.layers import Points

_NAPARI_COORD_COLS = ("frame", "y", "x")


# ---- Layer data/feature handling utilities ---------------------------------------------------------
def extract_layer_data_and_features(layer: Points) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return a defensive copy of Points layer data and aligned feature rows.

    Fallback policy
    ---------------
    - prefer `layer.features`
    - fall back to `layer.properties` if feature row count mismatches data
    """
    data = np.asarray(getattr(layer, "data", np.empty((0, 3))), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Points layer {getattr(layer, 'name', layer)!r} has invalid data shape: {data.shape!r}")

    features = getattr(layer, "features", None)
    if isinstance(features, pd.DataFrame):
        feat_df = features.reset_index(drop=True).copy()
    elif features is None:
        feat_df = pd.DataFrame(index=range(len(data)))
    else:
        feat_df = pd.DataFrame(features).reset_index(drop=True).copy()

    if len(feat_df) != len(data):
        props = getattr(layer, "properties", {}) or {}
        feat_df = pd.DataFrame(props).reset_index(drop=True)
        if len(feat_df) != len(data):
            raise ValueError(
                f"Points layer {getattr(layer, 'name', layer)!r} has mismatched data/features lengths: "
                f"{len(data)} rows vs {len(feat_df)} features."
            )

    return data.copy(), feat_df


def coord_columns_for_data(data: np.ndarray) -> list[str]:
    """
    Return canonical coordinate column names for a napari Points data array.

    Examples
    --------
    - shape (N, 3) -> ["frame", "y", "x"]
    - shape (N, 4) -> ["frame", "y", "x", "coord_3"]
    """
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected Points data with at least 3 columns [frame, y, x], got {data.shape!r}.")
    if data.shape[1] == 3:
        return list(_NAPARI_COORD_COLS)
    extra = [f"coord_{i}" for i in range(3, data.shape[1])]
    return [*_NAPARI_COORD_COLS, *extra]


def pick_semantic_series(
    df: pd.DataFrame,
    layer: Points,
    *,
    primary: str,
    fallback_prop: str,
) -> pd.Series:
    """
    Return a semantic per-row series from features first, then properties fallback.
    """
    if primary in df.columns:
        out = df[primary]
        if len(out) == len(df):
            return out.reset_index(drop=True)

    props = getattr(layer, "properties", {}) or {}
    vals = props.get(fallback_prop, None)
    if vals is None:
        return pd.Series([""] * len(df), index=df.index, dtype=object)

    arr = np.asarray(vals, dtype=object).ravel()
    if len(arr) != len(df):
        return pd.Series([""] * len(df), index=df.index, dtype=object)

    return pd.Series(arr, index=df.index, dtype=object)


def coerce_frame_series(series: pd.Series | Any) -> pd.Series:
    """
    Coerce a series-like object to rounded nullable integer-like frames.
    """
    if isinstance(series, pd.Series):
        out = pd.to_numeric(series, errors="coerce")
    else:
        out = pd.to_numeric(pd.Series(series), errors="coerce")
    return out.round().astype("Float64")


def normalize_slot_id(value: Any) -> str:
    """
    Normalize an id/individual value to a stable string key.
    """
    if value in ("", None):
        return ""
    try:
        if np.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def build_slot_key(*, frame: Any, slot_id: Any, label: Any) -> tuple[int, str, str]:
    """
    Build canonical semantic key for a tracked/DLC keypoint row.
    """
    return (int(frame), normalize_slot_id(slot_id), str(label).strip())


def duplicate_slot_row_indices(df: pd.DataFrame) -> tuple[int, ...]:
    """
    Return original row indices for duplicate semantic slot keys.
    """
    if df.empty:
        return ()
    dup_mask = df.duplicated(subset=["_slot_key"], keep=False)
    if not dup_mask.any():
        return ()
    return tuple(sorted_int_tuple(df.loc[dup_mask, "_source_row_index"]))


def sorted_int_tuple(values: pd.Series | list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    """
    Return values normalized to sorted ints.
    """
    return tuple(sorted(int(v) for v in values))


def format_frame_label(frame: Any) -> str:
    try:
        return str(int(frame))
    except Exception:
        return str(frame)


def format_slot_label(label: Any, slot_id: Any) -> str:
    slot_id = normalize_slot_id(slot_id)
    label = str(label)
    return f"{label} (id: {slot_id})" if slot_id else label


def format_coords_text(row: dict[str, Any] | pd.Series) -> str:
    x = float(row["x"])
    y = float(row["y"])
    return f"(x={x:.3f}, y={y:.3f})"


def normalize_points_layer_for_tracking(
    layer: Points,
    *,
    valid_flag_column: str = "_is_valid_slot_row",
) -> pd.DataFrame:
    """
    Return a normalized dataframe for semantic slot-based tracking operations.

    Output columns
    --------------
    - frame
    - y
    - x
    - label
    - id
    - _slot_key
    - _source_row_index
    - <valid_flag_column>

    Notes
    -----
    This helper is intentionally generic and can be reused by:
    - merge preview/apply
    - tracked prediction refinement / bulk delete
    """
    data, feat_df = extract_layer_data_and_features(layer)

    coord_cols = coord_columns_for_data(data)
    coords = pd.DataFrame(data[:, : len(coord_cols)], columns=coord_cols)

    df = pd.concat([coords.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    df["_source_row_index"] = np.arange(len(df), dtype=int)

    df["frame"] = coerce_frame_series(df.get("frame"))
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["label"] = pick_semantic_series(df, layer, primary="label", fallback_prop="label")
    df["id"] = pick_semantic_series(df, layer, primary="id", fallback_prop="id").map(normalize_slot_id)

    finite_xy = np.isfinite(df[["y", "x"]].to_numpy(dtype=float)).all(axis=1)
    valid_frame = df["frame"].notna()
    valid_label = df["label"].astype(str).str.strip().ne("")
    df[valid_flag_column] = finite_xy & valid_frame & valid_label

    df["_slot_key"] = [
        build_slot_key(frame=f, slot_id=i, label=l) if ok else None
        for f, i, l, ok in zip(
            df["frame"].tolist(),
            df["id"].tolist(),
            df["label"].tolist(),
            df[valid_flag_column].tolist(),
            strict=False,
        )
    ]

    return df


# ---- Layer names ----------------------------------------------------------
def _strip_tracked_prefix(name: str) -> str:
    """
    Best-effort cleanup for repeated tracked prefixes in display names.

    Examples
    --------
    "[Tracked v1] CollectedData_me" -> "CollectedData_me"
    "[Tracked] [Tracked v2] foo"    -> "foo"
    """
    text = str(name).strip()
    text = re.sub(r"^(?:\[Tracked(?: v\d+)?\]\s*)+", "", text)
    return text.strip()


def _base_tracking_source_name(self, source: Points) -> str:
    """
    Return the stable human-facing source name for a new tracking-result layer.

    If the current seed source is already a tracking-result layer, prefer the
    original recorded DLC source layer name so names do not become nested.
    """
    if self.lifecycle_manager.is_tracking_result_layer(source):
        original = self.lifecycle_manager.tracking_result_source_layer_name(source)
        if original:
            return self._strip_tracked_prefix(original)

    return self._strip_tracked_prefix(str(getattr(source, "name", "Unnamed layer")))


def _tracking_name_suffix(
    tracker_name: str,
    ref_frame_idx: int,
    source: Points,
) -> str:
    """
    Return the suffix part shared by all iterations of the same tracking run.

    Example
    -------
    "CollectedData_me - t0 - Cotracker3"
    """
    base_source_name = _base_tracking_source_name(source)
    return f"{base_source_name} - t{ref_frame_idx} - {tracker_name}"


def make_tracking_iteration_name(
    viewer: napari.Viewer,
    tracker_name: str,
    ref_frame_idx: int,
    source: Points,
) -> str:
    """
    Build a unique tracked-layer name with an incrementing version prefix.

    Examples
    --------
    [Tracked v1] CollectedData_me - t0 - Cotracker3
    [Tracked v2] CollectedData_me - t5 - Cotracker3
    """
    suffix = _tracking_name_suffix(
        tracker_name=tracker_name,
        ref_frame_idx=ref_frame_idx,
        source=source,
    )

    # Match names of the exact same tracking run description and collect versions.
    # Example matched names:
    #   [Tracked v1] CollectedData_me - t0 - Cotracker3
    #   [Tracked v2] CollectedData_me - t5 - Cotracker3
    pattern = re.compile(rf"^\[Tracked v(?P<version>\d+)\]\s+{re.escape(suffix)}$")

    versions: list[int] = []
    for layer in viewer.layers:
        name = str(getattr(layer, "name", ""))
        m = pattern.match(name)
        if m is None:
            continue
        try:
            versions.append(int(m.group("version")))
        except Exception:
            pass

    next_version = (max(versions) + 1) if versions else 1
    return f"[Tracked v{next_version}] {suffix}"
