# src/napari_deeplabcut/tracking/core/merge.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from napari.layers import Points

from .utils import (
    coord_columns_for_data,
    duplicate_slot_row_indices,
    extract_layer_data_and_features,
    format_coords_text,
    format_frame_label,
    format_slot_label,
    normalize_points_layer_for_tracking,
    sorted_int_tuple,
)

_COORD_TOL_DEFAULT = 1e-6


class TrackingMergePolicy(str, Enum):
    """Supported merge policies for tracking-result -> DLC points merges."""

    FILL_MISSING = "fill_missing"


@dataclass(frozen=True)
class LayerFingerprint:
    """Lightweight layer snapshot used to detect stale preview/apply state."""

    layer_name: str
    n_rows: int
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class TrackingMergeConflictEntry:
    """One conflicting semantic slot between source tracking layer and target DLC layer."""

    frame_label: str
    keypoint_label: str
    source_coords_text: str
    target_coords_text: str


@dataclass(frozen=True)
class TrackingMergePreview:
    """
    Immutable merge preview / plan.

    Notes
    -----
    - `append_source_indices` etc. refer to original source-layer row indices.
    - `is_valid=False` means merge must not proceed.
    - Under FILL_MISSING, conflicts are reported and skipped.
    """

    source_layer_name: str
    target_layer_name: str
    policy: TrackingMergePolicy

    source_fingerprint: LayerFingerprint
    target_fingerprint: LayerFingerprint

    n_source_rows: int
    n_appendable: int
    n_identical: int
    n_conflicts: int
    n_invalid_source: int

    has_source_duplicates: bool
    has_target_duplicates: bool
    is_valid: bool
    invalid_reason: str | None

    append_source_indices: tuple[int, ...]
    identical_source_indices: tuple[int, ...]
    conflict_source_indices: tuple[int, ...]
    invalid_source_indices: tuple[int, ...]

    conflicts: tuple[TrackingMergeConflictEntry, ...]
    truncated_conflicts: int = 0


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def preview_tracking_merge(
    source_layer: Points,
    target_layer: Points,
    *,
    policy: TrackingMergePolicy = TrackingMergePolicy.FILL_MISSING,
    coord_tolerance: float = _COORD_TOL_DEFAULT,
    max_conflicts: int = 50,
) -> TrackingMergePreview:
    """
    Build an immutable merge preview from a tracking-result layer into a DLC points layer.

    Parameters
    ----------
    source_layer
        Source points layer. The caller is responsible for selecting a valid
        tracking-result layer using the lifecycle manager.
    target_layer
        Target points layer. The caller is responsible for selecting a valid
        mergeable DLC points layer using the lifecycle manager.
    policy
        Merge policy. Only FILL_MISSING is supported in v1.
    coord_tolerance
        Absolute tolerance used to classify a slot as identical vs conflicting.
    max_conflicts
        Maximum number of conflict entries to materialize for UI display.

    Returns
    -------
    TrackingMergePreview
        Complete preview and apply plan.

    Invariants
    ----------
    - Merge identity is semantic: (frame, normalized id, label)
    - Source/target duplicate semantic slots invalidate the preview
    - Tracking-only feature columns must not become authoritative target schema
    """
    if source_layer is target_layer:
        return _invalid_preview(
            source_layer=source_layer,
            target_layer=target_layer,
            policy=policy,
            reason="Source and target layers must be different.",
        )

    source_df = _normalize_points_layer_for_merge(source_layer)
    target_df = _normalize_points_layer_for_merge(target_layer)

    source_fp = fingerprint_points_layer(source_layer)
    target_fp = fingerprint_points_layer(target_layer)

    # Invalid source rows are skipped but do not invalidate the whole preview.
    source_invalid_mask = ~source_df["_is_valid_merge_row"].astype(bool)
    invalid_source_indices = tuple(sorted_int_tuple(source_df.loc[source_invalid_mask, "_source_row_index"]))

    source_valid = source_df.loc[~source_invalid_mask].copy()
    target_valid = target_df.loc[target_df["_is_valid_merge_row"].astype(bool)].copy()

    # Duplicate semantic slots are integrity issues, not ordinary merge conflicts.
    src_dup = duplicate_slot_row_indices(source_valid)
    if src_dup:
        return _invalid_preview(
            source_layer=source_layer,
            target_layer=target_layer,
            policy=policy,
            reason="Source tracking layer contains duplicate semantic slots.",
            source_fingerprint=source_fp,
            target_fingerprint=target_fp,
            n_source_rows=len(source_df),
            n_invalid_source=len(invalid_source_indices),
            invalid_source_indices=invalid_source_indices,
            has_source_duplicates=True,
            has_target_duplicates=False,
        )

    tgt_dup = duplicate_slot_row_indices(target_valid)
    if tgt_dup:
        return _invalid_preview(
            source_layer=source_layer,
            target_layer=target_layer,
            policy=policy,
            reason="Target DLC points layer contains duplicate semantic slots.",
            source_fingerprint=source_fp,
            target_fingerprint=target_fp,
            n_source_rows=len(source_df),
            n_invalid_source=len(invalid_source_indices),
            invalid_source_indices=invalid_source_indices,
            has_source_duplicates=False,
            has_target_duplicates=True,
        )

    target_by_key = {row["_slot_key"]: row for row in target_valid.to_dict(orient="records")}

    append_source_indices: list[int] = []
    identical_source_indices: list[int] = []
    conflict_source_indices: list[int] = []
    conflicts: list[TrackingMergeConflictEntry] = []

    for row in source_valid.to_dict(orient="records"):
        src_idx = int(row["_source_row_index"])
        slot_key = row["_slot_key"]

        target_row = target_by_key.get(slot_key)
        if target_row is None:
            append_source_indices.append(src_idx)
            continue

        if _rows_have_same_coords(row, target_row, tol=coord_tolerance):
            identical_source_indices.append(src_idx)
            continue

        conflict_source_indices.append(src_idx)
        if len(conflicts) < max_conflicts:
            conflicts.append(
                TrackingMergeConflictEntry(
                    frame_label=format_frame_label(row["frame"]),
                    keypoint_label=format_slot_label(row["label"], row["id"]),
                    source_coords_text=format_coords_text(row),
                    target_coords_text=format_coords_text(target_row),
                )
            )

    truncated = max(0, len(conflict_source_indices) - len(conflicts))

    return TrackingMergePreview(
        source_layer_name=str(getattr(source_layer, "name", "Source layer")),
        target_layer_name=str(getattr(target_layer, "name", "Target layer")),
        policy=policy,
        source_fingerprint=source_fp,
        target_fingerprint=target_fp,
        n_source_rows=int(len(source_df)),
        n_appendable=int(len(append_source_indices)),
        n_identical=int(len(identical_source_indices)),
        n_conflicts=int(len(conflict_source_indices)),
        n_invalid_source=int(len(invalid_source_indices)),
        has_source_duplicates=False,
        has_target_duplicates=False,
        is_valid=True,
        invalid_reason=None,
        append_source_indices=tuple(sorted(append_source_indices)),
        identical_source_indices=tuple(sorted(identical_source_indices)),
        conflict_source_indices=tuple(sorted(conflict_source_indices)),
        invalid_source_indices=tuple(sorted(invalid_source_indices)),
        conflicts=tuple(conflicts),
        truncated_conflicts=truncated,
    )


def apply_tracking_merge(
    source_layer: Points,
    target_layer: Points,
    *,
    preview: TrackingMergePreview,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Apply a previously computed merge preview.

    Parameters
    ----------
    source_layer
        Source tracking-result layer.
    target_layer
        Target DLC points layer.
    preview
        Preview produced by `preview_tracking_merge(...)`.

    Returns
    -------
    (new_data, new_features)
        Data and features to assign back to `target_layer`.

    Raises
    ------
    ValueError
        If the preview is invalid, stale, or incompatible with the current layer state.
    """
    if not preview.is_valid:
        reason = preview.invalid_reason or "Unknown reason."
        raise ValueError(f"Cannot apply an invalid tracking merge preview: {reason}")

    if preview.policy is not TrackingMergePolicy.FILL_MISSING:
        raise ValueError(f"Unsupported merge policy: {preview.policy!r}")

    current_source_fp = fingerprint_points_layer(source_layer)
    current_target_fp = fingerprint_points_layer(target_layer)
    if current_source_fp != preview.source_fingerprint:
        raise ValueError("Source layer changed after preview was built; please refresh the merge preview.")
    if current_target_fp != preview.target_fingerprint:
        raise ValueError("Target layer changed after preview was built; please refresh the merge preview.")

    target_data, target_features = extract_layer_data_and_features(target_layer)

    if not preview.append_source_indices:
        return target_data.copy(), target_features.copy()

    source_norm = _normalize_points_layer_for_merge(source_layer)
    append_df = (
        source_norm.loc[source_norm["_source_row_index"].isin(preview.append_source_indices)]
        .sort_values("_source_row_index")
        .reset_index(drop=True)
    )

    if append_df.empty:
        return target_data.copy(), target_features.copy()

    coord_cols = coord_columns_for_data(target_data)
    missing_coord_cols = [col for col in coord_cols if col not in append_df.columns]
    if missing_coord_cols:
        raise ValueError(
            "Source and target point dimensionality are incompatible for merge: "
            f"target requires coordinate column(s) {missing_coord_cols!r}, "
            f"but the source provides only "
            f"{[c for c in append_df.columns if str(c).startswith('coord_')]!r}. "
            "Please refresh the preview or merge into a compatible target layer."
        )

    new_append_data = append_df.loc[:, coord_cols].to_numpy(dtype=float, copy=True)

    append_features = _build_append_features_from_source(
        source_rows=append_df,
        target_features=target_features,
    )

    merged_data = np.vstack([target_data, new_append_data]) if len(target_data) else new_append_data
    merged_features = pd.concat([target_features.reset_index(drop=True), append_features], ignore_index=True)

    return merged_data, merged_features


def fingerprint_points_layer(layer: Points) -> LayerFingerprint:
    """Return a lightweight fingerprint for stale preview detection."""
    data, features = extract_layer_data_and_features(layer)
    return LayerFingerprint(
        layer_name=str(getattr(layer, "name", "")),
        n_rows=int(len(data)),
        feature_columns=tuple(map(str, features.columns.tolist())),
    )


# -----------------------------------------------------------------------------#
# Internal helpers
# -----------------------------------------------------------------------------#


def _invalid_preview(
    *,
    source_layer: Points,
    target_layer: Points,
    policy: TrackingMergePolicy,
    reason: str,
    source_fingerprint: LayerFingerprint | None = None,
    target_fingerprint: LayerFingerprint | None = None,
    n_source_rows: int = 0,
    n_invalid_source: int = 0,
    invalid_source_indices: tuple[int, ...] = (),
    has_source_duplicates: bool = False,
    has_target_duplicates: bool = False,
) -> TrackingMergePreview:
    return TrackingMergePreview(
        source_layer_name=str(getattr(source_layer, "name", "Source layer")),
        target_layer_name=str(getattr(target_layer, "name", "Target layer")),
        policy=policy,
        source_fingerprint=source_fingerprint or fingerprint_points_layer(source_layer),
        target_fingerprint=target_fingerprint or fingerprint_points_layer(target_layer),
        n_source_rows=int(n_source_rows),
        n_appendable=0,
        n_identical=0,
        n_conflicts=0,
        n_invalid_source=int(n_invalid_source),
        has_source_duplicates=bool(has_source_duplicates),
        has_target_duplicates=bool(has_target_duplicates),
        is_valid=False,
        invalid_reason=reason,
        append_source_indices=(),
        identical_source_indices=(),
        conflict_source_indices=(),
        invalid_source_indices=tuple(invalid_source_indices),
        conflicts=(),
        truncated_conflicts=0,
    )


def _normalize_points_layer_for_merge(layer: Points) -> pd.DataFrame:
    """
    Return a normalized dataframe for semantic merge classification.

    Required normalized columns
    ---------------------------
    - frame
    - y
    - x
    - label
    - id
    - _slot_key
    - _source_row_index
    - _is_valid_merge_row
    """
    return normalize_points_layer_for_tracking(layer, valid_flag_column="_is_valid_merge_row")


def _rows_have_same_coords(source_row: dict[str, Any], target_row: dict[str, Any], *, tol: float) -> bool:
    return (
        abs(float(source_row["x"]) - float(target_row["x"])) <= tol
        and abs(float(source_row["y"]) - float(target_row["y"])) <= tol
    )


def _build_append_features_from_source(
    *,
    source_rows: pd.DataFrame,
    target_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build appended features for the target layer while preserving target schema ownership.

    Policy
    ------
    - Only columns already present in `target_features` are written.
    - tracking_* columns from the source are ignored unless the target already has such a column.
    - Missing target columns remain NA/default.
    """
    n = len(source_rows)
    if n == 0:
        return target_features.iloc[:0].copy()

    append_features = target_features.iloc[:0].reindex(range(n)).copy().reset_index(drop=True)

    shared_cols = [c for c in target_features.columns if c in source_rows.columns]
    for col in shared_cols:
        append_features[col] = source_rows[col].reset_index(drop=True)

    return append_features
