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

# NOTE: @C-Achard 2026-07-15 This system could be reused for machine labels -> GT merges,
# and would give more control to users to choose whether to overwrite existing GT or only fill missing slots.
# It would also simplify the whole machinery for machine to GT promotion.


class TrackingMergePolicy(str, Enum):
    """Supported merge policies for tracking-result -> DLC points merges."""

    FILL_MISSING = "fill_missing"
    OVERWRITE_EXISTING = "overwrite_existing"


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
    - Under FILL_MISSING, mismatching occupied slots are reported as conflicts and skipped.
    - Under OVERWRITE_EXISTING, mismatching occupied slots are reported as overwrites.
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
    n_overwriteable: int
    n_invalid_source: int

    has_source_duplicates: bool
    has_target_duplicates: bool
    is_valid: bool
    invalid_reason: str | None

    append_source_indices: tuple[int, ...]
    identical_source_indices: tuple[int, ...]
    conflict_source_indices: tuple[int, ...]
    overwrite_source_indices: tuple[int, ...]
    invalid_source_indices: tuple[int, ...]

    conflicts: tuple[TrackingMergeConflictEntry, ...]
    overwrites: tuple[TrackingMergeConflictEntry, ...]
    truncated_conflicts: int = 0
    truncated_overwrites: int = 0


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

    source_invalid_mask = ~source_df["_is_valid_merge_row"].astype(bool)
    invalid_source_indices = tuple(sorted_int_tuple(source_df.loc[source_invalid_mask, "_source_row_index"]))

    source_valid = source_df.loc[~source_invalid_mask].copy()
    target_valid = target_df.loc[target_df["_is_valid_merge_row"].astype(bool)].copy()

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
    overwrite_source_indices: list[int] = []

    conflict_entries: list[TrackingMergeConflictEntry] = []
    overwrite_entries: list[TrackingMergeConflictEntry] = []

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

        entry = TrackingMergeConflictEntry(
            frame_label=format_frame_label(row["frame"]),
            keypoint_label=format_slot_label(row["label"], row["id"]),
            source_coords_text=format_coords_text(row),
            target_coords_text=format_coords_text(target_row),
        )

        if policy is TrackingMergePolicy.FILL_MISSING:
            conflict_source_indices.append(src_idx)
            if len(conflict_entries) < max_conflicts:
                conflict_entries.append(entry)
        elif policy is TrackingMergePolicy.OVERWRITE_EXISTING:
            overwrite_source_indices.append(src_idx)
            if len(overwrite_entries) < max_conflicts:
                overwrite_entries.append(entry)
        else:
            return _invalid_preview(
                source_layer=source_layer,
                target_layer=target_layer,
                policy=policy,
                reason=f"Unsupported merge policy: {policy!r}",
                source_fingerprint=source_fp,
                target_fingerprint=target_fp,
                n_source_rows=len(source_df),
                n_invalid_source=len(invalid_source_indices),
                invalid_source_indices=invalid_source_indices,
            )

    truncated_conflicts = max(0, len(conflict_source_indices) - len(conflict_entries))
    truncated_overwrites = max(0, len(overwrite_source_indices) - len(overwrite_entries))

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
        n_overwriteable=int(len(overwrite_source_indices)),
        n_invalid_source=int(len(invalid_source_indices)),
        has_source_duplicates=False,
        has_target_duplicates=False,
        is_valid=True,
        invalid_reason=None,
        append_source_indices=tuple(sorted(append_source_indices)),
        identical_source_indices=tuple(sorted(identical_source_indices)),
        conflict_source_indices=tuple(sorted(conflict_source_indices)),
        overwrite_source_indices=tuple(sorted(overwrite_source_indices)),
        invalid_source_indices=tuple(sorted(invalid_source_indices)),
        conflicts=tuple(conflict_entries),
        overwrites=tuple(overwrite_entries),
        truncated_conflicts=truncated_conflicts,
        truncated_overwrites=truncated_overwrites,
    )


def apply_tracking_merge(
    source_layer: Points,
    target_layer: Points,
    *,
    preview: TrackingMergePreview,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Apply a previously computed merge preview.
    """
    if not preview.is_valid:
        reason = preview.invalid_reason or "Unknown reason."
        raise ValueError(f"Cannot apply an invalid tracking merge preview: {reason}")

    if preview.policy not in {
        TrackingMergePolicy.FILL_MISSING,
        TrackingMergePolicy.OVERWRITE_EXISTING,
    }:
        raise ValueError(f"Unsupported merge policy: {preview.policy!r}")

    current_source_fp = fingerprint_points_layer(source_layer)
    current_target_fp = fingerprint_points_layer(target_layer)
    if current_source_fp != preview.source_fingerprint:
        raise ValueError("Source layer changed after preview was built; please refresh the merge preview.")
    if current_target_fp != preview.target_fingerprint:
        raise ValueError("Target layer changed after preview was built; please refresh the merge preview.")

    target_data, target_features = extract_layer_data_and_features(target_layer)

    if not preview.append_source_indices and not preview.overwrite_source_indices:
        return target_data.copy(), target_features.copy()

    source_norm = _normalize_points_layer_for_merge(source_layer)
    target_norm = _normalize_points_layer_for_merge(target_layer)

    coord_cols = coord_columns_for_data(target_data)

    merged_data = target_data.copy()
    merged_features = target_features.reset_index(drop=True).copy()

    # Overwrite existing rows (only for relevant policies)
    if preview.policy is TrackingMergePolicy.OVERWRITE_EXISTING and preview.overwrite_source_indices:
        overwrite_df = (
            source_norm.loc[source_norm["_source_row_index"].isin(preview.overwrite_source_indices)]
            .sort_values("_source_row_index")
            .reset_index(drop=True)
        )

        if not overwrite_df.empty:
            missing_coord_cols = [col for col in coord_cols if col not in overwrite_df.columns]
            if missing_coord_cols:
                raise ValueError(
                    "Source and target point dimensionality are incompatible for merge: "
                    f"target requires coordinate column(s) {missing_coord_cols!r}, "
                    f"but the source provides only "
                    f"{[c for c in overwrite_df.columns if str(c).startswith('coord_')]!r}. "
                    "Please refresh the preview or merge into a compatible target layer."
                )

            target_valid = target_norm.loc[target_norm["_is_valid_merge_row"].astype(bool)].copy()
            target_row_by_key = {
                row["_slot_key"]: int(row["_source_row_index"]) for row in target_valid.to_dict(orient="records")
            }

            overwrite_target_indices: list[int] = []
            for row in overwrite_df.to_dict(orient="records"):
                tgt_idx = target_row_by_key.get(row["_slot_key"])
                if tgt_idx is None:
                    raise ValueError(
                        "Target layer no longer matches the overwrite preview; please refresh the merge preview."
                    )
                overwrite_target_indices.append(tgt_idx)

            overwrite_coords = overwrite_df.loc[:, coord_cols].to_numpy(dtype=float, copy=True)
            merged_data[np.asarray(overwrite_target_indices, dtype=int)] = overwrite_coords

            shared_cols = [c for c in merged_features.columns if c in overwrite_df.columns]
            for col in shared_cols:
                merged_features.loc[overwrite_target_indices, col] = (
                    overwrite_df[col].reset_index(drop=True).to_numpy(copy=True)
                )

    if preview.append_source_indices:
        append_df = (
            source_norm.loc[source_norm["_source_row_index"].isin(preview.append_source_indices)]
            .sort_values("_source_row_index")
            .reset_index(drop=True)
        )

        if not append_df.empty:
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
                target_features=merged_features,
            )

            merged_data = np.vstack([merged_data, new_append_data]) if len(merged_data) else new_append_data
            merged_features = pd.concat(
                [merged_features.reset_index(drop=True), append_features],
                ignore_index=True,
            )

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
        n_overwriteable=0,
        n_invalid_source=int(n_invalid_source),
        has_source_duplicates=bool(has_source_duplicates),
        has_target_duplicates=bool(has_target_duplicates),
        is_valid=False,
        invalid_reason=reason,
        append_source_indices=(),
        identical_source_indices=(),
        conflict_source_indices=(),
        overwrite_source_indices=(),
        invalid_source_indices=tuple(invalid_source_indices),
        conflicts=(),
        overwrites=(),
        truncated_conflicts=0,
        truncated_overwrites=0,
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
