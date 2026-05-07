from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from napari.layers import Points

from .utils import (
    extract_layer_data_and_features,
    normalize_points_layer_for_tracking,
)


@dataclass(frozen=True)
class TrackingFutureDeletePreview:
    layer_name: str
    anchor_frame: int

    n_selected_reference_rows: int
    n_selected_slot_keys: int
    n_rows_to_delete: int

    delete_row_indices: tuple[int, ...]
    ambiguous_slot_frames: tuple[tuple[int, str, str], ...]  # (frame, id, label)

    is_valid: bool
    invalid_reason: str | None


def preview_delete_tracking_points_in_future(
    layer: Points,
    *,
    selected_indices: set[int] | list[int] | tuple[int, ...],
    anchor_frame: int,
) -> TrackingFutureDeletePreview:
    layer_name = str(getattr(layer, "name", "Tracking layer"))

    data, _features = extract_layer_data_and_features(layer)
    n_rows = len(data)

    try:
        selected = tuple(sorted(set(int(i) for i in selected_indices)))
    except Exception:
        selected = ()

    if not selected:
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason="Select one or more tracked points on the current frame first.",
        )

    if any(i < 0 or i >= n_rows for i in selected):
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason="The selected points are no longer valid for this tracking layer.",
        )

    df = normalize_points_layer_for_tracking(
        layer,
        valid_flag_column="_is_valid_delete_row",
    )

    selected_df = df.loc[list(selected)].copy()
    if selected_df.empty:
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason="Select one or more tracked points on the current frame first.",
        )

    invalid_selected = selected_df.loc[~selected_df["_is_valid_delete_row"].astype(bool)]
    if not invalid_selected.empty:
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason="The selected points do not all have a valid label/id identity.",
        )

    selected_frames = selected_df["frame"].astype(int).unique().tolist()
    if any(int(f) != int(anchor_frame) for f in selected_frames):
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason="All selected reference points must be on the current frame.",
        )

    # For future deletion, identity must be stable across time.
    # Do NOT use _slot_key here because it includes frame.
    df["_delete_identity_key"] = list(
        zip(
            df["id"].astype(str).tolist(),
            df["label"].astype(str).str.strip().tolist(),
            strict=False,
        )
    )

    selected_identity_keys = selected_df.assign(
        _delete_identity_key=list(
            zip(
                selected_df["id"].astype(str).tolist(),
                selected_df["label"].astype(str).str.strip().tolist(),
                strict=False,
            )
        )
    )["_delete_identity_key"].tolist()

    if len(set(selected_identity_keys)) != len(selected_identity_keys):
        return _invalid_preview(
            layer_name=layer_name,
            anchor_frame=anchor_frame,
            reason=(
                "The selected points are not semantically unique on the current frame. "
                "Select at most one row per keypoint identity."
            ),
        )

    selected_identity_key_set = set(selected_identity_keys)

    future_df = df.loc[df["frame"].astype(int) > int(anchor_frame)].copy()
    future_matches = future_df.loc[future_df["_delete_identity_key"].isin(selected_identity_key_set)].copy()

    if future_matches.empty:
        return TrackingFutureDeletePreview(
            layer_name=layer_name,
            anchor_frame=int(anchor_frame),
            n_selected_reference_rows=int(len(selected_df)),
            n_selected_slot_keys=int(len(selected_identity_key_set)),
            n_rows_to_delete=0,
            delete_row_indices=(),
            ambiguous_slot_frames=(),
            is_valid=True,
            invalid_reason=None,
        )

    dup_mask = future_matches.duplicated(subset=["frame", "_delete_identity_key"], keep=False)
    if dup_mask.any():
        ambiguous = (
            future_matches.loc[dup_mask, ["frame", "id", "label"]]
            .drop_duplicates()
            .sort_values(["frame", "id", "label"])
        )

        ambiguous_slot_frames = tuple(
            (
                int(row.frame),
                str(row.id),
                str(row.label),
            )
            for row in ambiguous.itertuples(index=False)
        )

        return TrackingFutureDeletePreview(
            layer_name=layer_name,
            anchor_frame=int(anchor_frame),
            n_selected_reference_rows=int(len(selected_df)),
            n_selected_slot_keys=int(len(selected_identity_key_set)),
            n_rows_to_delete=0,
            delete_row_indices=(),
            ambiguous_slot_frames=ambiguous_slot_frames,
            is_valid=False,
            invalid_reason=(
                "Some selected keypoints are ambiguous in future frames because duplicate "
                "tracked rows exist for the same keypoint identity."
            ),
        )

    delete_row_indices = tuple(sorted(int(i) for i in future_matches["_source_row_index"].tolist()))

    return TrackingFutureDeletePreview(
        layer_name=layer_name,
        anchor_frame=int(anchor_frame),
        n_selected_reference_rows=int(len(selected_df)),
        n_selected_slot_keys=int(len(selected_identity_key_set)),
        n_rows_to_delete=int(len(delete_row_indices)),
        delete_row_indices=delete_row_indices,
        ambiguous_slot_frames=(),
        is_valid=True,
        invalid_reason=None,
    )


def apply_delete_tracking_points_in_future(
    layer: Points,
    *,
    preview: TrackingFutureDeletePreview,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not preview.is_valid:
        reason = preview.invalid_reason or "Unknown reason."
        raise ValueError(f"Cannot apply an invalid future-delete preview: {reason}")

    data, features = extract_layer_data_and_features(layer)

    if not preview.delete_row_indices:
        return data.copy(), features.copy()

    n_rows = len(data)
    delete_indices = tuple(sorted(set(int(i) for i in preview.delete_row_indices)))

    bad = [i for i in delete_indices if i < 0 or i >= n_rows]
    if bad:
        raise ValueError(
            "The tracking layer no longer matches the delete preview; some rows to delete are out of bounds."
        )

    keep_mask = np.ones(n_rows, dtype=bool)
    keep_mask[list(delete_indices)] = False

    new_data = data[keep_mask].copy()
    new_features = features.loc[keep_mask].reset_index(drop=True).copy()

    return new_data, new_features


def _invalid_preview(
    *,
    layer_name: str,
    anchor_frame: int,
    reason: str,
    n_selected_reference_rows: int = 0,
    n_selected_slot_keys: int = 0,
    ambiguous_slot_frames: tuple[tuple[int, str, str], ...] = (),
) -> TrackingFutureDeletePreview:
    return TrackingFutureDeletePreview(
        layer_name=layer_name,
        anchor_frame=int(anchor_frame),
        n_selected_reference_rows=int(n_selected_reference_rows),
        n_selected_slot_keys=int(n_selected_slot_keys),
        n_rows_to_delete=0,
        delete_row_indices=(),
        ambiguous_slot_frames=tuple(ambiguous_slot_frames),
        is_valid=False,
        invalid_reason=reason,
    )
