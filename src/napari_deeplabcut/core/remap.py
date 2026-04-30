# src/napari_deeplabcut/core/remap.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from napari_deeplabcut.core.project_paths import PathMatchPolicy, canonicalize_path, find_matching_depth

logger = logging.getLogger(__name__)
# NOTE: if relevant we could add unmapped annotations as a separate "unmapped" layer

_WARN_OLD_PATH_COVERAGE = 0.80
_SAMPLE_N = 5  # Number of examples to include in warnings about duplicate keys.


class RemapOutcome(str, Enum):
    SKIPPED = "skipped"
    NOOP = "noop"
    APPLIED_FULL = "applied_full"
    APPLIED_PARTIAL = "applied_partial"
    REJECTED = "rejected"


class RemapReason(str, Enum):
    NONE = "none"
    NO_DATA = "no_data"
    EMPTY_MAPPING = "empty_mapping"
    NO_OLD_PATHS = "no_old_paths"
    NO_NEW_PATHS = "no_new_paths"
    NO_OVERLAP = "no_overlap"
    ALREADY_ALIGNED = "already_aligned"
    AMBIGUOUS_MATCH = "ambiguous_match"
    NO_MAPPABLE_ROWS = "no_mappable_rows"
    INVALID_TIME_COLUMN = "invalid_time_column"
    REMAP_FAILED = "remap_failed"
    REMAPPED = "remapped"
    PARTIAL_ROWS_DROPPED = "partial_rows_dropped"
    USED_INDICES_UNMAPPED = "used_indices_unmapped"


@dataclass(frozen=True)
class RemapResult:
    """
    Result of an attempted time/frame remapping.

    Attributes
    ----------
    outcome:
        Overall outcome of the remap attempt, e.g. "skipped", "applied
        partial", "rejected", etc.
    reason:
        More specific reason for the outcome.
    depth_used:
        Canonicalization depth used to match paths (e.g. 3, 2, 1), or None.
    mapped_count:
        Number of old frame indices that had a mapping into new indices.
    message:
        Human-readable summary suitable for logs / UI.
    data:
        Remapped data object (same type shape intent as input), or None if not applied.
    warnings:
        Tuple of warning strings describing ambiguity/risk detected during remap decision.
    dropped_row_count:
        For partial remaps where some rows had unmapped indices and were dropped, the count of dropped rows.
    dropped_frame_indices:
        For partial remaps where some rows had unmapped indices and were dropped,
        example frame indices that were dropped.
    """

    outcome: RemapOutcome
    reason: RemapReason
    depth_used: int | None
    mapped_count: int
    message: str
    data: Any | None
    warnings: tuple[str, ...] = ()
    dropped_row_count: int = 0  # for partial remaps where some rows had unmapped indices and were dropped
    dropped_frame_indices: tuple[int, ...] = ()  # example frame indices that were dropped due to unmapped indices

    @property
    def applied(self) -> bool:
        return self.outcome in {RemapOutcome.APPLIED_FULL, RemapOutcome.APPLIED_PARTIAL}

    @property
    def paths_updated(self) -> bool:
        return self.outcome in {RemapOutcome.APPLIED_FULL, RemapOutcome.APPLIED_PARTIAL, RemapOutcome.NOOP}


def _remap_array(values: np.ndarray, idx_map: Mapping[int, int]) -> np.ndarray:
    """
    Remap time indices in an array of indices.

    Parameters
    ----------
    values:
        Array-like of integer time/frame indices to remap.
    idx_map:
        Mapping from old integer frame index -> new integer frame index.
        Indices not present in the mapping raise a KeyError.
    """
    values = np.asarray(values)
    if values.size == 0:
        return values

    try:
        values_int = values.astype(int, copy=False)
    except Exception:
        values_int = values.astype(int)

    missing = [int(v) for v in values_int if int(v) not in idx_map]
    if missing:
        raise KeyError(f"Unmapped frame indices encountered during remap: {missing[:10]}")

    mapped = np.fromiter(
        (idx_map[int(v)] for v in values_int),
        dtype=values_int.dtype,
        count=len(values_int),
    )
    return mapped


def _filter_and_remap_array(
    arr: np.ndarray,
    *,
    time_col: int,
    idx_map: dict[int, int],
) -> tuple[np.ndarray | None, int, tuple[int, ...], RemapReason, str]:

    if arr.ndim < 2 or arr.shape[1] <= time_col:
        return None, 0, (), RemapReason.INVALID_TIME_COLUMN, "Invalid time column."

    t = arr[:, time_col].astype(int, copy=False)
    keep_mask = np.array([v in idx_map for v in t], dtype=bool)
    drop_mask = ~keep_mask

    if not np.any(keep_mask):
        return (
            None,
            int(drop_mask.sum()),
            tuple(sorted(set(t))),
            RemapReason.NO_MAPPABLE_ROWS,
            ("No rows could be mapped to the new image stack."),
        )

    dropped_rows = int(drop_mask.sum())
    dropped_frames = tuple(sorted(set(t[drop_mask])))

    kept = arr[keep_mask].copy()
    kept[:, time_col] = np.array([idx_map[int(v)] for v in kept[:, time_col]])

    if dropped_rows > 0:
        return (
            kept,
            dropped_rows,
            dropped_frames,
            RemapReason.PARTIAL_ROWS_DROPPED,
            (
                f"Partially remapped data; dropped {dropped_rows} row(s) whose frame indices "
                "were not found in the new image stack."
            ),
        )
    if not np.any(keep_mask):
        return (
            None,
            int(drop_mask.sum()),
            tuple(sorted(set(t))),
            RemapReason.NO_MAPPABLE_ROWS,
            ("No rows could be mapped to the new image stack."),
        )

    return kept, 0, (), RemapReason.REMAPPED, "Remapped all rows."


def _find_duplicates(keys: list[str]) -> dict[str, int]:
    """Return dict of duplicate key -> count (only for keys occurring > 1)."""
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return {k: c for k, c in counts.items() if c > 1}


def _used_time_indices(data: Any, time_col: int) -> set[int]:
    if data is None:
        return set()

    if isinstance(data, list):
        used = set()
        for verts in data:
            arr = np.asarray(verts)
            if arr.size == 0 or arr.ndim < 2 or arr.shape[1] <= time_col:
                continue
            used.update(int(v) for v in arr[:, time_col])
        return used

    arr = np.asarray(data)
    if arr.size == 0 or arr.ndim < 2 or arr.shape[1] <= time_col:
        return set()
    return set(int(v) for v in arr[:, time_col])


def build_frame_index_map(
    *,
    old_paths: Iterable[str],
    new_paths: Iterable[str],
    policy: PathMatchPolicy = PathMatchPolicy.ORDERED_DEPTHS,
) -> tuple[dict[int, int], int | None]:
    """
    Build a mapping from old frame indices -> new frame indices using canonicalized path overlap.

    Returns
    -------
    (idx_map, depth_used)
    """
    old_paths = list(old_paths or [])
    new_paths = list(new_paths or [])
    if not old_paths or not new_paths:
        return {}, None

    depth = find_matching_depth(old_paths, new_paths, policy=policy)
    if depth is None:
        return {}, None

    old_keys = [canonicalize_path(p, depth) for p in old_paths]
    new_keys = [canonicalize_path(p, depth) for p in new_paths]

    key_to_new_idx = {k: i for i, k in enumerate(new_keys)}

    idx_map: dict[int, int] = {}
    for old_idx, k in enumerate(old_keys):
        new_idx = key_to_new_idx.get(k)
        if new_idx is not None:
            idx_map[old_idx] = new_idx

    return idx_map, depth


def remap_time_indices(
    *,
    data: Any,
    time_col: int,
    idx_map: Mapping[int, int],
    allow_partial: bool = False,
) -> RemapResult:
    """
    Remap time indices in a data container.

    If allow_partial is False, all used frame indices must be present in
    idx_map.

    If allow_partial is True, rows whose frame index is not present in idx_map
    are dropped and reported as APPLIED_PARTIAL.
    """
    mapped_count = len(idx_map)

    def _result(
        outcome: RemapOutcome,
        reason: RemapReason,
        message: str,
        data_out: Any | None = None,
        *,
        dropped_row_count: int = 0,
        dropped_frame_indices: tuple[int, ...] = (),
    ) -> RemapResult:
        return RemapResult(
            outcome=outcome,
            reason=reason,
            depth_used=None,
            mapped_count=mapped_count,
            message=message,
            data=data_out,
            dropped_row_count=dropped_row_count,
            dropped_frame_indices=dropped_frame_indices,
        )

    if data is None:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.NO_DATA,
            "No data to remap (data is None).",
        )

    if not idx_map:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.EMPTY_MAPPING,
            "No index mapping available (empty idx_map).",
        )

    # Array-like path. This is the common Points-layer case.
    if not isinstance(data, list):
        arr = np.asarray(data)

        if arr.size == 0:
            return _result(
                RemapOutcome.SKIPPED,
                RemapReason.NO_DATA,
                "No data to remap (empty array).",
            )

        if arr.ndim < 2 or arr.shape[1] <= time_col:
            return _result(
                RemapOutcome.SKIPPED,
                RemapReason.INVALID_TIME_COLUMN,
                "Data shape does not contain a valid time column.",
            )

        if allow_partial:
            remapped, dropped_rows, dropped_frames, reason, message = _filter_and_remap_array(
                arr,
                time_col=time_col,
                idx_map=dict(idx_map),
            )

            if remapped is None:
                outcome = (
                    RemapOutcome.REJECTED
                    if reason
                    in {
                        RemapReason.NO_MAPPABLE_ROWS,
                        RemapReason.INVALID_TIME_COLUMN,
                    }
                    else RemapOutcome.SKIPPED
                )
                return _result(
                    outcome,
                    reason,
                    message,
                    dropped_row_count=dropped_rows,
                    dropped_frame_indices=dropped_frames[:_SAMPLE_N],
                )

            outcome = RemapOutcome.APPLIED_PARTIAL if dropped_rows else RemapOutcome.APPLIED_FULL
            return _result(
                outcome,
                reason,
                message,
                remapped,
                dropped_row_count=dropped_rows,
                dropped_frame_indices=dropped_frames[:_SAMPLE_N],
            )

        arr2 = np.array(arr, copy=True)
        t = arr2[:, time_col]

        try:
            t2 = _remap_array(t, idx_map)
        except Exception as e:
            return _result(
                RemapOutcome.REJECTED,
                RemapReason.REMAP_FAILED,
                f"Failed to remap time column: {e}",
            )

        try:
            unchanged = np.array_equal(t2, np.asarray(t).astype(int, copy=False))
        except Exception:
            unchanged = False

        if unchanged:
            return _result(
                RemapOutcome.NOOP,
                RemapReason.ALREADY_ALIGNED,
                "Time column already aligned; no remap needed.",
            )

        arr2[:, time_col] = t2
        return _result(
            RemapOutcome.APPLIED_FULL,
            RemapReason.REMAPPED,
            "Remapped array-like data.",
            arr2,
        )

    # Existing list-like behavior can remain strict for now.
    new_data = []
    changed = False

    for verts in data:
        arr = np.asarray(verts)

        if arr.size == 0:
            new_data.append(arr)
            continue

        if arr.ndim < 2 or arr.shape[1] <= time_col:
            new_data.append(arr)
            continue

        arr2 = np.array(arr, copy=True)
        t = arr2[:, time_col]

        try:
            t2 = _remap_array(t, idx_map)
        except Exception as e:
            return _result(
                RemapOutcome.REJECTED,
                RemapReason.REMAP_FAILED,
                f"Failed to remap list-like vertices: {e}",
            )

        try:
            if not np.array_equal(t2, np.asarray(t).astype(int, copy=False)):
                changed = True
        except Exception:
            changed = True

        arr2[:, time_col] = t2
        new_data.append(arr2)

    if changed:
        return _result(
            RemapOutcome.APPLIED_FULL,
            RemapReason.REMAPPED,
            "Remapped list-like vertices.",
            new_data,
        )

    return _result(
        RemapOutcome.NOOP,
        RemapReason.ALREADY_ALIGNED,
        "List-like vertices already aligned; no remap needed.",
    )


def remap_layer_data_by_paths(
    *,
    data: Any,
    old_paths: Iterable[str] | None,
    new_paths: Iterable[str] | None,
    time_col: int,
    policy: PathMatchPolicy = PathMatchPolicy.ORDERED_DEPTHS,
    allow_partial: bool = False,
) -> RemapResult:
    """
    Remap layer time/frame indices after the image stack path list changes.

    Purpose
    -------
    This function reconciles annotation data whose time/frame column refers to
    indices in `old_paths` with a newly loaded image path list `new_paths`.

    In DeepLabCut-style projects, annotation paths would typically be something like:

        labeled-data/<video-or-session-name>/img070.png

    The primary safety invariant is whether all actually-used
    annotation frame indices can be mapped uniquely and unambiguously.

    Invariants
    ----------
    A remap is applied only if all of the following hold:

    1. A canonicalization depth can be found for matching old/new paths.
    2. Canonical keys at that depth are not ambiguous:
       - no duplicate keys in old paths,
       - no duplicate keys in new paths,
       - no non-bijective old-index -> new-index mapping.
    3. Every frame index actually used by `data[:, time_col]` has a mapping.
    4. The data has a valid time column when data remapping is required.

    Assumptions
    -----------
    - The layer's time/frame column stores integer indices into `old_paths`.
    - `old_paths[i]` describes the image frame originally referenced by time
      index `i`.
    - `new_paths[j]` describes the image frame currently loaded at time index
      `j`.
    - Canonicalized path equality means "same semantic frame" for the selected
      matching depth.
    - Sparse annotations are normal. It is acceptable for many image paths to
      have no annotation rows.

    Outcomes
    --------
    - SKIPPED:
        Required information is missing, or no path overlap can be found.
    - NOOP:
        Paths are already aligned, or a safe mapping exists but no data rewrite
        is needed.
    - APPLIED_FULL:
        All relevant annotation rows were remapped.
    - APPLIED_PARTIAL:
        Some rows were dropped during a permissive/partial remap.
        This high-level function currently prefers rejection over partial remap
        when used frame indices are unmapped.
    - REJECTED:
        A remap was possible in principle, but unsafe due to ambiguity or
        unmapped used frame indices.

    Diagnostics
    -----------
    The function may report warnings for suspicious but non-fatal situations,
    such as low old-path coverage. These warnings do not determine correctness.
    The hard safety checks above do.
    """
    old_paths = list(old_paths or [])
    new_paths = list(new_paths or [])

    def _result(
        outcome: RemapOutcome,
        reason: RemapReason,
        message: str,
        *,
        depth_used: int | None = None,
        mapped_count: int = 0,
        data_out: Any | None = None,
        warnings: list[str] | tuple[str, ...] = (),
        dropped_row_count: int = 0,
        dropped_frame_indices: tuple[int, ...] = (),
    ) -> RemapResult:
        return RemapResult(
            outcome=outcome,
            reason=reason,
            depth_used=depth_used,
            mapped_count=mapped_count,
            message=message,
            data=data_out,
            warnings=tuple(warnings),
            dropped_row_count=dropped_row_count,
            dropped_frame_indices=dropped_frame_indices,
        )

    # --- Initial checks -----------------------------------------------------

    if not old_paths:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.NO_OLD_PATHS,
            "No old paths present; cannot remap.",
        )

    if not new_paths:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.NO_NEW_PATHS,
            "No new paths present; cannot remap.",
        )

    depth = find_matching_depth(old_paths, new_paths, policy=policy)
    if depth is None:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.NO_OVERLAP,
            "No overlap between old and new paths; skipping remap.",
        )

    # --- Canonicalize  -------------------------------------------------

    old_keys = [canonicalize_path(p, depth) for p in old_paths]
    new_keys = [canonicalize_path(p, depth) for p in new_paths]

    # Fast path: already aligned => no data rewrite needed, but paths metadata
    # may be considered safe/updatable by callers.
    if old_keys == new_keys:
        return _result(
            RemapOutcome.NOOP,
            RemapReason.ALREADY_ALIGNED,
            "Path keys already aligned; no remap needed.",
            depth_used=depth,
            mapped_count=len(old_keys),
        )

    # --- Build idx_map -----------------------------------------------------

    # Keep first occurrence for deterministic behavior.
    # If duplicates exist, we will reject later anyway.
    key_to_new_idx: dict[str, int] = {}
    for i, k in enumerate(new_keys):
        key_to_new_idx.setdefault(k, i)

    idx_map: dict[int, int] = {}
    for old_idx, k in enumerate(old_keys):
        new_idx = key_to_new_idx.get(k)
        if new_idx is not None:
            idx_map[old_idx] = new_idx

    if not idx_map:
        return _result(
            RemapOutcome.SKIPPED,
            RemapReason.NO_OVERLAP,
            "No overlap between old and new paths after canonicalization; skipping remap.",
            depth_used=depth,
        )

    # --- Coverage / safety checks -----------------------------------------
    used = _used_time_indices(data, time_col)
    unmapped_used = used.difference(idx_map)
    mapped_used = used.intersection(idx_map)

    logger.debug(
        "Remap used-frame coverage: depth=%s used=%s mapped_used=%s unmapped_used=%s "
        "used_examples=%s unmapped_examples=%s",
        depth,
        len(used),
        len(mapped_used),
        len(unmapped_used),
        list(mapped_used)[:10],
        list(unmapped_used)[:10],
    )

    warnings: list[str] = []

    dup_old = _find_duplicates(old_keys)
    dup_new = _find_duplicates(new_keys)

    if dup_old:
        examples = ", ".join(list(dup_old.keys())[:_SAMPLE_N])
        warnings.append(f"Duplicate canonical keys in old_paths at depth={depth} (examples: {examples}).")

    if dup_new:
        examples = ", ".join(list(dup_new.keys())[:_SAMPLE_N])
        warnings.append(f"Duplicate canonical keys in new_paths at depth={depth} (examples: {examples}).")

    matched_old = set(idx_map)
    matched_new = set(idx_map.values())

    old_path_coverage = len(matched_old) / max(1, len(old_keys))
    new_path_coverage = len(matched_new) / max(1, len(new_keys))
    if old_path_coverage < _WARN_OLD_PATH_COVERAGE:
        warnings.append(
            f"Low old-path coverage at depth={depth}: {old_path_coverage:.2f} "
            f"({len(matched_old)} of {len(old_keys)} old paths map to the new stack). "
            "This may be still normal for DLC annotations if all used frame indices are mapped."
        )

    logger.debug(
        "Remap new-path coverage: %.2f (%s of %s new paths represented by old paths). "
        "Low values are expected when annotations are sparse.",
        new_path_coverage,
        len(matched_new),
        len(new_keys),
    )

    non_bijective = len(matched_new) < len(idx_map)
    if non_bijective:
        warnings.append("Non-bijective mapping detected (multiple old indices map to the same new index).")

    for w in warnings:
        logger.warning("Remap may be ambiguous/risky: %s", w)

    # Reject if the layer actually uses frame indices that do not map.
    if unmapped_used and not allow_partial:
        msg = "Rejected remap because some labeled frame indices have no mapping in the new image stack."
        logger.warning(
            "%s depth=%s unmapped_used_count=%s examples=%s",
            msg,
            depth,
            len(unmapped_used),
            list(unmapped_used)[:10],
        )
        warnings = [
            *warnings,
            (
                f"Used frame indices without mapping at depth={depth}: "
                f"{list(unmapped_used)[:10]}{'...' if len(unmapped_used) > 10 else ''}"
            ),
        ]
        return _result(
            RemapOutcome.REJECTED,
            RemapReason.USED_INDICES_UNMAPPED,
            msg,
            depth_used=depth,
            mapped_count=len(idx_map),
            warnings=warnings,
        )

    if unmapped_used and allow_partial:
        warnings.append(
            f"Partial remap: {len(unmapped_used)} used frame index/indices have no mapping "
            f"at depth={depth}; rows on those frames will be dropped. "
            f"Examples: {sorted(unmapped_used)[:_SAMPLE_N]}"
            f"{'...' if len(unmapped_used) > _SAMPLE_N else ''}"
        )

    # Reject ambiguous canonicalization/matching.
    if dup_old or dup_new or non_bijective:
        msg = f"Rejected ambiguous remap at depth={depth}; keeping original frame indices and paths."
        logger.warning(msg)
        return _result(
            RemapOutcome.REJECTED,
            RemapReason.AMBIGUOUS_MATCH,
            msg,
            depth_used=depth,
            mapped_count=len(idx_map),
            warnings=warnings,
        )

    # Safe mapping exists, but there is no data to rewrite. This is still a
    # successful metadata/path-alignment case for callers.
    if data is None:
        return _result(
            RemapOutcome.NOOP,
            RemapReason.NO_DATA,
            "Safe path mapping found, but there is no data to remap.",
            depth_used=depth,
            mapped_count=len(idx_map),
            warnings=warnings,
        )

    # --- Apply

    res = remap_time_indices(data=data, time_col=time_col, idx_map=idx_map, allow_partial=allow_partial)

    return _result(
        res.outcome,
        res.reason,
        res.message,
        depth_used=depth,
        mapped_count=len(idx_map),
        data_out=res.data,
        warnings=warnings,
        dropped_row_count=res.dropped_row_count,
        dropped_frame_indices=res.dropped_frame_indices,
    )
