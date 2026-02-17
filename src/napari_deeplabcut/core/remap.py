# src/napari_deeplabcut/core/remap.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from napari_deeplabcut.core.paths import PathMatchPolicy, canonicalize_path, find_matching_depth

logger = logging.getLogger(__name__)

# Heuristic thresholds for "risky remap" warnings.
# These do NOT change behavior; they only control warning emissions.
_WARN_OVERLAP_RATIO = 0.80  # Warn if canonicalized path overlap is below this ratio (relative to smaller set size).
_WARN_MAPPED_RATIO = 0.80  # Warn if mapping coverage of old paths is below this ratio (mapped / old).
_SAMPLE_N = 5  # Number of examples to include in warnings about duplicate keys.


@dataclass(frozen=True)
class RemapResult:
    """
    Result of an attempted time/frame remapping.

    Attributes
    ----------
    changed:
        True if output data differs from input data (remap applied).
    depth_used:
        Canonicalization depth used to match paths (e.g. 3, 2, 1), or None.
    mapped_count:
        Number of old frame indices that had a mapping into new indices.
    message:
        Human-readable summary suitable for debug logs.
    data:
        Remapped data object (same type shape intent as input), or None if not remapped.
    warnings:
        Tuple of warning strings describing ambiguity/risk detected during remap decision.
        This is informational only and does not affect behavior.
    """

    changed: bool
    depth_used: int | None
    mapped_count: int
    message: str
    data: Any | None
    warnings: tuple[str, ...] = ()


def _remap_array(values: np.ndarray, idx_map: Mapping[int, int]) -> np.ndarray:
    """
    Remap time indices in a data container.

    Parameters
    ----------
    data:
        Either:
        - array-like (N, D) numeric
        - list-like of vertex arrays (Shapes-like), each with (M, D)
    time_col:
        Which column contains time/frame indices.
    idx_map:
        Mapping from old integer frame index -> new integer frame index.
    """
    values = np.asarray(values)
    if values.size == 0:
        return values

    try:
        values_int = values.astype(int, copy=False)
    except Exception:
        values_int = values.astype(int)

    mapped = np.fromiter(
        (idx_map.get(int(v), int(v)) for v in values_int),
        dtype=values_int.dtype,
        count=len(values_int),
    )
    return mapped


def _find_duplicates(keys: list[str]) -> dict[str, int]:
    """Return dict of duplicate key -> count (only for keys occurring > 1)."""
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return {k: c for k, c in counts.items() if c > 1}


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
) -> RemapResult:
    """
    Remap time indices in a data container (array-like or list-of-arrays).

    Returns
    -------
    RemapResult
    """
    if data is None:
        return RemapResult(False, None, 0, "No data to remap (data is None).", None)

    if not idx_map:
        return RemapResult(False, None, 0, "No index mapping available (empty idx_map).", None)

    # Shapes-like: list of arrays
    if isinstance(data, list):
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
            except Exception:
                new_data.append(arr2)
                continue

            # changed detection (best-effort)
            try:
                if not np.array_equal(t2, np.asarray(t).astype(int, copy=False)):
                    changed = True
            except Exception:
                changed = True

            arr2[:, time_col] = t2
            new_data.append(arr2)

        return RemapResult(
            changed=changed,
            depth_used=None,
            mapped_count=len(idx_map),
            message="Remapped list-like vertices." if changed else "List-like vertices unchanged.",
            data=new_data if changed else None,
        )

    # Array-like
    arr = np.asarray(data)
    if arr.size == 0:
        return RemapResult(False, None, len(idx_map), "No data to remap (empty array).", None)

    if arr.ndim < 2 or arr.shape[1] <= time_col:
        return RemapResult(False, None, len(idx_map), "Data shape does not contain a time column.", None)

    arr2 = np.array(arr, copy=True)
    t = arr2[:, time_col]

    try:
        t2 = _remap_array(t, idx_map)
    except Exception:
        return RemapResult(False, None, len(idx_map), "Failed to remap time column.", None)

    try:
        unchanged = np.array_equal(t2, np.asarray(t).astype(int, copy=False))
    except Exception:
        unchanged = False

    if unchanged:
        return RemapResult(False, None, len(idx_map), "Time column unchanged after remap.", None)

    arr2[:, time_col] = t2
    return RemapResult(True, None, len(idx_map), "Remapped array-like data.", arr2)


def remap_layer_data_by_paths(
    *,
    data: Any,
    old_paths: Iterable[str] | None,
    new_paths: Iterable[str] | None,
    time_col: int,
    policy: PathMatchPolicy = PathMatchPolicy.ORDERED_DEPTHS,
) -> RemapResult:
    """
    High-level remap: infer idx_map from old/new paths, then remap `data`.

    Adds ambiguity detection: emits warnings when remapping is risky.
    Does NOT change behavior (still remaps when possible).
    """
    old_paths = list(old_paths or [])
    new_paths = list(new_paths or [])
    if not old_paths:
        return RemapResult(False, None, 0, "No old paths present; cannot remap.", None)
    if not new_paths:
        return RemapResult(False, None, 0, "No new paths present; cannot remap.", None)

    depth = find_matching_depth(old_paths, new_paths, policy=policy)
    if depth is None:
        return RemapResult(False, None, 0, "No overlap between old and new paths; skipping remap.", None)

    old_keys = [canonicalize_path(p, depth) for p in old_paths]
    new_keys = [canonicalize_path(p, depth) for p in new_paths]

    overlap = set(old_keys) & set(new_keys)
    overlap_ratio = (len(overlap) / max(1, min(len(old_keys), len(new_keys)))) if overlap else 0.0

    # Build mapping: canonical key -> new index (first occurrence)
    key_to_new_idx = {k: i for i, k in enumerate(new_keys)}
    idx_map: dict[int, int] = {}
    for old_idx, k in enumerate(old_keys):
        new_idx = key_to_new_idx.get(k)
        if new_idx is not None:
            idx_map[old_idx] = new_idx

    if not idx_map:
        return RemapResult(False, None, 0, "No overlap between old and new paths; skipping remap.", None)

    # If ordering already matches, skip
    if old_keys == new_keys:
        return RemapResult(False, depth, len(idx_map), "Path keys already aligned; no remap needed.", None)

    # -----------------------------
    # Ambiguity / risk detection
    # -----------------------------
    warnings: list[str] = []

    dup_old = _find_duplicates(old_keys)
    dup_new = _find_duplicates(new_keys)
    if dup_old:
        examples = ", ".join(list(dup_old.keys())[:_SAMPLE_N])
        warnings.append(f"Duplicate canonical keys in old_paths at depth={depth} (examples: {examples}).")
    if dup_new:
        examples = ", ".join(list(dup_new.keys())[:_SAMPLE_N])
        warnings.append(f"Duplicate canonical keys in new_paths at depth={depth} (examples: {examples}).")

    mapped_ratio = len(idx_map) / max(1, len(old_keys))
    if overlap_ratio < _WARN_OVERLAP_RATIO:
        warnings.append(
            f"Low path overlap ratio at depth={depth}: {overlap_ratio:.2f} (overlap={len(overlap)}, "
            f"old={len(old_keys)}, new={len(new_keys)})."
        )
    if mapped_ratio < _WARN_MAPPED_RATIO:
        warnings.append(f"Low mapping coverage: {mapped_ratio:.2f} (mapped={len(idx_map)} of old={len(old_keys)}).")

    # Non-bijective mapping is a strong ambiguity signal (multiple old -> same new)
    if len(set(idx_map.values())) < len(idx_map):
        warnings.append("Non-bijective mapping detected (multiple old indices map to the same new index).")

    # Emit warnings (does not change behavior)
    for w in warnings:
        logger.warning("Remap may be ambiguous/risky: %s", w)

    # Apply remap to data
    res = remap_time_indices(data=data, time_col=time_col, idx_map=idx_map)

    return RemapResult(
        changed=res.changed,
        depth_used=depth,
        mapped_count=res.mapped_count,
        message=res.message,
        data=res.data,
        warnings=tuple(warnings),
    )
