# src/napari_deeplabcut/core/remap.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from napari_deeplabcut.core.paths import PathMatchPolicy, canonicalize_path, find_matching_depth

logger = logging.getLogger(__name__)


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
    """

    changed: bool
    depth_used: int | None
    mapped_count: int
    message: str
    data: Any | None


def _remap_array(values: np.ndarray, idx_map: Mapping[int, int]) -> np.ndarray:
    """
    Remap integer indices using idx_map, leaving unknown indices unchanged.
    """
    values = np.asarray(values)
    if values.size == 0:
        return values

    # Convert to int indices safely (napari time column can be float)
    try:
        values_int = values.astype(int, copy=False)
    except Exception:
        values_int = values.astype(int)

    mapped = np.fromiter(
        (idx_map.get(int(v), int(v)) for v in values_int), dtype=values_int.dtype, count=len(values_int)
    )
    return mapped


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
        idx_map maps old numeric indices (based on old_paths order) to new numeric indices.
        depth_used is the canonicalization depth used, or None if no overlap.
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

    Returns
    -------
    RemapResult
        Contains remapped data if remap applied, else data=None and message explains why.
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

            if not np.array_equal(t2, t.astype(int, copy=False) if np.issubdtype(t.dtype, np.number) else t2):
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

    if np.array_equal(t2, t.astype(int, copy=False) if np.issubdtype(t.dtype, np.number) else t2):
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

    Returns
    -------
    RemapResult
    """
    old_paths = list(old_paths or [])
    new_paths = list(new_paths or [])
    if not old_paths:
        return RemapResult(False, None, 0, "No old paths present; cannot remap.", None)
    if not new_paths:
        return RemapResult(False, None, 0, "No new paths present; cannot remap.", None)

    idx_map, depth_used = build_frame_index_map(old_paths=old_paths, new_paths=new_paths, policy=policy)
    if depth_used is None or not idx_map:
        return RemapResult(False, None, 0, "No overlap between old and new paths; skipping remap.", None)

    # If ordering already matches, skip
    old_keys = [canonicalize_path(p, depth_used) for p in old_paths]
    new_keys = [canonicalize_path(p, depth_used) for p in new_paths]
    if old_keys == new_keys:
        return RemapResult(False, depth_used, len(idx_map), "Path keys already aligned; no remap needed.", None)

    res = remap_time_indices(data=data, time_col=time_col, idx_map=idx_map)
    # Inject depth into result
    return RemapResult(
        changed=res.changed,
        depth_used=depth_used,
        mapped_count=res.mapped_count,
        message=res.message,
        data=res.data,
    )
