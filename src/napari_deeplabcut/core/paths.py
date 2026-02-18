# src/napari_deeplabcut/core/paths.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Canonicalization (REUSED, behavior-preserving)
# -----------------------------------------------------------------------------


def canonicalize_path(p: str | Path, n: int = 3) -> str:
    """
    Return canonical POSIX path built from the last n path components.

    This implementation is intentionally identical to the legacy behavior
    in misc.canonicalize_path to preserve remapping semantics.

    Parameters
    ----------
    p : str | Path
        Input path.
    n : int
        Number of trailing components to keep.

    Returns
    -------
    str
        Canonicalized POSIX-style path, or empty string on failure.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    try:
        s = str(p)
    except Exception:
        logger.debug("Failed to stringify path %r", p, exc_info=True)
        return ""

    s = s.replace("\\", "/")
    s = s.rstrip("/")
    parts = [part for part in s.split("/") if part and part not in (".", "..")]

    if not parts:
        return ""

    return "/".join(parts[-n:])


# -----------------------------------------------------------------------------
# Path matching policy
# -----------------------------------------------------------------------------


class PathMatchPolicy(Enum):
    """
    Policy controlling how image paths are matched across datasets.

    ORDERED_DEPTHS means:
    - Try matching with depth=3
    - If no overlap, try depth=2
    - If still no overlap, try depth=1
    """

    ORDERED_DEPTHS = "ordered_depths"

    @property
    def depths(self) -> tuple[int, ...]:
        if self is PathMatchPolicy.ORDERED_DEPTHS:
            return (3, 2, 1)
        raise NotImplementedError(f"Unhandled PathMatchPolicy: {self}")


def find_matching_depth(
    old_paths: Iterable[str | Path],
    new_paths: Iterable[str | Path],
    policy: PathMatchPolicy = PathMatchPolicy.ORDERED_DEPTHS,
) -> int | None:
    """
    Find the first canonicalization depth producing overlapping path keys.

    Returns
    -------
    int | None
        Depth used for matching, or None if no overlap found.
    """
    old_paths = list(old_paths)
    new_paths = list(new_paths)

    if not old_paths or not new_paths:
        return None

    for depth in policy.depths:
        old_keys = {canonicalize_path(p, depth) for p in old_paths}
        new_keys = {canonicalize_path(p, depth) for p in new_paths}
        if old_keys & new_keys:
            return depth

    return None


# -----------------------------------------------------------------------------
# DLC path heuristics
# -----------------------------------------------------------------------------


def is_config_yaml(path: str | Path) -> bool:
    """Return True if path points to a DLC config.yaml file."""
    try:
        p = Path(path)
    except TypeError:
        return False
    return p.is_file() and p.name.lower() == "config.yaml"


def has_dlc_datafiles(folder: str | Path) -> bool:
    """
    True if folder contains DLC label artifacts such as:
    - CollectedData*.h5 / .csv
    - machinelabels*.h5 / .csv
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return False

    patterns = (
        "CollectedData*.h5",
        "CollectedData*.csv",
        "machinelabels*.h5",
        "machinelabels*.csv",
    )
    return any(any(p.glob(pat)) for pat in patterns)


def looks_like_dlc_labeled_folder(folder: str | Path) -> bool:
    """
    Heuristic for DLC labeled-data folders.

    True if:
    - DLC artifacts are present, OR
    - Folder is inside a 'labeled-data' directory.
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return False

    if has_dlc_datafiles(p):
        return True

    return any(part.lower() == "labeled-data" for part in p.parts)


def should_force_dlc_reader(paths: str | Path | Iterable[str | Path]) -> bool:
    """
    Decide whether napari-deeplabcut reader should be preferred.

    Rules (unchanged from legacy behavior):
    - Any config.yaml -> DLC reader
    - Any folder that looks like DLC labeled-data -> DLC reader
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    paths = list(paths)
    if not paths:
        return False

    if any(is_config_yaml(p) for p in paths):
        return True

    if any(looks_like_dlc_labeled_folder(p) for p in paths):
        return True

    return False
