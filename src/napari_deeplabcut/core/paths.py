"""
Regarding root anchor :
The "root anchor" is a configurable directory used to resolve project-relative
paths in IO provenance.

Motivation
----------
We may not always load a full DLC project (config.yaml may be missing).
In particular there may only be a labeled-data folder containing images + h5/csv files.
Therefore the root anchor must be inferable from what the user opened:

- If the user opens a file: the default anchor is the file's parent directory.
- If the user opens a folder: the anchor is that folder.
- If a config.yaml exists nearby: the anchor *may* be elevated to the project
  root, but must remain configurable and must not be required.
"""

# src/napari_deeplabcut/core/paths.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Canonicalization (REUSED for now, behavior-preserving)
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


# -----------------------------------------------------------------------------
# Root-anchor inference utilities
# -----------------------------------------------------------------------------


def infer_root_anchor(
    opened: str | Path,
    *,
    explicit_root: str | Path | None = None,
) -> str | None:
    """Infer a best-effort root anchor given a user-opened path.

    Parameters
    ----------
    opened:
        Path that the user opened (file or directory).
    explicit_root:
        If provided, this wins and is returned as-is.

    Returns
    -------
    str | None
        A directory path to use as anchor, or None if inference fails.

    Notes
    -----
    This is intentionally conservative and does not search globally.
    """
    if explicit_root:
        return str(Path(explicit_root))

    try:
        p = Path(opened)
    except Exception:
        return None

    if p.is_dir():
        return str(p)

    if p.is_file():
        return str(p.parent)

    # If path does not exist (e.g., virtual/remote), give up.
    return None


def find_nearest_project_root(
    start: str | Path,
    *,
    max_levels: int = 5,
) -> str | None:
    """Walk parents upwards to find a folder containing config.yaml.

    This is used as an *optional* enhancement. The result should be treated as
    a candidate anchor, not as a required project root.

    Parameters
    ----------
    start:
        Starting file or directory.
    max_levels:
        Maximum number of parent levels to inspect.

    Returns
    -------
    str | None
        Directory containing config.yaml if found, else None.
    """
    try:
        p = Path(start)
    except Exception:
        return None

    if p.is_file():
        p = p.parent

    cur = p
    for _ in range(max_levels + 1):
        cfg = cur / "config.yaml"
        if is_config_yaml(cfg):
            return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent

    return None


def choose_anchor_candidate(
    *,
    opened: str | Path,
    explicit_root: str | Path | None = None,
    prefer_project_root: bool = False,
) -> str | None:
    """Choose a root anchor candidate.

    Strategy
    --------
    1) If explicit_root provided -> use it.
    2) Infer anchor from opened path (file parent or directory).
    3) If prefer_project_root is True and a nearby config.yaml exists ->
       return the nearest project root.

    This does *not* validate existence of expected DLC files; callers may.
    """
    if explicit_root:
        return str(Path(explicit_root))

    anchor = infer_root_anchor(opened)
    if not anchor:
        return None

    if prefer_project_root:
        pr = find_nearest_project_root(anchor)
        if pr:
            return pr

    return anchor


def anchor_contains_dlc_artifacts(anchor: str | Path) -> bool:
    """Return True if the anchor folder appears to contain DLC artifacts."""
    try:
        return has_dlc_datafiles(anchor)
    except Exception:
        logger.debug("Failed to check DLC artifacts for %r", anchor, exc_info=True)
        return False
