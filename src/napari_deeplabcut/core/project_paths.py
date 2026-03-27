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

# src/napari_deeplabcut/core/project_paths.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from napari_deeplabcut.config.models import DLCProjectContext, PointsMetadata

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Canonicalization
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
        logger.debug("Failed to stringify path of type %s", type(p).__name__, exc_info=True)
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
    # FUTURE NOTE @C-Achard 2026-02-17: Do not hardcode these patterns
    # and clearly expose these if data file formats change or expand.
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
def _collect_anchor_candidates(
    *values: str | Path | None,
) -> list[Path]:
    anchors: list[Path] = []
    for value in values:
        anchor = normalize_anchor_candidate(value)
        if anchor is not None and anchor not in anchors:
            anchors.append(anchor)
    return anchors


def _is_labeled_data_dataset_folder(path: Path | None) -> bool:
    if path is None:
        return False
    lowered = [part.lower() for part in path.parts]
    return "labeled-data" in lowered and path.name.lower() != "labeled-data"


def _extract_dataset_name_from_paths(paths: Iterable[str | Path]) -> str | None:
    for value in paths:
        try:
            text = str(value).replace("\\", "/")
        except Exception:
            continue
        parts = [p for p in text.split("/") if p]
        lowered = [p.lower() for p in parts]
        try:
            idx = lowered.index("labeled-data")
        except ValueError:
            continue
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def normalize_anchor_candidate(value: str | Path | None) -> Path | None:
    """Return a normalized directory anchor from a file/folder candidate."""
    if value is None:
        return None

    try:
        p = Path(value).expanduser().resolve()
    except Exception:
        try:
            p = Path(value)
        except Exception:
            return None

    # If this is an existing file, anchor on its parent directory.
    if p.is_file():
        return p.parent

    # For non-existent paths, heuristically treat file-like paths (with a suffix)
    # as files so that their parent directory is used as the anchor. This avoids
    # searching for config files under a non-existent "<file>/config.yaml".
    if not p.exists() and p.suffix:
        return p.parent

    # Existing directories (or suffix-less paths) are used as-is.
    return p


def infer_dlc_project(
    *,
    anchor_candidates: list[str | Path] | tuple[str | Path, ...] = (),
    dataset_candidates: list[str | Path] | tuple[str | Path, ...] = (),
    explicit_root: str | Path | None = None,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    """
    Infer a best-effort DLC project context from generic path-like hints.

    Parameters
    ----------
    anchor_candidates:
        Ordered candidates that may indicate a project anchor, project root,
        file parent, source directory, etc.
    dataset_candidates:
        Ordered candidates that may already point at a labeled-data dataset folder.
    explicit_root:
        Strongest hint. If provided, used first.
    prefer_project_root:
        If True, root_anchor prefers the folder containing config.yaml.
        Otherwise it prefers the first valid anchor candidate.
    """
    anchors = _collect_anchor_candidates(explicit_root, *anchor_candidates)

    dataset_folder = None
    for cand in dataset_candidates:
        d = normalize_anchor_candidate(cand)
        if d is not None:
            dataset_folder = d
            break

    for anchor in anchors:
        cfg = find_nearest_config(anchor, max_levels=max_levels)
        if cfg is not None:
            project_root = cfg.parent
            root_anchor = project_root if prefer_project_root else anchor
            return DLCProjectContext(
                root_anchor=root_anchor,
                project_root=project_root,
                config_path=cfg,
                dataset_folder=dataset_folder,
            )

    return DLCProjectContext(
        root_anchor=anchors[0] if anchors else dataset_folder,
        project_root=None,
        config_path=None,
        dataset_folder=dataset_folder,
    )


def infer_labeled_data_folder_from_paths(
    paths: Iterable[str | Path],
    *,
    project_root: str | Path | None = None,
    fallback_root: str | Path | None = None,
) -> Path | None:
    """
    Infer a DLC labeled-data/<dataset> folder from path hints.
    """
    fallback = normalize_anchor_candidate(fallback_root)
    if _is_labeled_data_dataset_folder(fallback):
        return fallback

    dataset_name = _extract_dataset_name_from_paths(paths)
    if not dataset_name:
        return None

    proj = normalize_anchor_candidate(project_root)
    if proj is None:
        return None

    return proj / "labeled-data" / dataset_name


def find_nearest_config(
    start: str | Path | None,
    *,
    max_levels: int = 5,
) -> Path | None:
    """
    Walk upward from start to find the nearest config.yaml.
    """
    anchor = normalize_anchor_candidate(start)
    if anchor is None:
        return None

    cur = anchor
    for _ in range(max_levels + 1):
        cfg = cur / "config.yaml"
        if cfg.is_file() and cfg.name.lower() == "config.yaml":
            return cfg
        if cur.parent == cur:
            break
        cur = cur.parent

    return None


def infer_dlc_project_from_config(config_path: str | Path) -> DLCProjectContext:
    root = resolve_project_root_from_config(config_path)
    if root is None:
        raise ValueError(f"Not a valid DLC config.yaml: {config_path!r}")
    cfg = Path(config_path).expanduser().resolve(strict=False)
    return DLCProjectContext(
        root_anchor=root,
        project_root=root,
        config_path=cfg,
        dataset_folder=None,
    )


# -----------------------------------------------------------------------------
# Explicit config-based DLC path normalization for project-less labeled folders
# -----------------------------------------------------------------------------
def resolve_project_root_from_config(config_path: str | Path | None) -> Path | None:
    """
    Return the DLC project root (= parent directory) from an explicit config.yaml path.
    """
    if config_path is None:
        return None

    try:
        p = Path(config_path).expanduser().resolve(strict=False)
    except Exception:
        try:
            p = Path(config_path)
        except Exception:
            return None

    if p.name.lower() != "config.yaml":
        return None

    if not p.is_file():
        return None

    return p.parent


def coerce_paths_to_dlc_row_keys(
    paths: Iterable[str | Path],
    *,
    source_root: str | Path,
    dataset_name: str | None = None,
) -> tuple[list[str], tuple[int, ...]]:
    """
    Rewrite paths from a project-less labeled folder into canonical DLC row-key form:

        labeled-data/<dataset_name>/<image_name>

    Intended use
    ------------
    This is for the workflow where the user labeled a folder outside any DLC
    project, then chooses a target config.yaml at save time to associate the
    labels with a DLC project.

    Rules
    -----
    - If a path is already a DLC row key (`labeled-data/<dataset>/<image>`),
      normalize it to POSIX and keep it.
    - If a path is an absolute file directly inside `source_root`,
      rewrite it to `labeled-data/<dataset_name>/<basename>`.
    - If a path is a relative basename (e.g. `img001.png`),
      rewrite it similarly.
    - All other paths are preserved unchanged (POSIX-normalized) and reported as unresolved.

    This deliberately does NOT:
    - invent nested DLC row keys for subdirectories,
    - coerce multi-folder or ambiguous layouts,
    - validate against the selected project root.
    """
    root = normalize_anchor_candidate(source_root)
    if root is None:
        raise ValueError("source_root must resolve to a valid directory-like anchor")

    try:
        root = root.expanduser().resolve(strict=False)
    except Exception:
        pass

    ds_name = (dataset_name or root.name).strip()
    if not ds_name:
        raise ValueError("dataset_name must be non-empty")

    rewritten: list[str] = []
    unresolved: list[int] = []

    for i, value in enumerate(paths):
        try:
            text = str(value).replace("\\", "/")
        except Exception:
            text = ""

        if not text:
            rewritten.append(text)
            unresolved.append(i)
            continue

        parts = [p for p in text.split("/") if p]

        # Already canonical-ish DLC row key -> preserve from labeled-data onward
        lowered = [p.lower() for p in parts]
        if "labeled-data" in lowered:
            try:
                idx = lowered.index("labeled-data")
                if idx + 2 < len(parts):
                    rewritten.append("/".join(parts[idx:]))
                    continue
            except Exception:
                pass

        try:
            p = Path(value)
        except Exception:
            p = None

        # Relative basename only -> coerce safely
        if p is not None and not p.is_absolute():
            rel_parts = [part for part in p.parts if str(part) not in ("", ".", "..")]
            if len(rel_parts) == 1:
                rewritten.append(f"labeled-data/{ds_name}/{rel_parts[0]}")
            else:
                rewritten.append(text)
                unresolved.append(i)
            continue

        # Absolute file directly under source_root -> coerce safely
        try:
            abs_path = Path(value).expanduser().resolve(strict=False)
        except Exception:
            rewritten.append(text)
            unresolved.append(i)
            continue

        try:
            rel_to_root = abs_path.relative_to(root)
        except Exception:
            rewritten.append(abs_path.as_posix())
            unresolved.append(i)
            continue

        # Only direct children of source_root are coerced in this lightweight version.
        if len(rel_to_root.parts) == 1:
            rewritten.append(f"labeled-data/{ds_name}/{rel_to_root.name}")
        else:
            rewritten.append(abs_path.as_posix())
            unresolved.append(i)

    return rewritten, tuple(unresolved)


def target_dataset_folder_for_config(
    config_path: str | Path,
    *,
    dataset_name: str,
) -> Path | None:
    """
    Return the target DLC dataset folder under the chosen project:

        <project_root>/labeled-data/<dataset_name>
    """
    project_root = resolve_project_root_from_config(config_path)
    if project_root is None:
        return None
    return project_root / "labeled-data" / dataset_name


def dataset_folder_has_files(folder: str | Path | None) -> bool:
    """
    Return True if the given folder exists and contains any files.

    This is intentionally conservative: any existing file content means we refuse
    the project-association override to avoid colliding with an existing dataset.
    """
    if folder is None:
        return False

    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return False

    return any(child.is_file() for child in p.iterdir())


# -----------------------------------------------------------------------------
# Source-specific adapters
# -----------------------------------------------------------------------------
def infer_dlc_project_from_opened(
    opened: str | Path,
    *,
    explicit_root: str | Path | None = None,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    return infer_dlc_project(
        anchor_candidates=[opened],
        explicit_root=explicit_root,
        prefer_project_root=prefer_project_root,
        max_levels=max_levels,
    )


def infer_dlc_project_from_points_meta(
    pts_meta: PointsMetadata,
    *,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    """
    Infer DLC dataset folder (…/labeled-data/<dataset>) from PointsMetadata.

    Uses:
      - pts_meta.project (config parent) as project root
      - pts_meta.paths (canonicalized relpaths like labeled-data/test/img000.png)
      - pts_meta.root as a fallback hint

    Args:
        pts_meta: PointsMetadata object containing project-related metadata.
        prefer_project_root: If True, root_anchor prefers the folder containing config.yaml.
        max_levels: Maximum number of levels to search upward for config.yaml.


    Returns a DLCProjectContext object representing the inferred project context.
    """
    project = getattr(pts_meta, "project", None)
    root = getattr(pts_meta, "root", None)
    paths = getattr(pts_meta, "paths", None) or []

    dataset_folder = infer_labeled_data_folder_from_paths(
        paths,
        project_root=project,
        fallback_root=root,
    )

    return infer_dlc_project(
        anchor_candidates=[project, root, dataset_folder],
        dataset_candidates=[dataset_folder],
        explicit_root=None,
        prefer_project_root=prefer_project_root,
        max_levels=max_levels,
    )


def infer_dlc_project_from_image_layer(
    layer,
    *,
    prefer_project_root: bool = True,
    max_levels: int = 5,
) -> DLCProjectContext:
    """Best-effort inference of the DLC project context from an Image/video layer using its source metadata.

    Uses:
      - layer.metadata.project as project root
      - layer.metadata.root as a fallback hint
      - layer.source.path as a fallback hint

    Returns a DLCProjectContext object representing the inferred project context.
    """
    md = getattr(layer, "metadata", {}) or {}

    candidates: list[str | Path] = []

    project = md.get("project")
    if isinstance(project, str) and project:
        candidates.append(project)

    root = md.get("root")
    if isinstance(root, str) and root:
        candidates.append(root)

    try:
        src = getattr(getattr(layer, "source", None), "path", None)
    except Exception:
        src = None

    if src:
        candidates.append(src)

    return infer_dlc_project(
        anchor_candidates=candidates,
        dataset_candidates=[],
        explicit_root=None,
        prefer_project_root=prefer_project_root,
        max_levels=max_levels,
    )
