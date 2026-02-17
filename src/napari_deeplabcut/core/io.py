"""Deterministic discovery of DeepLabCut annotation artifacts in a folder.

This module is pure logic (no napari imports). It does not change behavior until
reader/writer code starts using it.

Why this exists
---------------
DeepLabCut folders often contain multiple H5/CSV artifacts side-by-side
(e.g. multiple CollectedData_*.h5, and machinelabels*.h5). Any "first match wins"
logic is unsafe and non-deterministic across OS/filesystems.

This discovery layer:
- Enumerates all relevant files.
- Classifies them (GT vs machine).
- Returns results in a deterministic order.

Conventions supported
---------------------
- Ground-truth: CollectedData*.h5 / CollectedData*.csv
- Machine:      machinelabels*.h5 / machinelabels*.csv

CSV-only folders are supported (e.g. shared data without H5). We treat those as
valid sources, with an optional inferred H5 companion if present.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from natsort import natsorted

from napari_deeplabcut.config.models import AnnotationKind

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationArtifact:
    """A discovered annotation artifact (H5/CSV) with inferred semantics.

    Attributes
    ----------
    kind:
        Annotation kind (gt or machine), if identifiable by filename.
    h5_path:
        Path to the H5 file, if present.
    csv_path:
        Path to the CSV file, if present.
    stem:
        Filename stem used for pairing (e.g. CollectedData_John).
    """

    kind: AnnotationKind | None
    h5_path: Path | None
    csv_path: Path | None
    stem: str

    @property
    def primary_path(self) -> Path | None:
        """Return a preferred path for opening (H5 preferred over CSV)."""
        return self.h5_path or self.csv_path


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------


def _infer_kind_from_name(name: str) -> AnnotationKind | None:
    """Infer artifact kind from filename."""
    low = name.lower()
    if low.startswith("collecteddata"):
        return AnnotationKind.GT
    if low.startswith("machinelabels"):
        return AnnotationKind.MACHINE
    return None


def _is_relevant_artifact(p: Path) -> bool:
    """Return True if a path looks like a DLC annotation artifact."""
    if not p.is_file():
        return False
    low = p.name.lower()
    if low.endswith(".h5") or low.endswith(".csv"):
        return low.startswith("collecteddata") or low.startswith("machinelabels")
    return False


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def discover_annotation_artifacts(folder: str | Path) -> list[AnnotationArtifact]:
    """Discover DLC annotation artifacts in a folder.

    Parameters
    ----------
    folder:
        Directory to scan.

    Returns
    -------
    list[AnnotationArtifact]
        Deterministically ordered list of artifacts. Ordering is stable across OS:
        natsorted by primary filename.
    """
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    # Collect relevant H5 and CSV files
    files = [p for p in root.iterdir() if _is_relevant_artifact(p)]

    # Deterministic ordering by filename
    files = natsorted(files, key=lambda p: p.name)

    # Group by stem
    by_stem: dict[str, dict[str, Path]] = {}
    for p in files:
        st = p.stem
        entry = by_stem.setdefault(st, {})
        if p.suffix.lower() == ".h5":
            entry["h5"] = p
        elif p.suffix.lower() == ".csv":
            entry["csv"] = p

    artifacts: list[AnnotationArtifact] = []
    for stem in natsorted(by_stem.keys()):
        entry = by_stem[stem]
        h5 = entry.get("h5")
        csv = entry.get("csv")
        kind = _infer_kind_from_name(stem)
        artifacts.append(AnnotationArtifact(kind=kind, h5_path=h5, csv_path=csv, stem=stem))

    # Stable order again by primary filename (H5 preferred, else CSV)
    artifacts = natsorted(
        artifacts,
        key=lambda a: (a.primary_path.name if a.primary_path else a.stem),
    )
    return artifacts


def discover_annotation_paths(folder: str | Path) -> list[Path]:
    """Convenience wrapper: return primary paths to open (H5 preferred else CSV)."""
    return [a.primary_path for a in discover_annotation_artifacts(folder) if a.primary_path is not None]


def iter_annotation_candidates(paths: Iterable[str | Path]) -> list[Path]:
    """Given user-provided paths (files or folders), return annotation file candidates.

    This is useful for future reader logic: a single entry point that expands folders.
    The returned list is deterministic.
    """
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(discover_annotation_paths(pp))
        elif _is_relevant_artifact(pp):
            out.append(pp)

    # Stable ordering
    return natsorted(out, key=lambda x: x.name)
