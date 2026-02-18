# src/napari_deeplabcut/core/discovery.py
"""
Deterministic discovery of DeepLabCut annotation artifacts in a folder.

This module is pure filesystem logic (no napari imports).
It enumerates all relevant files and classifies them into AnnotationKind.

# FUTURE NOTE hardcoded DLC structure:
# DLC naming conventions (CollectedData*, machinelabels*) are hardcoded here.
# If DLC expands formats/patterns, update ONLY this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from natsort import natsorted

from napari_deeplabcut.config.models import AnnotationKind


@dataclass(frozen=True)
class AnnotationArtifact:
    """A discovered annotation artifact (H5/CSV) with inferred semantics."""

    kind: AnnotationKind | None
    h5_path: Path | None
    csv_path: Path | None
    stem: str

    @property
    def primary_path(self) -> Path | None:
        """Preferred path for opening: H5 preferred over CSV."""
        return self.h5_path or self.csv_path


def _infer_kind_from_stem(stem: str) -> AnnotationKind | None:
    """Infer kind from filename stem."""
    low = stem.lower()

    # FUTURE NOTE hardcoded DLC structure:
    if low.startswith("collecteddata"):
        return AnnotationKind.GT
    if low.startswith("machinelabels"):
        return AnnotationKind.MACHINE
    return None


def _is_relevant_artifact(p: Path) -> bool:
    """Return True if path looks like a DLC annotation artifact."""
    if not p.is_file():
        return False
    low = p.name.lower()

    # FUTURE NOTE hardcoded DLC structure:
    if low.endswith(".h5") or low.endswith(".csv"):
        return low.startswith("collecteddata") or low.startswith("machinelabels")
    return False


def discover_annotations(folder: str | Path) -> list[AnnotationArtifact]:
    """Discover DLC annotation artifacts in a folder (deterministic order)."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    files = [p for p in root.iterdir() if _is_relevant_artifact(p)]
    files = natsorted(files, key=lambda p: p.name)

    by_stem: dict[str, dict[str, Path]] = {}
    for p in files:
        entry = by_stem.setdefault(p.stem, {})
        if p.suffix.lower() == ".h5":
            entry["h5"] = p
        elif p.suffix.lower() == ".csv":
            entry["csv"] = p

    artifacts: list[AnnotationArtifact] = []
    for stem in natsorted(by_stem.keys()):
        entry = by_stem[stem]
        kind = _infer_kind_from_stem(stem)
        artifacts.append(
            AnnotationArtifact(
                kind=kind,
                h5_path=entry.get("h5"),
                csv_path=entry.get("csv"),
                stem=stem,
            )
        )

    # Stable ordering by primary filename
    return natsorted(artifacts, key=lambda a: (a.primary_path.name if a.primary_path else a.stem))


def discover_annotation_paths(folder: str | Path) -> list[Path]:
    """Return primary paths to open (H5 preferred else CSV)."""
    return [a.primary_path for a in discover_annotations(folder) if a.primary_path is not None]


def iter_annotation_candidates(paths: Iterable[str | Path]) -> list[Path]:
    """Expand folders to annotation candidates (deterministic)."""
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(discover_annotation_paths(pp))
        elif _is_relevant_artifact(pp):
            out.append(pp)
    return natsorted(out, key=lambda x: x.name)


def infer_annotation_kind_for_file(file_path: str | Path) -> AnnotationKind | None:
    """Infer kind for a specific file path by scanning its parent folder."""
    fp = Path(file_path)
    parent = fp.parent
    for art in discover_annotations(parent):
        if art.h5_path == fp or art.csv_path == fp:
            return art.kind
    return None
