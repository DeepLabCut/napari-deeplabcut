from __future__ import annotations

from pathlib import Path

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.io import (
    discover_annotation_artifacts,
    discover_annotation_paths,
    iter_annotation_candidates,
)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_discover_annotation_artifacts_groups_h5_and_csv(tmp_path: Path):
    # Two GT + one machine, with mixed presence of CSV companions.
    _touch(tmp_path / "CollectedData_John.h5")
    _touch(tmp_path / "CollectedData_John.csv")
    _touch(tmp_path / "CollectedData_Jane.h5")  # no csv
    _touch(tmp_path / "machinelabels-iter0.h5")
    _touch(tmp_path / "machinelabels-iter0.csv")

    arts = discover_annotation_artifacts(tmp_path)
    assert len(arts) == 3

    # Deterministic ordering by filename
    assert [a.stem for a in arts] == ["CollectedData_Jane", "CollectedData_John", "machinelabels-iter0"]

    # Kind inference
    assert arts[0].kind == AnnotationKind.GT
    assert arts[1].kind == AnnotationKind.GT
    assert arts[2].kind == AnnotationKind.MACHINE

    # Pairing behavior
    by_stem = {a.stem: a for a in arts}
    assert by_stem["CollectedData_John"].h5_path.name.endswith(".h5")
    assert by_stem["CollectedData_John"].csv_path.name.endswith(".csv")
    assert by_stem["CollectedData_Jane"].csv_path is None
    assert by_stem["machinelabels-iter0"].csv_path is not None


def test_discover_annotation_paths_prefers_h5(tmp_path: Path):
    _touch(tmp_path / "CollectedData_John.csv")
    _touch(tmp_path / "CollectedData_John.h5")
    paths = discover_annotation_paths(tmp_path)
    assert len(paths) == 1
    assert paths[0].suffix.lower() == ".h5"


def test_discover_annotation_paths_supports_csv_only(tmp_path: Path):
    _touch(tmp_path / "CollectedData_John.csv")
    paths = discover_annotation_paths(tmp_path)
    assert len(paths) == 1
    assert paths[0].suffix.lower() == ".csv"


def test_iter_annotation_candidates_expands_folders_and_files(tmp_path: Path):
    folder = tmp_path / "shared"
    folder.mkdir()

    _touch(folder / "CollectedData_John.h5")
    _touch(folder / "machinelabels-iter0.h5")

    # Provide a mixture of folder and direct file input
    extra_file = _touch(tmp_path / "CollectedData_Jane.h5")

    out = iter_annotation_candidates([folder, extra_file])

    # Deterministic order by filename
    assert [p.name for p in out] == ["CollectedData_Jane.h5", "CollectedData_John.h5", "machinelabels-iter0.h5"]
