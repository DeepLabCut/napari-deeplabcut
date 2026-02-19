from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.errors import AmbiguousSaveError, MissingProvenanceError
from napari_deeplabcut.core.io import _resolve_output_path_from_metadata, write_hdf


def test_resolve_output_path_returns_none_for_machine_without_save_target():
    md = {
        "metadata": {
            "io": {
                "schema_version": 1,
                "project_root": str(Path.cwd()),
                "source_relpath_posix": "machinelabels-iter0.h5",
                "kind": AnnotationKind.MACHINE,
                "dataset_key": "keypoints",
            }
        }
    }
    out_path, scorer, kind = _resolve_output_path_from_metadata(md)
    assert out_path is None
    assert scorer is None
    assert kind == AnnotationKind.MACHINE


def test_write_hdf_refuses_machine_without_promotion(tmp_path: Path):
    # minimal points + metadata for a machine source
    data = np.zeros((1, 3), dtype=float)
    attrs = {
        "metadata": {
            "root": str(tmp_path),
            "io": {
                "schema_version": 1,
                "project_root": str(tmp_path),
                "source_relpath_posix": "machinelabels-iter0.h5",
                "kind": AnnotationKind.MACHINE,
                "dataset_key": "keypoints",
            },
            # header is required by writer
            "header": {
                "columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")],
            },
        },
        "properties": {"label": ["bp1"], "id": [""], "likelihood": [1.0]},
    }

    with pytest.raises(MissingProvenanceError):
        write_hdf("__dlc__.h5", data, attrs)


def test_write_hdf_raises_ambiguous_when_multiple_gt_candidates_and_no_provenance(tmp_path: Path):
    # Create two GT candidates in root folder
    (tmp_path / "CollectedData_John.h5").write_bytes(b"dummy")
    (tmp_path / "CollectedData_Jane.h5").write_bytes(b"dummy")

    data = np.zeros((1, 3), dtype=float)
    attrs = {
        "metadata": {
            "root": str(tmp_path),
            "header": {
                "columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")],
            },
            # No io/source_h5 => triggers GT fallback scan => ambiguous
        },
        "properties": {"label": ["bp1"], "id": [""], "likelihood": [1.0]},
    }

    with pytest.raises(AmbiguousSaveError):
        write_hdf("__dlc__.h5", data, attrs)
