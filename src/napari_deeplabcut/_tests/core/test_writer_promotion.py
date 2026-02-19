from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut import _writer, misc
from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core import io
from napari_deeplabcut.core.errors import MissingProvenanceError


def _make_minimal_points_metadata(
    root: Path, header, *, name: str, kind: AnnotationKind, save_target: dict | None = None
):
    # Minimal metadata payload compatible with form_df usage
    md = {
        "name": name,
        "properties": {
            "label": ["bodypart1", "bodypart2"],
            "id": ["", ""],
            "likelihood": [1.0, 1.0],
        },
        "metadata": {
            "header": header,
            "paths": ["img000.png", "img000.png"],  # indices 0 and 1 refer to same img for simplicity
            "root": str(root),
            "io": {
                "schema_version": 1,
                "project_root": str(root),
                "source_relpath_posix": f"{name}.h5",
                "kind": kind,
                "dataset_key": "keypoints",
            },
        },
    }
    if save_target is not None:
        md["metadata"]["save_target"] = save_target
    return md


def _read_keypoints_h5(p: Path) -> pd.DataFrame:
    return pd.read_hdf(p, key="keypoints")


def test_writer_aborts_if_machine_source_without_save_target(tmp_path: Path):
    # Create minimal header for single animal: scorer/bodyparts/coords
    cols = pd.MultiIndex.from_product(
        [["John"], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    header = misc.DLCHeader(cols)

    metadata = _make_minimal_points_metadata(
        tmp_path, header, name="machinelabels-iter0", kind=AnnotationKind.MACHINE, save_target=None
    )

    points = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 30.0, 40.0],
        ],
        dtype=float,
    )

    with pytest.raises(MissingProvenanceError):
        _writer.write_hdf_napari_dlc("ignored.h5", points, metadata)


def test_writer_promotion_writes_collecteddata_and_rewrites_scorer(tmp_path: Path, monkeypatch):
    # Build minimal header for single animal
    cols = pd.MultiIndex.from_product(
        [["machine"], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    header = misc.DLCHeader(cols)

    # Ensure overwrite confirm always returns True
    monkeypatch.setattr(io, "maybe_confirm_overwrite", lambda *args, **kwargs: True)

    # Pretend we loaded from a machine file but will promote to GT file CollectedData_Alice.h5
    save_target = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "source_relpath_posix": "CollectedData_Alice.h5",
        "kind": AnnotationKind.GT,
        "dataset_key": "keypoints",
        "scorer": "Alice",
    }

    metadata = _make_minimal_points_metadata(
        tmp_path,
        header,
        name="machinelabels-iter0",
        kind=AnnotationKind.MACHINE,
        save_target=save_target,
    )

    # Create a dummy machine file and snapshot it (writer must not touch it)
    machine_path = tmp_path / "machinelabels-iter0.h5"
    df_machine = pd.DataFrame(np.nan, columns=cols, index=["img000.png"])
    df_machine.to_hdf(machine_path, key="keypoints", mode="w")
    machine_before = _read_keypoints_h5(machine_path)

    points = np.array(
        [
            [0.0, 33.0, 44.0],  # bodypart1
            [0.0, 55.0, 66.0],  # bodypart2
        ],
        dtype=float,
    )

    fnames = _writer.write_hdf_napari_dlc("ignored.h5", points, metadata)
    assert Path(fnames[0]).name == "CollectedData_Alice.h5"
    assert Path(fnames[1]).name == Path(fnames[0]).with_suffix(".csv").name

    gt_path = tmp_path / "CollectedData_Alice.h5"
    assert gt_path.exists()

    df_gt = _read_keypoints_h5(gt_path)

    # Scorer level should be rewritten to Alice
    assert "scorer" in df_gt.columns.names
    assert set(df_gt.columns.get_level_values("scorer")) == {"Alice"}

    # Machine file should be unchanged
    machine_after = _read_keypoints_h5(machine_path)
    pd.testing.assert_frame_equal(machine_before, machine_after)
