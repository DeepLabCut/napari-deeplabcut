from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from skimage.io import imread

from napari_deeplabcut import _writer, misc
from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core import io as napari_dlc_io
from napari_deeplabcut.core.dataframes import guarantee_multiindex_rows
from napari_deeplabcut.core.errors import MissingProvenanceError

rng = np.random.default_rng(42)


#  Basic tests
def test_write_config(tmp_path):
    cfg = {"a": 1, "b": 2}
    path = tmp_path / "config.yaml"

    napari_dlc_io.write_config(str(path), cfg)

    assert path.exists()
    text = path.read_text()
    loaded = yaml.safe_load(text)
    assert loaded == cfg


def test_write_image(tmp_path):
    img = (rng.random((10, 10)) * 255).astype(np.uint8)
    output = tmp_path / "test.png"

    _writer._write_image(img, str(output))

    assert output.exists()
    loaded = imread(str(output))
    np.testing.assert_array_equal(loaded, img)


#  Form_df — multi-animal + single-animal
def _fake_metadata_for_df(df, paths):
    """Helper for metadata for form_df.

    IMPORTANT: The writer assigns properties row-wise, so we must provide
    per-row arrays (length == n_rows). We cycle through (individual, bodypart)
    combinations so that all header columns are represented across the dataset.
    """
    from itertools import cycle, islice, product

    header = misc.DLCHeader(df.columns)
    n_rows = len(df)

    # Build a cyclic sequence of (id, label) pairs from header
    pairs = list(product(header.individuals, header.bodyparts))
    if not pairs:
        # Degenerate case; still produce per-row arrays
        ids = ["" for _ in range(n_rows)]
        labels = ["" for _ in range(n_rows)]
    else:
        cyc = cycle(pairs)
        sel = list(islice(cyc, n_rows))
        ids = [p[0] for p in sel]
        labels = [p[1] for p in sel]

    props = {
        "label": labels,  # length == n_rows
        "id": ids,  # length == n_rows
        "likelihood": [1.0] * n_rows,  # length == n_rows
    }

    meta = {
        "header": header,
        "paths": paths,
    }

    return {
        "properties": props,
        "metadata": meta,
    }


def _add_source_io(metadata: dict, *, root: Path, kind: AnnotationKind, source_name: str) -> None:
    """Attach minimal PointsMetadata.io dict to metadata['metadata']."""
    md = metadata.setdefault("metadata", {})
    md["io"] = {
        "schema_version": 1,
        "project_root": str(root),
        "source_relpath_posix": source_name.replace("\\", "/"),
        "kind": kind,  # AnnotationKind.GT or AnnotationKind.MACHINE
        "dataset_key": "keypoints",
    }
    # legacy migration compatibility (optional but good)
    md["source_h5"] = str((root / source_name).resolve())
    md["source_h5_name"] = source_name
    md["source_h5_stem"] = Path(source_name).stem


def _add_save_target(metadata: dict, *, root: Path, scorer: str) -> None:
    """Attach promotion save_target (GT) to metadata['metadata']."""
    md = metadata.setdefault("metadata", {})
    md["save_target"] = {
        "schema_version": 1,
        "project_root": str(root),
        "source_relpath_posix": f"CollectedData_{scorer}.h5",
        "kind": "gt",
        "dataset_key": "keypoints",
        "scorer": scorer,
    }


def test_form_df_multi_animal(fake_keypoints):
    n = len(fake_keypoints)
    metadata = _fake_metadata_for_df(fake_keypoints, [f"img{i}.png" for i in range(n)])

    # inds + (x,y)
    data = np.column_stack([np.arange(n), rng.random(n), rng.random(n)])

    df = napari_dlc_io.form_df(data, layer_metadata=metadata["metadata"], layer_properties=metadata["properties"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n

    # MultiIndex expected for multi-animal
    lvls = df.columns.names
    assert lvls == ["scorer", "individuals", "bodyparts", "coords"]

    # Bodyparts appear correctly
    assert set(df.columns.get_level_values("bodyparts")) == set(fake_keypoints.columns.get_level_values("bodyparts"))


def test_form_df_single_animal(fake_keypoints):
    """Drop the individuals level and check that form_df handles it."""
    df_single = fake_keypoints.xs("animal_0", axis=1, level="individuals")
    scorer_values = df_single.columns.get_level_values("scorer").unique()
    bodyparts_values = df_single.columns.get_level_values("bodyparts").unique()
    coords_values = df_single.columns.get_level_values("coords").unique()
    df_single.columns = pd.MultiIndex.from_product(
        [
            [scorer_values[0]],  # scorer
            [""],  # empty individuals level
            bodyparts_values,  # bodyparts
            coords_values,  # coords
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    n = len(df_single)
    metadata = _fake_metadata_for_df(df_single, [f"img{i}.png" for i in range(n)])

    # inds + (x,y)
    points = np.column_stack([np.arange(n), rng.random(n), rng.random(n)])
    out = napari_dlc_io.form_df(points, layer_metadata=metadata["metadata"], layer_properties=metadata["properties"])

    assert isinstance(out, pd.DataFrame)
    assert len(out) == n

    if "individuals" in out.columns.names:
        assert all(val == "" for val in out.columns.get_level_values("individuals"))
    else:
        # level actually dropped — also OK
        assert "individuals" not in out.columns.names


#  Write_hdf — machine-prediction merge branch


def test_write_hdf_basic(tmp_path, fake_keypoints):
    """write_hdf should produce deterministic HDF and CSV."""
    root = tmp_path / "proj"
    root.mkdir()

    fake_keypoints.to_hdf(root / "data.h5", key="data")
    header = misc.DLCHeader(fake_keypoints.columns)

    # Build per-row properties (length == n_rows)
    n_rows = len(fake_keypoints)
    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "CollectedData_me",
        "properties": {
            "label": per_row_labels,
            "id": per_row_ids,
            "likelihood": [1.0] * n_rows,
        },
        "metadata": {
            "header": header,
            "paths": [f"img{i}.png" for i in range(n_rows)],
            "root": str(root),
        },
    }

    points = np.column_stack(
        [
            np.arange(n_rows),
            rng.random(n_rows),
            rng.random(n_rows),
        ]
    )

    fnames = _writer.write_hdf_napari_dlc("whatever.h5", points, metadata)

    h5_path = Path(fnames[0])
    csv_path = Path(fnames[1])
    assert h5_path.name == csv_path.with_suffix(".h5").name == "CollectedData_me.h5"

    assert h5_path.exists()
    assert csv_path.exists()

    df = pd.read_hdf(h5_path)
    assert isinstance(df, pd.DataFrame)
    # Ensure one row per index
    assert len(df) == n_rows


def test_write_hdf_promotion_merges_into_existing_gt(tmp_path, fake_keypoints, monkeypatch):
    """
    Promotion contract:
      - source is machine/prediction (io.kind is AnnotationKind.MACHINE)
      - save_target points to CollectedData_<scorer>.h5
      - writer must MERGE safely into GT (not overwrite blindly),
        and must NOT write back to prediction file.
    """
    root = tmp_path / "proj"
    root.mkdir()

    # Always allow overwrite confirmation in unit test
    monkeypatch.setattr(napari_dlc_io, "maybe_confirm_overwrite", lambda *args, **kwargs: True)

    header = misc.DLCHeader(fake_keypoints.columns)

    n_rows = len(fake_keypoints)
    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "machinelabels-iter0",
        "properties": {
            "label": per_row_labels,
            "id": per_row_ids,
            "likelihood": [1.0] * n_rows,
        },
        "metadata": {
            "header": header,
            "paths": [f"img{i}.png" for i in range(n_rows)],
            "root": str(root),
        },
    }

    # Source provenance: machine/prediction file
    _add_source_io(metadata, root=root, kind=AnnotationKind.MACHINE, source_name="machinelabels-iter0.h5")

    # Promotion target: existing GT
    _add_save_target(metadata, root=root, scorer="me")

    # Create existing GT file with DLC-like path-based index (not RangeIndex)
    gt_path = root / "CollectedData_me.h5"
    gt = fake_keypoints.copy()

    # Use the same "paths" convention as the writer uses when forming df_new
    gt.index = [f"img{i}.png" for i in range(len(gt))]

    # Convert to MultiIndex of path components (matches refactored indexing model)
    guarantee_multiindex_rows(gt)

    gt.to_hdf(gt_path, key="keypoints", mode="w")

    # Create a machine file too; it must remain untouched
    machine_path = root / "machinelabels-iter0.h5"
    df_machine = pd.DataFrame(np.nan, index=[0], columns=fake_keypoints.columns)
    df_machine.to_hdf(machine_path, key="keypoints", mode="w")
    machine_before = pd.read_hdf(machine_path, key="keypoints")

    points = np.column_stack([np.arange(n_rows), rng.random(n_rows), rng.random(n_rows)])

    fnames = _writer.write_hdf_napari_dlc("ignored.h5", points, metadata)
    assert Path(fnames[0]).name == "CollectedData_me.h5"

    # GT should exist and be readable
    df = pd.read_hdf(fnames[0], key="keypoints")
    assert isinstance(df, pd.DataFrame)

    # Must still be scored as "me" after promotion
    assert df.columns.get_level_values("scorer")[0] == "me"

    # Machine file must be unchanged
    machine_after = pd.read_hdf(machine_path, key="keypoints")
    pd.testing.assert_frame_equal(machine_before, machine_after)


def test_write_hdf_machine_source_without_save_target_aborts(tmp_path, fake_keypoints):
    """
    New contract:
    - machine/prediction sources must NEVER be written back.
    - if save_target is missing, writer must abort deterministically.
    """
    root = tmp_path / "proj"
    root.mkdir()

    header = misc.DLCHeader(fake_keypoints.columns)
    n_rows = len(fake_keypoints)

    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "machinelabels-iter0",
        "properties": {
            "label": per_row_labels,
            "id": per_row_ids,
            "likelihood": [1.0] * n_rows,
        },
        "metadata": {
            "header": header,
            "paths": [f"img{i}.png" for i in range(n_rows)],
            "root": str(root),
        },
    }

    _add_source_io(metadata, root=root, kind=AnnotationKind.MACHINE, source_name="machinelabels-iter0.h5")

    points = np.column_stack([np.arange(n_rows), rng.random(n_rows), rng.random(n_rows)])

    with pytest.raises(MissingProvenanceError):
        _writer.write_hdf_napari_dlc("ignored.h5", points, metadata)


def test_write_hdf_promotion_creates_gt_when_missing(tmp_path, fake_keypoints, monkeypatch):
    """
    Promotion contract:
    - machine source + save_target => create/update CollectedData_<scorer>.h5
    - scorer level should be rewritten to chosen scorer
    - machine file must not be created/modified by writer
    """
    root = tmp_path / "proj"
    root.mkdir()

    monkeypatch.setattr(napari_dlc_io, "maybe_confirm_overwrite", lambda *args, **kwargs: True)

    header = misc.DLCHeader(fake_keypoints.columns)
    n_rows = len(fake_keypoints)

    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "machinelabels-iter0",
        "properties": {
            "label": per_row_labels,
            "id": per_row_ids,
            "likelihood": [1.0] * n_rows,
        },
        "metadata": {
            "header": header,
            "paths": [f"img{i}.png" for i in range(n_rows)],
            "root": str(root),
        },
    }

    _add_source_io(metadata, root=root, kind=AnnotationKind.MACHINE, source_name="machinelabels-iter0.h5")
    _add_save_target(metadata, root=root, scorer="alice")

    points = np.column_stack([np.arange(n_rows), rng.random(n_rows), rng.random(n_rows)])

    fnames = _writer.write_hdf_napari_dlc("ignored.h5", points, metadata)
    assert Path(fnames[0]).name == "CollectedData_alice.h5"

    out_h5 = Path(fnames[0])
    assert out_h5.exists()

    df = pd.read_hdf(out_h5, key="keypoints")
    assert df.columns.get_level_values("scorer")[0] == "alice"

    # Ensure we still did NOT write back to a machine source file
    assert not (root / "machinelabels-iter0.h5").exists(), (
        "Writer should not create/overwrite prediction files during promotion."
    )
