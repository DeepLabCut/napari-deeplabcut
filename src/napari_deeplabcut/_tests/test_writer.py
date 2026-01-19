from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread

from napari_deeplabcut import _writer, misc


#  Basic tests
def test_write_config(tmp_path):
    cfg = {"a": 1, "b": 2}
    path = tmp_path / "config.yaml"

    _writer._write_config(str(path), cfg)

    assert path.exists()
    text = path.read_text()
    assert "a:" in text and "b:" in text


def test_write_image(tmp_path):
    img = (np.random.rand(10, 10) * 255).astype(np.uint8)
    output = tmp_path / "test.png"

    _writer._write_image(img, str(output))

    assert output.exists()
    loaded = imread(str(output))
    assert loaded.shape == img.shape


#  Form_df — multi-animal + single-animal


def _fake_metadata_for_df(df, paths):
    """Helper for metadata for _form_df.

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


def test_form_df_multi_animal(fake_keypoints):
    n = len(fake_keypoints)
    metadata = _fake_metadata_for_df(fake_keypoints, [f"img{i}.png" for i in range(n)])

    # inds + (x,y)
    data = np.column_stack([np.arange(n), np.random.rand(n), np.random.rand(n)])

    df = _writer._form_df(data, metadata)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n

    # MultiIndex expected for multi-animal
    lvls = df.columns.names
    assert lvls == ["scorer", "individuals", "bodyparts", "coords"]

    # Bodyparts appear correctly
    assert set(df.columns.get_level_values("bodyparts")) == set(fake_keypoints.columns.get_level_values("bodyparts"))


def test_form_df_single_animal(fake_keypoints):
    """Drop the individuals level and check that _form_df handles it."""
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
    points = np.column_stack([np.arange(n), np.random.rand(n), np.random.rand(n)])
    out = _writer._form_df(points, metadata)

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
            np.random.rand(n_rows),
            np.random.rand(n_rows),
        ]
    )

    fname = _writer.write_hdf("whatever.h5", points, metadata)

    h5_path = root / fname
    csv_path = h5_path.with_suffix(".csv")

    assert h5_path.exists()
    assert csv_path.exists()

    df = pd.read_hdf(h5_path)
    assert isinstance(df, pd.DataFrame)
    # Ensure one row per index
    assert len(df) == n_rows


def test_write_hdf_machine_prediction_merge(tmp_path, fake_keypoints):
    """
    Trigger the special 'machine' branch:
      - metadata["name"] contains 'machine'
      - an existing CollectedData*.h5 file exists
      -> data should be merged
    """
    root = tmp_path / "proj"
    root.mkdir()

    # --- FIX: make GT index consistent with writer output (single-level MultiIndex) ---
    gt = fake_keypoints.copy()
    gt_idx = [f"img{i}.png" for i in gt.index]
    gt.index = pd.MultiIndex.from_tuples([(x,) for x in gt_idx])  # ('img0.png',) etc.
    gt_path = root / "CollectedData_me.h5"
    gt.to_hdf(gt_path, key="data")

    header = misc.DLCHeader(fake_keypoints.columns)

    # Build per-row properties
    n_rows = len(fake_keypoints)
    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "machine_predictions",
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
            np.random.rand(n_rows),
            np.random.rand(n_rows),
        ]
    )

    fname = _writer.write_hdf("ignored.h5", points, metadata)
    out_h5 = root / fname

    df = pd.read_hdf(out_h5)

    # merged data must include at least as many rows as the original
    assert len(df) >= n_rows

    # scorer should match original GT scorer
    assert df.columns.get_level_values("scorer")[0] == "me"


def test_write_hdf_machine_pred_no_gt(tmp_path, fake_keypoints):
    """
    Trigger machine branch, but **without** a CollectedData*.h5 file.
    It should:
       - load config.yaml to get scorer
       - write under "CollectedData_{scorer}.h5"
    """
    project_root = tmp_path / "proj"
    project_root.mkdir()

    # The writer looks for config.yaml at Path(root).parents[1] / "config.yaml".
    # With root = str(project_root), that is two levels above 'proj'.
    cfg_path = project_root.parents[1] / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("scorer: alice")

    header = misc.DLCHeader(fake_keypoints.columns)

    # Build per-row properties
    n_rows = len(fake_keypoints)
    from itertools import cycle, islice, product

    pairs = list(product(header.individuals, header.bodyparts))
    sel = list(islice(cycle(pairs), n_rows))
    per_row_ids = [p[0] for p in sel]
    per_row_labels = [p[1] for p in sel]

    metadata = {
        "name": "machine_predictions",
        "properties": {
            "label": per_row_labels,
            "id": per_row_ids,
            "likelihood": [1.0] * n_rows,
        },
        "metadata": {
            "header": header,
            "paths": [f"img{i}.png" for i in range(n_rows)],
            "root": str(project_root),
        },
    }

    points = np.column_stack(
        [
            np.arange(n_rows),
            np.random.rand(n_rows),
            np.random.rand(n_rows),
        ]
    )

    fname = _writer.write_hdf("ignored.h5", points, metadata)

    # Should name file based on scorer
    assert fname.startswith("CollectedData_alice")

    out_h5 = project_root / fname
    df = pd.read_hdf(out_h5)

    # columns scorer should be "alice"
    assert df.columns.get_level_values("scorer")[0] == "alice"


#  Write_masks — verify masks & vertices
def test_write_masks(tmp_path):
    foldername = str(tmp_path / "masks.h5")

    # fake polygon: frame index always 0
    data = [
        np.array([[0, 5, 5], [0, 5, 2]]).T  # (inds, y, x)
    ]

    metadata = {
        "metadata": {
            "shape": (1, 10, 10),
            "paths": ["frame0.png"],
        }
    }

    output_dir = _writer.write_masks(foldername, data, metadata)
    out_path = Path(output_dir)

    assert out_path.exists()

    # mask files present
    mask_files = list(out_path.glob("*_obj_*.png"))
    assert mask_files

    # vertices.csv must be present
    assert (out_path / "vertices.csv").exists()
