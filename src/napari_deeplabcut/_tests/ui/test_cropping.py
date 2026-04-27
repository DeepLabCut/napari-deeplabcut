from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.ui import cropping as cropping_mod

# -----------------------------------------------------------------------------
# Small fake layer/viewer helpers (no qtbot, no napari viewer fixture)
# -----------------------------------------------------------------------------


class FakeShapes:
    def __init__(
        self,
        *,
        data,
        shape_type,
        metadata=None,
        name="shapes",
        selected_data=None,
    ):
        self.data = data
        self.shape_type = shape_type
        self.metadata = metadata or {}
        self.name = name
        self.selected_data = selected_data or set()


class FakeImage:
    def __init__(self, *, data=None, metadata=None, name="image", source_path=None):
        self.data = data
        self.metadata = metadata or {}
        self.name = name
        self.source = SimpleNamespace(path=source_path)


class FakePoints:
    def __init__(self, *, data=None, metadata=None, properties=None, name="points"):
        self.data = data
        self.metadata = metadata or {}
        self.properties = properties or {}
        self.name = name


class FakeLayerList(list):
    def __init__(self, layers=(), active=None):
        super().__init__(layers)
        self.selection = SimpleNamespace(active=active)


class DummyPanel:
    def __init__(self):
        self.text = None

    def set_context_text(self, text: str) -> None:
        self.text = text


# -----------------------------------------------------------------------------
# Basic schema validation
# -----------------------------------------------------------------------------


def test_viewer_crop_coords_accept_valid_tuple():
    coords = cropping_mod.ViewerCropCoords(values=(1, 10, 2, 20))
    assert coords.values == (1, 10, 2, 20)


@pytest.mark.parametrize(
    "bad",
    [
        (1, 1, 2, 20),  # x2 <= x1
        (1, 10, 5, 5),  # y2 <= y1
    ],
)
def test_viewer_crop_coords_reject_invalid_tuple(bad):
    with pytest.raises(ValueError, match="ViewerCropCoords"):
        cropping_mod.ViewerCropCoords(values=bad)


def test_dlc_config_crop_coords_accept_valid_tuple():
    coords = cropping_mod.DLCConfigCropCoords(values=(5, 15, 6, 30))
    assert coords.values == (5, 15, 6, 30)


def test_crop_save_plan_rejects_viewer_coords_for_config(tmp_path: Path):
    with pytest.raises(ValueError, match="Refusing to write napari/viewer crop coordinates"):
        cropping_mod.CropSavePlan(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            video_key="video.mp4",
            config_crop=cropping_mod.ViewerCropCoords(values=(1, 10, 2, 20)),
        )


# -----------------------------------------------------------------------------
# Rectangle resolution / coordinate conventions
# -----------------------------------------------------------------------------


def test_rectangle_spec_returns_viewer_and_DLC_config_coords(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    # rectangle vertices in [t, y, x]
    rect = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 60.0],
            [0.0, 40.0, 60.0],
            [0.0, 40.0, 20.0],
        ],
        dtype=float,
    )
    layer = FakeShapes(
        data=[rect],
        shape_type=["rectangle"],
        metadata={cropping_mod.DLC_CROP_LAYER_META_KEY: True},
        selected_data={0},
    )

    viewer = SimpleNamespace(
        dims=SimpleNamespace(
            # Y extent = 100, X extent = 200
            # This matches the [t, y, x] rectangle coordinates used in the test
            range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]
        )
    )

    spec = cropping_mod._rectangle_spec(viewer, layer, 0)
    assert spec is not None

    # raw napari/image-data coords
    assert spec.viewer_crop.values == (20, 60, 10, 40)

    #  DLC config coords (y flipped with h=100)
    assert spec.config_crop.values == (20, 60, 60, 90)


def test_find_crop_rectangle_prefers_dedicated_crop_layer(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    dedicated_rect = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 60.0],
            [0.0, 40.0, 60.0],
            [0.0, 40.0, 20.0],
        ],
        dtype=float,
    )
    other_rect = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 5.0],
            [0.0, 3.0, 5.0],
            [0.0, 3.0, 2.0],
        ],
        dtype=float,
    )

    dedicated = FakeShapes(
        data=[dedicated_rect],
        shape_type=["rectangle"],
        metadata={cropping_mod.DLC_CROP_LAYER_META_KEY: True},
        name=cropping_mod.DLC_CROP_LAYER_NAME,
        selected_data={0},
    )
    other = FakeShapes(
        data=[other_rect],
        shape_type=["rectangle"],
        metadata={},
        name="other",
        selected_data={0},
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([other, dedicated], active=other),
        dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]),
    )

    spec = cropping_mod.find_crop_rectangle(viewer, prefer_selected=True)
    assert spec is not None
    assert spec.viewer_crop.values == (20, 60, 10, 40)


def test_find_crop_rectangle_ignores_non_rectangles(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    poly = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 15.0, 25.0],
            [0.0, 12.0, 30.0],
        ],
        dtype=float,
    )
    rect = np.array(
        [
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 20.0],
            [0.0, 15.0, 20.0],
            [0.0, 15.0, 10.0],
        ],
        dtype=float,
    )

    poly_layer = FakeShapes(
        data=[poly],
        shape_type=["polygon"],
        metadata={},
        name="poly",
        selected_data={0},
    )
    rect_layer = FakeShapes(
        data=[rect],
        shape_type=["rectangle"],
        metadata={},
        name="rect",
        selected_data={0},
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([poly_layer, rect_layer], active=poly_layer),
        dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]),
    )

    spec = cropping_mod.find_crop_rectangle(viewer, prefer_selected=True)
    assert spec is not None
    assert spec.viewer_crop.values == (10, 20, 5, 15)


# -----------------------------------------------------------------------------
# Planning logic
# -----------------------------------------------------------------------------


def test_plan_frame_extraction_uses_viewer_crop(tmp_path: Path, monkeypatch):
    from napari.layers import Image

    monkeypatch.setattr(
        cropping_mod,
        "find_crop_rectangle",
        lambda viewer, prefer_selected=True: cropping_mod.CropRectangleSpec(
            viewer_crop=cropping_mod.ViewerCropCoords(values=(2, 8, 3, 9)),
            config_crop=cropping_mod.DLCConfigCropCoords(values=(2, 8, 91, 97)),
        ),
    )

    image = Image(
        np.zeros((10, 20, 30), dtype=np.uint8),
        name="demo.mp4",
        metadata={"root": str(tmp_path)},
    )

    viewer = SimpleNamespace(
        dims=SimpleNamespace(current_step=(4,), range=[(0, 10, 1), (0, 20, 1), (0, 100, 1)]),
    )

    plan, error = cropping_mod.plan_frame_extraction(
        viewer,
        image_layer=image,
        export_labels=False,
        apply_crop=True,
    )

    assert error is None
    assert plan is not None
    assert isinstance(plan.viewer_crop, cropping_mod.ViewerCropCoords)
    assert plan.viewer_crop.values == (2, 8, 3, 9)


def test_plan_crop_save_uses_config_crop(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        cropping_mod,
        "find_crop_rectangle",
        lambda viewer, prefer_selected=True: cropping_mod.CropRectangleSpec(
            viewer_crop=cropping_mod.ViewerCropCoords(values=(2, 8, 3, 9)),
            config_crop=cropping_mod.DLCConfigCropCoords(values=(2, 8, 91, 97)),
        ),
    )
    monkeypatch.setattr(
        cropping_mod,
        "infer_dlc_project_from_image_layer",
        lambda image_layer, prefer_project_root=True: SimpleNamespace(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            root_anchor=tmp_path,
        ),
    )

    image = FakeImage(
        data=np.zeros((10, 20, 30), dtype=np.uint8),
        metadata={},
        name="demo.mp4",
        source_path=str(tmp_path / "videos" / "demo.mp4"),
    )

    viewer = SimpleNamespace(
        dims=SimpleNamespace(range=[(0, 10, 1), (0, 20, 1), (0, 100, 1)]),
        layers=FakeLayerList([], active=None),
    )

    plan, error = cropping_mod.plan_crop_save(
        viewer,
        image_layer=image,
        explicit_project_path=None,
        fallback_video_name="demo.mp4",
    )

    assert error is None
    assert plan is not None
    assert isinstance(plan.config_crop, cropping_mod.DLCConfigCropCoords)
    assert plan.config_crop.values == (2, 8, 91, 97)


# -----------------------------------------------------------------------------
# Saving regression
# -----------------------------------------------------------------------------


def test_store_crop_coordinates_saves_DLC_config_coords(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    project = tmp_path / "project"
    project.mkdir()

    config_path = project / "config.yaml"
    config_path.write_text("video_sets: {}\n", encoding="utf-8")

    video_path = project / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.touch()

    rect = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 60.0],
            [0.0, 40.0, 60.0],
            [0.0, 40.0, 20.0],
        ],
        dtype=float,
    )

    crop_layer = FakeShapes(
        data=[rect],
        shape_type=["rectangle"],
        metadata={cropping_mod.DLC_CROP_LAYER_META_KEY: True},
        name=cropping_mod.DLC_CROP_LAYER_NAME,
        selected_data={0},
    )

    image_layer = FakeImage(
        data=np.zeros((10, 20, 30), dtype=np.uint8),
        metadata={"root": str(project / "labeled-data" / "demo")},
        name="demo.mp4",
        source_path=str(video_path),
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([crop_layer, image_layer], active=crop_layer),
        dims=SimpleNamespace(
            current_step=(0,),
            # Y extent is 100 (index -2), X extent is 200 (index -1 / old hardcoded index 2)
            range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)],
        ),
    )

    ok, msg = cropping_mod.store_crop_coordinates(
        viewer,
        image_layer=image_layer,
        explicit_project_path=str(project),
        fallback_video_name="demo.mp4",
    )

    assert ok is True
    assert "Saved crop" in msg

    cfg = cropping_mod.io.load_config(str(config_path))
    assert cfg["video_sets"][str(video_path)]["crop"] == "20, 60, 60, 90"


def test_rectangle_spec_uses_y_axis_extent_for_DLC_config_coords(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    # rectangle vertices in [t, y, x]
    rect = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 60.0],
            [0.0, 40.0, 60.0],
            [0.0, 40.0, 20.0],
        ],
        dtype=float,
    )
    layer = FakeShapes(
        data=[rect],
        shape_type=["rectangle"],
        metadata={cropping_mod.DLC_CROP_LAYER_META_KEY: True},
        selected_data={0},
    )

    # Make Y extent and X extent different so using the wrong axis would fail.
    # dims.range[-2][1] == 100  (Y)
    # dims.range[2][1]  == 200  (X)  <-- old buggy code would incorrectly use this
    viewer = SimpleNamespace(dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]))

    spec = cropping_mod._rectangle_spec(viewer, layer, 0)
    assert spec is not None

    # raw napari/image-data coords
    assert spec.viewer_crop.values == (20, 60, 10, 40)

    # DLC config coords must use Y extent = 100, not X extent = 200
    assert spec.config_crop.values == (20, 60, 60, 90)


# -----------------------------------------------------------------------------
# Context rendering (logic only, no Qt)
# -----------------------------------------------------------------------------


def test_update_video_panel_context_renders_current_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cropping_mod, "Image", FakeImage)
    monkeypatch.setattr(cropping_mod, "sync_crop_layer_autorefresh", lambda viewer, panel, refresh_callback: None)
    monkeypatch.setattr(
        cropping_mod,
        "get_crop_source_summary",
        lambda viewer: (
            "DLC crop layer",
            cropping_mod.CropRectangleSpec(
                viewer_crop=cropping_mod.ViewerCropCoords(values=(1, 10, 2, 20)),
                config_crop=cropping_mod.DLCConfigCropCoords(values=(1, 10, 80, 98)),
            ),
        ),
    )

    image = FakeImage(
        data=np.zeros((5, 20, 30), dtype=np.uint8),
        metadata={"root": str(tmp_path / "dataset")},
        name="demo.mp4",
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([image], active=image),
        dims=SimpleNamespace(current_step=(2,), range=[(0, 5, 1), (0, 20, 1), (0, 100, 1)]),
    )
    panel = DummyPanel()

    cropping_mod.update_video_panel_context(viewer, panel)

    assert "Frame 3/5" in panel.text
    assert f"Output folder: {tmp_path / 'dataset'}" in panel.text
    assert "Crop source: DLC crop layer" in panel.text


def test_execute_frame_extraction_keeps_new_labels_row_on_duplicate_index(monkeypatch, tmp_path: Path):
    from napari.layers import Image, Points

    # Avoid writing a real image file through skimage; just create the output file.
    monkeypatch.setattr(
        cropping_mod,
        "_write_image",
        lambda arr, path: Path(path).write_bytes(b"fake-image"),
    )

    image = Image(
        np.zeros((3, 20, 30), dtype=np.uint8),
        name="demo.mp4",
        metadata={"root": str(tmp_path)},
    )

    output_path = tmp_path / "img1.png"
    labels_path = tmp_path / "machinelabels-iter0.h5"

    # Existing row for the same extracted image path
    idx = pd.MultiIndex.from_tuples([("tmp", "pytest", "img1.png")])
    df_prev = pd.DataFrame({"bp1": [111.0]}, index=idx)
    df_prev.to_hdf(labels_path, key="df_with_missing")

    # New extracted row should overwrite the previous one
    df_new = pd.DataFrame({"bp1": [222.0]}, index=idx)

    monkeypatch.setattr(
        cropping_mod,
        "_build_extracted_frame_labels_df",
        lambda plan: (df_new, None),
    )

    points = Points(
        np.empty((0, 3), dtype=float),
        name="pts",
    )

    plan = cropping_mod.FrameExtractionPlan(
        image_layer=image,
        points_layer=points,
        frame_index=1,
        output_root=tmp_path,
        output_path=output_path,
        labels_path=labels_path,
        export_labels=True,
        viewer_crop=None,
    )

    written, note = cropping_mod.execute_frame_extraction(plan)

    assert note is None
    assert labels_path in written

    df_written = pd.read_hdf(labels_path, key="df_with_missing")
    assert float(df_written.iloc[0]["bp1"]) == 222.0


def test_ensure_dlc_crop_layer_reuses_existing(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    existing = FakeShapes(
        data=[],
        shape_type=[],
        metadata={cropping_mod.DLC_CROP_LAYER_META_KEY: True},
        name=cropping_mod.DLC_CROP_LAYER_NAME,
    )
    existing.visible = False
    existing.mode = None

    viewer = SimpleNamespace(
        layers=FakeLayerList([existing], active=None),
    )

    out = cropping_mod.ensure_dlc_crop_layer(viewer)

    assert out is existing
    assert existing.visible is True
    assert existing.mode == "add_rectangle"
    assert viewer.layers.selection.active is existing


def test_ensure_dlc_crop_layer_creates_new(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    created = FakeShapes(data=[], shape_type=[], metadata={}, name="created")
    created.mode = None

    class Viewer:
        def __init__(self):
            self.layers = FakeLayerList([], active=None)

        def add_shapes(self, name, metadata):
            created.name = name
            created.metadata = metadata
            self.layers.append(created)
            return created

    viewer = Viewer()

    out = cropping_mod.ensure_dlc_crop_layer(viewer)

    assert out is created
    assert out.name == cropping_mod.DLC_CROP_LAYER_NAME
    assert out.metadata[cropping_mod.DLC_CROP_LAYER_META_KEY] is True
    assert out.mode == "add_rectangle"
    assert viewer.layers.selection.active is out


def test_dlc_config_y_extent_falls_back_to_active_image(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Image", FakeImage)

    image = FakeImage(data=np.zeros((5, 40, 70), dtype=np.uint8))
    viewer = SimpleNamespace(
        dims=SimpleNamespace(range=None),
        layers=FakeLayerList([image], active=image),
    )

    assert cropping_mod._dlc_config_y_extent(viewer) == 40


def test_dlc_config_y_extent_falls_back_to_last_image(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Image", FakeImage)

    image = FakeImage(data=np.zeros((5, 55, 90), dtype=np.uint8))
    viewer = SimpleNamespace(
        dims=SimpleNamespace(range=None),
        layers=FakeLayerList([image], active=None),
    )

    assert cropping_mod._dlc_config_y_extent(viewer) == 55


def test_dlc_config_y_extent_returns_none_when_unavailable():
    viewer = SimpleNamespace(
        dims=SimpleNamespace(range=None),
        layers=FakeLayerList([], active=None),
    )

    assert cropping_mod._dlc_config_y_extent(viewer) is None


def test_find_rectangle_in_layer_falls_back_from_selected_to_last(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    bad_rect = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
        ],
        dtype=float,
    )
    good_rect = np.array(
        [
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 20.0],
            [0.0, 15.0, 20.0],
            [0.0, 15.0, 10.0],
        ],
        dtype=float,
    )

    layer = FakeShapes(
        data=[bad_rect, good_rect],
        shape_type=["rectangle", "rectangle"],
        selected_data={0},
    )
    viewer = SimpleNamespace(dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]))

    spec = cropping_mod._find_rectangle_in_layer(viewer, layer, prefer_selected=True)
    assert spec is not None
    assert spec.viewer_crop.values == (10, 20, 5, 15)


def test_get_crop_source_summary_uses_active_shapes_when_no_dedicated(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    rect = np.array(
        [
            [0.0, 5.0, 10.0],
            [0.0, 5.0, 20.0],
            [0.0, 15.0, 20.0],
            [0.0, 15.0, 10.0],
        ],
        dtype=float,
    )

    active = FakeShapes(
        data=[rect],
        shape_type=["rectangle"],
        metadata={},
        name="manual",
        selected_data={0},
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([active], active=active),
        dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]),
    )

    source, spec = cropping_mod.get_crop_source_summary(viewer)
    assert source == "active Shapes layer (manual)"
    assert spec is not None


def test_get_crop_source_summary_returns_none_when_no_valid_rectangles(monkeypatch):
    monkeypatch.setattr(cropping_mod, "Shapes", FakeShapes)

    poly = FakeShapes(
        data=[np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=float)],
        shape_type=["polygon"],
        metadata={},
        name="poly",
        selected_data={0},
    )

    viewer = SimpleNamespace(
        layers=FakeLayerList([poly], active=poly),
        dims=SimpleNamespace(range=[(0, 10, 1), (0, 100, 1), (0, 200, 1)]),
    )

    source, spec = cropping_mod.get_crop_source_summary(viewer)
    assert source == "none"
    assert spec is None


def test_plan_frame_extraction_rejects_missing_image_layer():
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0,)))
    plan, error = cropping_mod.plan_frame_extraction(viewer, image_layer=None)
    assert plan is None
    assert "No image/video layer is active" in error


def test_plan_frame_extraction_rejects_missing_output_root():
    image = FakeImage(data=np.zeros((3, 10, 10), dtype=np.uint8), metadata={})
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0,)))

    plan, error = cropping_mod.plan_frame_extraction(viewer, image_layer=image)
    assert plan is None
    assert "Could not determine the output folder" in error


def test_plan_frame_extraction_requires_points_layer_for_label_export(tmp_path: Path):
    image = FakeImage(
        data=np.zeros((3, 10, 10), dtype=np.uint8),
        metadata={"root": str(tmp_path)},
    )
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0,)))

    plan, error = cropping_mod.plan_frame_extraction(
        viewer,
        image_layer=image,
        export_labels=True,
        points_layer=None,
    )
    assert plan is None
    assert "no Points layer is available" in error


def test_plan_frame_extraction_requires_rectangle_when_crop_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cropping_mod, "find_crop_rectangle", lambda viewer, prefer_selected=True: None)

    image = FakeImage(
        data=np.zeros((3, 10, 10), dtype=np.uint8),
        metadata={"root": str(tmp_path)},
    )
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0,)))

    plan, error = cropping_mod.plan_frame_extraction(
        viewer,
        image_layer=image,
        apply_crop=True,
    )
    assert plan is None
    assert "no valid rectangle was found" in error


def test_plan_crop_save_rejects_missing_project_context(monkeypatch):
    monkeypatch.setattr(
        cropping_mod,
        "infer_dlc_project_from_image_layer",
        lambda image_layer, prefer_project_root=True: SimpleNamespace(
            config_path=None,
            project_root=None,
            root_anchor=None,
        ),
    )

    image = FakeImage(data=np.zeros((3, 10, 10), dtype=np.uint8))
    viewer = SimpleNamespace(layers=FakeLayerList([], active=None), dims=SimpleNamespace(range=[]))

    plan, error = cropping_mod.plan_crop_save(viewer, image_layer=image)
    assert plan is None
    assert "Could not determine a DLC config.yaml" in error


def test_plan_crop_save_rejects_missing_rectangle(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        cropping_mod,
        "infer_dlc_project_from_image_layer",
        lambda image_layer, prefer_project_root=True: SimpleNamespace(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            root_anchor=tmp_path,
        ),
    )
    monkeypatch.setattr(cropping_mod, "find_crop_rectangle", lambda viewer, prefer_selected=True: None)

    image = FakeImage(data=np.zeros((3, 10, 10), dtype=np.uint8), name="demo.mp4")
    viewer = SimpleNamespace(layers=FakeLayerList([], active=None), dims=SimpleNamespace(range=[]))

    plan, error = cropping_mod.plan_crop_save(viewer, image_layer=image)
    assert plan is None
    assert "No valid rectangle was found" in error


def test_execute_crop_save_replaces_non_dict_video_set_entry(monkeypatch, tmp_path: Path):
    cfg = {"video_sets": {"video.mp4": "old-value"}}

    monkeypatch.setattr(cropping_mod.io, "load_config", lambda path: cfg)
    written = {}

    def fake_write_config(path, out_cfg):
        written["path"] = path
        written["cfg"] = out_cfg

    monkeypatch.setattr(cropping_mod.io, "write_config", fake_write_config)

    plan = cropping_mod.CropSavePlan(
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        video_key="video.mp4",
        config_crop=cropping_mod.DLCConfigCropCoords(values=(1, 10, 20, 30)),
    )

    msg = cropping_mod.execute_crop_save(plan)

    assert "Saved crop" in msg
    assert written["cfg"]["video_sets"]["video.mp4"]["crop"] == "1, 10, 20, 30"


class HiddenPanel(DummyPanel):
    def isVisible(self):
        return False

    def parentWidget(self):
        return None


class VisiblePanel(DummyPanel):
    def isVisible(self):
        return True

    def parentWidget(self):
        return None


def test_update_video_panel_context_hidden_panel_detaches_and_returns(monkeypatch):
    panel = HiddenPanel()
    called = {"detach": 0}

    monkeypatch.setattr(
        cropping_mod, "_detach_crop_autorefresh", lambda p: called.__setitem__("detach", called["detach"] + 1)
    )

    viewer = SimpleNamespace(layers=FakeLayerList([], active=None), dims=SimpleNamespace(current_step=(0,)))

    cropping_mod.update_video_panel_context(viewer, panel)

    assert called["detach"] == 1
    assert panel.text is None


def test_update_video_panel_context_no_image_layer(monkeypatch):
    panel = VisiblePanel()
    monkeypatch.setattr(cropping_mod, "sync_crop_layer_autorefresh", lambda viewer, panel, refresh_callback: None)

    viewer = SimpleNamespace(
        layers=FakeLayerList([], active=None),
        dims=SimpleNamespace(current_step=(0,)),
    )

    cropping_mod.update_video_panel_context(viewer, panel)
    assert panel.text == "No active video/image layer."
