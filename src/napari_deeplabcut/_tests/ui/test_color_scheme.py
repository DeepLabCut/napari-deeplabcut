# src/napari_deeplabcut/_tests/ui/test_color_scheme.py
from __future__ import annotations

import numpy as np
import pytest

from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.config.settings import (
    DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP,
    DEFAULT_SINGLE_ANIMAL_CMAP,
)
from napari_deeplabcut.core import keypoints
from napari_deeplabcut.ui.color_scheme_display import (
    ColorSchemeDisplay,
    ColorSchemePanel,
    ColorSchemeResolver,
    _to_hex,
)


def _header_model_from_layer(layer) -> DLCHeaderModel:
    hdr = layer.metadata.get("header")
    assert hdr is not None, "Expected header in layer metadata"
    return hdr if isinstance(hdr, DLCHeaderModel) else DLCHeaderModel.model_validate(hdr)


def _is_multianimal_header(header: DLCHeaderModel) -> bool:
    inds = list(getattr(header, "individuals", []) or [])
    return bool(inds and str(inds[0]) != "")


def _config_colormap_from_layer(layer) -> str:
    md = layer.metadata or {}
    cmap = md.get("config_colormap")
    if isinstance(cmap, str) and cmap:
        return cmap
    return DEFAULT_SINGLE_ANIMAL_CMAP


def _expected_cycles_for_policy(layer) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute expected cycles from the new centralized policy.

    Source of truth:
    - layer header
    - metadata['config_colormap']
    - multi-animal policy for individual coloring
    """
    header = _header_model_from_layer(layer)
    config_cmap = _config_colormap_from_layer(layer)

    config_cycles = keypoints.build_color_cycles(header, config_cmap) or {}

    if _is_multianimal_header(header):
        individual_cycles = keypoints.build_color_cycles(header, DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP) or {}
    else:
        individual_cycles = config_cycles

    return {
        "label": config_cycles.get("label", {}),
        "id": individual_cycles.get("id", {}),
    }


def _expected_scheme(layer, *, prop: str, names: list[str]) -> dict[str, str]:
    cycles = _expected_cycles_for_policy(layer)
    mapping = cycles.get(prop, {})
    return {name: _to_hex(mapping[name]) for name in names if name in mapping}


def _expected_scheme_from_policy(layer, *, prop: str, names: list[str]) -> dict[str, str]:
    cycles = _expected_cycles_for_policy(layer)
    mapping = cycles.get(prop, {})
    return {name: _to_hex(mapping[name]) for name in names if name in mapping}


def test_to_hex_converts_rgb_and_rgba():
    assert _to_hex([1.0, 0.0, 0.0]) == "#ff0000"
    assert _to_hex([0.0, 1.0, 0.0, 0.5]) == "#00ff00"
    assert _to_hex([0.0, 0.0, 1.0, 1.0]) == "#0000ff"


def test_to_hex_returns_black_for_too_short_input():
    assert _to_hex([]) == "#000000"
    assert _to_hex([1.0, 0.0]) == "#000000"


@pytest.mark.usefixtures("qtbot")
def test_color_scheme_display_update_reuse_and_reset(qtbot):
    widget = ColorSchemeDisplay()
    qtbot.addWidget(widget)

    # Parent does not need to be shown for these tests; check hidden-state instead of isVisible().
    widget.update_color_scheme({"nose": "#ff0000", "tail": "#00ff00"})
    assert widget.scheme_dict == {"nose": "#ff0000", "tail": "#00ff00"}
    assert len(widget.labels) == 2
    assert widget.labels[0].part_name == "nose"
    assert widget.labels[0].color == "#ff0000"
    assert widget.labels[1].part_name == "tail"
    assert widget.labels[1].color == "#00ff00"
    assert widget.labels[0].isHidden() is False
    assert widget.labels[1].isHidden() is False

    # Reuse first widget, hide extra second widget
    widget.update_color_scheme({"ear": "#0000ff"})
    assert widget.scheme_dict == {"ear": "#0000ff"}
    assert len(widget.labels) == 2
    assert widget.labels[0].part_name == "ear"
    assert widget.labels[0].color == "#0000ff"
    assert widget.labels[0].isHidden() is False
    assert widget.labels[1].isHidden() is True

    widget.reset()
    assert widget.scheme_dict == {}
    assert all(w.isHidden() for w in widget.labels)


def test_resolver_get_target_layer_prefers_active_visible(fake_viewer, make_points_layer, get_header_model):
    layer1 = make_points_layer(labels=["nose"], bodyparts=["nose"], visible=True)
    layer2 = make_points_layer(labels=["tail"], bodyparts=["tail"], visible=True)

    fake_viewer.layers.append(layer1)
    fake_viewer.layers.append(layer2)
    fake_viewer.layers.selection.active = layer1

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )

    assert resolver.get_target_layer() is layer1


def test_resolver_get_target_layer_falls_back_to_topmost_visible_when_active_hidden(
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    layer1 = make_points_layer(labels=["nose"], bodyparts=["nose"], visible=True)
    layer2 = make_points_layer(labels=["tail"], bodyparts=["tail"], visible=True)

    fake_viewer.layers.append(layer1)
    fake_viewer.layers.append(layer2)

    # Active layer exists but is hidden -> should fall back to topmost visible
    layer1.visible = False
    fake_viewer.layers.selection.active = layer1

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )

    assert resolver.get_target_layer() is layer2


def test_resolver_get_color_property_prefers_id_in_multianimal_individual_mode(
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    layer = make_points_layer(
        labels=["bodypart1", "bodypart2"],
        ids=["animal1", "animal2"],
        bodyparts=["bodypart1", "bodypart2"],
        individuals=["animal1", "animal2"],
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.INDIVIDUAL),
        get_header_model=get_header_model,
    )

    assert resolver.get_color_property(layer) == "id"


def test_resolver_get_visible_categories_filters_frame_shown_and_deduplicates(
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    data = np.array(
        [
            [0, 0.0, 0.0],  # nose, shown
            [0, 1.0, 1.0],  # tail, hidden
            [1, 2.0, 2.0],  # nose, other frame
            [0, 3.0, 3.0],  # ear, shown
            [0, 4.0, 4.0],  # nose, duplicate in current frame
        ],
        dtype=float,
    )
    layer = make_points_layer(
        data=data,
        labels=["nose", "tail", "nose", "ear", "nose"],
        bodyparts=["nose", "tail", "ear"],
        shown=[True, False, True, True, True],
    )

    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer
    fake_viewer.dims.current_step = (0,)

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )

    visible = resolver.get_visible_categories(layer, "label")
    assert visible == ["nose", "ear"]


def test_resolver_get_config_categories_prefers_config_yaml_bodyparts(
    fake_viewer,
    make_points_layer,
    get_header_model,
    single_animal_project,
):
    project, _config_path = single_animal_project
    layer = make_points_layer(
        labels=["nose"],
        bodyparts=["nose", "tail"],
        individuals=[""],
        project=str(project),
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )

    assert resolver.get_config_categories(layer, "label") == ["cfg1", "cfg2"]


def test_resolver_get_config_categories_id_falls_back_to_bodyparts_for_single_animal_config(
    fake_viewer,
    make_points_layer,
    get_header_model,
    single_animal_project,
):
    project, _config_path = single_animal_project
    layer = make_points_layer(
        labels=["nose"],
        bodyparts=["nose", "tail"],
        individuals=[""],
        project=str(project),
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.INDIVIDUAL),
        get_header_model=get_header_model,
    )

    assert resolver.get_config_categories(layer, "id") == ["cfg1", "cfg2"]


@pytest.mark.usefixtures("qtbot")
def test_panel_initial_active_mode_updates_display_from_current_frame(
    qtbot,
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    data = np.array(
        [
            [0, 0.0, 0.0],  # nose
            [1, 1.0, 1.0],  # tail
        ],
        dtype=float,
    )
    layer = make_points_layer(
        data=data,
        labels=["nose", "tail"],
        bodyparts=["nose", "tail"],
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer
    fake_viewer.dims.current_step = (0,)

    panel = ColorSchemePanel(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )
    qtbot.addWidget(panel)

    expected = _expected_scheme_from_policy(layer, prop="label", names=["nose"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected)


@pytest.mark.usefixtures("qtbot")
def test_panel_reacts_to_frame_change_event(
    qtbot,
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    data = np.array(
        [
            [0, 0.0, 0.0],  # nose
            [1, 1.0, 1.0],  # tail
        ],
        dtype=float,
    )
    layer = make_points_layer(
        data=data,
        labels=["nose", "tail"],
        bodyparts=["nose", "tail"],
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer
    fake_viewer.dims.current_step = (0,)

    panel = ColorSchemePanel(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )
    qtbot.addWidget(panel)

    expected0 = _expected_scheme_from_policy(layer, prop="label", names=["nose"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected0)

    fake_viewer.dims.current_step = (1,)
    expected1 = _expected_scheme_from_policy(layer, prop="label", names=["tail"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected1)


@pytest.mark.usefixtures("qtbot")
def test_panel_toggle_switches_from_active_to_config_preview(
    qtbot,
    fake_viewer,
    make_points_layer,
    get_header_model,
    single_animal_project,
):
    project, _config_path = single_animal_project
    layer = make_points_layer(
        data=np.array([[0, 0.0, 0.0]], dtype=float),
        labels=["nose"],
        bodyparts=["nose", "tail"],
        project=str(project),
        extra_metadata={
            "config_colormap": "rainbow",
        },
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer
    fake_viewer.dims.current_step = (0,)

    panel = ColorSchemePanel(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )
    qtbot.addWidget(panel)

    # Active mode should show the currently visible label from the layer.
    expected_active = _expected_scheme_from_policy(layer, prop="label", names=["nose"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_active)

    # Config preview should show the configured bodyparts from config.yaml.
    panel._toggle.setChecked(True)
    expected_config = _expected_scheme_from_policy(layer, prop="label", names=["cfg1", "cfg2"])
    qtbot.waitUntil(lambda: panel.display.scheme_dict == expected_config)


def test_color_scheme_panel_delete_later_does_not_crash_on_pending_update(qtbot, fake_viewer, get_header_model):
    panel = ColorSchemePanel(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )
    qtbot.addWidget(panel)
    panel.schedule_update()
    qtbot.wait(50)
