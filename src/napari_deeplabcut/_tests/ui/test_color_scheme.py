# src/napari_deeplabcut/_tests/ui/test_color_scheme.py
from __future__ import annotations

import numpy as np
import pytest

from napari_deeplabcut import keypoints
from napari_deeplabcut.ui.color_scheme_display import (
    ColorSchemeDisplay,
    ColorSchemePanel,
    ColorSchemeResolver,
    _to_hex,
)


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
        # ensure cycles include config names so resolve() can use them later
        extra_metadata={
            "face_color_cycles": {
                "label": {
                    "cfg1": np.array([1.0, 0.0, 0.0, 1.0]),
                    "cfg2": np.array([0.0, 1.0, 0.0, 1.0]),
                }
            }
        },
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
        bodyparts=["cfg1", "cfg2"],
        individuals=[""],
        project=str(project),
        include_id_cycle=False,
        extra_metadata={
            "face_color_cycles": {
                "label": {
                    "cfg1": np.array([1.0, 0.0, 0.0, 1.0]),
                    "cfg2": np.array([0.0, 1.0, 0.0, 1.0]),
                }
            }
        },
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.INDIVIDUAL),
        get_header_model=get_header_model,
    )

    assert resolver.get_config_categories(layer, "id") == ["cfg1", "cfg2"]


def test_resolver_resolve_active_mode_returns_hex_for_visible_categories(
    fake_viewer,
    make_points_layer,
    get_header_model,
):
    data = np.array(
        [
            [0, 0.0, 0.0],  # nose
            [0, 1.0, 1.0],  # tail
            [1, 2.0, 2.0],  # nose, other frame
        ],
        dtype=float,
    )
    layer = make_points_layer(
        data=data,
        labels=["nose", "tail", "nose"],
        bodyparts=["nose", "tail"],
        shown=[True, False, True],  # hide tail in current frame
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer
    fake_viewer.dims.current_step = (0,)

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.BODYPART),
        get_header_model=get_header_model,
    )

    scheme = resolver.resolve(show_config_keypoints=False)
    assert scheme == {"nose": "#ff0000"}


def test_resolver_resolve_config_mode_uses_config_yaml_individuals_in_multianimal_mode(
    fake_viewer, make_points_layer, get_header_model, multianimal_config_project
):
    project, _config_path = multianimal_config_project
    layer = make_points_layer(
        data=np.array([[0, 0.0, 0.0], [0, 1.0, 1.0]], dtype=float),
        labels=["bodypart1", "bodypart2"],
        ids=["animal1", "animal2"],
        bodyparts=["bodypart1", "bodypart2"],
        individuals=["animal1", "animal2"],
        project=str(project),
    )
    fake_viewer.layers.append(layer)
    fake_viewer.layers.selection.active = layer

    resolver = ColorSchemeResolver(
        viewer=fake_viewer,
        get_color_mode=lambda: str(keypoints.ColorMode.INDIVIDUAL),
        get_header_model=get_header_model,
    )

    scheme = resolver.resolve(show_config_keypoints=True)
    assert scheme == {
        "animal1": "#ff00ff",
        "animal2": "#00ffff",
    }


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

    qtbot.waitUntil(lambda: panel.display.scheme_dict == {"nose": "#ff0000"})


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

    qtbot.waitUntil(lambda: panel.display.scheme_dict == {"nose": "#ff0000"})

    fake_viewer.dims.current_step = (1,)
    qtbot.waitUntil(lambda: panel.display.scheme_dict == {"tail": "#00ff00"})


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
            "face_color_cycles": {
                "label": {
                    "cfg1": np.array([1.0, 0.0, 0.0, 1.0]),
                    "cfg2": np.array([0.0, 1.0, 0.0, 1.0]),
                }
            }
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

    # Active mode: currently visible label is "nose", but it is absent from config-based cycle mapping
    # so active resolve should be empty with this specific setup.
    qtbot.waitUntil(lambda: panel.display.scheme_dict == {})

    panel._toggle.setChecked(True)
    qtbot.waitUntil(
        lambda: panel.display.scheme_dict
        == {
            "cfg1": "#ff0000",
            "cfg2": "#00ff00",
        }
    )
