"""Internal unit tests for compat module logic. Does not check actual napari integration."""

#  src/napari_deeplabcut/_tests/compat/test_compat_internal.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut.napari_compat import (
    apply_points_layer_ui_tweaks,
    install_add_wrapper,
    install_paste_patch,
)
from napari_deeplabcut.napari_compat.points_layer import make_paste_data


def test_apply_points_layer_ui_tweaks_returns_none_when_viewer_shape_is_missing():
    viewer = SimpleNamespace(window=SimpleNamespace(qt_viewer=SimpleNamespace()))
    layer = SimpleNamespace(metadata={})

    result = apply_points_layer_ui_tweaks(
        viewer,
        layer,
        dropdown_cls=object,
        plt_module=SimpleNamespace(colormaps=[]),
    )

    assert result is None


def test_install_add_wrapper_calls_add_impl_and_schedule_recolor():
    calls = []

    class Layer:
        pass

    layer = Layer()

    def add_impl(*args, **kwargs):
        calls.append(("add_impl", args, kwargs))
        return "added"

    def schedule_recolor(obj):
        calls.append(("schedule_recolor", obj))

    install_add_wrapper(layer, add_impl=add_impl, schedule_recolor=schedule_recolor)

    result = layer.add(1, 2, kind="point")

    assert result == "added"
    assert calls[0] == ("add_impl", (1, 2), {"kind": "point"})
    assert calls[1] == ("schedule_recolor", layer)


def test_install_add_wrapper_swallows_schedule_recolor_errors():
    class Layer:
        pass

    layer = Layer()
    add_called = []

    def add_impl(*args, **kwargs):
        add_called.append((args, kwargs))
        return 123

    def schedule_recolor(_layer):
        raise RuntimeError("boom")

    install_add_wrapper(layer, add_impl=add_impl, schedule_recolor=schedule_recolor)

    assert layer.add("x") == 123
    assert add_called == [(("x",), {})]


def test_install_paste_patch_binds_method_to_layer():
    class Layer:
        pass

    layer = Layer()
    seen = []

    def paste_func(this):
        seen.append(this)

    install_paste_patch(layer, paste_func=paste_func)

    layer._paste_data()

    assert seen == [layer]


def test_make_paste_data_returns_early_when_all_points_are_annotated(monkeypatch):
    import sys
    import types
    from dataclasses import dataclass

    fake_layer_utils = types.ModuleType("napari.layers.utils.layer_utils")
    fake_layer_utils._features_to_properties = lambda features: {
        col: features[col].to_numpy() for col in features.columns
    }
    monkeypatch.setitem(sys.modules, "napari.layers.utils.layer_utils", fake_layer_utils)

    @dataclass(frozen=True)
    class Keypoint:
        label: object
        id: object

    recolor_calls = []

    controls = SimpleNamespace(
        np=np,
        keypoints=SimpleNamespace(Keypoint=Keypoint),
        _schedule_recolor=lambda layer: recolor_calls.append(layer),
    )

    store = SimpleNamespace(
        annotated_keypoints={Keypoint("nose", 1)},
        layer=object(),
    )

    layer = SimpleNamespace(
        _clipboard={
            "features": pd.DataFrame({"label": ["nose"], "id": [1]}),
            "indices": np.array([0, 0]),
            "text": None,
        },
        data=np.array([[1.0, 2.0]]),
        shown=np.array([True]),
        size=np.array([1.0]),
        symbol=np.array(["o"], dtype=object),
        edge_width=np.array([1.0]),
        _view_data=np.array([[1.0, 2.0]]),
        refresh=lambda: pytest.fail("refresh should not be called"),
    )

    paste = make_paste_data(controls, store=store)
    paste(layer)

    assert recolor_calls == []
    # features are popped before the early return; smoke test just checks no crash/no recolor


def test_make_paste_data_pastes_only_unannotated_points_and_recolors(paste_env):
    paste = make_paste_data(paste_env.controls, store=paste_env.store)
    paste(paste_env.layer)

    layer = paste_env.layer

    # original 1 point + only 1 pasted point survives (tail, id=2)
    assert layer._data.shape == (2, 3)
    np.testing.assert_allclose(layer._data[0], [1.0, 2.0, 3.0])

    # clipboard indices were [0, 5, 0], current slice idx is [0, 7, 0], so +2 on axis 1
    np.testing.assert_allclose(layer._data[1], [40.0, 52.0, 60.0])

    np.testing.assert_array_equal(layer._shown, np.array([True, False]))
    np.testing.assert_allclose(layer._size, np.array([5.0, 4.0]))
    np.testing.assert_array_equal(layer._symbol, np.array(["o", "+"], dtype=object))
    np.testing.assert_allclose(layer._edge_width, np.array([0.5, 2.0]))

    assert len(layer._feature_table.appended) == 1
    appended_features = layer._feature_table.appended[0]
    assert list(appended_features["label"]) == ["tail"]
    assert list(appended_features["id"]) == [2]

    assert len(layer.text.calls) == 1
    assert layer.text.calls[0]["color"] == "white"
    np.testing.assert_array_equal(layer.text.calls[0]["string"], np.array(["tail-2"], dtype=object))

    assert len(layer._edge.calls) == 1
    assert len(layer._face.calls) == 1
    np.testing.assert_array_equal(layer._edge.calls[0]["colors"], np.array([[0.0, 1.0, 0.0, 1.0]]))
    np.testing.assert_array_equal(layer._face.calls[0]["colors"], np.array([[0.0, 0.5, 0.0, 1.0]]))

    assert layer._selected_view == [1]
    assert layer._selected_data == {1}
    assert layer.refresh_count == 1
    assert paste_env.recolor_calls == [paste_env.store.layer]
