from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import napari_deeplabcut.config.keybinds as keybinds


class DummyLayer:
    def __init__(self):
        self.bound = []

    def bind_key(self, key, callback, overwrite=False):
        self.bound.append(
            {
                "key": key,
                "callback": callback,
                "overwrite": overwrite,
            }
        )


def test_iter_shortcuts_returns_registry():
    shortcuts = tuple(keybinds.iter_shortcuts())
    assert shortcuts == keybinds.SHORTCUTS
    assert shortcuts, "SHORTCUTS should not be empty"


def test_shortcuts_registry_points_layer_entries_have_callbacks():
    for spec in keybinds.SHORTCUTS:
        assert spec.keys
        assert spec.description
        assert spec.group
        assert spec.scope in {"points-layer", "global-points"}

        if spec.scope == "points-layer":
            assert spec.get_callback is not None
            assert spec.action is not None


def test_shortcuts_registry_has_no_duplicate_keys_within_scope():
    seen = set()

    for spec in keybinds.SHORTCUTS:
        for key in spec.keys:
            item = (spec.scope, key)
            assert item not in seen, f"Duplicate shortcut declared for scope/key: {item}"
            seen.add(item)


def test_bind_each_key_binds_all_keys():
    layer = DummyLayer()

    def callback():
        return None

    keybinds._bind_each_key(layer, ("A", "B"), callback, overwrite=True)

    assert layer.bound == [
        {"key": "A", "callback": callback, "overwrite": True},
        {"key": "B", "callback": callback, "overwrite": True},
    ]


def test_install_points_layer_keybindings_binds_registry_declared_shortcuts():
    layer = DummyLayer()

    controls = SimpleNamespace(
        cycle_through_label_modes=object(),
        cycle_through_color_modes=object(),
    )
    store = SimpleNamespace(
        next_keypoint=object(),
        prev_keypoint=object(),
        _find_first_unlabeled_frame=object(),
    )

    keybinds.install_points_layer_keybindings(layer, controls, store)

    expected = []
    ctx = keybinds.BindingContext(controls=controls, store=store)

    for spec in keybinds.SHORTCUTS:
        if spec.scope != "points-layer":
            continue
        callback = spec.get_callback(ctx)
        for key in spec.keys:
            expected.append(
                {
                    "key": key,
                    "callback": callback,
                    "overwrite": spec.overwrite,
                }
            )

    assert layer.bound == expected


def test_callback_resolvers_return_expected_methods():
    controls = SimpleNamespace(
        cycle_through_label_modes=object(),
        cycle_through_color_modes=object(),
    )
    store = SimpleNamespace(
        next_keypoint=object(),
        prev_keypoint=object(),
        _find_first_unlabeled_frame=object(),
    )
    ctx = keybinds.BindingContext(controls=controls, store=store)

    assert keybinds._cycle_label_mode(ctx) is controls.cycle_through_label_modes
    assert keybinds._cycle_color_mode(ctx) is controls.cycle_through_color_modes
    assert keybinds._next_keypoint(ctx) is store.next_keypoint
    assert keybinds._prev_keypoint(ctx) is store.prev_keypoint
    assert keybinds._jump_unlabeled_frame(ctx) is store._find_first_unlabeled_frame


def test_toggle_edge_color_toggles_between_0_and_2():
    layer = SimpleNamespace(border_width=np.array([0, 2, 0, 2]))

    keybinds.toggle_edge_color(layer)
    np.testing.assert_array_equal(layer.border_width, np.array([2, 0, 2, 0]))

    keybinds.toggle_edge_color(layer)
    np.testing.assert_array_equal(layer.border_width, np.array([0, 2, 0, 2]))


def test_install_global_points_keybindings_installs_once(monkeypatch):
    calls = []

    class DummyPoints:
        @staticmethod
        def bind_key(key):
            calls.append(("bind_key", key))

            def decorator(callback):
                calls.append(("decorated", key, callback))
                return callback

            return decorator

    monkeypatch.setattr(keybinds, "Points", DummyPoints)
    monkeypatch.setattr(keybinds, "_global_points_bindings_installed", False)

    keybinds.install_global_points_keybindings()
    first_calls = list(calls)

    keybinds.install_global_points_keybindings()
    second_calls = list(calls)

    assert first_calls, "Expected at least one global binding registration"
    assert second_calls == first_calls, "Second install should be a no-op"


def test_global_shortcuts_are_not_missing_install_support():
    """
    Help ensure all global shortcuts have corresponding installation code
    by asserting that all global actions are accounted for.
    """
    global_actions = {spec.action for spec in keybinds.SHORTCUTS if spec.scope == "global-points"}
    assert global_actions == {keybinds.ShortcutAction.TOGGLE_EDGE_COLOR}
