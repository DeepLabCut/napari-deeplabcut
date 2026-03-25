from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary, ref

from napari.layers import Layer, Points


@dataclass
class LayerChangeGenerations:
    """
    Monotonic "versioning" tokens for a layer.
    This helps build any derived state that needs to be invalidated on upstream changes,
    without needing to inspect the nature of those changes.

    Notes
    -----
    - `content` is intended for semantic/model changes (data, properties, features).
    - `presentation` is intended for display/config changes (metadata, visual config).
    """

    content: int = 0
    presentation: int = 0

    def bump_content(self) -> None:
        self.content += 1

    def bump_presentation(self) -> None:
        self.presentation += 1


@dataclass
class _LayerState:
    generations: LayerChangeGenerations = field(default_factory=LayerChangeGenerations)
    connections: list[tuple[object, Callable]] = field(default_factory=list)
    installed: bool = False


class LayerChangeRegistry:
    """
    Centralized registry for per-layer change generations and reusable event hooks.
    Layer agnostic, but with some Points-specific hooks currently.
    If the hook list needs updated, check `_content_emitters` and `_presentation_emitters` below.

    Design goals
    ------------
    - O(1) version-token reads
    - one-time event hookup per layer
    - explicit detach support
    - no mutation of napari internals
    """

    def __init__(self) -> None:
        self._states: WeakKeyDictionary[Layer, _LayerState] = WeakKeyDictionary()

    def ensure_hooks(self, layer: Layer) -> LayerChangeGenerations:
        state = self._states.get(layer)
        if state is None:
            state = _LayerState()
            self._states[layer] = state

        if not state.installed:
            self._install_hooks(layer, state)
            state.installed = True

        return state.generations

    def generations_for(self, layer: Layer) -> LayerChangeGenerations:
        return self.ensure_hooks(layer)

    def mark_content_changed(self, layer: Layer) -> None:
        self.ensure_hooks(layer).bump_content()

    def mark_presentation_changed(self, layer: Layer) -> None:
        self.ensure_hooks(layer).bump_presentation()

    def detach(self, layer: Layer) -> None:
        state = self._states.get(layer)
        if state is None:
            return

        for emitter, callback in state.connections:
            try:
                emitter.disconnect(callback)
            except Exception:
                # Best-effort disconnect; emitter/layer may already be torn down.
                pass

        state.connections.clear()
        state.installed = False
        self._states.pop(layer, None)

    # ---------- Private ----------

    def _install_hooks(self, layer: Layer, state: _LayerState) -> None:
        layer_ref = ref(layer)

        def _with_layer(fn: Callable[[Layer], None]) -> Callable:
            def _callback(event=None) -> None:
                target = layer_ref()
                if target is None:
                    return
                fn(target)

            return _callback

        def _on_content_change(target: Layer) -> None:
            self._states[target].generations.bump_content()

        def _on_presentation_change(target: Layer) -> None:
            self._states[target].generations.bump_presentation()

        for emitter in self._content_emitters(layer):
            callback = _with_layer(_on_content_change)
            emitter.connect(callback)
            state.connections.append((emitter, callback))

        for emitter in self._presentation_emitters(layer):
            callback = _with_layer(_on_presentation_change)
            emitter.connect(callback)
            state.connections.append((emitter, callback))

    def _content_emitters(self, layer: Layer) -> list[object]:
        emitters = [
            getattr(layer.events, "data", None),
            getattr(layer.events, "set_data", None),
        ]

        # Points-specific semantic state
        if isinstance(layer, Points):
            emitters.extend(
                [
                    getattr(layer.events, "properties", None),
                    getattr(layer.events, "features", None),
                ]
            )

        return [emitter for emitter in emitters if emitter is not None]

    def _presentation_emitters(self, layer: Layer) -> list[object]:
        emitters = [
            getattr(layer.events, "metadata", None),
        ]
        return [emitter for emitter in emitters if emitter is not None]


_LAYER_CHANGES = LayerChangeRegistry()


def ensure_layer_change_hooks(layer: Layer) -> LayerChangeGenerations:
    return _LAYER_CHANGES.ensure_hooks(layer)


def layer_change_generations(layer: Layer) -> LayerChangeGenerations:
    return _LAYER_CHANGES.generations_for(layer)


def mark_layer_content_changed(layer: Layer) -> None:
    _LAYER_CHANGES.mark_content_changed(layer)


def mark_layer_presentation_changed(layer: Layer) -> None:
    _LAYER_CHANGES.mark_presentation_changed(layer)


def detach_layer_change_hooks(layer: Layer) -> None:
    _LAYER_CHANGES.detach(layer)
