# src/napari_deeplabcut/_tests/ui/conftest.py
from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points
from napari.utils.events import EmitterGroup, Event


class FakeSelection:
    def __init__(self):
        self._active = None
        self.events = EmitterGroup(source=self, active=Event)

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value
        self.events.active(value=value)


class FakeLayers(list):
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.events = EmitterGroup(source=self, inserted=Event, removed=Event)
        self.selection = FakeSelection()

    def append(self, layer):
        super().append(layer)
        self.events.inserted(value=layer)

    def remove(self, layer):
        super().remove(layer)
        self.events.removed(value=layer)


class FakeDims:
    def __init__(self, current_step=(0,)):
        self._current_step = tuple(current_step)
        self.events = EmitterGroup(source=self, current_step=Event)

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value):
        self._current_step = tuple(value)
        self.events.current_step(value=self._current_step)


class FakeViewer:
    def __init__(self, layers=None, current_step=(0,)):
        self.layers = FakeLayers(layers or [])
        self.dims = FakeDims(current_step=current_step)


@pytest.fixture
def fake_viewer():
    return FakeViewer()


@pytest.fixture
def make_header(make_real_header_factory):
    """
    Return a callable header factory, not a single pre-built header.
    """

    def _make_header(
        *,
        bodyparts=("nose", "tail"),
        individuals=("",),
        scorer="tester",
    ):
        return make_real_header_factory(
            bodyparts=bodyparts,
            individuals=individuals,
            scorer=scorer,
        )

    return _make_header


@pytest.fixture
def get_header_model():
    """
    Match the production callback signature used by ColorSchemeResolver/Panel.
    """

    def _get_header_model(md: dict):
        return md.get("header", None)

    return _get_header_model


def _make_cycles_for_bodyparts(bodyparts: list[str] | tuple[str, ...]):
    base = {
        "nose": np.array([1.0, 0.0, 0.0, 1.0]),  # red
        "tail": np.array([0.0, 1.0, 0.0, 1.0]),  # green
        "ear": np.array([0.0, 0.0, 1.0, 1.0]),  # blue
        "paw": np.array([1.0, 1.0, 0.0, 1.0]),  # yellow
        "cfg1": np.array([1.0, 0.0, 0.0, 1.0]),
        "cfg2": np.array([0.0, 1.0, 0.0, 1.0]),
        "bodypart1": np.array([1.0, 0.0, 0.0, 1.0]),
        "bodypart2": np.array([0.0, 1.0, 0.0, 1.0]),
        "bodypart3": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    return {bp: base[bp] for bp in bodyparts}


def _make_cycles_for_ids(ids: list[str] | tuple[str, ...]):
    base = {
        "animal1": np.array([1.0, 0.0, 1.0, 1.0]),  # magenta
        "animal2": np.array([0.0, 1.0, 1.0, 1.0]),  # cyan
        "animal3": np.array([0.5, 0.5, 0.5, 1.0]),  # gray
    }
    return {i: base[i] for i in ids}


@pytest.fixture
def make_points_layer(make_header):
    def _make_points_layer(
        *,
        data: np.ndarray | None = None,
        labels: list[str] | tuple[str, ...] = ("nose", "tail"),
        ids: list[str] | tuple[str, ...] | None = None,
        bodyparts: list[str] | tuple[str, ...] | None = None,
        individuals: list[str] | tuple[str, ...] = ("",),
        project: str | None = None,
        shown: list[bool] | np.ndarray | None = None,
        visible: bool = True,
        include_id_cycle: bool = True,
        extra_metadata: dict | None = None,
    ) -> Points:
        labels = list(labels)
        if ids is None:
            ids = [""] * len(labels)
        ids = list(ids)

        if bodyparts is None:
            bodyparts = tuple(dict.fromkeys(labels))

        header = make_header(bodyparts=bodyparts, individuals=individuals)

        if data is None:
            # default: frame, y, x
            data = np.array([[0, float(i), float(i)] for i in range(len(labels))], dtype=float)

        metadata = {
            "header": header,
            "face_color_cycles": {
                "label": _make_cycles_for_bodyparts(bodyparts),
            },
        }

        if include_id_cycle:
            non_empty_ids = [x for x in individuals if x != ""]
            if non_empty_ids:
                metadata["face_color_cycles"]["id"] = _make_cycles_for_ids(non_empty_ids)

        if project is not None:
            metadata["project"] = str(project)

        if extra_metadata:
            metadata.update(extra_metadata)

        properties = {
            "label": np.asarray(labels, dtype=object),
            "id": np.asarray(ids, dtype=object),
        }

        layer = Points(
            data=np.asarray(data, dtype=float),
            properties=properties,
            metadata=metadata,
            name="points",
        )
        layer.visible = visible

        if shown is not None:
            layer.shown = np.asarray(shown, dtype=bool)

        return layer

    return _make_points_layer
