from __future__ import annotations

import pytest
from qtpy.QtWidgets import QWidget

from napari_deeplabcut.ui.base_widget import ViewerSingletonWidget


class DummyWindow:
    def __init__(self):
        self.dock_widgets = {}


class DummyViewer:
    def __init__(self):
        self.window = DummyWindow()


class Wrapper:
    def __init__(self, wrapped=None, *, use_obj=False):
        if use_obj:
            self._obj = wrapped
        else:
            self.__wrapped__ = wrapped


class DockWrapper:
    def __init__(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget


class TestSingleton(ViewerSingletonWidget):
    def __init__(self, napari_viewer):
        if not self._singleton_prepare_init(napari_viewer=napari_viewer):
            return

        super().__init__()
        self._singleton_finalize_init()

        self.viewer = self.canonical_viewer(napari_viewer)
        self.init_count = getattr(self, "init_count", 0) + 1


class OtherSingleton(ViewerSingletonWidget):
    def __init__(self, napari_viewer):
        if not self._singleton_prepare_init(napari_viewer=napari_viewer):
            return

        super().__init__()
        self._singleton_finalize_init()

        self.viewer = self.canonical_viewer(napari_viewer)
        self.init_count = getattr(self, "init_count", 0) + 1


@pytest.fixture(autouse=True)
def clear_singleton_registry():
    """
    Keep tests isolated because the registry is class-level global state.
    """
    ViewerSingletonWidget._instances_by_cls.clear()
    yield
    ViewerSingletonWidget._instances_by_cls.clear()


def test_extract_viewer_from_call_positional():
    viewer = DummyViewer()
    assert ViewerSingletonWidget._extract_viewer_from_call((viewer,), {}) is viewer


def test_extract_viewer_from_call_napari_viewer_kwarg():
    viewer = DummyViewer()
    assert ViewerSingletonWidget._extract_viewer_from_call((), {"napari_viewer": viewer}) is viewer


def test_extract_viewer_from_call_viewer_kwarg():
    viewer = DummyViewer()
    assert ViewerSingletonWidget._extract_viewer_from_call((), {"viewer": viewer}) is viewer


def test_extract_viewer_from_call_none_when_missing():
    assert ViewerSingletonWidget._extract_viewer_from_call((), {}) is None


def test_canonical_viewer_unwraps_wrapped_chain():
    viewer = DummyViewer()
    wrapped = Wrapper(Wrapper(viewer))
    assert ViewerSingletonWidget.canonical_viewer(wrapped) is viewer


def test_canonical_viewer_unwraps_obj_chain():
    viewer = DummyViewer()
    wrapped = Wrapper(Wrapper(viewer, use_obj=True), use_obj=True)
    assert ViewerSingletonWidget.canonical_viewer(wrapped) is viewer


def test_canonical_viewer_stops_on_self_wrapped():
    wrapped = Wrapper(None)
    wrapped.__wrapped__ = wrapped
    assert ViewerSingletonWidget.canonical_viewer(wrapped) is wrapped


@pytest.mark.usefixtures("qtbot")
def test_same_viewer_same_subclass_returns_same_instance(qtbot):
    viewer = DummyViewer()

    w1 = TestSingleton(viewer)
    qtbot.addWidget(w1)

    w2 = TestSingleton(viewer)

    assert w1 is w2
    assert w1.init_count == 1
    assert w2.init_count == 1


@pytest.mark.usefixtures("qtbot")
def test_get_or_create_returns_existing_instance(qtbot):
    viewer = DummyViewer()

    w1 = TestSingleton(viewer)
    qtbot.addWidget(w1)

    w2 = TestSingleton.get_or_create(viewer)

    assert w1 is w2
    assert w1.init_count == 1


@pytest.mark.usefixtures("qtbot")
def test_get_existing_accepts_wrapped_viewer(qtbot):
    viewer = DummyViewer()
    proxy = Wrapper(viewer)

    w = TestSingleton(viewer)
    qtbot.addWidget(w)

    assert TestSingleton.get_existing(proxy) is w


@pytest.mark.usefixtures("qtbot")
def test_different_subclasses_have_independent_singletons(qtbot):
    viewer = DummyViewer()

    w1 = TestSingleton(viewer)
    qtbot.addWidget(w1)

    w2 = OtherSingleton(viewer)
    qtbot.addWidget(w2)

    assert w1 is not w2
    assert isinstance(w1, TestSingleton)
    assert isinstance(w2, OtherSingleton)


@pytest.mark.usefixtures("qtbot")
def test_is_docked_true_when_widget_directly_registered(qtbot):
    viewer = DummyViewer()
    widget = TestSingleton(viewer)
    qtbot.addWidget(widget)

    viewer.window.dock_widgets["a"] = widget

    assert TestSingleton.is_docked(viewer, widget) is True


@pytest.mark.usefixtures("qtbot")
def test_is_docked_true_when_wrapper_returns_widget(qtbot):
    viewer = DummyViewer()
    widget = TestSingleton(viewer)
    qtbot.addWidget(widget)

    viewer.window.dock_widgets["a"] = DockWrapper(widget)

    assert TestSingleton.is_docked(viewer, widget) is True


@pytest.mark.usefixtures("qtbot")
def test_is_docked_false_when_widget_not_present(qtbot):
    viewer = DummyViewer()
    widget = TestSingleton(viewer)
    qtbot.addWidget(widget)

    other = QWidget()
    qtbot.addWidget(other)

    assert TestSingleton.is_docked(viewer, widget) is False


@pytest.mark.usefixtures("qtbot")
def test_get_existing_returns_none_after_widget_deleted(qtbot):
    viewer = DummyViewer()

    widget = TestSingleton(viewer)
    qtbot.addWidget(widget)

    assert TestSingleton.get_existing(viewer) is widget

    widget.deleteLater()
    qtbot.waitUntil(lambda: TestSingleton.get_existing(viewer) is None, timeout=1000)

    assert TestSingleton.get_existing(viewer) is None
