from __future__ import annotations

import pytest
from qtpy.QtCore import QObject, Signal

from napari_deeplabcut.ui.base_widget._qt_timers import OwnedTimersMixin


class DummyOwner(QObject, OwnedTimersMixin):
    """Minimal QObject host for OwnedTimersMixin."""

    ping = Signal()

    def __init__(self):
        super().__init__()
        self._init_owned_timers()


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeTimer:
    def __init__(self, parent=None, *, fail_on_start=False, fail_on_stop=False, fail_on_delete=False):
        self.parent = parent
        self.fail_on_start = fail_on_start
        self.fail_on_stop = fail_on_stop
        self.fail_on_delete = fail_on_delete

        self.single_shot = None
        self.started_with = []
        self.stopped = 0
        self.deleted = 0
        self.timeout = FakeSignal()

    def setSingleShot(self, value):
        self.single_shot = value

    def start(self, msec):
        if self.fail_on_start:
            raise RuntimeError("timer already deleted")
        self.started_with.append(msec)

    def stop(self):
        if self.fail_on_stop:
            raise RuntimeError("timer already deleted")
        self.stopped += 1

    def deleteLater(self):
        if self.fail_on_delete:
            raise RuntimeError("timer already deleted")
        self.deleted += 1


@pytest.fixture
def owner():
    return DummyOwner()


@pytest.fixture
def fake_qtimer_state():
    """Mutable state used by the autouse QTimer patch."""
    return {
        "created": [],
        "fail_on_start": False,
        "fail_on_stop": False,
        "fail_on_delete": False,
    }


@pytest.fixture(autouse=True)
def patch_qtimer(monkeypatch, fake_qtimer_state):
    """Patch OwnedTimersMixin's QTimer dependency for all tests in this module."""

    def fake_qtimer(parent=None):
        timer = FakeTimer(
            parent=parent,
            fail_on_start=fake_qtimer_state["fail_on_start"],
            fail_on_stop=fake_qtimer_state["fail_on_stop"],
            fail_on_delete=fake_qtimer_state["fail_on_delete"],
        )
        fake_qtimer_state["created"].append(timer)
        return timer

    monkeypatch.setattr(
        "napari_deeplabcut.ui.base_widget._qt_timers.QTimer",
        fake_qtimer,
    )


def test_init_owned_timers_is_idempotent(owner):
    timers_before = owner._timers
    temp_before = owner._temp_timers

    owner._init_owned_timers()

    assert owner._owned_timers_initialized is True
    assert owner._timers is timers_before
    assert owner._temp_timers is temp_before
    assert owner._timers == {}
    assert owner._temp_timers == set()


def test_schedule_once_reuses_named_timer(owner, fake_qtimer_state):
    calls = []

    def cb():
        calls.append("called")

    owner._schedule_once("refresh", 0, cb)
    owner._schedule_once("refresh", 10, cb)

    created = fake_qtimer_state["created"]
    assert len(created) == 1

    timer = created[0]
    assert timer.parent is owner
    assert timer.single_shot is True
    assert timer.started_with == [0, 10]
    assert owner._timers["refresh"] is timer
    assert timer.timeout.callbacks == [cb]
    assert calls == []  # scheduling only; no auto-fire here


def test_schedule_once_creates_distinct_timers_for_distinct_names(owner, fake_qtimer_state):
    owner._schedule_once("refresh", 0, lambda: None)
    owner._schedule_once("status", 5, lambda: None)

    created = fake_qtimer_state["created"]
    assert len(created) == 2
    assert set(owner._timers) == {"refresh", "status"}
    assert owner._timers["refresh"] is created[0]
    assert owner._timers["status"] is created[1]


def test_single_shot_owned_tracks_timer_and_removes_it_after_fire(owner, fake_qtimer_state):
    calls = []

    def cb():
        calls.append("called")

    owner._single_shot_owned(7, cb)

    created = fake_qtimer_state["created"]
    assert len(created) == 1
    timer = created[0]
    assert timer.parent is owner
    assert timer.single_shot is True
    assert timer.started_with == [7]
    assert timer in owner._temp_timers
    assert len(timer.timeout.callbacks) == 1

    # Simulate timeout
    fire = timer.timeout.callbacks[0]
    fire()

    assert calls == ["called"]
    assert timer not in owner._temp_timers
    assert timer.deleted == 1


def test_single_shot_owned_discards_timer_if_start_raises(owner, fake_qtimer_state):
    fake_qtimer_state["fail_on_start"] = True

    owner._single_shot_owned(0, lambda: None)

    created = fake_qtimer_state["created"]
    assert len(created) == 1
    assert created[0] not in owner._temp_timers


def test_schedule_once_swallows_runtime_error_on_start(owner, fake_qtimer_state):
    fake_qtimer_state["fail_on_start"] = True

    owner._schedule_once("refresh", 0, lambda: None)

    created = fake_qtimer_state["created"]
    assert len(created) == 1
    assert owner._timers["refresh"] is created[0]


def test_cancel_scheduled_stops_named_timer(owner, fake_qtimer_state):
    owner._schedule_once("refresh", 0, lambda: None)
    owner._cancel_scheduled("refresh")
    owner._cancel_scheduled("missing")  # no-op

    created = fake_qtimer_state["created"]
    assert len(created) == 1
    assert created[0].stopped == 1


def test_cleanup_owned_timers_stops_and_deletes_all(owner, fake_qtimer_state):
    owner._schedule_once("refresh", 0, lambda: None)
    owner._schedule_once("status", 1, lambda: None)
    owner._single_shot_owned(2, lambda: None)
    owner._single_shot_owned(3, lambda: None)

    assert len(owner._timers) == 2
    assert len(owner._temp_timers) == 2

    owner._cleanup_owned_timers()

    assert owner._timers == {}
    assert owner._temp_timers == set()

    for timer in fake_qtimer_state["created"]:
        assert timer.stopped == 1
        assert timer.deleted == 1


def test_cleanup_owned_timers_swallows_runtime_error(owner, fake_qtimer_state):
    fake_qtimer_state["fail_on_stop"] = True
    fake_qtimer_state["fail_on_delete"] = True

    owner._schedule_once("refresh", 0, lambda: None)
    owner._single_shot_owned(0, lambda: None)

    # Should not raise
    owner._cleanup_owned_timers()

    assert owner._timers == {}
    assert owner._temp_timers == set()
