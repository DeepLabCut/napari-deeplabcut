# src/napari_deeplabcut/_tests/e2e/conftest.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qtpy.QtWidgets import QInputDialog, QMessageBox


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark all tests collected from this folder as @pytest.mark.e2e.
    """
    here = Path(__file__).resolve().parent

    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue

        if here in item_path.parents or item_path == here:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True)
def _auto_accept_qmessagebox(monkeypatch):
    """Prevent any QMessageBox modal dialogs from blocking tests."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)


@pytest.fixture
def inputdialog(monkeypatch):
    """
    Controller for QInputDialog.getText used by promotion-to-GT first save prompt.
    """
    state = {"value": "Alice", "ok": True, "calls": 0, "forbid": False}

    def _fake_getText(*args, **kwargs):
        state["calls"] += 1
        if state["forbid"]:
            raise AssertionError("QInputDialog.getText was called but forbid=True")
        return state["value"], state["ok"]

    monkeypatch.setattr(QInputDialog, "getText", _fake_getText)

    class Controller:
        @property
        def calls(self):
            return state["calls"]

        def set(self, value: str, ok: bool = True):
            state["value"] = value
            state["ok"] = ok
            return self

        def forbid(self):
            state["forbid"] = True
            return self

    return Controller()


@pytest.fixture(autouse=True)
def overwrite_confirm(monkeypatch):
    """
    Control the overwrite-confirmation path used by the writer.

    NOTE: the io module imports/uses maybe_confirm_overwrite at module scope,
    so we patch napari_deeplabcut.core.io.maybe_confirm_overwrite.

    API:
      - forbid(): fail test if confirmation is requested for a real overwrite
      - cancel(): return False (simulate user cancel)
      - capture(): record calls and return True
      - set_result(bool): return chosen bool
      - reset_calls(): clear recorded calls
    """
    calls = []
    state = {"mode": "always_true", "result": True}

    def _metadata_keys(meta: Any) -> list[str] | None:
        """Support dict metadata or pydantic models with model_dump()."""
        if meta is None:
            return None
        if isinstance(meta, dict):
            return sorted(list(meta.keys()))
        dump = getattr(meta, "model_dump", None)
        if callable(dump):
            try:
                d = dump()
                if isinstance(d, dict):
                    return sorted(list(d.keys()))
            except Exception:
                return None
        return None

    def _patched_maybe_confirm_overwrite(metadata, key_conflict):
        n_pairs = int(key_conflict.to_numpy().sum()) if hasattr(key_conflict, "to_numpy") else 0
        n_images = int(key_conflict.any(axis=1).to_numpy().sum()) if hasattr(key_conflict, "any") else None

        calls.append(
            {
                "metadata_keys": _metadata_keys(metadata),
                "n_pairs": n_pairs,
                "n_images": n_images,
            }
        )

        if state["mode"] == "forbid" and n_pairs > 0:
            raise AssertionError("maybe_confirm_overwrite was called unexpectedly for a real overwrite (n_pairs > 0).")

        return state["result"]

    import napari_deeplabcut.core.io as io_mod

    monkeypatch.setattr(io_mod, "maybe_confirm_overwrite", _patched_maybe_confirm_overwrite)

    class Controller:
        @property
        def calls(self):
            return calls

        def forbid(self):
            state["mode"] = "forbid"
            state["result"] = True
            return self

        def cancel(self):
            state["mode"] = "capture"
            state["result"] = False
            return self

        def capture(self):
            state["mode"] = "capture"
            state["result"] = True
            return self

        def set_result(self, value: bool):
            state["mode"] = "capture"
            state["result"] = bool(value)
            return self

        def reset_calls(self):
            calls.clear()
            return self

    return Controller()
