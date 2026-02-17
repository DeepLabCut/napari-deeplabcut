from __future__ import annotations

from pathlib import Path

import pytest
from napari_deeplabcut.core.sidecar import (
    get_default_scorer,
    read_sidecar,
    set_default_scorer,
    sidecar_path,
)


def test_sidecar_missing_returns_empty(tmp_path: Path):
    assert read_sidecar(tmp_path) == {}
    assert get_default_scorer(tmp_path) is None
    assert sidecar_path(tmp_path).name == ".napari-deeplabcut.json"


def test_sidecar_set_and_get_default_scorer(tmp_path: Path):
    set_default_scorer(tmp_path, "Alice")
    assert get_default_scorer(tmp_path) == "Alice"

    data = read_sidecar(tmp_path)
    assert data["default_scorer"] == "Alice"
    assert "schema_version" in data
    assert isinstance(data["schema_version"], int)


def test_sidecar_rejects_empty_scorer(tmp_path: Path):
    with pytest.raises(ValueError):
        set_default_scorer(tmp_path, "")

    with pytest.raises(ValueError):
        set_default_scorer(tmp_path, "   ")
