from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_deeplabcut.config.models import FolderUIState, TrailsDisplayConfig
from napari_deeplabcut.core.sidecar import (
    _migrate_sidecar_payload,
    get_default_scorer,
    get_trails_config,
    read_sidecar_state,
    set_default_scorer,
    set_trails_config,
    sidecar_path,
    update_sidecar_state,
    update_trails_config,
    write_sidecar_state,
)


def test_sidecar_path_joins_anchor_and_filename(tmp_path: Path):
    p = sidecar_path(tmp_path)
    assert p == tmp_path / ".napari-deeplabcut.json"


def test_read_sidecar_state_missing_file_returns_defaults(tmp_path: Path):
    state = read_sidecar_state(tmp_path)

    assert isinstance(state, FolderUIState)
    assert state.schema_version == 1
    assert state.default_scorer is None
    assert state.trails == TrailsDisplayConfig()


def test_read_sidecar_state_invalid_json_returns_defaults(tmp_path: Path):
    p = sidecar_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")

    state = read_sidecar_state(tmp_path)

    assert state.schema_version == 1
    assert state.default_scorer is None
    assert state.trails == TrailsDisplayConfig()


def test_read_sidecar_state_non_dict_json_returns_defaults(tmp_path: Path):
    p = sidecar_path(tmp_path)
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    state = read_sidecar_state(tmp_path)
    assert state == FolderUIState()


def test_migrate_sidecar_payload_normalizes_schema_version():
    payload = {"default_scorer": "John"}
    migrated = _migrate_sidecar_payload(payload)

    assert migrated["schema_version"] == 1
    assert migrated["default_scorer"] == "John"


def test_write_and_read_sidecar_state_roundtrip(tmp_path: Path):
    state = FolderUIState(
        default_scorer="John",
        trails=TrailsDisplayConfig(
            tail_length=70,
            head_length=10,
            tail_width=4.25,
            opacity=0.6,
            blending="opaque",
            visible=False,
        ),
    )

    write_sidecar_state(tmp_path, state)
    reloaded = read_sidecar_state(tmp_path)

    assert reloaded.schema_version == 1
    assert reloaded.default_scorer == "John"
    assert reloaded.trails == state.trails


def test_update_sidecar_state_shallow_patch_preserves_other_fields(tmp_path: Path):
    initial = FolderUIState(
        default_scorer="John",
        trails=TrailsDisplayConfig(tail_length=80, head_length=20, tail_width=5.0),
    )
    write_sidecar_state(tmp_path, initial)

    updated = update_sidecar_state(tmp_path, default_scorer="Jane")

    assert updated.default_scorer == "Jane"
    assert updated.trails.tail_length == 80
    assert updated.trails.head_length == 20
    assert updated.trails.tail_width == 5.0


def test_get_and_set_default_scorer_roundtrip(tmp_path: Path):
    assert get_default_scorer(tmp_path) is None

    set_default_scorer(tmp_path, "John")
    assert get_default_scorer(tmp_path) == "John"


def test_set_default_scorer_rejects_empty(tmp_path: Path):
    with pytest.raises(ValueError, match="default_scorer must be non-empty"):
        set_default_scorer(tmp_path, "   ")


def test_get_trails_config_returns_defaults_when_absent(tmp_path: Path):
    cfg = get_trails_config(tmp_path)
    assert cfg == TrailsDisplayConfig()


def test_set_trails_config_roundtrip(tmp_path: Path):
    cfg = TrailsDisplayConfig(
        tail_length=90,
        head_length=15,
        tail_width=3.75,
        opacity=0.85,
        blending="minimum",
        visible=False,
    )

    set_trails_config(tmp_path, cfg)
    reloaded = get_trails_config(tmp_path)

    assert reloaded == cfg


def test_set_trails_config_accepts_dict_payload(tmp_path: Path):
    set_trails_config(
        tmp_path,
        {
            "tail_length": 65,
            "head_length": 11,
            "tail_width": 2.5,
            "opacity": 0.7,
            "blending": "opaque",
            "visible": False,
        },
    )

    cfg = get_trails_config(tmp_path)
    assert cfg == TrailsDisplayConfig(
        tail_length=65,
        head_length=11,
        tail_width=2.5,
        opacity=0.7,
        blending="opaque",
        visible=False,
    )


def test_update_trails_config_patches_only_requested_fields(tmp_path: Path):
    set_trails_config(
        tmp_path,
        TrailsDisplayConfig(
            tail_length=50,
            head_length=50,
            tail_width=6.0,
            opacity=1.0,
            blending="translucent",
            visible=True,
        ),
    )

    updated = update_trails_config(tmp_path, tail_length=120, visible=False)

    assert updated == TrailsDisplayConfig(
        tail_length=120,
        head_length=50,
        tail_width=6.0,
        opacity=1.0,
        blending="translucent",
        visible=False,
    )

    reloaded = get_trails_config(tmp_path)
    assert reloaded == updated


def test_sidecar_json_written_with_schema_version(tmp_path: Path):
    set_default_scorer(tmp_path, "John")

    raw = json.loads(sidecar_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["default_scorer"] == "John"


def test_read_sidecar_state_old_payload_gets_default_trails(tmp_path: Path):
    sidecar_path(tmp_path).write_text(
        json.dumps({"default_scorer": "John"}),
        encoding="utf-8",
    )

    state = read_sidecar_state(tmp_path)
    assert state.schema_version == 1
    assert state.default_scorer == "John"
    assert state.trails == TrailsDisplayConfig()
