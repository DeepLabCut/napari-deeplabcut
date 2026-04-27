# src/napari_deeplabcut/core/sidecar.py
"""
Sidecar storage for folder-scoped napari-deeplabcut preferences.

This is intentionally non-invasive: DeepLabCut ignores unknown files in folders.
We store minimal, portable UI state (e.g. default scorer, trails display config)
to avoid repeated prompts and to restore per-folder preferences.

File name: .napari-deeplabcut.json
Location:  anchor folder (typically labeled-data/<video> folder)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from napari_deeplabcut.config.models import FolderUIState, TrailsDisplayConfig

logger = logging.getLogger(__name__)

_SIDECAR_NAME = ".napari-deeplabcut.json"
_SCHEMA_VERSION = 1


def sidecar_path(anchor: str | Path) -> Path:
    return Path(anchor) / _SIDECAR_NAME


# ---------------------------------------------------------------------
# Raw JSON I/O
# ---------------------------------------------------------------------
def _read_sidecar_json(anchor: str | Path) -> dict[str, Any]:
    """
    Read raw sidecar JSON.

    Returns an empty dict if the file is missing or unreadable.
    """
    p = sidecar_path(anchor)
    if not p.exists() or not p.is_file():
        return {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        logger.debug("Failed to read sidecar JSON from %s", p, exc_info=True)
        return {}


def _write_sidecar_json(anchor: str | Path, payload: dict[str, Any]) -> None:
    """
    Write raw sidecar JSON atomically-ish.

    Notes
    -----
    - Ensures parent folder exists.
    - Writes to a temp file in the same directory, then replaces.
    """
    p = sidecar_path(anchor)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(payload)
    payload.setdefault("schema_version", _SCHEMA_VERSION)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(p.parent),
        prefix=p.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")

    tmp_path.replace(p)


# ---------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------
def _migrate_sidecar_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate raw sidecar payload to the current schema version.

    This function must be:
    - pure
    - deterministic
    - tolerant of missing/partial/legacy input

    Migration strategy
    ------------------
    We currently support:
    - legacy flat payloads like:
        {"default_scorer": "John"}
      or
        {"schema_version": 1, "default_scorer": "John"}

    Since schema_version=1 already matches the current structure and the new
    `trails` field is optional via model defaults, migration is currently light.

    Keep migrations incremental:
    - v1 -> v2
    - v2 -> v3
    Never skip logic inline in callers.
    """
    if not isinstance(payload, dict):
        return {"schema_version": _SCHEMA_VERSION}

    out = dict(payload)
    version = out.get("schema_version", 1)

    # Future-proof migration chain template:
    while version < _SCHEMA_VERSION:
        if version == 1:
            # Example for a future v2 migration:
            # out = _migrate_v1_to_v2(out)
            # version = 2
            break
        # elif version == 2:
        #     out = _migrate_v2_to_v3(out)
        #     version = 3
        # ...
        else:
            logger.debug("Unknown sidecar schema_version=%r; falling back conservatively.", version)
            break

    out["schema_version"] = _SCHEMA_VERSION
    return out


# ---------------------------------------------------------------------
# Typed API
# ---------------------------------------------------------------------
def read_sidecar_state(anchor: str | Path) -> FolderUIState:
    """
    Read, migrate, and validate folder UI state from sidecar.

    Returns defaults on missing/invalid data.
    """
    raw = _read_sidecar_json(anchor)
    migrated = _migrate_sidecar_payload(raw)

    try:
        state = FolderUIState.model_validate(migrated)
    except ValidationError:
        logger.debug("Invalid sidecar payload at %s", sidecar_path(anchor), exc_info=True)
        state = FolderUIState()

    # Ensure schema_version is always normalized on read.
    if state.schema_version != _SCHEMA_VERSION:
        state = state.model_copy(update={"schema_version": _SCHEMA_VERSION})

    return state


def write_sidecar_state(anchor: str | Path, state: FolderUIState) -> None:
    """
    Validate and write folder UI state to sidecar.
    """
    normalized = state.model_copy(update={"schema_version": _SCHEMA_VERSION})
    _write_sidecar_json(anchor, normalized.model_dump(exclude_none=True))


def update_sidecar_state(anchor: str | Path, **patches: Any) -> FolderUIState:
    """
    Read existing state, apply a shallow patch, validate, and persist.

    Returns the updated normalized state.
    """
    current = read_sidecar_state(anchor)
    updated = current.model_copy(update=patches)
    write_sidecar_state(anchor, updated)
    return updated


# ---------------------------------------------------------------------
# Focused convenience helpers
# ---------------------------------------------------------------------
def get_default_scorer(anchor: str | Path) -> str | None:
    """
    Return default scorer stored in sidecar, if present and non-empty.
    """
    scorer = read_sidecar_state(anchor).default_scorer
    if isinstance(scorer, str) and scorer.strip():
        return scorer.strip()
    return None


def set_default_scorer(anchor: str | Path, scorer: str) -> None:
    """
    Persist a non-empty scorer to sidecar.
    """
    scorer = (scorer or "").strip()
    if not scorer:
        raise ValueError("default_scorer must be non-empty")

    update_sidecar_state(anchor, default_scorer=scorer)


def get_trails_config(anchor: str | Path) -> TrailsDisplayConfig:
    """
    Return folder-scoped trails display config.

    If absent, returns model defaults.
    """
    return read_sidecar_state(anchor).trails


def set_trails_config(anchor: str | Path, cfg: TrailsDisplayConfig) -> None:
    """
    Persist folder-scoped trails display config.
    """
    if not isinstance(cfg, TrailsDisplayConfig):
        cfg = TrailsDisplayConfig.model_validate(cfg)

    update_sidecar_state(anchor, trails=cfg)


def update_trails_config(anchor: str | Path, **fields: Any) -> TrailsDisplayConfig:
    """
    Patch the folder-scoped trails config and persist it.

    Example
    -------
    update_trails_config(anchor, tail_length=100, tail_width=4.0)
    """
    state = read_sidecar_state(anchor)
    updated_trails = state.trails.model_copy(update=fields)
    updated_state = state.model_copy(update={"trails": updated_trails})
    write_sidecar_state(anchor, updated_state)
    return updated_state.trails
