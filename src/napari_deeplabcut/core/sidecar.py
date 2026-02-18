# src/napari_deeplabcut/core/sidecar.py
"""
Sidecar storage for folder-scoped napari-deeplabcut preferences.

This is intentionally non-invasive: DeepLabCut ignores unknown files in folders.
We store minimal, portable info (e.g., default scorer) to avoid repeated prompts
when config.yaml is missing.

File name: .napari-deeplabcut.json
Location:  anchor folder (typically labeled-data/<video> folder)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SIDECAR_NAME = ".napari-deeplabcut.json"
_SCHEMA_VERSION = 1


def sidecar_path(anchor: str | Path) -> Path:
    return Path(anchor) / _SIDECAR_NAME


def read_sidecar(anchor: str | Path) -> dict[str, Any]:
    """Read sidecar JSON. Returns empty dict on missing/invalid."""
    p = sidecar_path(anchor)
    if not p.exists() or not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read sidecar %s", p, exc_info=True)
        return {}


def write_sidecar(anchor: str | Path, data: dict[str, Any]) -> None:
    """Write sidecar JSON atomically-ish."""
    p = sidecar_path(anchor)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(data)
    payload.setdefault("schema_version", _SCHEMA_VERSION)

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def get_default_scorer(anchor: str | Path) -> str | None:
    """Return default scorer stored in sidecar, if present and non-empty."""
    data = read_sidecar(anchor)
    scorer = data.get("default_scorer")
    if isinstance(scorer, str) and scorer.strip():
        return scorer.strip()
    return None


def set_default_scorer(anchor: str | Path, scorer: str) -> None:
    """Persist a non-empty scorer to sidecar."""
    scorer = (scorer or "").strip()
    if not scorer:
        raise ValueError("default_scorer must be non-empty")

    data = read_sidecar(anchor)
    data["schema_version"] = _SCHEMA_VERSION
    data["default_scorer"] = scorer
    write_sidecar(anchor, data)