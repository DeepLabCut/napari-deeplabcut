# src/napari_deeplabcut/core/schemas/layer_identity.py
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

DLC_LAYER_ROLE_KEY = "dlc_layer_role"
DLC_SAVE_BEHAVIOR_KEY = "dlc_save_behavior"
DLC_SOURCE_CONFIG_KEY = "dlc_source_config"


class LayerRole(str, Enum):
    CONFIG_PLACEHOLDER = "dlc_config_placeholder"
    DLC_ANNOTATION = "dlc_annotation"
    FRAMES = "frames"
    TRACKING_RESULT = "tracking_result"


class FrameLayerType(str, Enum):
    """Annotation subtypes, used to further specify the role of a FRAMES layer."""

    IMAGES = "image"
    VIDEO = "video"


class LayerSaveBehavior(str, Enum):
    """
    Save semantics for Points layers.

    REGULAR:
        The layer represents a normal annotation save. Missing keypoints may
        represent intentional deletions, depending on the existing destination
        file and preflight conflict report.

    PARTIAL_UPDATE:
        Missing keypoints must not be interpreted as deletions. Only explicitly
        present keypoints should be written/merged.
    """

    REGULAR = "regular"
    PARTIAL_UPDATE = "partial_update"


# ---- Metadata accessors / role helpers --------------------------------------------


def _enum_value(value: Any) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return value
    return None


def get_layer_role_from_metadata(metadata: dict[str, Any] | None) -> LayerRole | None:
    raw = _enum_value((metadata or {}).get(DLC_LAYER_ROLE_KEY))
    if raw is None:
        return None

    try:
        return LayerRole(raw)
    except ValueError:
        return None


def get_save_behavior_from_metadata(metadata: dict[str, Any] | None) -> LayerSaveBehavior | None:
    raw = _enum_value((metadata or {}).get(DLC_SAVE_BEHAVIOR_KEY))
    if raw is None:
        return None

    try:
        return LayerSaveBehavior(raw)
    except ValueError:
        return None


def set_layer_role_metadata(
    metadata: dict[str, Any] | None,
    *,
    role: LayerRole,
) -> dict[str, Any]:
    md = dict(metadata or {})
    md[DLC_LAYER_ROLE_KEY] = role.value
    return md


def set_layer_identity_metadata(
    metadata: dict[str, Any] | None,
    *,
    role: LayerRole,
    save_behavior: LayerSaveBehavior | None = None,
) -> dict[str, Any]:
    md = set_layer_role_metadata(metadata, role=role)

    if save_behavior is not None:
        md[DLC_SAVE_BEHAVIOR_KEY] = save_behavior.value

    return md


def save_behavior_disallows_deletions(metadata: dict[str, Any] | None) -> bool:
    return get_save_behavior_from_metadata(metadata) is LayerSaveBehavior.PARTIAL_UPDATE


# ---- Annotation layer ------------------------------------------------------------
def tag_dlc_annotation_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Mark metadata for a Points layer that represents a DLC annotation layer.
    """
    return set_layer_identity_metadata(
        metadata,
        role=LayerRole.DLC_ANNOTATION,
        save_behavior=LayerSaveBehavior.REGULAR,
    )


# ---- Frames layer ------------------------------------------------------------
def tag_frames_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Mark metadata for an image/video layer that acts as the DLC session frame source.

    The concrete frame source type is stored separately, currently in the
    existing session_role field used by the reader/manager.
    """
    return set_layer_role_metadata(metadata, role=LayerRole.FRAMES)


# ---- Config placeholder layer ------------------------------------------------
def tag_config_placeholder_metadata(
    metadata: dict[str, Any] | None,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Mark metadata for a Points layer created from config.yaml.

    This is a temporary lifecycle identity. The layer starts as a config-only
    placeholder and may later be merged into an existing DLC annotation layer
    or promoted/reinterpreted as a real annotation layer once it has concrete
    image/save scope.
    """
    md = set_layer_identity_metadata(
        metadata,
        role=LayerRole.CONFIG_PLACEHOLDER,
        save_behavior=LayerSaveBehavior.PARTIAL_UPDATE,
    )
    md[DLC_SOURCE_CONFIG_KEY] = str(Path(config_path).expanduser().resolve(strict=False))
    return md


# ---- Tracking result layer ----------------------------------------------------
def tag_tracking_result_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Mark metadata for a Points layer produced by a tracking workflow.
    """
    return set_layer_identity_metadata(
        metadata,
        role=LayerRole.TRACKING_RESULT,
    )
