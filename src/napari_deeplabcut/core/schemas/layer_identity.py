# src/napari_deeplabcut/core/schemas/layer_identity.py
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

DLC_LAYER_ROLE_KEY = "dlc_layer_role"
DLC_SAVE_BEHAVIOR_KEY = "dlc_save_behavior"
DLC_SOURCE_CONFIG_KEY = "dlc_source_config"


class LayerRole(str, Enum):
    """Metadata-level DLC layer roles.

    These roles describe how a layer participates in the DLC workflow.
    """

    CONFIG_PLACEHOLDER = "dlc_config_placeholder"
    DLC_ANNOTATION = "dlc_annotation"
    FRAMES = "frames"
    TRACKING_RESULT = "tracking_result"


class FrameLayerType(str, Enum):
    """Frame source subtypes used to further specify a FRAMES layer."""

    IMAGES = "image"
    VIDEO = "video"


class LayerSaveBehavior(str, Enum):
    """Save semantics for Points layers.

    PLUGIN_MANAGED:
        The layer represents a normal editable annotation save scope.
        Missing keypoints may represent intentional deletions, depending on
        the existing destination file and preflight conflict report.

    NAPARI_MANAGED:
        Napari owns the save workflow for this layer.
        No plugin code should operate on the save process.
    """

    PLUGIN_MANAGED = "plugin_managed"
    NAPARI_MANAGED = "napari_managed"


# ---- Internal normalization -------------------------------------------------------


def _enum_value(value: Any) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return value
    return None


# ---- Metadata accessors -----------------------------------------------------------


def get_layer_role_from_metadata(metadata: dict[str, Any] | None) -> LayerRole | None:
    raw = _enum_value((metadata or {}).get(DLC_LAYER_ROLE_KEY))
    if raw is None:
        return None

    try:
        return LayerRole(raw)
    except ValueError:
        return None


def get_save_behavior_from_metadata(
    metadata: dict[str, Any] | None,
) -> LayerSaveBehavior | None:
    raw = _enum_value((metadata or {}).get(DLC_SAVE_BEHAVIOR_KEY))
    if raw is None:
        return None

    try:
        return LayerSaveBehavior(raw)
    except ValueError:
        return None


def get_effective_save_behavior_from_metadata(
    metadata: dict[str, Any] | None,
) -> LayerSaveBehavior | None:
    """Return explicit or role-implied save ownership.

    Rules:
    - explicit save behavior wins;
    - DLC_ANNOTATION defaults to PLUGIN_MANAGED;
    - TRACKING_RESULT defaults to NAPARI_MANAGED;
    - other roles have no default save behavior here.

    CONFIG_PLACEHOLDER does not default to PLUGIN_MANAGED. Empty config
    placeholders are lifecycle objects; their promotion belongs in the manager.
    """
    explicit = get_save_behavior_from_metadata(metadata)
    if explicit is not None:
        return explicit

    role = get_layer_role_from_metadata(metadata)
    if role is LayerRole.DLC_ANNOTATION:
        return LayerSaveBehavior.PLUGIN_MANAGED

    if role is LayerRole.TRACKING_RESULT:
        return LayerSaveBehavior.NAPARI_MANAGED

    return None


def get_source_config_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    value = (metadata or {}).get(DLC_SOURCE_CONFIG_KEY)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    return None


# ---- Metadata predicates ----------------------------------------------------------


def has_layer_role_metadata(
    metadata: dict[str, Any] | None,
    role: LayerRole,
) -> bool:
    return get_layer_role_from_metadata(metadata) is role


# ---- Metadata mutators ------------------------------------------------------------


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


def clear_save_behavior_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    md = dict(metadata or {})
    md.pop(DLC_SAVE_BEHAVIOR_KEY, None)
    return md


def plugin_manages_save_from_metadata(metadata: dict[str, Any] | None) -> bool:
    return get_effective_save_behavior_from_metadata(metadata) is LayerSaveBehavior.PLUGIN_MANAGED


def napari_manages_save_from_metadata(metadata: dict[str, Any] | None) -> bool:
    return get_effective_save_behavior_from_metadata(metadata) is LayerSaveBehavior.NAPARI_MANAGED


def validate_plugin_managed_dlc_annotation_metadata(
    metadata: dict[str, Any] | None,
) -> None:
    """Raise if metadata is not eligible for the plugin DLC annotation writer.

    This is metadata-only. UI/session routing belongs in the lifecycle manager,
    but the writer and save preflight also need a hard safety gate because direct
    napari writer calls can bypass the UI.
    """
    role = get_layer_role_from_metadata(metadata)
    behavior = get_effective_save_behavior_from_metadata(metadata)

    if role is LayerRole.TRACKING_RESULT:
        raise ValueError(
            "Tracking-result layers cannot be saved through the napari-deeplabcut "
            "DLC annotation writer. Merge them into a DLC annotation layer first, "
            "or use napari's generic layer save."
        )

    if behavior is LayerSaveBehavior.NAPARI_MANAGED:
        raise ValueError(
            "This layer is marked as napari-managed and cannot be saved through "
            "the napari-deeplabcut DLC annotation writer."
        )

    if role is not LayerRole.DLC_ANNOTATION:
        raise ValueError("Only DLC annotation layers can be saved through the napari-deeplabcut DLC annotation writer.")

    if behavior is not LayerSaveBehavior.PLUGIN_MANAGED:
        raise ValueError("This DLC annotation layer is not marked as plugin-managed.")


# ---- Annotation layer -------------------------------------------------------------


def tag_dlc_annotation_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Mark metadata for a Points layer that represents DLC annotations.

    DLC annotation layers participate in the DLC annotation save workflow and
    use regular destructive-aware save semantics.
    """
    return set_layer_identity_metadata(
        metadata,
        role=LayerRole.DLC_ANNOTATION,
        save_behavior=LayerSaveBehavior.PLUGIN_MANAGED,
    )


# ---- Frames layer -----------------------------------------------------------------


def tag_frames_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Mark metadata for an image/video layer that acts as DLC frame context.

    The concrete frame source type is stored separately, currently in the
    existing session_role field used by the reader/manager.
    """
    return set_layer_role_metadata(metadata, role=LayerRole.FRAMES)


# ---- Config placeholder layer -----------------------------------------------------


def tag_config_placeholder_metadata(
    metadata: dict[str, Any] | None,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """Mark metadata for a Points layer created from config.yaml.

    This is a temporary lifecycle identity. The layer starts as an empty
    config-only placeholder carrying config-derived bodypart/header and
    presentation metadata.

    The lifecycle manager may later either:

    - merge this metadata into an existing annotation layer and remove the
      placeholder; or
    - keep/promote the placeholder as the scaffold for labeling from scratch.

    The placeholder should not be treated as an authoritative annotation layer
    while it is still empty and config-only.
    """
    md = set_layer_role_metadata(metadata, role=LayerRole.CONFIG_PLACEHOLDER)

    # Do not write PARTIAL_UPDATE here by default. Config-placeholder specialness
    # should be lifecycle/session interpretation, not a sticky save behavior.
    md.pop(DLC_SAVE_BEHAVIOR_KEY, None)

    md[DLC_SOURCE_CONFIG_KEY] = str(Path(config_path).expanduser().resolve(strict=False))
    return md


def promote_config_placeholder_to_annotation_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Promote a config-derived placeholder into a normal DLC annotation layer.

    This should be used once the layer has concrete frame/save context and is no
    longer merely a config-only placeholder.
    """
    return set_layer_identity_metadata(
        metadata,
        role=LayerRole.DLC_ANNOTATION,
        save_behavior=LayerSaveBehavior.PLUGIN_MANAGED,
    )


# ---- Tracking result layer --------------------------------------------------------


def tag_tracking_result_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Mark metadata for a Points layer produced by a tracking workflow.

    Tracking results are merge sources in the DLC workflow. They should not be
    routed through direct DLC annotation saving.
    """
    return set_layer_identity_metadata(
        metadata,
        role=LayerRole.TRACKING_RESULT,
        save_behavior=LayerSaveBehavior.NAPARI_MANAGED,
    )
