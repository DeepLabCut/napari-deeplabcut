# src/napari_deeplabcut/core/layer_lifecycle/merge.py
from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class PlaceholderConfigAction(str, Enum):
    APPLY_TO_CURRENT = "apply_to_current"
    KEEP_AS_SEPARATE_LAYER = "keep_as_separate_layer"
    CANCEL = "cancel"


class PlaceholderConfigDecisionProvider(Protocol):
    def resolve_placeholder_config_action(
        self,
        *,
        placeholder_layer: Any,
        managed_layers: tuple[Any, ...],
        added_keypoints: tuple[str, ...],
        message: str,
    ) -> PlaceholderConfigAction: ...
