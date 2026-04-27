# src/napari_deeplabcut/core/layer_lifecycle/merge.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class MergeDisposition(str, Enum):
    KEEP_BOTH = "keep_both"
    HIDE_EXISTING = "hide_existing"
    HIDE_NEW = "hide_new"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class MergeDecisionRequest:
    new_layer: Any
    existing_layers: tuple[Any, ...]
    added_keypoints: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class MergeDecisionResult:
    disposition: MergeDisposition


class MergeDecisionProvider(Protocol):
    def resolve_merge(self, request: MergeDecisionRequest) -> MergeDecisionResult: ...
