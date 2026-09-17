# src/napari_deeplabcut/widget_factory.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._widgets import KeypointControls


def get_existing_keypoint_controls(viewer) -> KeypointControls | None:
    from ._widgets import KeypointControls

    wdg = KeypointControls.get_existing(viewer)
    if wdg is None:
        return None
    if not KeypointControls.is_docked(viewer, wdg):
        return None
    return wdg
