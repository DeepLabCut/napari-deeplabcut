# src/napari_deeplabcut/widget_factory.py
from __future__ import annotations

from ._widgets import KeypointControls


def get_existing_keypoint_controls(viewer) -> KeypointControls | None:
    wdg = KeypointControls.get_existing(viewer)
    if wdg is None:
        return None
    if not KeypointControls.is_docked(viewer, wdg):
        return None
    return wdg


def get_or_create_keypoint_controls(viewer) -> KeypointControls:
    return KeypointControls(viewer)
