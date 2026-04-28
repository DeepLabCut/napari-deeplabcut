# src/napari_deeplabcut/widget_factory.py
from __future__ import annotations

from ._widgets import KeypointControls


def get_existing_keypoint_controls(viewer) -> KeypointControls | None:
    return KeypointControls.get_existing(viewer)


def get_or_create_keypoint_controls(viewer) -> KeypointControls:
    return KeypointControls(viewer)
