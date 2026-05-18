# src/napari_deeplabcut/ui/plots/plot_models.py
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectorySeries:
    individual: str
    bodypart: str
    x: np.ndarray
    y: np.ndarray
    color: object
    label: str


@dataclass(frozen=True)
class TrajectoryPlotState:
    df: pd.DataFrame | None
    series: tuple[TrajectorySeries, ...]
    frame_min: float
    frame_max: float
    image_height: float | None
