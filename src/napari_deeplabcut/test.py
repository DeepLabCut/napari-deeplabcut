# %% 
import napari
import napari_deeplabcut

import numpy as np
from skimage import data

viewer = napari.view_image(data.astronaut(), rgb=True)
points = np.array([[100, 100], [200, 200], [300, 100]])

points_layer = viewer.add_points(points, size=30)
# %%
