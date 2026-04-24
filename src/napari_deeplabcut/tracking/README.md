# Point-tracker assisted labeling

**EXPERIMENTAL FEATURE**

This feature allows to speed up the labeling process by using a simple point tracker.
Please note this is still a very basic implementation, and **it WILL overwrite existing annotations**
in your napari-DLC Points layer!

Always **backup your data** before using this feature, and **try it on a copy of your data first.**
We cannot be held responsible for any lost annotations!

The current intended workflow would be to annotate a single frame, use the tracker to propagate annotations,
and manually correct any mistakes before saving.

Based on interest, we may polish the user experience and add more advanced tracking algorithms in the future.

**Basic usage:**

- The tracking widget is opened via the "Plugin > napari-deeplabcut: Tracking controls" menu.
- In the layer selection lists, select both the video layer and the Points layer to be used for tracking.
- Select the starting frame for tracking by moving the viewer slider to the desired frame.
- Select how many frames you want to track forward and backward (relative to current frame or in absolute terms, termed respectively Rel and Abs.).
- Use the track forward/backward/both buttons to run the tracker.

**Key bindings:**

- Tracking Controls
  - **`l`** → Track **forward**
  - **`k`** → Track **forward (to end)**
  - **`h`** → Track **backward**
  - **`j`** → Track **backward (to end)**

- Frame Navigation
  - **`i`** → Move **forward one frame**
  - **`u`** → Move **backward one frame**

**Known issues:**
- After several runs, keypoint attributions may get shuffled. Do not run the tracker several times without checking the results in between.
- Can only be run on plugin-controlled Points layers. Creating a new Points layer manually will not allow tracking on it.
