# napari-deeplabcut

[![License](https://img.shields.io/pypi/l/napari-deeplabcut.svg?color=green)](https://github.com/DeepLabCut/napari-deeplabcut/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-deeplabcut.svg?color=green)](https://pypi.org/project/napari-deeplabcut)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-deeplabcut.svg?color=green)](https://python.org)
[![tests](https://github.com/DeepLabCut/napari-deeplabcut/workflows/tests/badge.svg)](https://github.com/DeepLabCut/napari-deeplabcut/actions)
[![codecov](https://codecov.io/gh/DeepLabCut/napari-deeplabcut/branch/main/graph/badge.svg)](https://codecov.io/gh/DeepLabCut/napari-deeplabcut)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-deeplabcut)](https://napari-hub.org/plugins/napari-deeplabcut)

napari + DeepLabCut annotation tool

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari-deeplabcut` via [pip]:

    pip install napari-deeplabcut



To install latest development version :

    pip install git+https://github.com/DeepLabCut/napari-deeplabcut.git


## Usage

To use the plugin, please run:

    napari

Then, activate the plugin in Plugins > napari-deeplabcut: Keypoint controls.

All accepted files (config.yaml, images, h5 data files) can be loaded
either by dropping them directly onto the canvas or via the File menu.

The easiest way to get started is to drop a folder (typically a folder from within a DeepLabCut's `labeled-data` directory), and, if labeling from scratch, drop the corresponding `config.yaml` to automatically add a `Points layer` and populate the dropdown menus.

**Tools & shortcuts are:**

- `2` and `3`, to easily switch between labeling and selection mode
- `4`, to enable pan & zoom (which is achieved using the mouse wheel or finger scrolling on the Trackpad)
- `M`, to cycle through regular (sequential), quick, and cycle annotation mode (see the description [here](https://github.com/DeepLabCut/DeepLabCut-label/blob/ee71b0e15018228c98db3b88769e8a8f4e2c0454/dlclabel/layers.py#L9-L19))
- `E`, to enable edge coloring (by default, if using this in refinement GUI mode, points with a confidence lower than 0.6 are marked
in red)
- `F`, to toggle between animal and body part color scheme.
- `backspace` to delete a point.
- Check the box "display text" to show the label names on the canvas.
- To move to another folder, be sure to save (Ctrl+S), then delete the layers, and re-drag/drop the next folder.


### Save Layers

Annotations and segmentations are saved with `File > Save Selected Layer(s)...`
(or its shortcut `Ctrl+S`). For convenience, the save file dialog opens automatically into the folder containing your images or your h5 data file.
- As a reminder, DLC will only use the H5 file; so be sure if you open already labeled images you save/overwrite the H5. If you label from scratch, you should save the file as `CollectedData_YourName.h5`
- Note, before saving a layer, make sure the points layer is selected. If the user clicked on the image(s) layer first, does `Save As`, then closes the window, any labeling work during that session will be lost!


## Workflow

Suggested workflows, depending on the image folder contents:

1. **Labeling from scratch** – the image folder does not contain `CollectedData_<ScorerName>.h5` file.

    Open *napari* as described in [Usage](#usage) and open an image folder together with the DeepLabCut project's `config.yaml`.
    The image folder creates an *image layer* with the images to label.
    Supported image formats are: `jpg`, `jpeg`, `png`.
    The `config.yaml` file creates a *Points layer*, which holds metadata (such as keypoints read from the config file) necessary for labeling.
    Select the *Points layer* in the layer list (lower left pane on the GUI) and click on the *+*-symbol in the layer controls menu (upper left pane) to start labeling.
    The current keypoint can be viewed/selected in the keypoints dropdown menu (bottom pane).
    The slider below the displayed image (or the left/right arrow keys) allows selecting the image to label.

    To save the labeling progress refer to [Save Layers](#save-layers).
    If the console window does not display any errors, the image folder should now contain a `CollectedData_<ScorerName>.h5` file.
    (Note: For convenience, a CSV file with the same name is also saved.)

1. **Resuming labeling** – the image folder contains a `CollectedData_<ScorerName>.h5` file.

    Open *napari* and open an image folder (which needs to contain a `CollectedData_<ScorerName>.h5` file).
    In this case, it is not necessary to open the DLC project's `config.yaml` file, as all necessary metadata is read from the `h5` data file.

    Saving works as described in *1*.

1. **Refining labels** – the image folder contains a `machinelabels-iter<#>.h5` file.

    The process is analog to *2*.

### Labeling multiple image folders

Labeling multiple image folders has to be done in sequence; i.e., only one image folder can be opened at a time.
After labeling the images of a particular folder is done and the associated *Points layer* has been saved, *all* layers should be removed from the layers list (lower left pane on the GUI) by selecting them and clicking on the trashcan icon.
Now, another image folder can be labeled, following the process described in *1*, *2*, or *3*, depending on the particular image folder.



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-deeplabcut" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/DeepLabCut/napari-deeplabcut/issues
