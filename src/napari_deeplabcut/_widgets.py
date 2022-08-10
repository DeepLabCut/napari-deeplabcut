import os
from collections import defaultdict
from math import ceil, log10
from types import MethodType
from typing import Optional, Sequence, Union

import numpy as np
from napari.layers import Image, Points, Shapes
from napari.layers.points._points_key_bindings import register_points_action
from napari.layers.utils import color_manager
from napari.utils.events import Event
from napari.utils.history import get_save_history, update_save_history
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut import keypoints
from napari_deeplabcut._reader import _load_config
from napari_deeplabcut._writer import _write_config, _write_image
from napari_deeplabcut.misc import to_os_dir_sep


def _get_and_try_preferred_reader(
    self,
    dialog,
    *args,
):
    try:
        self.viewer.open(
            dialog._current_file,
            plugin="napari-deeplabcut",
        )
    except ValueError:
        self.viewer.open(
            dialog._current_file,
            plugin="builtins",
        )


# Hack to avoid napari's silly variable type guess,
# where property is understood as continuous if
# there are more than 16 unique categories...
def guess_continuous(property):
    if issubclass(property.dtype.type, np.floating):
        return True
    else:
        return False


color_manager.guess_continuous = guess_continuous


# Hack to save a KeyPoints layer without showing the Save dialog
def _save_layers_dialog(self, selected=False):
    """Save layers (all or selected) to disk, using ``LayerList.save()``.
    Parameters
    ----------
    selected : bool
        If True, only layers that are selected in the viewer will be saved.
        By default, all layers are saved.
    """
    selected_layers = list(self.viewer.layers.selection)
    msg = ""
    if not len(self.viewer.layers):
        msg = "There are no layers in the viewer to save."
    elif selected and not len(selected_layers):
        msg = (
            "Please select one or more layers to save," '\nor use "Save all layers..."'
        )
    if msg:
        QMessageBox.warning(self, "Nothing to save", msg, QMessageBox.Ok)
        return
    if len(selected_layers) == 1 and isinstance(selected_layers[0], Points):
        self.viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
        self.viewer.status = "Data successfully saved"
    else:
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)
        filename, _ = dlg.getSaveFileName(
            caption=f'Save {"selected" if selected else "all"} layers',
            dir=hist[0],  # home dir by default
        )
        if filename:
            self.viewer.layers.save(filename, selected=selected)


class KeypointControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.on_insert)
        self.viewer.layers.events.removed.connect(self.on_remove)

        self.viewer.window.qt_viewer._get_and_try_preferred_reader = MethodType(
            _get_and_try_preferred_reader,
            self.viewer.window.qt_viewer,
        )

        self._label_mode = keypoints.LabelMode.default()

        # Hold references to the KeypointStores
        self._stores = {}

        # Storage for extra image metadata that are relevant to other layers.
        # These are updated anytime images are added to the Viewer
        # and passed on to the other layers upon creation.
        self._images_meta = dict()

        # Add some more controls
        self._layout = QVBoxLayout(self)
        self._menus = []

        self._video_group = self._form_video_action_menu()
        self._radio_group = self._form_mode_radio_buttons()

        # Substitute default menu action with custom one
        for action in self.viewer.window.file_menu.actions():
            if "save selected layer" in action.text().lower():
                action.triggered.disconnect()
                action.triggered.connect(
                    lambda: _save_layers_dialog(
                        self.viewer.window.qt_viewer,
                        selected=True,
                    )
                )
                break

    def _form_video_action_menu(self):
        group_box = QGroupBox("Video")
        layout = QVBoxLayout()
        extract_button = QPushButton("Extract frame")
        extract_button.clicked.connect(self._extract_single_frame)
        extract_button.setEnabled(False)
        layout.addWidget(extract_button)
        crop_button = QPushButton("Store crop coordinates")
        crop_button.clicked.connect(self._store_crop_coordinates)
        crop_button.setEnabled(False)
        layout.addWidget(crop_button)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)
        return extract_button, crop_button

    def _extract_single_frame(self, *args):
        layer = None
        for layer_ in self.viewer.layers:
            if isinstance(layer_, Image):
                layer = layer_
                break
        if layer is not None:
            ind = self.viewer.dims.current_step[0]
            frame = layer.data[ind]
            n_frames = layer.data.shape[0]
            name = f"img{str(ind).zfill(int(ceil(log10(n_frames))))}.png"
            output_path = os.path.join(layer.metadata["root"], name)
            _write_image(frame, str(output_path))

    def _store_crop_coordinates(self, *args):
        if not (project_path := self._images_meta.get("project")):
            return

        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                try:
                    ind = layer.shape_type.index("rectangle")
                except ValueError:
                    return
                bbox = layer.data[ind][:, 1:]
                h = self.viewer.dims.range[2][1]
                bbox[:, 0] = h - bbox[:, 0]
                bbox = np.clip(bbox, 0, a_max=None).astype(int)
                y1, x1 = bbox.min(axis=0)
                y2, x2 = bbox.max(axis=0)
                temp = {"crop": ", ".join(map(str, [x1, x2, y1, y2]))}
                config_path = os.path.join(project_path, "config.yaml")
                cfg = _load_config(config_path)
                cfg["video_sets"][
                    os.path.join(project_path, "videos", self._images_meta["name"])
                ] = temp
                _write_config(config_path, cfg)
                break

    def _form_dropdown_menus(self, store):
        menu = KeypointsDropdownMenu(store)
        self._menus.append(menu)
        layout = QVBoxLayout()
        layout.addWidget(menu)
        self._layout.addLayout(layout)

    def _form_mode_radio_buttons(self):
        group_box = QGroupBox("Labeling mode")
        layout = QHBoxLayout()
        group = QButtonGroup(self)
        for i, mode in enumerate(keypoints.LabelMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            btn.setToolTip(keypoints.TOOLTIPS[mode])
            group.addButton(btn, i)
            layout.addWidget(btn)
        group.button(1).setChecked(True)
        group_box.setLayout(layout)
        self._layout.addWidget(group_box)

        def _func():
            self.label_mode = group.checkedButton().text()

        group.buttonClicked.connect(_func)
        return group

    def _remap_frame_indices(self, layer):
        if not self._images_meta.get("paths"):
            return

        new_paths = [to_os_dir_sep(p) for p in self._images_meta["paths"]]
        paths = layer.metadata.get("paths")
        if paths is not None and np.any(layer.data):
            paths_map = dict(zip(range(len(paths)), map(to_os_dir_sep, paths)))
            # Discard data if there are missing frames
            missing = [i for i, path in paths_map.items() if path not in new_paths]
            if missing:
                if isinstance(layer.data, list):
                    inds_to_remove = [
                        i
                        for i, verts in enumerate(layer.data)
                        if verts[0, 0] in missing
                    ]
                else:
                    inds_to_remove = np.flatnonzero(np.isin(layer.data[:, 0], missing))
                layer.selected_data = inds_to_remove
                layer.remove_selected()
                for i in missing:
                    paths_map.pop(i)

            # Check now whether there are new frames
            temp = {k: new_paths.index(v) for k, v in paths_map.items()}
            data = layer.data
            if isinstance(data, list):
                for verts in data:
                    verts[:, 0] = np.vectorize(temp.get)(verts[:, 0])
            else:
                data[:, 0] = np.vectorize(temp.get)(data[:, 0])
            layer.data = data
        layer.metadata.update(self._images_meta)

    def on_insert(self, event):
        layer = event.source[-1]
        if isinstance(layer, Image):
            paths = layer.metadata.get("paths")
            if paths is None:  # Then it's a video file
                for widget in self._video_group:
                    widget.setEnabled(True)
            # Store the metadata and pass them on to the other layers
            self._images_meta.update(
                {
                    "paths": paths,
                    "shape": layer.level_shapes[0],
                    "root": layer.metadata["root"],
                    "name": layer.name,
                }
            )
            # FIXME Ensure the images are always underneath the other layers
            # self.viewer.layers.selection = []
            # if (ind := event.index) != 0:
            #     order = list(range(len(self.viewer.layers)))
            #     order.remove(ind)
            #     new_order = [ind] + order
            #     self.viewer.layers.move_multiple(new_order)
            # if (ind := event.index) != 0:
            #     self.viewer.layers.move_selected(ind, 0)
        elif isinstance(layer, Points):
            store = keypoints.KeypointStore(self.viewer, layer)
            self._stores[layer] = store
            # TODO Set default dir of the save file dialog
            if root := layer.metadata.get("root"):
                update_save_history(root)
            layer.metadata["controls"] = self
            layer.text.visible = False
            layer.bind_key("M", self.cycle_through_label_modes)
            layer.add = MethodType(keypoints._add, store)
            layer.events.add(query_next_frame=Event)
            layer.events.query_next_frame.connect(store._advance_step)
            layer.bind_key("Shift-Right", store._find_first_unlabeled_frame)
            layer.bind_key("Shift-Left", store._find_first_unlabeled_frame)
            self.viewer.dims.events.current_step.connect(
                store.smart_reset,
                position="last",
            )
            store.smart_reset(event=None)
            layer.bind_key("Down", store.next_keypoint, overwrite=True)
            layer.bind_key("Up", store.prev_keypoint, overwrite=True)
            layer.face_color_mode = "cycle"
            if not self._menus:
                self._form_dropdown_menus(store)
            self._images_meta.update(
                {
                    "project": layer.metadata.get("project"),
                }
            )
        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self._remap_frame_indices(layer_)

    def on_remove(self, event):
        layer = event.value
        if isinstance(layer, Points):
            self._stores.pop(layer, None)
            while self._menus:
                menu = self._menus.pop()
                self._layout.removeWidget(menu)
                menu.setParent(None)
                menu.destroy()
        elif isinstance(layer, Image):
            self._images_meta = dict()
            paths = layer.metadata.get("paths")
            if paths is None:
                for widget in self._video_group:
                    widget.setEnabled(False)

    @register_points_action("Change labeling mode")
    def cycle_through_label_modes(self, *args):
        self.label_mode = next(keypoints.LabelMode)

    @property
    def label_mode(self):
        return str(self._label_mode)

    @label_mode.setter
    def label_mode(self, mode: Union[str, keypoints.LabelMode]):
        self._label_mode = keypoints.LabelMode(mode)
        self.viewer.status = self.label_mode
        for btn in self._radio_group.buttons():
            if btn.text() == str(mode):
                btn.setChecked(True)
                break


@Points.bind_key("F")
def toggle_face_color(layer):
    if layer._face.color_properties.name == "id":
        layer.face_color = "label"
        layer.face_color_cycle = layer.metadata["face_color_cycles"]["label"]
    else:
        layer.face_color = "id"
        layer.face_color_cycle = layer.metadata["face_color_cycles"]["id"]
    layer.events.face_color()


@Points.bind_key("E")
def toggle_edge_color(layer):
    # Trick to toggle between 0 and 2
    layer.edge_width = np.bitwise_xor(layer.edge_width, 2)


class DropdownMenu(QComboBox):
    def __init__(self, labels: Sequence[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.addItems(labels)

    def update_to(self, text: str):
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def reset(self):
        self.setCurrentIndex(0)


class KeypointsDropdownMenu(QWidget):
    def __init__(
        self,
        store: keypoints.KeypointStore,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.store = store
        self.store.layer.events.current_properties.connect(self.update_menus)

        # Map individuals to their respective bodyparts
        self.id2label = defaultdict(list)
        for keypoint in store._keypoints:
            label = keypoint.label
            id_ = keypoint.id
            if label not in self.id2label[id_]:
                self.id2label[id_].append(label)

        self.menus = dict()
        if store.ids[0]:
            menu = create_dropdown_menu(store, list(self.id2label), "id")
            menu.currentTextChanged.connect(self.refresh_label_menu)
            self.menus["id"] = menu
        self.menus["label"] = create_dropdown_menu(
            store, self.id2label[store.ids[0]], "label"
        )
        layout1 = QVBoxLayout()
        layout1.addStretch(1)
        group_box = QGroupBox("Keypoint selection")
        layout2 = QVBoxLayout()
        for menu in self.menus.values():
            layout2.addWidget(menu)
        group_box.setLayout(layout2)
        layout1.addWidget(group_box)
        self.setLayout(layout1)

    def update_menus(self, event):
        keypoint = self.store.current_keypoint
        for attr, menu in self.menus.items():
            val = getattr(keypoint, attr)
            if menu.currentText() != val:
                menu.update_to(val)

    def refresh_label_menu(self, text: str):
        menu = self.menus["label"]
        menu.blockSignals(True)
        menu.clear()
        menu.addItems(self.id2label[text])
        menu.blockSignals(False)


def create_dropdown_menu(store, items, attr):
    menu = DropdownMenu(items)

    def item_changed(ind):
        current_item = menu.itemText(ind)
        if current_item is not None:
            setattr(store, f"current_{attr}", current_item)

    menu.currentIndexChanged.connect(item_changed)
    return menu
