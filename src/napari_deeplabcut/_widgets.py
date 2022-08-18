from ast import Not
from collections import defaultdict
from re import S
from functools import partial
from xml.etree.ElementInclude import XINCLUDE
import pandas as pd
from types import MethodType
from PIL import Image as pilImage
from typing import Optional, Sequence, Union
from PyQt5.QtCore import pyqtSlot, QObject, QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import napari
from typing import List, Tuple
from PyQt5 import QtCore, QtWidgets
from napari.types import ImageData
from napari.layers import Image, Points
from napari.layers.points._points_key_bindings import register_points_action
from napari.layers.utils import color_manager
from napari.utils.events import Event
from napari.utils.history import update_save_history, get_save_history
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QProgressBar
)
from napari_deeplabcut.kmeans import read_data
from napari_deeplabcut import keypoints
from napari_deeplabcut.misc import to_os_dir_sep

class Worker(QObject):

    started = pyqtSignal()
    percentageChanged = pyqtSignal(int)
    finished = QtCore.Signal()
    value = pyqtSignal(object)

    def __init__(self, func):
        super().__init__()
        self._percentage = 0
        self.func = func

    def __add__(self, other):
        if isinstance(other, int):
            self._percentage += other
            self.percentageChanged.emit(self._percentage)
            return self
        return super().__add__(other)

    def __lt__(self, other):
        if isinstance(other, int):
            return self._percentage < other
        return super().__lt__(other)

    def run(self):

        self._percentage = 0
        points_cluster , color, names = self.func()
        
        #thread.join()
        #self.data = points_cluster
        self.value.emit((points_cluster,color, names))
        #self.names.emit(names)
        #  QTimer.singleShot(0, self.func) # ????
        self.finished.emit()

    def move_to_separate_thread(func):
        thread = QtCore.QThread()
        worker = Worker(func)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        #self.worker.percentageChanged.connect(self.progress.setValue) # ??? #creo que esto no porque necesita ir abajo
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.deleteLater)

        return worker, thread



def _get_and_try_preferred_reader(
    self, dialog, *args,
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
            'Please select one or more layers to save,'
            '\nor use "Save all layers..."'
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
        self.file_path = str()
        self.viewer.window.qt_viewer._get_and_try_preferred_reader = MethodType(
            _get_and_try_preferred_reader, self.viewer.window.qt_viewer,)

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
        
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()
        self.show()
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        #self.ax.axis('off')
        #self.im = self.ax.imshow([[]])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.figure.set_tight_layout(True)
        self.img_refine = [[]]
        self.bdpts_refine = [[]]
        self.bodyparts_name = [[]]
        self.file_relabel = str()
        self.step = []
        self.collect_data = [[]] #empty df????
        #self.fig.subplots_adjust(0.2, 0.2, 0.8, 0.8)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        #self.show()

        
    def initUI(self):
        
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        button1 = QPushButton('cluster pose', self)
        #button.setToolTip('This is an example button')
        button1.move(5,70)
        button1.clicked.connect(self.on_click)
        self.setGeometry(self.left, self.top, self.width, self.height)
        button2 = QPushButton('show img', self)
        #button.setToolTip('This is an example button')
        button2.move(120,70)
        
        button3 = QPushButton('close img', self)
        #button.setToolTip('This is an example button')
        button3.move(230,70)
        button3.clicked.connect(self.on_click_close_img)      
        button2.clicked.connect(self.on_click_show_img)   
        #self.setWindowTitle('Progress Bar')
        self.progress = QProgressBar(self)
        self.progress.setGeometry(10,120, 300, 25)
        self.progress.setMaximum(100)
        self.show()

        return button2, button3

    def _plot(self,input):
        points_cluster ,color, names = input
        x = list(points_cluster[0])
        y = list(points_cluster[1])
        points_cluster1 = np.column_stack((y,x))
        color2 = [(i + 1)/(max(color)+1) for i in color]
        #print(color2)
        dict_prop_points = {'colorn':color2,'frame' : names}
        clust_layer = self.viewer.add_points(points_cluster1, size=8 , features=dict_prop_points, face_color='colorn',face_colormap = 'plasma',name='cluster',) 
        self.viewer.window.add_dock_widget(self.canvas,name = 'frames')
        clust_layer.mode = 'select'
        df = pd.read_hdf(self.file_path)
        df2 = df.reset_index()
        df = df.dropna()
        
        
        self.collect_data = df
        self.viewer.layers[0].visible = False #collect

        @clust_layer.mouse_drag_callbacks.append
        def get_event(clust_layer,event):
            print("click")
            inds = list(clust_layer.selected_data)
            if len(inds) == 1:
                ind = inds[0]
                filename = clust_layer.properties['frame'][ind]
                print(filename)
                self.file_relabel = filename
                path = self.file_path.split('training-datasets')[0] + filename # the user is going to use the h5 from training no?
                im = pilImage.open(path)
                bdpts = df.loc[filename].values
        
                
                self.step = df2.index[df2['index']==str(filename)].to_list()
                print(self.step)

                self.img_refine = np.array(im)
                xbdpts = bdpts[::2]
                ybdpts = bdpts[1::2]
                self.ax.clear()
                self.ax.set_xlim(0, np.array(im).shape[1])
                self.ax.set_ylim(0, np.array(im).shape[0])      
                #self.im.set_data #FIX!
                self.ax.imshow(im)
                self.ax.scatter(xbdpts,ybdpts)
                self.ax.invert_yaxis()
                self.canvas.draw()
      

    @pyqtSlot()
    def on_click(self):
        
        filename_path = list(self.viewer.layers[0]._source)[0][1]
        self.file_path = filename_path.replace("\\", "/")  #work in other os?
        print(list(self.viewer.layers[0]._source)[0][1])
        
        
        func = partial(read_data, self.file_path)

        self.worker, self.thread = Worker.move_to_separate_thread(func)
        #self.thread.finished.connect(self._show_success_message)
        self.worker.percentageChanged.connect(self.progress.setValue) # ?????
        self.worker.value.connect(self._plot)
        self.thread.start() #add progress bar!


    @pyqtSlot()
    def on_click_show_img(self):
        self.viewer.layers[0].visible = True #collect
        self.viewer.layers[1].visible = False #cluster
        self.viewer.dims.set_current_step(0,self.step[0])
        self.viewer.add_image(self.img_refine, name = 'image refine label')
        self.viewer.layers.move_selected(0,2)

    @pyqtSlot()
    def on_click_close_img(self):
        self.viewer.layers.remove('image refine label')
        self.viewer.layers.move_selected(0,1)
        #self.viewer.layers.remove('refine label')
        self.viewer.layers[0].visible = False
        self.viewer.layers[1].visible = True
           

    def _form_dropdown_menus(self, store):
        menu = KeypointsDropdownMenu(store)
        self._menus.append(menu)
        layout = QVBoxLayout()
        layout.addWidget(menu)
        self._layout.addLayout(layout)

    def _form_mode_radio_buttons(self):
        layout1 = QVBoxLayout()
        title = QLabel("Labeling mode")
        layout1.addWidget(title)
        layout2 = QHBoxLayout()
        group = QButtonGroup(self)

        for i, mode in enumerate(keypoints.LabelMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            btn.setToolTip(keypoints.TOOLTIPS[mode])
            group.addButton(btn, i)
            layout2.addWidget(btn)
        group.button(1).setChecked(True)
        layout1.addLayout(layout2)
        self._layout.addLayout(layout1)

        def _func():
            self.label_mode = group.checkedButton().text()

        group.buttonClicked.connect(_func)
        return group

    def _remap_frame_indices(self, layer):
        if "paths" not in self._images_meta:
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
            #print(data)
            if isinstance(data, list):
                for verts in data:
                    verts[:, 0] = np.vectorize(temp.get)(verts[:, 0])
            else:
                data[:, 0] = np.vectorize(temp.get)(data[:, 0])
            layer.data = data
        layer.metadata.update(self._images_meta)

    def on_insert(self, event):
        layer = event.source[-1]
        print(layer)
        if str(layer) != 'cluster' and str(layer) != 'image refine label' :
            if isinstance(layer, Image):
                paths = layer.metadata.get("paths")
                if paths is None:
                    return
                # Store the metadata and pass them on to the other layers
                self._images_meta.update(
                    {
                        "paths": paths,
                        "shape": layer.level_shapes[0],
                        "root": layer.metadata["root"],
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
                print(store)
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
        layout = QVBoxLayout()
        title = QLabel("Keypoint selection")
        layout.addWidget(title)
        for menu in self.menus.values():
            layout.addWidget(menu)
        layout.addStretch(1)
        self.setLayout(layout)

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

