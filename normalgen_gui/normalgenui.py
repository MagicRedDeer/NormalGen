import sys
import os
import cv2

from PySide2 import QtWidgets, QtUiTools, QtCore, QtGui

import normalgen

dirname = os.path.dirname(__file__)


def loadUI(filename, parent=None):
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(filename)
    uifile.open(QtCore.QFile.ReadOnly)
    ui = loader.load(uifile, parent)
    uifile.close()
    return ui


def convertToPixmap(cvImg):
    height, width, channel = cvImg.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(
            cvImg.data, width, height, bytesPerLine,
            QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap(qImg)


class NormalGenDialog(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.dialog = loadUI(os.path.join(dirname, 'normalgenui.ui'))
        self.setupUI()
        self.image = None
        self.hmap = None
        self.normalmap = None
        self.aomap = None
        self.maps = {
                'AO': 'aomap',
                'NMAP': 'normalmap',
                'HMAP': 'hmap',
                'ORIG': 'image'}
        self._output_filename = None

    def setupUI(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.dialog)
        self.setLayout(self.layout)
        self.setWindowTitle('NormalGen')
        self.dialog.load_image_button.clicked.connect(self.load_image)
        self.dialog.generate_normal_button.clicked.connect(
                self.generate_normals)
        self.dialog.generate_ao_button.clicked.connect(
                self.generate_ao)
        self.dialog.generate_all_button.clicked.connect(
                self.generate_ao)
        self.dialog.close_button.clicked.connect(
                self.deleteLater)
        self.connect_sliders_and_spinboxes()
        self.dialog.save_aomap_button.clicked.connect(self.save_aomap)
        self.dialog.save_normalmap_button.clicked.connect(self.save_normalmap)
        self.dialog.save_all_button.clicked.connect(self.save_all)

    def connect_slider_to_spinbox(self, slider, spinbox):
        if isinstance(spinbox, QtWidgets.QSpinBox):
            spinbox.valueChanged[int].connect(slider.setValue)
            slider.valueChanged[int].connect(spinbox.setValue)
        elif isinstance(spinbox, QtWidgets.QDoubleSpinBox):
            spinbox.valueChanged[float].connect(
                    lambda : slider.setValue(spinbox.value()*10))
            slider.valueChanged[int].connect(
                    lambda : spinbox.setValue(slider.value()/10))

    def connect_sliders_and_spinboxes(self):
        names = ['normal_strength', 'normal_level',
                 'ao_size', 'ao_hscale', 'ao_xscale', 'ao_yscale',
                 'ao_intensity', 'ao_samples', 'ao_seed']
        for name in names:
            self.connect_slider_to_spinbox(
                    getattr(self.dialog, name + '_slider'),
                    getattr(self.dialog, name + '_spinbox'))

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", os.path.expanduser('~'),
            "Image Files (*.png *.jpg *.bmp)")
        if filename:
            self.filename = filename
            self.read_image()
            self.show_image()

    def get_output_filename(self, postfix=''):
        _, filename = os.path.split(self.filename)
        if postfix:
            fn, ext = os.path.splitext(filename)
            filename = fn + '_' + postfix + ext
        self._output_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Image",
                os.path.join(os.path.expanduser('~'), filename),
                "Image Files (*.png *.jpg *.bmp)")
        return self._output_filename

    def save_image(self, image, output_path=None, postfix=''):
        if image is None:
            return
        if output_path is None:
            output_path = self.get_output_filename(postfix)
        if cv2.imwrite(output_path, image):
            return output_path
        else:
            raise IOError('Could not Write file to %s' % output_path)

    def _map_out_filename(self, output_path, postfix):
        output_dir, output_file = os.path.split(output_path)
        output_file, output_ext = os.path.splitext(output_file)
        return os.path.join(
                output_dir, output_file + '_' + postfix + output_ext)

    def save_aomap(self, *args, output_path=None):
        return self.save_image(self.aomap, output_path, 'AO')

    def save_normalmap(self, *args, output_path=None):
        return self.save_image(self.normalmap, output_path, 'NMAP')

    def save_all(self, output_path=None):
        output_path = self.get_output_filename()

        mapfilenames = {}
        ask = False
        for postfix, mapname in self.maps.items():
            image = getattr(self, mapname, None)
            if image is not None:
                mapfilename = self._map_out_filename(output_path, postfix)
                mapfilenames[mapname] = mapfilename
                if os.path.isfile(mapfilename):
                    ask = True

        if not mapfilenames:
            return

        if ask:
            if QtWidgets.QMessageBox.question(
                    self,
                    'File Already Exists',
                    'One or more of the maps already exist!'
                    ' Overwrite?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    ) == QtWidgets.QMessageBox.No:
                return

        try:
            for mapname, mapfilename in mapfilenames.items():
                image = getattr(self, mapname, None)
                self.save_image(image, mapfilename)
        except IOError as error:
            QtWidgets.QMessageBox.critical(
                    self, str(error), QtWidgets.QMessageBox.Abort)
            return

        return mapfilenames.values()

    def read_image(self):
        self.empty_maps()
        self.image = normalgen.load_image(self.filename)
        self.hmap = normalgen.makeGray(self.image)

    def show_image(self):
        pixmap = convertToPixmap(self.image)
        scaledPixmap = pixmap.scaled(
                self.dialog.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.dialog.image_label.setPixmap(scaledPixmap)

    def empty_maps(self):
        self.normalmap = None
        self.aomap = None
        self.dialog.normal_image_label.clear()
        self.dialog.ao_image_label.clear()

    def show_normalmap(self):
        if self.normalmap is None:
            return
        pixmap = convertToPixmap(self.normalmap)
        scaledPixmap = pixmap.scaled(
                self.dialog.normal_image_label.size(),
                QtCore.Qt.KeepAspectRatio)
        self.dialog.normal_image_label.setPixmap(scaledPixmap)
        self.dialog.output_tab_widget.setCurrentIndex(0)

    def show_aomap(self):
        if self.aomap is None:
            return
        pixmap = convertToPixmap(self.aomap)
        scaledPixmap = pixmap.scaled(
                self.dialog.ao_image_label.size(),
                QtCore.Qt.KeepAspectRatio)
        self.dialog.ao_image_label.setPixmap(scaledPixmap)
        self.dialog.output_tab_widget.setCurrentIndex(1)

    def generate_normals(self):
        self.empty_maps()
        if self.image is None:
            raise ValueError('No Image Loaded')
        self.normalmap = normalgen.generateNormals(
                self.hmap,
                method=self.dialog.normal_method_combobox.currentText(),
                strength=self.dialog.normal_strength_spinbox.value(),
                level=self.dialog.normal_level_spinbox.value())
        self.show_normalmap()

    def generate_ao(self):
        if self.normalmap is None:
            self.generate_normals()
        self.aomap = normalgen.generateAmbientOcclusion(
                self.hmap, self.normalmap,
                size=self.dialog.ao_size_spinbox.value(),
                height_scaling=self.dialog.ao_hscale_spinbox.value(),
                scale=(self.dialog.ao_xscale_spinbox.value(),
                       self.dialog.ao_yscale_spinbox.value()),
                intensity=self.dialog.ao_intensity_spinbox.value(),
                max_samples=self.dialog.ao_samples_spinbox.value(),
                seed=self.dialog.ao_seed_spinbox)
        self.show_aomap()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = NormalGenDialog()
    main_window.show()
    sys.exit(app.exec_())
